# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent geometry checks and real native H200 execution, not GPU accuracy."""

import json
import math
import pickle

import pytest

from aisimulate_core.sdk import ForwardPassPerfModelConfig, RustForwardPassPerfModel, common
from aisimulate_core.sdk.config import ModelConfig
from aisimulate_core.sdk.models import check_is_moe, get_model
from aisimulate_core.sdk.models.glm5_next import Glm5NextModel
from aisimulate_core.sdk.speculation.base import SpeculationConfig

pytestmark = pytest.mark.unit
MODEL = "zai-org/GLM-5.3-Flash"


def model(tp=4, **changes):
    kwargs = {"tp_size": tp, "moe_tp_size": tp, "moe_ep_size": 1, **changes}
    return get_model(MODEL, ModelConfig(**kwargs), "vllm")


def canonical(mode="SOL", tp=4, **changes):
    return ForwardPassPerfModelConfig(
        **{
            "model": MODEL,
            "system": "h200_sxm",
            "backend": "vllm",
            "worker_type": "aggregated",
            "tp": tp,
            "moe_tp_size": tp,
            "moe_ep_size": 1,
            "backend_version": "0.24.0",
            "estimation_mode": "op_level",
            "database_mode": mode,
            **changes,
        }
    )


def test_graph_geometry_and_runtime_precisions():
    m = model()
    assert isinstance(m, Glm5NextModel)
    assert check_is_moe(MODEL)
    assert m.config.gemm_quant_mode == common.GEMMQuantMode.fp8_block
    assert m.config.moe_quant_mode == common.MoEQuantMode.fp8_block
    assert m.config.kvcache_quant_mode == common.KVCacheQuantMode.bfloat16
    assert m.config.fmha_quant_mode == common.FMHAQuantMode.bfloat16
    assert not m.encoder_ops
    for phase, group in [("context", m.context_ops), ("generation", m.generation_ops)]:
        specs = [json.loads(op._spec_json()) for op in group]
        kda = [s["Glm5NextKda"] for s in specs if "Glm5NextKda" in s]
        sparse = [s["Glm5NextSparseAttention"] for s in specs if "Glm5NextSparseAttention" in s]
        assert kda[0]["scale_factor"] == 34
        assert kda[0]["num_heads"] == 16
        assert sparse[0]["scale_factor"] == 11
        assert sparse[0]["index_n_heads"] == 32  # replicated, NOT 8
        assert sum(s["Glm5NextMhc"]["scale_factor"] for s in specs if "Glm5NextMhc" in s) == 90
        by_name = {op._name: op for op in group}
        assert by_name[f"{phase}_kda_input_gemm"]._n == 6416  # 3*2048 + 16 + 2*128
        assert by_name[f"{phase}_mla_down_gemm"]._n == 2048
        assert by_name[f"{phase}_mla_q_b_gemm"]._n == 4096
        for name, op in by_name.items():
            if any(tag in name for tag in ("_kda_", "_mla_", "_index_")) and "gemm" in name:
                assert op._quant_mode == common.GEMMQuantMode.bfloat16
        assert by_name[f"{phase}_moe"]._scale_factor == 42
        assert by_name[f"{phase}_router_gemm"]._quant_mode == common.GEMMQuantMode.bfloat16
        assert by_name[f"{phase}_dense_gate_up_gemm"]._scale_factor == 3
        assert f"{phase}_moe_pre_dispatch" not in by_name


def test_cache_inventory_from_independent_tensor_shapes():
    m = model()
    # 34 KDA: 16*128*128 FP32 recurrent + 3*16*128*3 BF16 conv.
    # 11 sparse layers: one 4-slot K/gate BF16 tail per request.
    fixed = 34 * (1_048_576 + 36_864) + 11 * 2048
    assert fixed == 36_927_488
    assert m._fixed_state_bytes() == fixed
    assert m.get_kvcache_bytes_per_sequence(0) == 0
    # 11*512 BF16 latent values/token; pooled entry 128 FP8 + 4-byte scale.
    assert m.get_kvcache_bytes_per_sequence(4) == fixed + 45_056 + 1452
    assert m.get_kvcache_bytes_per_sequence(5) == fixed + 56_320 + 2904
    for seq in (1, 3, 4, 5, 2048, 8192, 131072, 1048576):
        budget = m.get_kvcache_bytes_per_sequence(seq)
        assert m.get_kvcache_max_tokens(budget) == seq
        assert m.get_kvcache_max_tokens(budget - 1) == seq - 1
    assert m.get_kvcache_max_tokens(fixed - 1) == 0
    assert m.get_kvcache_max_tokens(-1) == 0
    assert model(8)._fixed_state_bytes() == (fixed - 11 * 2048) / 2 + 11 * 2048
    # Growing cache does not divide by attention TP; only recurrent state does.
    assert m.get_kvcache_bytes_per_sequence(1028) - m.get_kvcache_bytes_per_sequence(1024) == (
        model(8).get_kvcache_bytes_per_sequence(1028) - model(8).get_kvcache_bytes_per_sequence(1024)
    )


def test_batch_capacity_reserves_state_per_scheduler_slot():
    m = model()
    budget = m.get_kvcache_bytes_per_sequence(8192) * 16
    capacities = [m.get_kvcache_batch_capacity(budget, b) for b in (1, 16, 64)]
    assert capacities[0] > capacities[1] > capacities[2]
    assert capacities[1] <= 8192 * 16
    assert m.get_kvcache_batch_capacity(1, 16) == 0
    for bad in (math.inf, math.nan):
        with pytest.raises(ValueError, match="finite"):
            m.get_kvcache_max_tokens(bad)


@pytest.mark.parametrize("tp", [1, 2, 4, 8])
def test_resident_weight_inventory_and_tp_scaling(tp):
    m = model(tp)
    resident = m.get_resident_weights_bytes()
    assert resident > sum(op.get_weights() for op in m.context_ops)
    assert resident > 42 * 288 * 3 * 4096 * 2048 / tp  # all experts, not only top-8
    # Independently counted TP1 tensors from the pinned vLLM shapes (bytes):
    # embeddings/logits; KDA projections/state params; sparse projections/params;
    # mHC; RMSNorm; routers+bias; dense/shared/routed FFNs and block128 scales.
    inventory = (
        2_537_553_920,
        9_358_540_800,
        27_878_912,
        2_753_901_568,
        141_567_480,
        745_472,
        99_138_816,
        452_984_832,
        110_592,
        1_056_964_608,
        258_048,
        304_405_807_104,
        74_317_824,
    )
    # Replicated: mHC, routers, MLA down-proj/indexer/norms, decoder norms,
    # KDA f_a/g_a and output norm. All other resident tensors shard over TP.
    replicated = sum((141_567_480, 99_138_816, 184_549_376, 170_131_456, 78_848, 745_472, 71_303_168, 17_408))
    assert resident == (sum(inventory) - replicated) / tp + replicated
    if tp > 1:
        # Replicated projections and mHC mean memory does not divide perfectly.
        assert resident > model(1).get_resident_weights_bytes() / tp


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"kvcache_quant_mode": common.KVCacheQuantMode.fp8}, "bfloat16"),
        ({"fmha_quant_mode": common.FMHAQuantMode.fp8}, "bfloat16"),
        ({"gemm_quant_mode": common.GEMMQuantMode.nvfp4}, "fp8_block"),
        ({"moe_quant_mode": common.MoEQuantMode.bfloat16}, "fp8_block"),
        ({"nextn": 1}, "speculative"),
        ({"speculation": SpeculationConfig("ngram", {"depth": 2})}, "speculative"),
        ({"pp_size": 3}, "PP=CP"),
        ({"overwrite_num_layers": 44}, "layer overrides"),
        ({"moe_tp_size": 1, "moe_ep_size": 4}, "EP=1"),
        ({"enable_eplb": True}, "EPLB"),
        ({"moe_comm_backend": {"context": "deepep_ht"}}, "large EP"),
    ],
)
def test_unsupported_settings_fail_loudly(changes, match):
    with pytest.raises((ValueError, NotImplementedError), match=match):
        model(**changes)


@pytest.mark.parametrize("backend", ["sglang", "trtllm"])
def test_unaudited_backends_rejected(backend):
    with pytest.raises(ValueError, match="vLLM"):
        get_model(MODEL, ModelConfig(tp_size=4, moe_tp_size=4, moe_ep_size=1), backend)


def test_native_ops_pickle_and_json_round_trip():
    import aisimulate_core
    from aisimulate_core.sdk import operations

    for op in model().context_ops:
        # Existing base-wrapped op contract is JSON-only (as with V4.1 ops).
        rebuilt = aisimulate_core.op_from_spec_json(op._spec_json())
        assert rebuilt._spec_json() == op._spec_json()
        if isinstance(op, (operations.Glm5NextKDA, operations.Glm5NextSparseAttention, operations.Glm5NextMHC)):
            assert pickle.loads(pickle.dumps(op))._spec_json() == op._spec_json()


@pytest.mark.parametrize("mode", ["SOL", "HYBRID"])
def test_canonical_h200_forward_pass_and_config_reload(mode):
    estimate = RustForwardPassPerfModel.best_available(canonical(mode))
    diag = estimate.diagnostics()
    assert diag["readiness"] == "ready"
    resolved = diag["provenance"]["config"]
    reloaded = RustForwardPassPerfModel.best_available(resolved)
    assert reloaded.diagnostics()["provenance"]["config"] == resolved
    for prefill in (True, False):
        rows = estimate.static_phase_diagnostics(batch_size=2, context_length=4096, prefill=prefill)
        assert rows
        assert sum(row["latency_ms"] for row in rows) > 0
        assert all(math.isfinite(row["latency_ms"]) and row["latency_ms"] >= 0 for row in rows)
    assert "analytic_unvalidated" in json.dumps(estimate.diagnostics())


def test_canonical_silicon_missing_data_is_not_a_false_pass():
    from aisimulate_core.sdk.errors import PerfDataNotAvailableError

    with pytest.raises(PerfDataNotAvailableError, match="GLM5NEXT.*SILICON"):
        RustForwardPassPerfModel.best_available(canonical("SILICON"))


def test_canonical_invalid_quantization_cannot_fallback():
    with pytest.raises(ValueError, match="bfloat16"):
        RustForwardPassPerfModel.best_available(
            canonical(
                "HYBRID",
                estimation_mode="auto",
                fallback_policy="allow",
                kvcache_quant_mode="fp8",
            )
        )


def test_h200_public_memory_path_and_single_gpu_oom():
    from aisimulate_core.sdk.memory import estimate_kv_cache

    kwargs = dict(
        backend_version="0.24.0",
        max_num_tokens=2048,
        max_batch_size=16,
        memory_fraction_kind="of_total",
        memory_fraction_value=0.9,
    )
    result = estimate_kv_cache(MODEL, "h200_sxm", "vllm", tp_size=4, moe_tp_size=4, moe_ep_size=1, **kwargs)
    assert result["source"] == "native"
    assert result["total_kv_size_tokens"] > 0
    assert result["memory_breakdown"]["weights_bytes"] == model().get_resident_weights_bytes()
    with pytest.raises(ValueError, match="no KV budget"):
        estimate_kv_cache(MODEL, "h200_sxm", "vllm", tp_size=1, moe_tp_size=1, moe_ep_size=1, **kwargs)


@pytest.mark.parametrize("changes", [{"image_height": 448}, {"num_image_tokens": 256}, {"video_frames": 1}])
def test_visual_inputs_are_not_silently_ignored(changes):
    from aisimulate_core.sdk.backends.base_backend import BaseBackend
    from aisimulate_core.sdk.config import RuntimeConfig

    runtime = RuntimeConfig(isl=128, **changes)
    with pytest.raises(ValueError, match="text-only"):
        BaseBackend.effective_prefill_isl(MODEL, runtime)
    with pytest.raises(ValueError, match="text-only"):
        BaseBackend._visual_context_tokens(model(), runtime)


def test_python_provenance_preserves_unvalidated_as_worst_tier():
    from aisimulate_core.sdk.operations.util_empirical import worst_provenance

    assert worst_provenance(["xop", "analytic_unvalidated", "silicon"]) == "analytic_unvalidated"


def test_afd_cannot_partition_hybrid_state_even_with_unknown_ops_allowed():
    from aisimulate_core.sdk.afd_partition import (
        AFDPartitionError,
        build_afd_ops_partition,
        validate_afd_model_architecture,
    )

    with pytest.raises(AFDPartitionError, match="GLM5NEXT"):
        validate_afd_model_architecture("Glm5NextForConditionalGeneration")
    with pytest.raises(AFDPartitionError, match="GLM5NEXT"):
        build_afd_ops_partition(model(), allow_unknown_ops=True)


@pytest.mark.parametrize(
    "field,value",
    [
        ("index_kpool_compress", False),
        ("index_kpool_always_select_tail", False),
        ("tie_word_embeddings", True),
        ("hidden_act", "gelu"),
        ("moe_router_dtype", "bfloat16"),
    ],
)
def test_altered_checkpoint_execution_contract_is_rejected(monkeypatch, field, value):
    from copy import deepcopy

    from aisimulate_core.sdk import models

    info = deepcopy(models._get_model_info(MODEL))
    info["raw_config"]["text_config"][field] = value
    monkeypatch.setattr(models, "_get_model_info", lambda _: info)
    with pytest.raises(ValueError, match="native compressed-indexer/KDA/FFN contract"):
        model()


def test_custom_text_ffn_exclusion_is_rejected(monkeypatch):
    from copy import deepcopy

    from aisimulate_core.sdk import models

    info = deepcopy(models._get_model_info(MODEL))
    info["raw_config"]["quantization_config"]["modules_to_not_convert"].append("model.layers.3.mlp.experts")
    monkeypatch.setattr(models, "_get_model_info", lambda _: info)
    with pytest.raises(ValueError, match="custom FFN"):
        model()
