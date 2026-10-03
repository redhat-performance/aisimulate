# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native review repairs: raw coordinates, fail-closed controls and serialization.

Requires a coordinator-rebuilt extension; no GPU or downloaded weights needed.
No frozen timing constants: compare identical scheduled work across public APIs.
"""

import json
import pickle
from copy import deepcopy

import pytest

import aisimulate_core
from aisimulate_core.sdk import ForwardPassPerfModelConfig, RustForwardPassPerfModel, models
from aisimulate_core.sdk.config import ModelConfig
from aisimulate_core.sdk.engine import EngineHandle
from aisimulate_core.sdk.errors import InvalidEngineConfigurationError, MissingSystemFlopsError
from aisimulate_core.sdk.operations import (
    Glm5NextFp32Linear,
    Glm5NextKDA,
    Glm5NextMHC,
    Glm5NextSparseAttention,
)

pytestmark = pytest.mark.unit
MODEL = "zai-org/GLM-5.3-Flash"


def config(**changes):
    return ForwardPassPerfModelConfig(
        **{
            "model": MODEL,
            "system": "h200_sxm",
            "backend": "vllm",
            "backend_version": "0.24.0",
            "worker_type": "aggregated",
            "tp": 4,
            "moe_tp_size": 4,
            "moe_ep_size": 1,
            "estimation_mode": "op_level",
            "database_mode": "SOL",
            **changes,
        }
    )


@pytest.fixture(scope="module")
def handle():
    return EngineHandle.compile(
        MODEL,
        "h200_sxm",
        "vllm",
        backend_version="0.24.0",
        tp_size=4,
        moe_tp_size=4,
        moe_ep_size=1,
        database_mode="SOL",
    )


def flash_model():
    return models.get_model(MODEL, ModelConfig(tp_size=4, moe_tp_size=4, moe_ep_size=1), "vllm")


@pytest.mark.parametrize("model", [MODEL, "Qwen/Qwen3-32B"])
@pytest.mark.parametrize("past", [0, 3, 2048])
def test_raw_canonical_decode_matches_same_workload_static(model, past):
    estimator = RustForwardPassPerfModel.best_available(config(model=model))
    batch = 64
    raw = estimator.estimate_forward_pass_time_ms(
        {"scheduled_requests": {"num_decode_requests": batch, "sum_decode_kv_tokens": batch * past}}
    )
    rows = estimator.static_phase_diagnostics(batch_size=batch, context_length=past, prefill=False)
    assert raw == pytest.approx(sum(row["latency_ms"] for row in rows), rel=1e-12, abs=1e-12)


def test_raw_mixed_zero_past_executes_both_attention_phases(handle):
    estimator = RustForwardPassPerfModel.best_available(config())
    actual = estimator.estimate_forward_pass_time_ms(
        {
            "scheduled_requests": {
                "num_prefill_requests": 2,
                "sum_prefill_tokens": 6,
                "num_decode_requests": 2,
                "sum_decode_kv_tokens": 0,
            }
        }
    )
    model = flash_model()
    shared = [json.loads(op._spec_json()) for op in model.context_ops if op._name != "context_attention"]
    prefill = [json.loads(op._spec_json()) for op in model.context_ops if op._name == "context_attention"]
    decode = [json.loads(op._spec_json()) for op in model.generation_ops if op._name == "generation_attention"]
    expected = 0.0
    for ops, context, batch, length in [(shared, True, 1, 8), (prefill, True, 2, 3), (decode, False, 2, 1)]:
        rows = handle.evaluate_ops_json(json.dumps(ops), is_context=context, batch_size=batch, s=length)
        expected += sum(row[1] for row in rows)
        if ops is decode:
            assert rows and all(row[1] > 0 for row in rows)
    assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("mixed", [False, True])
def test_raw_decode_overflow_is_rejected_before_query(mixed):
    estimator = RustForwardPassPerfModel.best_available(config())
    scheduled = {"num_decode_requests": 1, "sum_decode_kv_tokens": 2**32 - 1}
    if mixed:
        scheduled.update(num_prefill_requests=1, sum_prefill_tokens=1)
    with pytest.raises(ValueError, match="decode.*u32"):
        estimator.estimate_forward_pass_time_ms({"scheduled_requests": scheduled})


def test_empty_decode_phase_has_no_synthetic_request(handle):
    estimator = RustForwardPassPerfModel.best_available(config())
    assert estimator.estimate_forward_pass_time_ms({"scheduled_requests": {}}) == 0
    assert (
        estimator.estimate_forward_pass_time_ms(
            {"scheduled_requests": {"num_prefill_requests": 1, "sum_prefill_tokens": 0, "sum_prefill_kv_tokens": 3}}
        )
        == 0
    )
    assert handle.run_static(batch_size=1, isl=2**32 - 1, osl=1, mode="static_gen") == (0, 0, 0)
    assert handle.run_static(batch_size=0, isl=2**32 - 1, osl=2, mode="static_gen") == (0, 0, 0)
    with pytest.raises(ValueError, match="inclusive decode"):
        handle.run_static(batch_size=1, isl=2**32 - 1, osl=2, mode="static_gen")


def test_empty_decode_diagnostics_do_not_require_an_inclusive_coordinate():
    estimator = RustForwardPassPerfModel.best_available(config())
    assert estimator.static_phase_diagnostics(batch_size=0, context_length=2**32 - 1, prefill=False) == []


@pytest.mark.parametrize("stride", [1, 32, 2**32 - 1])
def test_static_decode_checks_unsampled_trajectory_overflow(handle, stride):
    # The first step fits, but the repeated second step would need 2**32.
    with pytest.raises(ValueError, match="inclusive decode"):
        handle.run_static(batch_size=1, isl=2**32 - 2, osl=3, mode="static_gen", stride=stride)


def test_low_level_kda_zero_inclusive_decode_is_not_a_silent_noop(handle):
    decode = Glm5NextKDA("generation_attention", 1, False, 16)
    ops = json.dumps([json.loads(decode._spec_json())])
    with pytest.raises(ValueError, match="positive inclusive"):
        handle.evaluate_ops_json(ops, is_context=False, batch_size=2, s=0)
    assert handle.evaluate_ops_json(ops, is_context=False, batch_size=0, s=0)[0][1] == 0
    prefill = Glm5NextKDA("context_attention", 1, True, 16)
    assert (
        handle.evaluate_ops_json(json.dumps([json.loads(prefill._spec_json())]), is_context=True, batch_size=2, s=0)[0][
            1
        ]
        == 0
    )


def test_low_level_context_token_overflow_is_typed_and_explicit_x_is_honored(handle):
    op = Glm5NextKDA("context_attention", 1, True, 16)
    ops = json.dumps([json.loads(op._spec_json())])
    with pytest.raises(ValueError, match="token count.*u32"):
        handle.evaluate_ops_json(ops, is_context=True, batch_size=2, s=2**32 - 1)
    assert handle.evaluate_ops_json(ops, is_context=True, batch_size=2, s=2**32 - 1, x=1)[0][1] > 0


@pytest.mark.parametrize(
    "field,mode",
    [("seq_imbalance_correction_scale", "static_ctx"), ("gen_seq_imbalance_correction_scale", "static_gen")],
)
@pytest.mark.parametrize("scale", [2.0, 0.0, -1.0, float("nan"), float("inf")])
def test_sparse_nondefault_imbalance_is_rejected(handle, field, mode, scale):
    with pytest.raises(ValueError, match="imbalance.*unsupported"):
        handle.run_static(batch_size=2, isl=8192, osl=2, mode=mode, **{field: scale})


@pytest.mark.parametrize(
    "field,mode",
    [("seq_imbalance_correction_scale", "static_gen"), ("gen_seq_imbalance_correction_scale", "static_ctx")],
)
def test_sparse_imbalance_does_not_reject_an_unexecuted_phase(handle, field, mode):
    assert handle.run_static(batch_size=2, isl=8192, osl=2, mode=mode, **{field: 2}) == handle.run_static(
        batch_size=2, isl=8192, osl=2, mode=mode
    )


@pytest.mark.parametrize("mode,policy", [("op_level", "deny"), ("auto", "deny"), ("auto", "allow")])
def test_mandatory_fp32_rate_fails_during_canonical_construction(mode, policy):
    with pytest.raises(MissingSystemFlopsError, match="fp32_flops"):
        RustForwardPassPerfModel.best_available(
            config(system="h100_pcie", tp=8, moe_tp_size=8, estimation_mode=mode, fallback_policy=policy)
        )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: Glm5NextSparseAttention("sparse", 1, True, 16, 512, 256, 256, 32, 64, 2048, 4),
        lambda: Glm5NextMHC("mhc", -1, 4096, 4, 20),
        lambda: Glm5NextKDA("kda", 1, True, 0),
        lambda: Glm5NextFp32Linear("linear", 1, 0, 4096),
    ],
)
def test_native_invalid_geometry_retains_compile_bridge_exception(factory):
    with pytest.raises(InvalidEngineConfigurationError, match="GLM5NEXT"):
        factory()


@pytest.mark.parametrize(
    "mode,policy", [("auto", "deny"), ("auto", "allow"), ("op_level", "deny"), ("op_level", "allow")]
)
@pytest.mark.parametrize("dimension", [0, -1, 2**32, None])
def test_invalid_native_json_geometry_cannot_trigger_canonical_fallback(monkeypatch, mode, policy, dimension):
    # Inject at the native JSON boundary during compilation, not in checkpoint
    # parsing: semantic and serde geometry errors must both remain hard errors.
    from aisimulate_core.sdk import engine

    def invalid_model(*args, **kwargs):
        return aisimulate_core.op_from_spec_json(
            json.dumps({"Glm5NextFp32Linear": {"name": "invalid", "scale_factor": 1, "n": dimension, "k": 4096}})
        )

    monkeypatch.setattr(engine, "get_model", invalid_model)
    with pytest.raises(ValueError, match="GLM5NEXT|invalid opspec"):
        RustForwardPassPerfModel.best_available(config(estimation_mode=mode, fallback_policy=policy))


@pytest.mark.parametrize("dimension", [0, -1, 2**32, None])
def test_native_json_invalid_geometry_is_typed(dimension):
    with pytest.raises(InvalidEngineConfigurationError):
        aisimulate_core.op_from_spec_json(
            json.dumps({"Glm5NextFp32Linear": {"name": "invalid", "scale_factor": 1, "n": dimension, "k": 4096}})
        )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: Glm5NextSparseAttention("sparse", 1, True, 16, 512, 256, 256, 32, 128, 2048, 4),
        lambda: Glm5NextMHC("mhc", 1, 4096, 4, 20),
        lambda: Glm5NextKDA("kda", 1, True, 16),
        lambda: Glm5NextFp32Linear("linear", 1, 32, 4096),
    ],
)
@pytest.mark.parametrize("scale", [-1, float("nan"), float("inf")])
def test_glm_native_scale_mutation_is_validated_before_serialization(factory, scale):
    op = factory()
    original = op._spec_json()
    with pytest.raises(InvalidEngineConfigurationError, match="scale_factor"):
        op._scale_factor = scale
    assert op._spec_json() == original


@pytest.mark.parametrize("protocol", range(2, pickle.HIGHEST_PROTOCOL + 1))
def test_fourth_op_native_json_pickle_deepcopy_and_live_fields(protocol):
    op = Glm5NextFp32Linear("index_head_weight", 11, 32, 4096)
    op._name = "renamed"
    op._scale_factor = 3
    rebuilt = aisimulate_core.op_from_spec_json(op._spec_json())
    assert isinstance(rebuilt, aisimulate_core._native.Glm5NextFp32Linear)
    for candidate in [
        rebuilt,
        pickle.loads(pickle.dumps(op, protocol=protocol)),
        pickle.loads(pickle.dumps(rebuilt, protocol=protocol)),
        deepcopy(op),
    ]:
        assert candidate._spec_json() == op._spec_json()
        assert candidate._n == 32 and candidate._k == 4096
        assert candidate.get_weights() == 3 * 32 * 4096 * 4
        assert candidate._ENGINE_QUERY_SHAPE == "tokens"


def test_complete_flash_model_pickle_and_deepcopy_include_every_native_op():
    model = flash_model()
    assert any("Glm5NextFp32Linear" in op._spec_json() for op in model.context_ops)
    for restored in [pickle.loads(pickle.dumps(model)), deepcopy(model)]:
        assert restored.get_resident_weights_bytes() == model.get_resident_weights_bytes()
        for phase in ["context_ops", "generation_ops"]:
            assert [op._spec_json() for op in getattr(restored, phase)] == [
                op._spec_json() for op in getattr(model, phase)
            ]
