# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local review reproductions for 96ba7b3; intentionally RED, not regression baselines.

Run explicitly; this filename is not automatically discovered as a test module:
  PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES='' \
    python/aisimulate/.venv/bin/python -B -m pytest -o addopts='' \
    -p no:cacheprovider docs/glm5_next_adversarial_repros.py

Alternate checkpoint fixtures patch only config loading, never estimator math.
The assertions express safe/consistent public behavior rather than pinning the
implementation's current answers. Profile fixes may require adapting the tests
to the final typed API. See glm5-next-adversarial-review.md for source evidence.
No GPU, serving-framework installation, weights, or network access is required.
The fixed-KV-pool control records a withdrawn OOM allegation: that explicit pool
is not necessarily the complete hybrid-state memory budget. Decode comparisons
also expose an inherited telemetry boundary mismatch, not exclusively new code.
"""

import pickle
from copy import deepcopy

import pytest

from aisimulate_core.sdk import (
    ForwardPassPerfModelConfig,
    RustForwardPassPerfModel,
    models,
    utils,
)
from aisimulate_core.sdk.config import ModelConfig
from aisimulate_core.sdk.engine import EngineHandle
from aisimulate_core.sdk.memory import estimate_kv_cache

MODEL = "zai-org/GLM-5.3-Flash"


def model(*, tp=4, language_only=True):
    return models.get_model(
        MODEL,
        ModelConfig(tp_size=tp, moe_tp_size=tp, moe_ep_size=1, language_only=language_only),
        "vllm",
    )


def canonical(**changes):
    config = {
        "model": MODEL,
        "system": "h200_sxm",
        "backend": "vllm",
        "backend_version": "0.24.0",
        "worker_type": "aggregated",
        "tp": 4,
        "moe_tp_size": 4,
        "moe_ep_size": 1,
        "database_mode": "SOL",
        "estimation_mode": "op_level",
        **changes,
    }
    return RustForwardPassPerfModel.best_available(ForwardPassPerfModelConfig(**config))


@pytest.fixture
def checkpoint(monkeypatch):
    """Present alternate metadata through the normal parse/build paths."""

    def edit_checkpoint(edit):
        raw = deepcopy(utils._load_pre_downloaded_hf_config(MODEL))
        edit(raw)
        utils._attach_inferred_quant_fields(raw)
        utils.get_model_config_from_model_path.cache_clear()
        models._get_model_info.cache_clear()
        monkeypatch.setattr(utils, "_load_model_config_from_model_path", lambda _: raw)

    yield edit_checkpoint
    utils.get_model_config_from_model_path.cache_clear()
    models._get_model_info.cache_clear()


def test_control_native_h200_baseline_executes():
    estimator = canonical()
    assert estimator.diagnostics()["readiness"] == "ready"
    assert estimator.static_phase_latency(batch_size=1, input_tokens=128, output_tokens=2, prefill=False) > 0


def test_control_fixed_kv_pool_is_not_the_entire_hybrid_memory_budget():
    from aisimulate.compiler import prediction_to_replay_spec
    from aisimulate.config import CorePredictionConfig
    from aisimulate.runner import EngineReplayRunnerFactory

    cache_budget = 1_488_256
    assert cache_budget < model().get_kvcache_bytes_per_sequence(34)
    config = CorePredictionConfig.model_validate(
        {
            "engine": {
                "model": MODEL,
                "hardware": "h200_sxm",
                "backend": "vllm",
                "backend_version": "0.24.0",
                "context_length": 128,
                "database_mode": "SOL",
                "estimation_mode": "op_level",
                "workers": {
                    "aggregated": {
                        "parallelism": {"tensor": 4, "moe_tensor": 4, "moe_expert": 1},
                        "scheduler": {"max_sequences": 1, "max_batched_tokens": 128},
                        "kv_cache": {
                            "block_size": 128,
                            "prefix_caching": False,
                            "bytes_per_token": 11_627,
                            "capacity": {"type": "fixed", "bytes": cache_budget},
                        },
                    }
                },
            },
            "traffic": {
                "source": {"type": "synthetic", "input_tokens": 32, "output_tokens": 2},
                "load": {"type": "concurrency", "concurrency": 1},
                "stop": {"requests": 1},
            },
        }
    )
    runner = EngineReplayRunnerFactory().create(0)
    try:
        spec = prediction_to_replay_spec(config)
        result = runner.run(spec)
    finally:
        runner.close()
    # With state_cache absent, the explicitly supplied pool is token KV only.
    # A successful request does not prove omitted live state or impossible GPU
    # residency; state can be provisioned outside this token pool. Automatic
    # capacity already reserves live state through get_kvcache_batch_capacity.
    assert spec.backend_deployment.agg_engine_args.get("state_cache") is None
    assert result.metrics["completed_requests"] == 1


@pytest.mark.parametrize("field", ["ignored_layers", "modules_to_not_convert"])
@pytest.mark.parametrize("prefix", ["language_model.model.layers", "model.language_model.layers"])
def test_f2_effective_expert_exclusions_must_not_be_priced_as_fp8(checkpoint, field, prefix):
    def edit(raw):
        quant = raw["quantization_config"]
        quant[field] = quant["modules_to_not_convert"] + [f"{prefix}.{i}.mlp.experts" for i in range(3, 45)]

    checkpoint(edit)
    with pytest.raises(ValueError, match="FFN|excl|unsupported"):
        estimate_kv_cache(
            MODEL,
            "h200_sxm",
            "vllm",
            backend_version="0.24.0",
            tp_size=4,
            moe_tp_size=4,
            moe_ep_size=1,
            max_num_tokens=2048,
            max_batch_size=16,
            memory_fraction_kind="of_total",
            memory_fraction_value=0.9,
            allow_naive_fallback=False,
        )


@pytest.mark.parametrize("past_kv", [0, 3, 2048])
def test_f3_canonical_decode_agrees_with_same_workload_static_decode(past_kv):
    estimator = canonical()
    batch = 64
    telemetry = estimator.estimate_forward_pass_time_ms(
        {
            "scheduled_requests": {
                "num_decode_requests": batch,
                "sum_decode_kv_tokens": batch * past_kv,
            },
        }
    )
    # The telemetry field is past KV, excluding the current token. Static decode
    # at context_length=past_kv adds the current token itself.
    rows = estimator.static_phase_diagnostics(batch_size=batch, context_length=past_kv, prefill=False)
    expected = sum(row["latency_ms"] for row in rows)
    assert telemetry == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_control_general_decode_boundary_mismatch_predates_flash():
    estimator = canonical(model="Qwen/Qwen3-32B")
    actual = estimator.estimate_forward_pass_time_ms(
        {
            "scheduled_requests": {
                "num_decode_requests": 64,
                "sum_decode_kv_tokens": 64 * 2048,
            },
        }
    )
    same_workload = sum(
        row["latency_ms"]
        for row in estimator.static_phase_diagnostics(
            batch_size=64,
            context_length=2048,
            prefill=False,
        )
    )
    shifted = sum(
        row["latency_ms"]
        for row in estimator.static_phase_diagnostics(
            batch_size=64,
            context_length=2047,
            prefill=False,
        )
    )
    # This control deliberately records an inherited defect, not desired API
    # behavior. Update it when the shared boundary is corrected. The Flash-only
    # amplification is its KDA zero-s guard, which drops fixed recurrent work.
    assert actual == pytest.approx(shifted, rel=1e-12)
    assert actual != pytest.approx(same_workload, rel=1e-12)


def test_f4_language_only_false_is_not_silently_reinterpreted():
    language_only_weights = model(language_only=True).get_resident_weights_bytes()
    try:
        with_tower = model(language_only=False)
    except ValueError as error:
        assert any(word in str(error).lower() for word in ("language", "vision", "profile", "text-only"))
        return
    assert with_tower.get_resident_weights_bytes() > language_only_weights


@pytest.mark.parametrize(
    "field,value",
    [
        ("num_attention_heads", 65),
        ("layer_types", []),
        ("index_head_dim", 64),
    ],
)
def test_f5_invalid_checkpoint_geometry_must_not_trigger_estimator_fallback(checkpoint, field, value):
    checkpoint(lambda raw: raw["text_config"].update({field: value}))
    with pytest.raises(ValueError, match="invalid|must|requires|GLM5NEXT"):
        canonical(estimation_mode="auto", fallback_policy="deny")


@pytest.mark.parametrize(
    "field,value",
    [
        ("activation_scheme", "static"),
        ("activation_scheme", "bad"),
        ("store_dtype", "mxfp4"),
    ],
)
def test_f6_non_native_fp8_execution_metadata_is_rejected(checkpoint, field, value):
    checkpoint(lambda raw: raw["quantization_config"].update({field: value}))
    with pytest.raises(ValueError, match="activation|storage|dtype|native|unsupported|GLM5NEXT"):
        canonical()


def test_f7_equivalent_nested_quantization_uses_the_shared_resolver(checkpoint):
    def edit(raw):
        raw["text_config"]["quantization_config"] = raw.pop("quantization_config")

    checkpoint(edit)
    assert utils.get_model_config_from_model_path(MODEL)["raw_config"]["quant_algo"] == "fp8_block"
    assert canonical().diagnostics()["readiness"] == "ready"


def test_f8_nondefault_attention_imbalance_controls_have_an_effect_or_are_rejected():
    handle = EngineHandle.compile(
        MODEL,
        "h200_sxm",
        "vllm",
        backend_version="0.24.0",
        tp_size=4,
        moe_tp_size=4,
        moe_ep_size=1,
        database_mode="SOL",
    )
    baseline = handle.run_static(batch_size=2, isl=8192, osl=2)
    try:
        corrected = handle.run_static(
            batch_size=2,
            isl=8192,
            osl=2,
            seq_imbalance_correction_scale=2.0,
            gen_seq_imbalance_correction_scale=2.0,
        )
    except ValueError as error:
        assert any(word in str(error).lower() for word in ("imbalance", "scale", "unsupported"))
        return
    assert corrected != baseline, "Both public correction controls were silently ignored"


def test_f9_mandatory_fp32_hardware_metadata_is_checked_before_ready():
    from aisimulate_core.sdk.errors import MissingSystemFlopsError

    with pytest.raises((ValueError, MissingSystemFlopsError), match="fp32|flops|GLM5NEXT"):
        canonical(system="h100_pcie", tp=8, moe_tp_size=8)


@pytest.mark.parametrize(
    "tp,expected",
    [
        (1, 320_909_761_272),
        (2, 160_788_642_296),
        (4, 80_728_082_808),
        (8, 40_697_803_064),
    ],
)
def test_f10_kda_norm_residency_follows_runtime_parameter_dtype(tp, expected):
    # Pinned vLLM FusedRMSNormGated uses dtype=None, inside the model loader's
    # BF16 default-dtype context. Its FP32 arithmetic does not change storage.
    # Independent tensor inventory: 34 * 128 output-norm weights use 2 B,
    # not 4 B. The discrepancy is 8,704 bytes/rank for every TP size.
    assert model(tp=tp).get_resident_weights_bytes() == expected


def test_limitation_complete_model_pickle_includes_the_fourth_native_op():
    # Compatibility limitation, not a demonstrated current production caller.
    current = model()
    restored = pickle.loads(pickle.dumps(current))
    assert restored.get_resident_weights_bytes() == current.get_resident_weights_bytes()
