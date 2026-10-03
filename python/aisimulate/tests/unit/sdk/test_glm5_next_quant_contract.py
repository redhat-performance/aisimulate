# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial metadata tests through real public parse/build/canonical paths.

Backend semantics checked against vllm-project/vllm commit
0a30bc3f9ac3cc1a9339e115377a99d32252aed0: fp8.py field selection,
quant_utils.py expert/fused matching, glm5next/common/model.py prefixes.
No runtime imports, weights downloads, graph mocks, or golden updates.
"""

from copy import deepcopy

import pytest

from aisimulate_core.sdk import ForwardPassPerfModelConfig, RustForwardPassPerfModel, common, utils
from aisimulate_core.sdk.config import ModelConfig
from aisimulate_core.sdk.errors import InvalidEngineConfigurationError
from aisimulate_core.sdk.glm5_next import (
    effective_glm5_next_quantization,
    parse_glm5_next_config,
    validate_glm5_next_ffn_exclusions,
)

pytestmark = pytest.mark.unit
MODEL = "zai-org/GLM-5.3-Flash"
PREFIXES = ("language_model.model.layers", "model.language_model.layers")


@pytest.fixture
def checkpoint(monkeypatch):
    from aisimulate_core.sdk import models

    raw = deepcopy(utils._load_pre_downloaded_hf_config(MODEL))
    utils.get_model_config_from_model_path.cache_clear()
    models._get_model_info.cache_clear()

    def load(_):
        return utils._attach_inferred_quant_fields(raw)

    def no_download(*args, **kwargs):
        pytest.fail("GLM metadata regression must remain offline")

    monkeypatch.setattr(utils, "_load_model_config_from_model_path", load)
    monkeypatch.setattr(utils, "_download_hf_json", no_download)
    yield raw
    utils.get_model_config_from_model_path.cache_clear()
    models._get_model_info.cache_clear()


def build_model():
    from aisimulate_core.sdk import models

    return models.get_model(MODEL, ModelConfig(tp_size=4, moe_tp_size=4, moe_ep_size=1), "vllm")


def canonical(mode="op_level", policy="deny"):
    return RustForwardPassPerfModel.best_available(
        ForwardPassPerfModelConfig(
            model=MODEL,
            system="h200_sxm",
            backend="vllm",
            backend_version="0.24.0",
            worker_type="aggregated",
            tp=4,
            moe_tp_size=4,
            moe_ep_size=1,
            database_mode="SOL",
            estimation_mode=mode,
            fallback_policy=policy,
        )
    )


def memory():
    from aisimulate_core.sdk.memory import estimate_kv_cache

    return estimate_kv_cache(
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


@pytest.mark.parametrize("nested", [False, True])
def test_effective_language_quantization_preserves_scope_and_native_baseline(checkpoint, nested):
    if nested:
        checkpoint["text_config"]["quantization_config"] = checkpoint.pop("quantization_config")
    before = deepcopy(checkpoint)
    quant = effective_glm5_next_quantization(checkpoint)
    assert quant is utils._get_language_quantization_config(checkpoint)
    assert utils._infer_quantization_fields(checkpoint) == {"quant_algo": "fp8_block", "quant_dynamic": True}
    assert checkpoint == before
    m = build_model()
    assert m.config.gemm_quant_mode == common.GEMMQuantMode.fp8_block
    assert m.config.moe_quant_mode == common.MoEQuantMode.fp8_block
    assert m.get_resident_weights_bytes() == 80_728_082_808
    assert canonical().diagnostics()["readiness"] == "ready"
    assert memory()["memory_breakdown"]["weights_bytes"] == 80_728_082_808


@pytest.mark.parametrize("nested_quant", [None, [], {}, {"quant_method": "bf16"}, {"quant_method": "fp8"}])
def test_root_quantization_wins_without_merging_nested_metadata(checkpoint, nested_quant):
    checkpoint["text_config"]["quantization_config"] = nested_quant
    assert build_model().get_resident_weights_bytes() == 80_728_082_808


@pytest.mark.parametrize("root_quant", [None, [], {}, {"quant_method": "fp8"}])
def test_invalid_root_cannot_fall_through_to_valid_nested_quantization(checkpoint, root_quant):
    checkpoint["text_config"]["quantization_config"] = checkpoint["quantization_config"]
    checkpoint["quantization_config"] = root_quant
    with pytest.raises(InvalidEngineConfigurationError):
        parse_glm5_next_config(checkpoint)


@pytest.mark.parametrize("field", ["ignored_layers", "modules_to_not_convert"])
@pytest.mark.parametrize("prefix", PREFIXES)
@pytest.mark.parametrize("suffix", ["experts", "experts.0.gate_proj", "shared_experts.down_proj"])
@pytest.mark.parametrize("nested", [False, True])
def test_effective_ffn_exclusions_rejected_before_public_memory(checkpoint, field, prefix, suffix, nested):
    quant = checkpoint["quantization_config"]
    quant[field] = [f"{prefix}.{i}.mlp.{suffix}" for i in range(3, 45)]
    if nested:
        checkpoint["text_config"]["quantization_config"] = checkpoint.pop("quantization_config")
    with pytest.raises(InvalidEngineConfigurationError, match="FFN.*exclusions"):
        build_model()
    with pytest.raises(ValueError, match="FFN.*exclusions"):
        memory()


@pytest.mark.parametrize("prefix", PREFIXES)
@pytest.mark.parametrize(
    "layer,module",
    [
        (0, "gate_up_proj"),
        (0, "gate_proj"),
        (0, "up_proj"),
        (0, "down_proj"),
        (3, "shared_experts.gate_up_proj"),
        (3, "shared_experts.gate_proj"),
        (3, "shared_experts.up_proj"),
    ],
)
def test_fused_and_unfused_ffn_exclusions_are_unsupported(checkpoint, prefix, layer, module):
    checkpoint["quantization_config"]["ignored_layers"] = [f"{prefix}.{layer}.mlp.{module}"]
    with pytest.raises(InvalidEngineConfigurationError, match="FFN"):
        build_model()


@pytest.mark.parametrize("ignored", [[], ["language_model.model.layers.3.mlp.gate"]])
def test_backend_ignored_layers_precedence_not_union(checkpoint, ignored):
    quant = checkpoint["quantization_config"]
    quant["modules_to_not_convert"] = ["language_model.model.layers.3.mlp.experts"]
    quant["ignored_layers"] = ignored
    if ignored:
        assert build_model().get_resident_weights_bytes() == 80_728_082_808
        assert canonical().diagnostics()["readiness"] == "ready"
    else:
        with pytest.raises(InvalidEngineConfigurationError, match="FFN"):
            build_model()


@pytest.mark.parametrize(
    "name",
    [
        "model.layers.3.mlp.experts",  # Not the conditional-generation runtime namespace.
        "language_model.model.layers.*.mlp.experts",  # No wildcard expansion.
        "language_model.model.layers.3.mlp",  # No generic parent-module matching.
        "language_model.model.layers.0.mlp.gate_proj.weight",  # Dense exact matching.
        "language_model.model.layers.45.mlp.experts",  # MTP is not resident.
        "language_model.model.layers.3.mlp.gate",
        "language_model.model.layers.3.mlp.gate.e_score_correction_bias",
        "visual.blocks.0.mlp.down_proj",
    ],
)
def test_ineffective_or_baseline_exclusions_do_not_change_inventory(checkpoint, name):
    checkpoint["quantization_config"]["ignored_layers"] = [name]
    assert build_model().get_resident_weights_bytes() == 80_728_082_808


INVALID_QUANT = (
    [("activation_scheme", value) for value in ("static", "bad", "DYNAMIC", None, True, 1, [], {})]
    + [("quant_method", value) for value in ("bf16", "bad", "mxfp4", None, True, 1, [], {})]
    + [("store_dtype", value) for value in ("mxfp4", True, 1, [], {})]
    + [
        (field, value)
        for field in ("ignored_layers", "modules_to_not_convert")
        for value in (None, "experts", True, 1, {}, [None], [1], [[]])
    ]
)


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("field,value", INVALID_QUANT)
def test_invalid_effective_quant_fields_fail_closed(checkpoint, field, value, nested):
    quant = checkpoint["quantization_config"]
    quant[field] = value
    if nested:
        checkpoint["text_config"]["quantization_config"] = checkpoint.pop("quantization_config")
    with pytest.raises(InvalidEngineConfigurationError, match=field):
        parse_glm5_next_config(checkpoint)
    with pytest.raises(InvalidEngineConfigurationError, match=field):
        build_model()


INVALID_CANONICAL = [
    ("text", "num_attention_heads", 65),
    ("text", "layer_types", []),
    ("text", "index_head_dim", 64),
    ("text", "index_kpool", 1),
    ("text", "index_topk", 2049),
    ("text", "intermediate_size", 12289),
    ("text", "moe_intermediate_size", 2049),
    ("text", "vocab_size", 154881),
    ("text", "hidden_size", 2**32),
    ("linear", "head_dim", 64),
    ("linear", "short_conv_kernel_size", 3),
    ("linear", "num_heads", 65),
    ("quant", "activation_scheme", "static"),
    ("quant", "activation_scheme", "bad"),
    ("quant", "quant_method", "bad"),
    ("quant", "quant_method", "bf16"),
    ("quant", "store_dtype", "mxfp4"),
    ("quant", "ignored_layers", ["language_model.model.layers.3.mlp.experts"]),
    ("quant", "modules_to_not_convert", ["model.language_model.layers.3.mlp.experts.0.down_proj"]),
    ("quant", "ignored_layers", "not-a-list"),
]


@pytest.mark.parametrize("mode", ["auto", "op_level"])
@pytest.mark.parametrize("policy", ["deny", "allow"])
@pytest.mark.parametrize("scope,field,value", INVALID_CANONICAL)
def test_public_canonical_invalid_metadata_never_returns_fallback(checkpoint, scope, field, value, mode, policy):
    target = {
        "text": checkpoint["text_config"],
        "linear": checkpoint["text_config"]["linear_attn_config"],
        "quant": checkpoint["quantization_config"],
    }[scope]
    target[field] = value
    # The native canonical API translates the typed Python rejection to its
    # public ValueError; any returned estimator (including not-ready) is wrong.
    with pytest.raises(ValueError, match="GLM5NEXT"):
        canonical(mode, policy)


@pytest.mark.parametrize("storage", [None, "fp8"])
def test_native_storage_metadata_does_not_invent_mxfp4(checkpoint, storage):
    checkpoint["quantization_config"]["store_dtype"] = storage
    assert build_model().get_resident_weights_bytes() == 80_728_082_808


def test_missing_activation_scheme_is_not_inferred_as_dynamic(checkpoint):
    del checkpoint["quantization_config"]["activation_scheme"]
    with pytest.raises(InvalidEngineConfigurationError, match="activation_scheme"):
        build_model()


@pytest.fixture
def quant_metadata():
    """Pure metadata coverage remains runnable with a stale native extension."""
    return deepcopy(utils._load_pre_downloaded_hf_config(MODEL))


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("field,value", INVALID_QUANT)
def test_pure_invalid_effective_quant_contract(quant_metadata, nested, field, value):
    quant_metadata["quantization_config"][field] = value
    if nested:
        quant_metadata["text_config"]["quantization_config"] = quant_metadata.pop("quantization_config")
    with pytest.raises(InvalidEngineConfigurationError, match=field):
        parse_glm5_next_config(quant_metadata)


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("field", ["ignored_layers", "modules_to_not_convert"])
@pytest.mark.parametrize("prefix", PREFIXES)
@pytest.mark.parametrize("suffix", ["experts", "experts.0.down_proj", "experts.*", "shared_experts.down_proj"])
def test_pure_effective_ffn_exclusion_matching(quant_metadata, nested, field, prefix, suffix):
    quant_metadata["quantization_config"][field] = [f"{prefix}.3.mlp.{suffix}"]
    if nested:
        quant_metadata["text_config"]["quantization_config"] = quant_metadata.pop("quantization_config")
    plan = parse_glm5_next_config(quant_metadata)["extra_params"]
    quant = effective_glm5_next_quantization(quant_metadata)
    with pytest.raises(InvalidEngineConfigurationError, match="FFN"):
        validate_glm5_next_ffn_exclusions(quant, plan.mlp_layer_types, plan.num_shared_experts)


@pytest.mark.parametrize("ignored", [[], ["language_model.model.layers.3.mlp.gate"]])
def test_pure_backend_field_precedence(quant_metadata, ignored):
    quant = quant_metadata["quantization_config"]
    quant["modules_to_not_convert"] = ["language_model.model.layers.3.mlp.experts"]
    quant["ignored_layers"] = ignored
    plan = parse_glm5_next_config(quant_metadata)["extra_params"]
    if ignored:
        validate_glm5_next_ffn_exclusions(quant, plan.mlp_layer_types, plan.num_shared_experts)
    else:
        with pytest.raises(InvalidEngineConfigurationError, match="FFN"):
            validate_glm5_next_ffn_exclusions(quant, plan.mlp_layer_types, plan.num_shared_experts)


@pytest.mark.parametrize("nested", [False, True])
def test_pure_resolver_and_inference_agree_without_mutation(quant_metadata, nested):
    if nested:
        quant_metadata["text_config"]["quantization_config"] = quant_metadata.pop("quantization_config")
    before = deepcopy(quant_metadata)
    quant = effective_glm5_next_quantization(quant_metadata)
    assert quant is utils._get_language_quantization_config(quant_metadata)
    assert utils._infer_quantization_fields(quant_metadata) == {"quant_algo": "fp8_block", "quant_dynamic": True}
    assert len(quant["modules_to_not_convert"]) == 1509
    assert quant_metadata == before


@pytest.mark.parametrize("nested_quant", [None, [], {}, {"quant_method": "fp8", "activation_scheme": "static"}])
def test_pure_root_presence_overrides_even_malformed_nested_quant(quant_metadata, nested_quant):
    quant_metadata["text_config"]["quantization_config"] = nested_quant
    quant = effective_glm5_next_quantization(quant_metadata)
    assert quant is quant_metadata["quantization_config"]
    assert parse_glm5_next_config(quant_metadata)["layers"] == 45


@pytest.mark.parametrize("root_quant", [None, [], {}, {"quant_method": "fp8"}])
def test_pure_invalid_root_does_not_fall_through(quant_metadata, root_quant):
    quant_metadata["text_config"]["quantization_config"] = quant_metadata["quantization_config"]
    quant_metadata["quantization_config"] = root_quant
    with pytest.raises(InvalidEngineConfigurationError):
        parse_glm5_next_config(quant_metadata)


def test_pure_inactive_exclusion_field_still_requires_valid_type(quant_metadata):
    quant_metadata["quantization_config"].update(
        ignored_layers=["language_model.model.layers.3.mlp.gate"], modules_to_not_convert="not-a-list"
    )
    with pytest.raises(InvalidEngineConfigurationError, match="modules_to_not_convert"):
        parse_glm5_next_config(quant_metadata)


@pytest.mark.parametrize(
    "name",
    [
        "model.layers.3.mlp.experts",
        "language_model.model.layers.*.mlp.experts",
        "language_model.model.layers.3.mlp",
        "language_model.model.layers.0.mlp.gate_proj.weight",
        "language_model.model.layers.45.mlp.experts",
        "language_model.model.layers.3.mlp.gate",
        "language_model.model.layers.3.mlp.gate.e_score_correction_bias",
        "visual.blocks.0.mlp.down_proj",
    ],
)
def test_pure_matching_does_not_invent_effective_exclusions(quant_metadata, name):
    quant = quant_metadata["quantization_config"]
    quant["ignored_layers"] = [name]
    plan = parse_glm5_next_config(quant_metadata)["extra_params"]
    validate_glm5_next_ffn_exclusions(quant, plan.mlp_layer_types, plan.num_shared_experts)
