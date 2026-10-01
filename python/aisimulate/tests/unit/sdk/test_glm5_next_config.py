# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Config-only tests; fixture provenance is recorded alongside the bundled JSON."""

from collections import Counter
from copy import deepcopy
from dataclasses import FrozenInstanceError

import pytest

from aisimulate_core.sdk import common, utils

pytestmark = pytest.mark.unit
MODEL = "zai-org/GLM-5.3-Flash"


@pytest.fixture
def raw():
    return utils._load_pre_downloaded_hf_config(MODEL)


def test_registration_and_offline_loader(monkeypatch):
    def no_download(*args, **kwargs):
        pytest.fail("Bundled Flash config must load offline")

    monkeypatch.setattr(utils, "_download_hf_json", no_download)
    utils.get_model_config_from_model_path.cache_clear()
    utils._load_model_config_from_model_path.cache_clear()
    info = utils.get_model_config_from_model_path(MODEL)
    assert MODEL in common.DefaultHFModels
    assert {m for m in common.DefaultHFModels if "GLM-5.3-Flash" in m} == {MODEL}
    assert common.ARCHITECTURE_TO_MODEL_FAMILY[info["architecture"]] == "GLM5NEXT"
    assert "GLM5NEXT" in common.ModelFamily
    assert info["raw_config"]["quant_algo"] == "fp8_block"
    assert info["raw_config"]["quant_dynamic"] is True
    assert info["raw_config"]["quantization_config"]["weight_block_size"] == [128, 128]
    assert info["encoder_config"] is None
    assert info["raw_config"]["vision_config"]["depth"] == 24


def test_pinned_geometry_and_immutable_contract(raw):
    before = deepcopy(raw)
    info = utils._parse_hf_config_json(raw)
    assert raw == before
    assert (info["layers"], info["hidden_size"], info["n"], info["n_kv"], info["d"]) == (45, 4096, 64, 64, 256)
    assert (info["topk"], info["num_experts"], info["moe_inter_size"], info["inter_size"]) == (8, 288, 2048, 12288)
    assert (info["vocab"], info["context"]) == (154880, 1048576)
    extra = info["extra_params"]
    assert isinstance(extra, common.Glm5NextConfig)
    assert Counter(extra.layer_types) == {"linear_attention": 34, "deepseek_sparse_attention": 11}
    assert extra.mlp_layer_types == ("dense",) * 3 + ("sparse",) * 42
    assert (extra.kda_num_heads, extra.kda_head_dim, extra.kda_conv_kernel) == (64, 128, 4)
    assert (extra.q_lora_rank, extra.kv_lora_rank) == (1536, 512)
    assert (extra.qk_nope_head_dim, extra.qk_rope_head_dim, extra.v_head_dim) == (256, 0, 256)
    assert (extra.index_head_dim, extra.index_n_heads, extra.index_topk, extra.index_kpool) == (128, 32, 2048, 4)
    assert (extra.hc_mult, extra.hc_sinkhorn_iters, extra.hc_eps) == (4, 20, 1e-6)
    assert (extra.topk, extra.num_experts, extra.moe_inter_size, extra.num_shared_experts) == (8, 288, 2048, 1)
    assert (extra.first_k_dense_replace, extra.dense_inter_size) == (3, 12288)
    with pytest.raises(FrozenInstanceError):
        extra.index_topk = 1


@pytest.mark.parametrize("value", [None, [], "text", 0])
def test_invalid_nested_text(raw, value):
    raw["text_config"] = value
    with pytest.raises(ValueError, match="text_config"):
        utils._parse_hf_config_json(raw)


@pytest.mark.parametrize("value", [None, [], "Glm5NextForConditionalGeneration", [None], [123], [{}]])
def test_invalid_architectures(raw, value):
    raw["architectures"] = value
    with pytest.raises(ValueError, match="architectures"):
        utils._parse_hf_config_json(raw)


REQUIRED_INTS = (
    "num_hidden_layers",
    "hidden_size",
    "num_attention_heads",
    "num_key_value_heads",
    "vocab_size",
    "max_position_embeddings",
    "intermediate_size",
    "moe_intermediate_size",
    "n_routed_experts",
    "num_experts_per_tok",
    "q_lora_rank",
    "kv_lora_rank",
    "qk_nope_head_dim",
    "v_head_dim",
    "index_head_dim",
    "index_n_heads",
    "index_topk",
    "index_kpool",
    "hc_mult",
    "hc_sinkhorn_iters",
)


@pytest.mark.parametrize("field", REQUIRED_INTS)
@pytest.mark.parametrize("value", [None, 0, -1, True, 1.5, "64", [], {}])
def test_invalid_positive_integers(raw, field, value):
    raw["text_config"][field] = value
    with pytest.raises(ValueError, match=field):
        utils._parse_hf_config_json(raw)


@pytest.mark.parametrize(
    "field",
    (
        *REQUIRED_INTS,
        "head_dim",
        "qk_rope_head_dim",
        "n_shared_experts",
        "first_k_dense_replace",
        "layer_types",
        "mlp_layer_types",
        "linear_attn_config",
        "hc_eps",
        "mhc",
        "mla_use_nope",
    ),
)
def test_missing_fields(raw, field):
    del raw["text_config"][field]
    with pytest.raises(ValueError, match=field):
        utils._parse_hf_config_json(raw)


@pytest.mark.parametrize("field", ["layer_types", "mlp_layer_types"])
@pytest.mark.parametrize("value", [None, "linear_attention", [], ["dense"] * 44, ["bad"] * 45, [{}] * 45])
def test_invalid_plans(raw, field, value):
    raw["text_config"][field] = value
    with pytest.raises(ValueError, match=field):
        utils._parse_hf_config_json(raw)


@pytest.mark.parametrize("field", ["kda_layers", "full_attn_layers"])
@pytest.mark.parametrize("value", [None, "0", [], [-1], [45], [True], [0.0], ["0"], [[0]], [0, 0]])
def test_invalid_zero_based_layer_ids(raw, field, value):
    raw["text_config"]["linear_attn_config"][field] = value
    with pytest.raises(ValueError, match=field):
        utils._parse_hf_config_json(raw)


def test_rejects_one_based_kda_plan(raw):
    linear = raw["text_config"]["linear_attn_config"]
    linear["kda_layers"] = [i + 1 for i in linear["kda_layers"]]
    with pytest.raises(ValueError, match="zero-based"):
        utils._parse_hf_config_json(raw)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("first_k_dense_replace", 4, "mlp_layer_types"),
        ("first_k_dense_replace", 46, "mlp_layer_types"),
        ("num_experts_per_tok", 289, "num_experts_per_tok"),
        ("qk_head_dim", 128, "qk_head_dim"),
        ("qk_rope_head_dim", -1, "qk_rope_head_dim"),
        ("head_dim", -1, "head_dim"),
        ("hc_eps", float("nan"), "hc_eps"),
        ("hc_eps", float("inf"), "hc_eps"),
        ("hc_eps", 0, "hc_eps"),
        ("hc_eps", True, "hc_eps"),
        ("mhc", "true", "mhc"),
        ("mla_use_nope", None, "mla_use_nope"),
        ("linear_attn_config", [], "linear_attn_config"),
    ],
)
def test_inconsistent_or_invalid_geometry(raw, field, value, match):
    raw["text_config"][field] = value
    with pytest.raises(ValueError, match=match):
        utils._parse_hf_config_json(raw)


@pytest.mark.parametrize("field", ["num_heads", "head_dim", "short_conv_kernel_size"])
@pytest.mark.parametrize("value", [None, 0, -1, True, "128", 1.5])
def test_invalid_kda_geometry(raw, field, value):
    raw["text_config"]["linear_attn_config"][field] = value
    with pytest.raises(ValueError, match=field):
        utils._parse_hf_config_json(raw)


@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        {},
        {"quant_method": 1},
        {"quant_method": "fp8", "weight_block_size": [True, 128]},
        {"quant_method": "fp8", "weight_block_size": [0, 128]},
    ],
)
def test_invalid_quantization(raw, value):
    raw["quantization_config"] = value
    with pytest.raises(ValueError, match="quantization_config|weight_block_size"):
        utils._parse_hf_config_json(raw)


def test_public_loader_preserves_full_root_quantization(raw, monkeypatch):
    raw["quantization_config"]["modules_to_not_convert"] = ["model.layers.0.self_attn.q_proj", "hyper_connection"]
    # A conflicting nested quantization must not replace the root's scope/default.
    raw["text_config"]["quantization_config"] = {"quant_method": "bf16"}
    expected = deepcopy(raw["quantization_config"])
    utils._attach_inferred_quant_fields(raw)
    monkeypatch.setattr(utils, "_load_model_config_from_model_path", lambda _: raw)
    utils.get_model_config_from_model_path.cache_clear()
    info = utils.get_model_config_from_model_path("test/flash-root-quant")
    assert info["raw_config"]["quantization_config"] == expected
    assert info["raw_config"]["quant_algo"] == "fp8_block"
    assert info["raw_config"]["text_config"]["dtype"] == "bfloat16"


def test_absent_quantization_is_not_invented_from_architecture(raw):
    del raw["quantization_config"]
    assert utils._infer_quantization_fields(raw) == {}
    assert utils._parse_hf_config_json(raw)["d"] == 256


def test_no_fabricated_silicon_support():
    result = common.check_support(MODEL, "h200_sxm", architecture="Glm5NextForConditionalGeneration")
    assert not result.agg_supported
    assert not result.disagg_supported


@pytest.mark.parametrize("field", ["head_dim", "qk_rope_head_dim", "n_shared_experts", "first_k_dense_replace"])
@pytest.mark.parametrize("value", [None, -1, True, "0", 0.5, []])
def test_invalid_nonnegative_integers(raw, field, value):
    raw["text_config"][field] = value
    with pytest.raises(ValueError, match=field):
        utils._parse_hf_config_json(raw)


def test_missing_text_config(raw):
    del raw["text_config"]
    with pytest.raises(ValueError, match="text_config"):
        utils._parse_hf_config_json(raw)


def test_kda_ids_must_agree_even_with_correct_count(raw):
    raw["text_config"]["linear_attn_config"]["kda_layers"][0] = 3
    with pytest.raises(ValueError, match="kda_layers"):
        utils._parse_hf_config_json(raw)


@pytest.mark.parametrize("value", [None, "full", ["full"] * 44, ["reuse"] * 45, [False] * 45])
def test_invalid_indexer_plan(raw, value):
    raw["text_config"]["indexer_types"] = value
    with pytest.raises(ValueError, match="indexer_types"):
        utils._parse_hf_config_json(raw)


def test_valid_optional_indexer_plan(raw):
    raw["text_config"]["indexer_types"] = ["full"] * 45
    assert utils._parse_hf_config_json(raw)["layers"] == 45


def test_unrepresentable_hc_epsilon(raw):
    raw["text_config"]["hc_eps"] = 10**1000
    with pytest.raises(ValueError, match="hc_eps"):
        utils._parse_hf_config_json(raw)


@pytest.mark.parametrize("model", sorted(common.DefaultHFModels))
def test_all_catalog_configs_still_parse(model):
    # Registration-only regression: do not instantiate model graphs or infer PASS.
    info = utils.get_model_config_from_model_path(model)
    assert info["architecture"] in common.ARCHITECTURE_TO_MODEL_FAMILY
    assert info["layers"] > 0
