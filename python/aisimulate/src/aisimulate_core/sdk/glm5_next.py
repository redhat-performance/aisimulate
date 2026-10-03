# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Strict GLM5NEXT metadata parsing; performance arithmetic lives in Rust.

Schema source and fixture license: ../model_configs/zai-org--GLM-5.3-Flash_README.md.
"""

import math

from aisimulate_core.sdk.common import Glm5NextConfig
from aisimulate_core.sdk.errors import InvalidEngineConfigurationError


def validate_glm5_next_text_workload(runtime_config) -> None:
    """Do not silently accept visual input when only the decoder is modeled."""
    fields = (
        "image_height",
        "image_width",
        "num_image_tokens",
        "video_height",
        "video_width",
        "video_frames",
        "num_video_tokens",
        "num_videos_per_request",
    )
    if any(getattr(runtime_config, field, 0) for field in fields):
        raise InvalidEngineConfigurationError(
            "GLM5NEXT currently supports text-only estimation, not image/video workloads"
        )


def expand_bundled_glm5_next_config(config: dict) -> dict:
    """Expand the pinned fixture's lossless exclusion groups before any consumer.

    This is a storage encoding only, not wildcard matching: the returned standard
    HF modules_to_not_convert list is identical (including order) to upstream.
    Complete downloaded/local HF configs never need this transformation.
    """
    quant = config.get("quantization_config")
    key = "_aisimulate_modules_to_not_convert_groups"
    if not isinstance(quant, dict) or key not in quant:
        return config
    if config.get("architectures") != ["Glm5NextForConditionalGeneration"]:
        raise InvalidEngineConfigurationError("Compact exclusion groups are only supported for GLM5NEXT")
    names = quant.get("modules_to_not_convert")
    groups = quant[key]
    if not isinstance(names, list) or any(not isinstance(name, str) for name in names):
        raise InvalidEngineConfigurationError("GLM5NEXT modules_to_not_convert must be a list of strings")
    if not isinstance(groups, list):
        raise InvalidEngineConfigurationError("GLM5NEXT compact exclusion groups must be a list")
    expanded = list(names)
    for group in groups:
        if not isinstance(group, dict) or set(group) != {"prefix", "layers", "suffixes"}:
            raise InvalidEngineConfigurationError("GLM5NEXT exclusion group requires prefix, layers and suffixes")
        prefix, layers, suffixes = group["prefix"], group["layers"], group["suffixes"]
        if (
            prefix not in ("model.layers", "visual.blocks")
            or not isinstance(layers, list)
            or any(type(layer) is not int or layer < 0 for layer in layers)
            or not isinstance(suffixes, list)
            or any(not isinstance(suffix, str) or not suffix for suffix in suffixes)
        ):
            raise InvalidEngineConfigurationError("GLM5NEXT invalid compact exclusion group")
        expanded.extend(f"{prefix}.{layer}.{suffix}" for layer in layers for suffix in suffixes)
    if len(set(expanded)) != len(expanded):
        raise InvalidEngineConfigurationError("GLM5NEXT compact exclusion groups contain duplicate module names")
    normalized = {name: value for name, value in quant.items() if name != key}
    normalized["modules_to_not_convert"] = sorted(expanded)
    return {**config, "quantization_config": normalized}


def _integer(config: dict, name: str, *, minimum: int = 1) -> int:
    value = config.get(name)
    if type(value) is not int or not minimum <= value <= 2**32 - 1:
        raise InvalidEngineConfigurationError(
            f"GLM5NEXT {name} must be an integer >= {minimum} fitting native u32, got {value!r}"
        )
    return value


def _layer_plan(config: dict, name: str, layers: int, allowed: tuple[str, ...]) -> tuple[str, ...]:
    values = config.get(name)
    if (
        not isinstance(values, list)
        or len(values) != layers
        or any(not isinstance(value, str) or value not in allowed for value in values)
    ):
        raise InvalidEngineConfigurationError(f"GLM5NEXT {name} must contain {layers} entries from {allowed}")
    return tuple(values)


def effective_glm5_next_quantization(root: dict) -> dict:
    """Validate the same language declaration used by shared quant inference.

    Root presence wins, even for malformed/empty declarations; only absence
    enables nested fallback. Do not merge scopes or rewrite the HF fixture.
    """
    # utils imports this parser, so resolve the shared helper lazily.
    from aisimulate_core.sdk.utils import _get_language_quantization_config

    quant = _get_language_quantization_config(root)
    if not isinstance(quant, dict):
        raise InvalidEngineConfigurationError("GLM5NEXT quantization_config must be an object")
    if not quant and "quantization_config" not in root and "quantization_config" not in root["text_config"]:
        return quant
    method = quant.get("quant_method")
    if not isinstance(method, str) or not method:
        raise InvalidEngineConfigurationError("GLM5NEXT quantization_config.quant_method must be a non-empty string")
    if method != "fp8":
        raise InvalidEngineConfigurationError("GLM5NEXT native checkpoint requires quant_method='fp8'")
    block = quant.get("weight_block_size")
    if not isinstance(block, list) or len(block) != 2 or any(type(value) is not int or value <= 0 for value in block):
        raise InvalidEngineConfigurationError("GLM5NEXT FP8 weight_block_size must contain two positive integers")
    if quant.get("activation_scheme") != "dynamic":
        raise InvalidEngineConfigurationError("GLM5NEXT native block FP8 requires activation_scheme='dynamic'")
    storage = quant.get("store_dtype")
    if storage is not None and not isinstance(storage, str):
        raise InvalidEngineConfigurationError("GLM5NEXT store_dtype must be a string or null")
    if storage == "mxfp4":
        raise InvalidEngineConfigurationError("GLM5NEXT store_dtype='mxfp4' expert storage is not modeled")
    for field in ("ignored_layers", "modules_to_not_convert"):
        if field in quant:
            excluded = quant[field]
            if not isinstance(excluded, list) or any(not isinstance(value, str) for value in excluded):
                raise InvalidEngineConfigurationError(f"GLM5NEXT {field} must be a list of strings")
    return quant


def validate_glm5_next_ffn_exclusions(quant: dict, mlp_types: tuple[str, ...], shared_experts: int) -> None:
    """Reject effective native-FP8 skips, without generic glob/parent matching.

    Backend evidence: vLLM@0a30bc3f9ac3cc1a9339e115377a99d32252aed0,
    fp8.py:Fp8Config and quant_utils.py:is_layer_skipped. This is a baseline
    rejection check, not a serving implementation or a general quant resolver.
    """
    excluded = quant.get("ignored_layers") or quant.get("modules_to_not_convert") or []
    # The inherited conditional-generation mapper rewrites this HF alias.
    excluded = [
        "language_model.model." + name.removeprefix("model.language_model.")
        if name.startswith("model.language_model.")
        else name
        for name in excluded
    ]
    for layer, kind in enumerate(mlp_types):
        prefix = f"language_model.model.layers.{layer}.mlp"
        if kind == "sparse" and any(f"{prefix}.experts" in name for name in excluded):
            raise InvalidEngineConfigurationError("GLM5NEXT custom FFN quantization exclusions are not modeled")
        if kind == "sparse" and not shared_experts:
            continue
        linear_prefix = f"{prefix}.shared_experts" if kind == "sparse" else prefix
        # A fused gate/up exclusion or either unfused shard changes execution
        # (a partial shard exclusion is rejected by vLLM itself).
        projections = ("gate_up_proj", "gate_proj", "up_proj")
        skipped = any(f"{linear_prefix}.{projection}" in excluded for projection in projections)
        down = f"{linear_prefix}.down_proj"
        skipped |= any(down in name for name in excluded) if kind == "sparse" else down in excluded
        if skipped:
            raise InvalidEngineConfigurationError("GLM5NEXT custom FFN quantization exclusions are not modeled")


def parse_glm5_next_config(root: dict) -> dict:
    """Normalize nested HF metadata without mutating it or inferring GQA shapes."""
    if not isinstance(root, dict):
        raise InvalidEngineConfigurationError("GLM5NEXT config must be an object")
    text = root.get("text_config")
    if not isinstance(text, dict):
        raise InvalidEngineConfigurationError("GLM5NEXT text_config must be an object")
    if "vision_config" in root and not isinstance(root["vision_config"], dict):
        raise InvalidEngineConfigurationError("GLM5NEXT vision_config must be an object")
    effective_glm5_next_quantization(root)
    layers = _integer(text, "num_hidden_layers")
    base = {
        target: _integer(text, source)
        for target, source in (
            ("hidden_size", "hidden_size"),
            ("n", "num_attention_heads"),
            ("n_kv", "num_key_value_heads"),
            ("vocab", "vocab_size"),
            ("context", "max_position_embeddings"),
            ("inter_size", "intermediate_size"),
            ("topk", "num_experts_per_tok"),
            ("num_experts", "n_routed_experts"),
            ("moe_inter_size", "moe_intermediate_size"),
        )
    }
    if base["topk"] > base["num_experts"]:
        raise InvalidEngineConfigurationError("GLM5NEXT num_experts_per_tok must not exceed n_routed_experts")
    layer_types = _layer_plan(text, "layer_types", layers, ("linear_attention", "deepseek_sparse_attention"))
    mlp_types = _layer_plan(text, "mlp_layer_types", layers, ("dense", "sparse"))
    if "indexer_types" in text:
        _layer_plan(text, "indexer_types", layers, ("full",))
    first_dense = _integer(text, "first_k_dense_replace", minimum=0)
    if first_dense > layers or any(
        kind != ("dense" if i < first_dense else "sparse") for i, kind in enumerate(mlp_types)
    ):
        raise InvalidEngineConfigurationError("GLM5NEXT mlp_layer_types must agree with first_k_dense_replace")
    linear = text.get("linear_attn_config")
    if not isinstance(linear, dict):
        raise InvalidEngineConfigurationError("GLM5NEXT linear_attn_config must be an object")
    for name, kind in (("kda_layers", "linear_attention"), ("full_attn_layers", "deepseek_sparse_attention")):
        ids = linear.get(name)
        if (
            not isinstance(ids, list)
            or any(type(i) is not int or not 0 <= i < layers for i in ids)
            or len(set(ids)) != len(ids)
            or set(ids) != {i for i, value in enumerate(layer_types) if value == kind}
        ):
            raise InvalidEngineConfigurationError(
                f"GLM5NEXT linear_attn_config.{name} must match zero-based layer_types without duplicates"
            )
    dims = {
        name: _integer(text, name)
        for name in (
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
    }
    dims["qk_rope_head_dim"] = _integer(text, "qk_rope_head_dim", minimum=0)
    # head_dim=0 is the published sentinel, not hidden_size/num_heads.
    _integer(text, "head_dim", minimum=0)
    if "qk_head_dim" in text and _integer(text, "qk_head_dim") != (dims["qk_nope_head_dim"] + dims["qk_rope_head_dim"]):
        raise InvalidEngineConfigurationError("GLM5NEXT qk_head_dim must equal qk_nope_head_dim + qk_rope_head_dim")
    eps = text.get("hc_eps")
    if not isinstance(eps, (float, int)) or isinstance(eps, bool):
        raise InvalidEngineConfigurationError("GLM5NEXT hc_eps must be finite and positive")
    try:
        valid_eps = math.isfinite(eps) and eps > 0
    except OverflowError:
        valid_eps = False
    if not valid_eps:
        raise InvalidEngineConfigurationError("GLM5NEXT hc_eps must be finite and positive")
    for name in ("mhc", "mla_use_nope"):
        if type(text.get(name)) is not bool:
            raise InvalidEngineConfigurationError(f"GLM5NEXT {name} must be boolean")
    if not text["mhc"] or not text["mla_use_nope"] or dims["qk_rope_head_dim"] != 0:
        raise InvalidEngineConfigurationError("GLM5NEXT currently requires mHC and NoPE with qk_rope_head_dim=0")
    extra = Glm5NextConfig(
        layer_types=layer_types,
        mlp_layer_types=mlp_types,
        kda_num_heads=_integer(linear, "num_heads"),
        kda_head_dim=_integer(linear, "head_dim"),
        kda_conv_kernel=_integer(linear, "short_conv_kernel_size"),
        **dims,
        hc_eps=float(eps),
        topk=base["topk"],
        num_experts=base["num_experts"],
        moe_inter_size=base["moe_inter_size"],
        num_shared_experts=_integer(text, "n_shared_experts", minimum=0),
        first_k_dense_replace=first_dense,
        dense_inter_size=base["inter_size"],
    )
    # Native constructors historically surface ordinary ValueError. Preflight
    # their shape requirements here so canonical selection cannot fall back.
    if extra.kda_head_dim != 128 or extra.kda_conv_kernel != 4:
        raise InvalidEngineConfigurationError("GLM5NEXT KDA requires head_dim=128 and short_conv_kernel_size=4")
    if extra.index_head_dim != 128 or extra.index_kpool < 2 or extra.index_topk % extra.index_kpool:
        raise InvalidEngineConfigurationError(
            "GLM5NEXT sparse indexer requires index_head_dim=128, index_kpool>=2, "
            "and index_topk divisible by index_kpool"
        )
    return {
        "architecture": "Glm5NextForConditionalGeneration",
        "layers": layers,
        **base,
        "d": extra.v_head_dim,
        "extra_params": extra,
        "encoder_config": None,
    }
