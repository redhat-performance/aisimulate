# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified: CPU-only model composition and memory inventory, not serving code.
# Source: vllm-project/vllm@0a30bc3f9ac3cc1a9339e115377a99d32252aed0,
# vllm/models/glm5next/common/{model,kda,attention}.py. See THIRD_PARTY_NOTICES.md.

"""GLM-5.3-Flash text baseline. Analytical attention is NOT silicon validated.

Runtime contract: vLLM, native FP8 FFNs, BF16 attention/KV, TP-only, no MTP,
no sequence parallelism, sparse_mla_force_mqa=True, prefix caching disabled.
Memory describes logical resident tensors, not vLLM's padded hybrid allocator.
"""

from __future__ import annotations

import json
import math

import aisimulate_core._native as core
import aisimulate_core.sdk.operations as ops
from aisimulate_core.sdk import common
from aisimulate_core.sdk.errors import InvalidEngineConfigurationError
from aisimulate_core.sdk.models.base import BaseModel, register_model
from aisimulate_core.sdk.models.blocks.moe import MoEBlockShape, build_moe_block_ops
from aisimulate_core.sdk.models.helpers import power_law_distribution, quant_exclude_patterns


def _fp32(name: str, count: int, n: int, k: int):
    return core.op_from_spec_json(
        json.dumps(
            {
                "Glm5NextFp32Linear": {
                    "name": name,
                    "scale_factor": count,
                    "n": n,
                    "k": k,
                }
            }
        )
    )


@register_model("GLM5NEXT")
class Glm5NextModel(BaseModel):
    extra_params: common.Glm5NextConfig

    @classmethod
    def create(cls, model_info, model_config, backend_name):
        return cls(model_info, model_config, backend_name)

    def __init__(self, info, cfg, backend_name):
        d = info["extra_params"]
        if not isinstance(d, common.Glm5NextConfig):
            raise TypeError("GLM5NEXT requires Glm5NextConfig metadata")
        if backend_name != "vllm":
            raise InvalidEngineConfigurationError("GLM5NEXT currently models the vLLM text-only baseline")
        if cfg.pp_size != 1 or cfg.cp_size != 1 or cfg.attention_dp_size != 1:
            raise InvalidEngineConfigurationError("GLM5NEXT requires PP=CP=attention DP=1")
        if cfg.nextn or (cfg.speculation is not None and cfg.speculation.kind != "none"):
            raise InvalidEngineConfigurationError("GLM5NEXT speculative decoding is not modeled; use nextn=0")
        if cfg.overwrite_num_layers or info["layers"] != len(d.layer_types):
            raise ValueError("GLM5NEXT layer overrides cannot preserve the hybrid layer plan")
        if cfg.moe_backend or cfg.moe_comm_backend or cfg.enable_eplb or cfg.wideep_num_slots or cfg.decoder_replay:
            raise InvalidEngineConfigurationError("GLM5NEXT does not model large EP, EPLB or decoder replay")
        if cfg.attention_backend is not None:
            raise InvalidEngineConfigurationError(
                "GLM5NEXT uses its dedicated forced-MQA analytical attention contract"
            )
        if cfg.tp_size not in (1, 2, 4, 8) or cfg.moe_ep_size != 1 or cfg.moe_tp_size != cfg.tp_size:
            raise InvalidEngineConfigurationError("GLM5NEXT baseline requires TP in {1,2,4,8}, MoE TP=TP and EP=1")
        if d.kda_num_heads % cfg.tp_size or info["n"] % cfg.tp_size:
            raise ValueError("GLM5NEXT attention and KDA heads must divide TP")
        if info["inter_size"] % cfg.tp_size or d.moe_inter_size % cfg.tp_size or info["vocab"] % cfg.tp_size:
            raise ValueError("GLM5NEXT FFN widths and vocabulary must divide TP")
        if cfg.gemm_quant_mode != common.GEMMQuantMode.fp8_block or cfg.moe_quant_mode != common.MoEQuantMode.fp8_block:
            raise InvalidEngineConfigurationError("GLM5NEXT baseline requires native fp8_block GEMM and MoE weights")
        if (
            cfg.kvcache_quant_mode != common.KVCacheQuantMode.bfloat16
            or cfg.fmha_quant_mode != common.FMHAQuantMode.bfloat16
        ):
            raise InvalidEngineConfigurationError("GLM5NEXT baseline requires bfloat16 KV cache and attention compute")
        if cfg.comm_quant_mode != common.CommQuantMode.half:
            raise InvalidEngineConfigurationError("GLM5NEXT baseline requires half communication")
        raw = info["raw_config"]
        text = raw["text_config"]
        if (
            text.get("index_kpool_compress") is not True
            or text.get("index_kpool_always_select_tail") is not True
            or text.get("hidden_act") != "silu"
            or text.get("moe_router_dtype") != "float32"
            or text.get("tie_word_embeddings") is not False
            or text["linear_attn_config"].get("gate_lower_bound") != -5.0
        ):
            raise InvalidEngineConfigurationError("GLM5NEXT requires the native compressed-indexer/KDA/FFN contract")
        quant = raw.get("quantization_config", {})
        if quant.get("quant_method") != "fp8" or quant.get("weight_block_size") != [128, 128]:
            raise InvalidEngineConfigurationError("GLM5NEXT baseline requires the native block128 FP8 checkpoint")
        if any(
            p.startswith("model.layers.")
            and ".mlp." in p
            and not p.endswith((".mlp.gate", ".mlp.gate.e_score_correction_bias"))
            for p in quant_exclude_patterns(raw)
        ):
            raise InvalidEngineConfigurationError("GLM5NEXT custom FFN quantization exclusions are not modeled")
        super().__init__(
            info["model_path"],
            info["model_family"],
            info["architecture"],
            info["layers"],
            info["n"],
            info["n_kv"],
            info["d"],
            info["hidden_size"],
            info["inter_size"],
            info["vocab"],
            info["context"],
            cfg,
            d,
        )
        self.raw_config = raw
        self._topk, self._num_experts, self._moe_inter_size = d.topk, d.num_experts, d.moe_inter_size
        self._linear_layers = d.layer_types.count("linear_attention")
        self._sparse_layers = d.layer_types.count("deepseek_sparse_attention")
        self.context_ops = self._phase_ops(True)
        self.generation_ops = self._phase_ops(False)
        self._resident_weight_bytes = self._weight_inventory()

    @property
    def activation_hidden_size(self):
        return self._hidden_size

    def _phase_ops(self, context):
        d, cfg = self.extra_params, self.config
        h, tp = self._hidden_size, cfg.tp_size
        phase = "context" if context else "generation"
        bf16 = common.GEMMQuantMode.bfloat16
        local_heads = d.kda_num_heads // tp
        p = local_heads * d.kda_head_dim
        linear, sparse, layers = self._linear_layers, self._sparse_layers, self._num_layers
        result = [
            ops.Embedding(f"{phase}_embedding", 1, self._vocab_size // tp, h),
            ops.CustomAllReduce(f"{phase}_embedding_ar", 1, h, tp),
            ops.ElementWise(f"{phase}_hc_expand", 1, h, d.hc_mult * h),
            # Each site includes pre AND post, so exactly two sites per layer.
            ops.Glm5NextMHC(f"{phase}_mhc", 2 * layers, h, d.hc_mult, d.hc_sinkhorn_iters),
            ops.ElementWise(f"{phase}_decoder_norm", 2 * layers, h, h),
            # Two replicated low-rank shards (f_a, g_a), not TP-divided.
            ops.GEMM(f"{phase}_kda_input_gemm", linear, 3 * p + local_heads + 2 * d.kda_head_dim, h, bf16),
            ops.GEMM(f"{phase}_kda_f_b_gemm", linear, p, d.kda_head_dim, bf16),
            ops.GEMM(f"{phase}_kda_g_b_gemm", linear, p, d.kda_head_dim, bf16),
            ops.Glm5NextKDA(f"{phase}_attention", linear, context, local_heads, d.kda_head_dim, d.kda_conv_kernel),
            ops.GEMM(f"{phase}_kda_output_gemm", linear, h, p, bf16),
            ops.CustomAllReduce(f"{phase}_kda_ar", linear, h, tp),
            # vLLM explicitly dequantizes ALL MLA projections to BF16 on load.
            ops.GEMM(f"{phase}_mla_down_gemm", sparse, d.q_lora_rank + d.kv_lora_rank, h, bf16),
            ops.ElementWise(
                f"{phase}_mla_latent_norm", sparse, d.q_lora_rank + d.kv_lora_rank, d.q_lora_rank + d.kv_lora_rank
            ),
            ops.GEMM(f"{phase}_mla_q_b_gemm", sparse, self._num_heads // tp * d.qk_nope_head_dim, d.q_lora_rank, bf16),
            ops.GEMM(f"{phase}_index_q_gemm", sparse, d.index_n_heads * d.index_head_dim, d.q_lora_rank, bf16),
            ops.GEMM(f"{phase}_index_key_weight_gemm", sparse, d.index_head_dim + d.index_n_heads, h, bf16),
            _fp32(f"{phase}_index_weight_fp32", sparse, d.index_n_heads, h),
            ops.GEMM(f"{phase}_index_pool_gate_gemm", sparse, d.index_head_dim, h, bf16),
            ops.Glm5NextSparseAttention(
                f"{phase}_attention",
                sparse,
                context,
                self._num_heads // tp,
                d.kv_lora_rank,
                d.qk_nope_head_dim,
                d.v_head_dim,
                d.index_n_heads,
                d.index_head_dim,
                d.index_topk,
                d.index_kpool,
            ),
            ops.GEMM(f"{phase}_mla_output_gemm", sparse, h, self._num_heads // tp * d.v_head_dim, bf16),
            ops.CustomAllReduce(f"{phase}_mla_ar", sparse, h, tp),
        ]
        dense = d.first_k_dense_replace
        result.extend(
            [
                ops.GEMM(f"{phase}_dense_gate_up_gemm", dense, 2 * d.dense_inter_size // tp, h, cfg.gemm_quant_mode),
                ops.ElementWise(
                    f"{phase}_dense_act_gate", dense, 2 * d.dense_inter_size // tp, d.dense_inter_size // tp
                ),
                ops.GEMM(f"{phase}_dense_down_gemm", dense, h, d.dense_inter_size // tp, cfg.gemm_quant_mode),
                ops.CustomAllReduce(f"{phase}_dense_ar", dense, h, tp),
            ]
        )
        moe_layers = layers - dense
        shape = MoEBlockShape(h, d.moe_inter_size, d.topk, d.num_experts, d.num_shared_experts, moe_layers)
        block = build_moe_block_ops(
            phase,
            shape,
            cfg,
            cfg.moe_quant_mode,
            power_law_distribution(cfg.workload_distribution, 1.01),
            scale_factor=moe_layers,
            backend_name="vllm",
            inference_phase=phase,
            model_family="GLM5NEXT",
        )
        # Attention is already reduced; TP-only serving needs no pre-dispatch.
        # Shared/routed outputs use one reduction, not an additional shared AR.
        for op in block:
            if op._name == f"{phase}_router_gemm":
                # GateLinear's out_dtype=float32 does not set params_dtype.
                # Weights/compute remain BF16; account for the additional two
                # output bytes per logit (generic GEMM assumes BF16 outputs).
                # The epilogue's fusion is not independently calibrated.
                result.append(op)
                result.append(ops.ElementWise(f"{phase}_router_fp32_output", moe_layers, 0, d.num_experts))
            elif op._name == f"{phase}_moe_pre_dispatch":
                continue
            elif op._name == f"{phase}_moe_post_dispatch":
                result.append(ops.CustomAllReduce(f"{phase}_moe_ar", moe_layers, h, tp))
            else:
                result.append(op)
        result.extend(
            [
                ops.ElementWise(f"{phase}_hc_contract", 1, d.hc_mult * h, h),
                ops.ElementWise(f"{phase}_final_norm", 1, h, h),
                ops.GEMM(f"{phase}_logits_gemm", 1, self._vocab_size // tp, h, bf16),
            ]
        )
        return result

    def _weight_inventory(self):
        """Resident decoder tensors, including untimed parameters and FP8 scales.

        MTP and vision towers are not resident in this language-only baseline.
        No empirical checkpoint-size fudge factor is used.
        """
        d, tp, h = self.extra_params, self.config.tp_size, self._hidden_size
        local_heads = d.kda_num_heads // tp
        p = local_heads * d.kda_head_dim
        total = sum(op.get_weights() for op in self.context_ops)
        # Original FP32 Q/K/V conv weights and vLLM's retained concatenated copy.
        total += self._linear_layers * (
            2 * 3 * p * d.kda_conv_kernel * 4 + 4 * p + 4 * local_heads + 4 * d.kda_head_dim
        )
        # Absorbed kv_b is resident but its BMM work is owned by sparse attention.
        total += self._sparse_layers * (
            2 * self._num_heads // tp * (d.qk_nope_head_dim + d.v_head_dim) * d.kv_lora_rank
            + 2 * (d.q_lora_rank + d.kv_lora_rank)
            + 8 * d.index_head_dim
            + 4 * d.index_kpool * d.index_head_dim
        )
        total += (2 * self._num_layers + 1) * h * 2  # decoder/final RMSNorm
        total += (self._num_layers - d.first_k_dense_replace) * d.num_experts * 4  # router bias
        # Each 128x128 weight tile owns an FP32 scale. GEMM/MoE's generic
        # inventory excludes these; only FFNs remain quantized at runtime.
        for op in self.context_ops:
            if isinstance(op, ops.GEMM) and op._quant_mode == common.GEMMQuantMode.fp8_block:
                total += op._scale_factor * math.ceil(op._n / 128) * math.ceil(op._k / 128) * 4
        total += (
            (self._num_layers - d.first_k_dense_replace)
            * d.num_experts
            * 3
            * math.ceil(h / 128)
            * math.ceil(d.moe_inter_size / tp / 128)
            * 4
        )
        return float(total)

    def get_resident_weights_bytes(self):
        return self._resident_weight_bytes

    def _fixed_state_bytes(self):
        d = self.extra_params
        heads = d.kda_num_heads // self.config.tp_size
        recurrent = heads * d.kda_head_dim * d.kda_head_dim * 4
        convolution = 3 * heads * d.kda_head_dim * (d.kda_conv_kernel - 1) * 2
        tail = self._sparse_layers * d.index_kpool * d.index_head_dim * 2 * 2
        return self._linear_layers * (recurrent + convolution) + tail

    def get_kvcache_elements_per_token(self):
        # BF16 latent KV only; pooled FP8 index/state are accounted separately.
        return self._sparse_layers * self.extra_params.kv_lora_rank

    def get_kvcache_bytes_per_sequence(self, seq_len):
        seq_len = max(0, seq_len)
        if seq_len == 0:
            return 0.0
        d = self.extra_params
        # Conservatively reserve the incomplete pool's next compressed slot.
        pools = (seq_len + d.index_kpool - 1) // d.index_kpool
        return float(
            self._fixed_state_bytes()
            + seq_len * self.get_kvcache_elements_per_token() * 2
            + self._sparse_layers * pools * common.indexer_cache_entry_bytes(d.index_head_dim)
        )

    def get_kvcache_max_tokens(self, kv_budget_bytes):
        if not math.isfinite(kv_budget_bytes):
            raise ValueError("GLM5NEXT cache budget must be finite")
        return self._binary_search_kvcache_max_tokens(kv_budget_bytes)

    def get_kvcache_batch_capacity(self, kv_budget_bytes, max_batch_size):
        if not math.isfinite(kv_budget_bytes) or max_batch_size < 1:
            raise ValueError("GLM5NEXT requires a finite cache budget and positive max_batch_size")
        d = self.extra_params
        entry = self._sparse_layers * common.indexer_cache_entry_bytes(d.index_head_dim)
        slope = self.get_kvcache_elements_per_token() * 2 + entry / d.index_kpool
        # Per-request state plus worst-case ceil(pool count) fragmentation.
        fixed = max_batch_size * (self._fixed_state_bytes() + entry * (d.index_kpool - 1) / d.index_kpool)
        return max(0, int((kv_budget_bytes - fixed) // slope))

    def get_additional_activation_bytes(self, num_tokens):
        d = self.extra_params
        topk_width = ((d.index_topk + d.index_kpool - 2) // 128 + 1) * 128
        residual = 2 * d.hc_mult * self._hidden_size * 2
        return float(num_tokens * (residual + topk_width * 4))
