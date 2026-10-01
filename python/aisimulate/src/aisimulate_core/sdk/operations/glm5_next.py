# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified: typed AISimulate declarations; no GPU implementation or Python timing math.
# Serving contract: https://github.com/vllm-project/vllm/tree/0a30bc3f9ac3cc1a9339e115377a99d32252aed0/vllm/models/glm5next
# Original paths: common/{model,kda,attention,sparse_indexer}.py and nvidia/sparse_indexer.py.

"""GLM5NEXT typed native operations, analytical and unvalidated on hardware.

SILICON fails with missing data. SOL is a staged peak roofline. HYBRID and
EMPIRICAL use the hardware spec's memory efficiency/latency constants, NOT a
measured utilization sample. Native provenance is ``analytic_unvalidated``.
All geometry, validation, weight accounting and timing live in Rust.
"""

import aisimulate_core._native as _core
from aisimulate_core.sdk.operations.base import OpShellKit


class Glm5NextSparseAttention(_core.Glm5NextSparseAttention, OpShellKit):
    """Sparse core with LOCAL MLA heads and REPLICATED indexer heads.

    Signature: (name, num, is_context, num_heads, kv_lora_rank, qk_head_dim,
    v_head_dim, index_n_heads, index_head_dim, index_topk, index_kpool).
    BF16 cache, NoPE, forced-MQA prefill only: validation serving MUST set
    ``sparse_mla_force_mqa=True``. No CP, PP, MTP or beam search.

    Includes absorbed kv_b BMMs, latent attention, key norm/FWHT/conversion,
    pooling/tail, logits, selection and index expansion. The model separately
    adds ALL projection GEMMs (including the indexer's FP32 head-weight
    recomputation), q/kv latent norms and resident kv_b/indexer parameters.
    ``get_weights()`` is zero; do not also time kv_b expansion or MLA BMM ops.
    """


class Glm5NextMHC(_core.Glm5NextMHC, OpShellKit):
    """ONE pre+post site: (name, num, hidden_size, hc_mult, hc_sinkhorn_iters).

    Includes FP32 projection, flatten norm, Sinkhorn, BF16 stream traffic and
    FP32 mixing. Counts fn/base/scale once per site. Emit two sites/layer, not
    two pre+post pairs; do not additionally bill projection GEMMs. Decoder
    input/post-attention RMSNorm and initial expansion/final mean stay outside.
    """


class Glm5NextKDA(_core.Glm5NextKDA, OpShellKit):
    """(name, num, is_context, num_heads, head_dim=128, conv_kernel=4).

    LOCAL heads, BF16 conv states and FP32 recurrent state. Includes packed
    QKV convolution, bounded-gate recurrence and gated output norm; exclude
    these from the model's other ops. Projection GEMMs and ALL resident KDA
    parameters stay outside (get_weights=0). Avoids KDAKernel's nearest-Kimi
    shape selection and its different fused-decode boundary. No speculation.
    """
