<!-- SPDX-License-Identifier: Apache-2.0 -->

# GLM-5.3-Flash text baseline

`zai-org/GLM-5.3-Flash` is registered as `GLM5NEXT`, not as older GLM/DeepSeek
DSA. This is **initial analytical model support, not an accuracy certification**.
The next gate is the accuracy estimator against ground-truth H200 data.
No measured Flash tables or support-matrix PASS rows are added.

## Scope

- vLLM text-only, native FP8 block128 checkpoint; TP 1/2/4/8, MoE TP=TP, EP=1.
- 45 decoder layers: 34 KDA, 11 NoPE sparse MLA, 3 dense FFNs, 42 top-8-of-288
  MoE FFNs with one shared expert, and two mHC sites per layer.
- All attention projections execute/reside in BF16, including sparse projections
  upcast from FP8 at load by the pinned vLLM implementation. Dense/shared/routed
  FFNs remain FP8. mHC and indexer head-weight projections use FP32. The MoE
  router retains BF16 weights/compute with FP32 output logits; `moe_router_dtype`
  selects the output dtype, not the parameter dtype. Its output traffic is
  modeled separately from the generic BF16 GEMM (fusion is uncalibrated).
- BF16 MLA KV and convolution state, FP32 recurrent state, FP8 pooled index keys
  plus FP32 scales and a BF16 raw-key/gate tail.
- No PP, CP, attention DP, EP, EPLB, speculative decoding, image/video execution,
  sequence-parallel MoE, or calibrated disaggregation/replay allocator behavior.
  Model-level unsupported settings are rejected instead of silently approximated.

The baseline assumes a language-only worker: exclude the vision tower and MTP
weights in the serving setup. Merely sending text to a server that still loads
the vision tower is a different resident-memory contract.

## Estimation contract

Use the canonical `RustForwardPassPerfModel.best_available` interface:

```python
from aisimulate_core.sdk import ForwardPassPerfModelConfig, RustForwardPassPerfModel

estimator = RustForwardPassPerfModel.best_available(ForwardPassPerfModelConfig(
    model="zai-org/GLM-5.3-Flash",
    system="h200_sxm",
    backend="vllm",
    backend_version="0.24.0",
    worker_type="aggregated",
    tp=4,
    moe_tp_size=4,
    moe_ep_size=1,
    estimation_mode="op_level",
    database_mode="HYBRID",
))
prefill = estimator.static_phase_diagnostics(
    batch_size=1, context_length=8192, prefill=True,
)
decode = estimator.static_phase_diagnostics(
    batch_size=1, context_length=8192, prefill=False,
)
print(estimator.diagnostics())
```

Here `0.24.0` identifies the repository's existing H200 **performance profiles**
used for generic GEMM/MoE/communication. It does **not** mean that vLLM 0.24.0
can serve Flash. Runtime ground truth must use a Flash-capable build, recording
its exact commit/image separately; see the [official recipe](https://recipes.vllm.ai/zai-org/GLM-5.3-Flash).
Do not relabel the older profiles as new measurements.

Flash-specific native operators are tableless. `SOL` uses serial stage rooflines;
`HYBRID`/`EMPIRICAL` use peak compute with the system specification's memory
efficiency and launch-latency constants. They report `analytic_unvalidated`
provenance and **reject `SILICON`**, including during estimator readiness checks.
No Kimi KDA, older GLM DSA or DeepSeek mHC timing table is misidentified as Flash.
Generic operations retain their existing measured/transfer policies.

The sparse prefill formula assumes `sparse_mla_force_mqa=True`. The initial
validation workload should disable speculative decoding, sequence parallelism,
prefix-cache reuse, and CUDA graphs (or supply the measured graph reservation).
KDA chunk-intermediate traffic, top-k heap work, kernel fusion, padding and
cache reuse are explicit uncalibrated approximations. Nonlinear scalar operations
are counted as single operations; their measured cost must be checked later.

## Memory and capacity

Memory inventories count all routed experts, not only active experts. They add
the FP8 block scales, BF16 absorbed `kv_b` weights, replicated low-rank KDA
projections, indexer parameters, norms, mHC parameters and retained conv copy.
The cache inverse reserves state **per scheduler slot**, not once per GPU.
Sparse latent KV/index keys are TP-replicated; KDA heads/state are TP-sharded.

The memory API (`estimate_kv_cache`) uses these model-owned inventories. Specify
`backend_version="0.24.0"`, `tp_size=4`, `moe_tp_size=4`, `moe_ep_size=1` and
`allow_naive_fallback=False`. One H200 cannot hold the GPU-resident checkpoint;
TP1 memory estimation fails with no KV budget.

Cache sizes are logical tensor payloads with rounded compressed slots and one
live recurrent state per request. They do not reproduce vLLM hybrid page/group
padding, prefix-checkpoint retention, or boot-time pool partitioning. Compare
memory against the target server before using it for capacity planning.

## Validation checklist for ground truth

1. Pin the checkpoint revision, vLLM image/commit, actual dtypes, TP/EP,
   sparse-MQA selection, and resident vision/MTP status.
2. Measure prefill and decode separately across batch size and context length,
   including below/at/above 2048 selected tokens and four-token pool boundaries.
3. Compare TTFT, TPOT, goodput and memory with the accuracy estimator. Separate
   scheduler/queue effects from forward-pass error.
4. Collect exact native operation shapes or matched whole-forward profiles where
   analytical errors dominate; only then consider measured support/accuracy claims.
5. Bump ConfigIQ's SDK dependency and add its frontend/catalog changes in a later PR.

## Sources

- [Checkpoint config and MIT license provenance](../python/aisimulate/src/aisimulate_core/model_configs/zai-org--GLM-5.3-Flash_README.md).
- [Pinned vLLM model implementation](https://github.com/vllm-project/vllm/tree/0a30bc3f9ac3cc1a9339e115377a99d32252aed0/vllm/models/glm5next).
- [Third-party notices](../THIRD_PARTY_NOTICES.md).
