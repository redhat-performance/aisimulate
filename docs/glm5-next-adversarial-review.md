# GLM-5.3-Flash adversarial review

## Status and verdict

- Reviewed implementation: `96ba7b3fbd723a2b8e083fe0e8eb41f2ff8b061e`.
- Base: `0fc975f36e5b98427985b76a8edf62fea2e5011e`.
- [PR #5](https://github.com/redhat-performance/aisimulate/pull/5) is **closed**, not deleted. Its branch and commits are preserved.
- This review does not change production code, make commits, push changes, or reopen the PR.
- **Verdict: changes required before reopening.** There are nine actionable findings, including an inherited API boundary issue exposed by this integration, plus one compatibility limitation. No P0/P1 memory defect was substantiated after challenging the initial review.

The review used three independent read-only passes (native architecture, memory/runtime, and API/validation), followed by independent CPU reproductions and challenges to the reviewers' conclusions. Passing tests and newly generated goldens were not treated as independent evidence of correctness.

The explicitly deferred GPU accuracy calibration is **not itself a defect**. Findings below concern ignored controls, inconsistent contracts, invalid-input handling, or source-verifiable tensor accounting. No GPU, weights download, or serving-runtime installation was used.

## Reproduce locally

From the AISimulate repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
CUDA_VISIBLE_DEVICES='' python/aisimulate/.venv/bin/python -B -m pytest \
  --color=no -q --tb=short -o addopts='' -p no:cacheprovider \
  docs/glm5_next_adversarial_repros.py
```

At the reviewed commit, the final reproduction suite has **22 failing checks and 3 passing controls**. These are not 22 independent defects: parametrization covers several inputs per finding, and one failure records the pickle limitation. The failing assertions are deliberate review evidence, not updated regression goldens or expected-failure exemptions.

The file is outside the normal test tree and does not have a `test_` filename, so it runs only when explicitly selected. Alternate checkpoint tests replace only metadata loading in memory; parsing, model construction, memory estimation, and native execution use the actual implementation.

## Findings

### F2 — P2: effective expert exclusions bypass the native-FP8 guard

**Code:** `python/aisimulate/src/aisimulate_core/sdk/models/glm5_next.py:96–105`; `sdk/models/helpers.py:48–71` under the same SDK root.

The validator only examines selected exclusion spellings and names beginning with `model.layers.`. Pinned vLLM instead gives a nonempty `ignored_layers` precedence over `modules_to_not_convert`, and its conditional-generation expert prefix is `language_model.model.layers.N.mlp.experts`. The HF alias `model.language_model.layers.N.mlp.experts` maps to that same runtime prefix.

All four combinations of those two names and those two exclusion fields are accepted by AISimulate, even when all 42 routed-expert modules are excluded from FP8. The public memory API still prices them as FP8 and returns the unchanged TP4 weight inventory, **80,728,091,512 bytes/rank**, with positive H200 KV capacity.

For an actual artifact retaining those experts in BF16, the tensor payload difference is **76,101,451,776 bytes/rank** before removing FP8 scales. This could change a supposedly fitting configuration into one that does not fit. This delta is source-derived, not measured GPU memory. Merely editing metadata on unchanged FP8 weight files may instead make loading fail; the review does not claim to have served such an artifact.

**Fix:** resolve backend-effective exclusion-field precedence and module-name mapping, then reject unsupported FFN exclusions before estimating. Do not expand this into speculative glob support: pinned native FP8 uses its own exact matching rules.

**Repro:** `test_f2_effective_expert_exclusions_must_not_be_priced_as_fp8` (four cases).

### F3 — P2: raw telemetry decode has a boundary mismatch; GLM KDA amplifies zero-past behavior

**Code:** `crates/core/src/perfmodel/operators/glm5_next.rs:218–233,421–428`; existing dispatch in `crates/core/src/perfmodel/engine/runtime.rs:2174–2188,2209–2216`.

The raw canonical telemetry API supplies past-KV length, excluding the current token. Static decode supplies an inclusive sequence length. The shared op-level telemetry dispatcher does not normalize this boundary. **That general one-token defect predates this PR:** Qwen3-32B is a control demonstrating it.

The Flash-specific amplification is the new KDA operator's unconditional `ctx.s == 0` early return: scheduled decode work with zero past KV skips the fixed recurrent computation. At H200 TP4, batch 64, SOL:

| Past KV/request | Telemetry, ms | Same-workload static decode, ms |
| ---: | ---: | ---: |
| 0 | 17.136696211667 | 18.182463198333 |
| 3 | 18.183065118333 | 18.183685811667 |
| 2048 | 18.798848051667 | 18.810262238333 |

At zero past KV, the omitted 1.045766986667 ms comprises about **1.010824533 ms KDA recurrence** plus 0.034942453 ms sparse attention.

**Important limit:** the native production replay adapter already normalizes its inclusive scheduler coordinates before calling static generation. These probes do **not** establish that production replay drops this KDA work.

**Fix:** address past/inclusive coordinates at the correct shared telemetry boundary with dense and GLM regressions. Make empty/inconsistent GLM generation handling explicit. Do not blindly add one inside every sparse query, alter whole-forward interpolation's past-KV coordinates, or double-adjust the already-correct production adapter.

**Repro:** `test_f3_canonical_decode_agrees_with_same_workload_static_decode` (three cases), with `test_control_general_decode_boundary_mismatch_predates_flash` documenting inherited causality.

### F4 — P2: the assumed runtime profile is not enforced or preserved

**Code:** `python/aisimulate/src/aisimulate_core/sdk/models/glm5_next.py:52–105,230–235`; `sdk/glm5_next.py:206–212`; canonical identity in `crates/core/src/perfmodel/fpm/config.rs:83–156`.

The graph assumes a tower-free, forced-MQA, non-sequence-parallel baseline, but canonical identity cannot express those choices. In particular:

- Explicit `ModelConfig(language_only=False)` produces exactly the same weights and zero encoder ops as `language_only=True`: **80,728,091,512 bytes/rank** at TP4.
- The canonical API rejects unknown `language_only` and `sparse_mla_force_mqa` fields rather than offering a validated profile that preserves them through reload.
- Pinned vLLM's `sparse_mla_force_mqa` default is **false**, while this graph assumes true.
- Pinned vLLM skips the vision tower when limits for both image and video are zero; sending only text does not alone remove tower residency.

Replay also accepts prefix caching outside the declared baseline. An independently reproduced automatic-capacity run with two sequential 256-token prompts sharing 128 tokens reported 384 computed prefill tokens and a 0.25 reuse ratio, versus 512/0 with prefix caching disabled. No recurrent checkpoint/restore accounting was enabled. This is an **unsupported-profile acceptance issue**, not proof of an out-of-memory event.

**Fix:** add a Rust-owned, validated execution profile or equivalent typed identity through canonical construction, serialization, and relevant callers. Honor or reject explicit residency choices. Reject unsupported prefix/retention behavior until modeled; do not rely solely on the guide's launch instructions.

**Repro:** `test_f4_language_only_false_is_not_silently_reinterpreted`. Canonical identity and runtime defaults were also source-checked; replay observations are bounded as described above.

### F5 — P2: malformed checkpoint geometry is treated as fallback-safe

**Code:** `python/aisimulate/src/aisimulate_core/sdk/models/glm5_next.py:72–75`; `sdk/glm5_next.py:69–85`; native wrapping in `crates/core/src/perfmodel/py_ops.rs:134–152`.

Several invalid-input paths raise ordinary `ValueError` rather than the typed invalid-configuration error recognized by the compilation bridge. With `estimation_mode="auto", fallback_policy="deny"`, 65 attention heads at TP4, an empty layer plan, and an unsupported index head dimension of 64 all return an untrained `fpm_regression` model rather than rejecting the invalid configuration.

The returned model reports `unsupported_config` and estimates `None`. This is not a fabricated numeric result, but it violates the repository's explicit rule that invalid configuration must not trigger estimator fallback.

**Fix:** preserve invalid-configuration classification through GLM parsing, construction, and native validation. Keep genuine coverage/missing-data failures fallback-safe. Test both automatic selection and explicit fallback policies.

**Repro:** `test_f5_invalid_checkpoint_geometry_must_not_trigger_estimator_fallback` (three cases).

### F6 — P2: backend-significant FP8 execution metadata is ignored

**Code:** `python/aisimulate/src/aisimulate_core/sdk/glm5_next.py:87–108`; `sdk/models/glm5_next.py:96–98`; `sdk/models/helpers.py:613–615`.

`activation_scheme="static"` and even `"bad"` are accepted as the dynamic block128 baseline. Pinned vLLM rejects unsupported schemes and specifically requires dynamic activations for block-wise FP8.

Likewise, `store_dtype="mxfp4"` is accepted with unchanged FP8 expert modes, weights and timing. Pinned vLLM uses that field to select `Mxfp4MoEMethod` for routed experts. No successful MXFP4 serving claim on H200 is needed to establish that the estimator has ignored a meaningful execution control.

**Fix:** validate the effective activation and expert-storage contract and reject non-native variants with a non-fallback-safe configuration error until explicitly supported.

**Repro:** `test_f6_non_native_fp8_execution_metadata_is_rejected` (three cases).

### F7 — P2: equivalent nested quantization metadata is rejected

**Code:** `python/aisimulate/src/aisimulate_core/sdk/models/glm5_next.py:96–98`.

Moving the unchanged quantization object into `text_config.quantization_config` is correctly recognized by the shared loader and inferred as `fp8_block`. The new constructor then bypasses that resolver and reads only the root, rejecting the otherwise equivalent metadata as not being the native block128 checkpoint.

**Fix:** use one effective language-quantization resolver for inference, validation and exclusions; preserve root precedence when both declarations exist.

**Repro:** `test_f7_equivalent_nested_quantization_uses_the_shared_resolver`.

### F8 — P2: public attention-imbalance controls are silently ignored

**Code:** `crates/core/src/perfmodel/operators/glm5_next.rs:313–322`.

The sparse operator receives `RuntimeContext` but does not consume its context or generation imbalance multipliers. At H200 TP4, batch 2, ISL 8192, OSL 2, changing both public multipliers from 1 to 2 gives the identical tuple:

```text
(352.7216970092121, 1.8392281633333332, 354.56092517254547)
```

**Fix:** implement the defined correction semantics where applicable, or explicitly reject nondefault values. A silent no-op is not equivalent to deferred calibration.

**Repro:** `test_f8_nondefault_attention_imbalance_controls_have_an_effect_or_are_rejected`.

### F9 — P2: readiness ignores mandatory hardware metadata

**Code:** `crates/core/src/perfmodel/engine/readiness.rs:124–138`; mandatory query dependency in `operators/glm5_next.rs:112–118` under the same perfmodel root.

The shipped H100 PCIe spec lacks scalar `fp32_flops`. GLM construction still reports `ready` under SOL, then the first nonempty estimate fails with `MissingSystemFlopsError`. This dependency is workload-independent, not a shape-specific table miss.

**Fix:** validate required hardware capabilities/metadata during readiness, or explicitly restrict the supported hardware profile. Do not pin a ready estimator that cannot execute any nonempty forward.

**Repro:** `test_f9_mandatory_fp32_hardware_metadata_is_checked_before_ready`.

### F10 — P3: KDA output-norm storage is counted at the compute dtype

**Code:** `python/aisimulate/src/aisimulate_core/sdk/models/glm5_next.py:240–243`; current inventory oracle in `tests/unit/sdk/models/test_glm5_next.py:117–138` under `python/aisimulate/`.

The inventory charges `4 * kda_head_dim` for each KDA output norm. Pinned `FusedRMSNormGated` constructs the parameter with `dtype=None`, within the loader's BF16 default-dtype context. Its FP32 arithmetic does not make its resident parameter FP32.

This overcounts **34 × 128 × 2 = 8,704 bytes per rank**, at every supported TP size. Capacity impact is tiny, hence P3; the important testing lesson is that the current supposedly independent inventory repeats the same dtype mistake.

**Fix:** distinguish parameter storage from arithmetic dtype and correct the source-derived oracle. At TP4 the corrected inventory is **80,728,082,808 bytes**.

**Repro:** `test_f10_kda_norm_residency_follows_runtime_parameter_dtype` (four TP cases).

## Compatibility limitation, not a demonstrated production regression

**C1: complete-model pickle/deepcopy fails.** The fourth native op, `Glm5NextFp32Linear`, is wrapped as a bare native `Operation` at `crates/core/src/perfmodel/py_ops.rs:149–152`. Pickling or deep-copying the complete Flash model raises `TypeError: cannot pickle 'aisimulate_core._native.Operation' object`.

EngineSpec JSON/bincode remains functional. The existing test explicitly checks pickle only for the other three new op types. No current in-repo caller passing a whole `BaseModel` through pickle/deepcopy was demonstrated; actual sweep callers copy `ModelConfig`. This is therefore recorded as a limitation/test gap, **not a merge-blocking P2 production regression**. Add a wrapper/reduction if full Python model serialization is an intended guarantee.

**Repro:** `test_limitation_complete_model_pickle_includes_the_fourth_native_op`.

## Withdrawn or bounded allegations

1. **Withdrawn F1: tiny fixed KV pool proves OOM.** A fixed token pool with `state_cache` absent is not necessarily the entire hybrid memory budget. Automatic capacity already subtracts **36,927,488 bytes of live state per slot at TP4**, plus fragmentation, via `get_kvcache_batch_capacity`. Charging it again against the resulting token pool would double-count it. The successful 1,488,256-byte token-pool example is now a passing scope control, not a failing defect test.
2. **Corrected exclusion namespace.** The initial `model.layers.N.mlp.experts` mutation does not match the pinned conditional-generation runtime prefix. F2 uses the verified effective and mapped HF names, both of which still bypass the estimator's guard.
3. **Inherited decode mismatch.** Dense controls show that the general one-token telemetry boundary mismatch predates this patch. The GLM-specific zero-past recurrence omission and its raw-telemetry scope are separated in F3.
4. **Mixed KDA phase grouping.** Pinned serving may send decode tokens through a chunked KDA path when a batch contains prefill. The model composes separate phases. Its latency impact was not established beyond declared analytical approximations, so this is a calibration risk rather than a confirmed additional defect.
5. **Missing GPU calibration and allocator padding.** These are declared limitations, not newly discovered correctness failures by themselves.

## Checks that held up

- The existing GLM parser/model selection still passes **435 tests** during review; those passing tests coexist with the adversarial failures.
- Existing compiled-engine parity selection passed **85 tests**; the non-FPM engine-step selection passed **279**, with **24 FPM tests deselected** for that read-only review run.
- The expanded bundled config matches the pinned HF config exactly, including **1,509 quantization exclusions**.
- Independent memory review checked **8,192 inverse boundaries** and **7,380 batch-capacity bounds** without finding an error in those inverses.
- The BF16 router weights with FP32 output logits, BF16 conv state, FP32 recurrent state, pooled index-entry sizing, and TP replication/sharding were confirmed against pinned source.
- Native unvalidated provenance, explicit SILICON rejection, and EngineSpec reload work on the tested baseline.

## Recommended repair order

1. Unify quantization resolution, field precedence and effective exclusions; reject unsupported schemes/storage (F2, F6, F7).
2. Define and enforce a canonical runtime profile through FPE and relevant replay callers (F4). Preserve the existing correct live-state reservation.
3. Preserve hard configuration errors and perform mandatory hardware preflight (F5, F9).
4. Correct decode coordinates at the appropriate boundary and add independent phase/empty-work tests; honor or reject imbalance controls (F3, F8).
5. Correct the small dtype inventory error, and decide whether complete-model pickle is promised (F10, C1).
6. Turn accepted review cases into permanent regressions, rerun the existing suites, and review deliberate numerical golden changes rather than refreshing them automatically.
7. Perform the requested ground-truth accuracy comparison separately. Reopen or replace the PR only after local fixes/re-review and user approval.

## Immutable upstream evidence

All vLLM references use commit `0a30bc3f9ac3cc1a9339e115377a99d32252aed0`:

- [FP8 field precedence, scheme checks and expert dispatch](https://github.com/vllm-project/vllm/blob/0a30bc3f9ac3cc1a9339e115377a99d32252aed0/vllm/model_executor/layers/quantization/fp8.py).
- [Exclusion matching](https://github.com/vllm-project/vllm/blob/0a30bc3f9ac3cc1a9339e115377a99d32252aed0/vllm/model_executor/layers/quantization/utils/quant_utils.py).
- [GLM module composition/prefixes](https://github.com/vllm-project/vllm/blob/0a30bc3f9ac3cc1a9339e115377a99d32252aed0/vllm/models/glm5next/common/model.py) and [inherited HF name mapper](https://github.com/vllm-project/vllm/blob/0a30bc3f9ac3cc1a9339e115377a99d32252aed0/vllm/model_executor/models/glm4_1v.py).
- [KDA construction](https://github.com/vllm-project/vllm/blob/0a30bc3f9ac3cc1a9339e115377a99d32252aed0/vllm/models/glm5next/common/kda.py), [output normalization](https://github.com/vllm-project/vllm/blob/0a30bc3f9ac3cc1a9339e115377a99d32252aed0/vllm/third_party/flash_linear_attention/ops/kda.py), and [loader dtype context](https://github.com/vllm-project/vllm/blob/0a30bc3f9ac3cc1a9339e115377a99d32252aed0/vllm/model_executor/model_loader/base_loader.py).
- [Tower residency controls](https://github.com/vllm-project/vllm/blob/0a30bc3f9ac3cc1a9339e115377a99d32252aed0/vllm/model_executor/models/interfaces.py) and [attention configuration defaults](https://github.com/vllm-project/vllm/blob/0a30bc3f9ac3cc1a9339e115377a99d32252aed0/vllm/config/attention.py).
- [Pinned HF config](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/eb9eb208eb0d988989d07a6a12d0fdeb5f52574a/config.json).

Local contracts: [canonical core API](core-api.md), [fixed/state-cache capacity semantics](cli/user-guide.md#manual-state-cache-sizing), and [current baseline guide](glm5-next.md).
