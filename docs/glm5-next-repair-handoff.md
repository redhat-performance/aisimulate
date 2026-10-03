# GLM-5.3-Flash repair handoff

Implementation state for the repairs of
[glm5-next-adversarial-review.md](glm5-next-adversarial-review.md), recorded for
handoff. All changes are **uncommitted working-tree edits** (no commits/pushes
were made, per constraints). No golden fixtures were refreshed, no CodeGraph
index was created, no dependencies were installed.

## Repaired findings

| ID | Fix | Primary files |
| --- | --- | --- |
| F2 | Backend-effective exclusion precedence (`ignored_layers` over `modules_to_not_convert`), vLLM conditional-gen prefix `language_model.model.layers.N.mlp.experts` + HF alias mapping; rejects unsupported FFN exclusions before estimating | `python/aisimulate/src/aisimulate_core/sdk/models/glm5_next.py` |
| F3 | Raw-telemetry past-KV vs inclusive decode coordinate handled at the shared telemetry boundary; explicit empty/inconsistent GLM generation handling; production replay adapter (already correct) left untouched. Inherited general mismatch documented, not hidden | `crates/core/src/perfmodel/operators/glm5_next.rs`; `crates/core/src/perfmodel/fpm/tests.rs` (coordinate comment/arg fixed) |
| F4 | Rust-owned typed execution profile (`glm5_next_text_tp_v1`) + persisted `model_family` through canonical construction, serialization, and reload. Enforced contract: vllm, TP in {1,2,4,8}, PP=attnDP=1, MoE TP=TP/EP=1, fixed quant modes, no speculation/EPLB/slots/backend overrides, FpmInterpolation rejected. Alternate checkpoints resolve via typed family (survives no-Python reload; available metadata still rejects family mismatch). Partial MoE dims (`moe_tp_size` or `moe_ep_size` alone) resolve. Resolution happens **before** estimator selection so an invalid profile can never become a fallback. Replay prefix caching/retention/host-G3/prefix-match rejected | `crates/core/src/perfmodel/fpm/config.rs`, `model.rs`, `crates/core/src/perfmodel/py.rs` (`normalize_config(config_json, model_family=None)`, `validate_replay_cache(...)`, legacy initializer `model_family: None`), `python/aisimulate/src/aisimulate_core/sdk/config.py` (tri-state `language_only`, `ModelConfig.resolve_execution_profile`), `sdk/models/__init__.py`, `sdk/models/glm5_next.py` (direct-constructor guard after raw-metadata validation), `python/aisimulate/src/aisimulate/compiler.py` (replay must set `prefix_caching=False` explicitly), `python/aisimulate/src/aisimulate/config/engine.py` (recommendation cannot drop an explicit profile), `sdk/engine.py`, `_native.pyi` |
| F5 | Invalid checkpoint geometry (typed errors) no longer swallowed by `.ok()`; no untrained `fpm_regression` fallback on `estimation_mode="auto"` or explicit policies; corrupted-download metadata is a hard error (residual resolved 2026-10-02, see open item 1) | `crates/core/src/perfmodel/fpm/config.rs` |
| F6 | Effective `activation_scheme` / `store_dtype` validated against the native block128 baseline; non-native variants rejected with a non-fallback-safe error | `python/aisimulate/src/aisimulate_core/sdk/glm5_next.py` |
| F7 | Single effective language-quantization resolver (root precedence, nested `text_config.quantization_config` accepted) | `sdk/glm5_next.py`, `sdk/models/glm5_next.py` |
| F8 | Sparse operator consumes (or explicitly rejects) the public attention-imbalance multipliers; no silent no-op | `crates/core/src/perfmodel/operators/glm5_next.rs` |
| F9 | Ready-state preflight validates mandatory `fp32_flops` hardware metadata before reporting ready | `crates/core/src/perfmodel/engine/readiness.rs` |
| F10 | KDA output-norm parameters counted at their storage dtype (BF16), not compute dtype; inventory oracle corrected (TP4 = 80,728,082,808 bytes) | `sdk/models/glm5_next.py`; `tests/unit/sdk/models/test_glm5_next.py` |
| C1 | `Glm5NextFp32Linear` wrapped for full-model pickle/deepcopy | `crates/core/src/perfmodel/py_ops.rs`; `sdk/operations/glm5_next.py`, `sdk/operations/__init__.py` |

### F4 follow-up items fixed after independent review

1. Hard metadata-loader errors (e.g. bad `index_head_dim`) propagate as
   invalid-configuration instead of `insufficient_data`, in both explicit
   regression mode and auto selection.
2. Resolved alternate-checkpoint family identity persists in the canonical
   config, so serialize → reload → `best_available` works without Python and
   with metadata unavailable; mismatch against *available* metadata still
   rejects.
3. Either-MoE-dimension omission (`moe_tp_size=4` alone, or `moe_ep_size=1`
   alone) resolves to the supported baseline instead of failing.

## Verification completed

Commands run from `/home/akamra/workspace/aisimulate`, offline environment
(`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=''
PYTHONDONTWRITEBYTECODE=1`), existing venv at `python/aisimulate/.venv`; the
extension at `python/aisimulate/src/aisimulate/_runtime.abi3.so` was
**rebuilt after the F5 residual fix** (2026-10-02) before re-running:

| Suite | Result |
| --- | --- |
| GLM regressions: `tests/unit/sdk/test_glm5_next_config.py`, `models/test_glm5_next.py`, `test_glm5_next_quant_contract.py`, `test_glm5_next_native_repairs.py`, `test_glm5_next_execution_profile_f4.py` | **942 passed** |
| Parity: `crates/core/parity_tests/perfmodel/test_engine_step_parity.py` + `test_compile_engine_parity.py` (release build) | **388 passed** (≈4m37s, re-run 2026-10-02) |
| Broad SDK unit (`tests/unit/sdk`, excl. torch-dependent `database/test_moe_dispatch.py`, `-k 'not parity'`) | **4,659 passed**, 32 deselected, 1 environmental failure (host RAM pressure, `test_fpm_forward.py::test_fpm_detail_distinguishes_memory_budget_from_runtime_capacity`; passes in isolation, 3/3 repeats) |
| Broad unit incl. collector (`tests/unit`, `-k 'not parity'`) | **8,714 passed**; the 96 collector failures are **pre-existing** (identical on clean HEAD, reproduced via `git worktree`; e.g. `collector/sglang/test_collect_mla_module.py` 85f, `collector/test_version_resolver.py` 5f) and the 1 remaining failure is the host-RAM case above |
| Root compiler/replay/core/config regressions (`tests/test_aic*.py`, `tests/sweeper/*`, etc.) | **1,347 passed** |
| Focused model/FPM/opspec/single-oracle recheck | **2,424 passed** |
| Native core lib, no-Python (`cargo test -p aisimulate-core --lib`) | **1,680 passed**, 1 ignored (re-run 2026-10-02) |
| Ruff check/format on touched src+tests; `git diff --check` | pass (1 pre-existing error elsewhere, listed below) |

Two independent read-only review rounds were run: findings #1–#3 (P2) above
were reproduced and repaired; the latest round's single remaining finding
(F5 residual) is resolved and verified — see open item 1.

Stale test expectations reconciled (tests-only, coverage retained):
`sweep/test_sweep.py` (2, omission `language_only=None` semantics),
`test_rust_engine_step.py` (dataclass snapshot now includes keyword-only
`execution_profile`/`model_family`, absent when unset),
`test_util_empirical.py` (`analytic_unvalidated` tier),
`test_v41_native_bridge.py` (invalid-metadata cause is `ValueError`).

## Open items (next steps)

1. **F5 residual — RESOLVED and VERIFIED (2026-10-02).**
   `fpm/config.rs::discover_model_family` routes every availability-candidate
   error through `validate_metadata_availability_chain` (`fpm/config.rs:592`,
   `607–670`), which walks `__cause__` first, then unsuppressed `__context__`
   (capped at 32 links, identity cycle detection). A
   `HuggingFaceDownloadError` caused/contexted by a parser failure
   (`JSONDecodeError`, `UnicodeDecodeError`) is a hard
   `InvalidEngineConfig`; genuine transport failures (`URLError`, 404,
   `TimeoutError`, `IncompleteRead`, suppressed context) stay fallback-safe.
   Permanent coverage:
   `crates/core/tests/glm5_execution_profile_f4.rs::embedded_real_download_parser_corruption_and_transport_controls`
   drives the real `get_model_config_from_model_path` → `_download_hf_json`
   boundary with a mocked `urllib.request.urlopen` (corruption cases in
   `fpm_regression` + `auto` × `deny`/`allow` policies, transport controls in
   `fpm_regression`).
2. **Extension rebuild + re-run — DONE (2026-10-02).** The
   `_runtime.abi3.so` predated the fix and was rebuilt
   (`maturin develop --release --offline --skip-install`). Re-verification:
   GLM suites **942 passed** (up from 926 — the new boundary tests), parity
   **388 passed**, broad `tests/unit/sdk` **4,659 passed** (one host-RAM
   environmental failure that passes in isolation), native lib
   **1,680 passed** / 1 ignored, ruff clean on touched files.
3. **Embedded Rust F4 tests — RESOLVED: 13/13 passing (2026-10-02).**
   Root cause of the recorded failures was invocation-only: every
   `PYTHONPATH` entry must be **absolute** — the test binary does not inherit
   the repository root as cwd, so a relative `python/aisimulate/src` entry
   silently never resolves (`No module named 'yaml'` was the first missing
   import it exposed; `No module named 'aisimulate_core'` is the same bug).
   The command below already uses absolute paths.
4. **Pre-existing, unrelated lint error** (not introduced by this work):
   ruff I001 import-order at
   `python/aisimulate/tests/unit/sdk/database/test_collection_meta_schema.py:112`.
   Pre-existing collector test failures (96, reproduced on clean HEAD) are
   similarly out of scope — see the verification table above.
5. **No GPU-accuracy claim.** All verification is CPU-side contract/parity
   evidence. H200 serving behavior (KV capacity, mixed KDA phase, FP8/MXFP4
   kernel selection) remains calibration-only, as bounded in the review.

## Build / run commands

```bash
# Rebuild extension in-place, offline, no dependency installs
cd python/aisimulate
PATH="$HOME/.cargo/bin:$PATH" \
  VIRTUAL_ENV="$PWD/.venv" \
  PYO3_PYTHON="$PWD/.venv/bin/python" \
  maturin develop --release --offline --skip-install

# Embedded (python feature) Rust F4 tests — both library paths required.
# PYTHONPATH entries MUST be absolute: the test binary does not run with the
# repository root as cwd, so a relative src entry silently never resolves
# (observed as ModuleNotFoundError for yaml/aisimulate_core).
cd /home/akamra/workspace/aisimulate
PATH="$HOME/.cargo/bin:$PATH" \
  LD_LIBRARY_PATH="$HOME/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/lib:$LD_LIBRARY_PATH" \
  PYTHONPATH="$PWD/python/aisimulate/src:$(python/aisimulate/.venv/bin/python -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')" \
  cargo test --offline -p aisimulate-core --features embed-python --test glm5_execution_profile_f4

# GLM regressions
python/aisimulate/.venv/bin/python -B -m pytest -c python/aisimulate/pytest.ini \
  -o addopts='' -p no:cacheprovider -q --tb=short \
  python/aisimulate/tests/unit/sdk/test_glm5_next_config.py \
  python/aisimulate/tests/unit/sdk/models/test_glm5_next.py \
  python/aisimulate/tests/unit/sdk/test_glm5_next_quant_contract.py \
  python/aisimulate/tests/unit/sdk/test_glm5_next_native_repairs.py \
  python/aisimulate/tests/unit/sdk/test_glm5_next_execution_profile_f4.py

# Full parity (release build required; ~4 min)
python/aisimulate/.venv/bin/python -B -m pytest -o addopts='' -p no:cacheprovider -q \
  crates/core/parity_tests/perfmodel/test_engine_step_parity.py \
  crates/core/parity_tests/perfmodel/test_compile_engine_parity.py
```

## File inventory (uncommitted)

Modified:
`crates/core/src/perfmodel/{engine/readiness.rs,engine/runtime.rs,fpm/config.rs,fpm/model.rs,fpm/tests.rs,operators/glm5_next.rs,operators/mod.rs,py.rs,py_ops.rs,session.rs}`,
`crates/core/src/python.rs`,
`python/aisimulate/src/aisimulate/{compiler.py,config/engine.py}`,
`python/aisimulate/src/aisimulate_core/{_native.pyi}`,
`python/aisimulate/src/aisimulate_core/sdk/{config.py,engine.py,glm5_next.py,models/__init__.py,models/glm5_next.py,operations/__init__.py,operations/glm5_next.py,rust_engine_step.py}`,
`python/aisimulate/tests/unit/sdk/{models/test_glm5_next.py,sweep/test_sweep.py,test_glm5_next_config.py,test_rust_engine_step.py,test_util_empirical.py,test_v41_native_bridge.py}`.

New: `crates/core/tests/glm5_execution_profile_f4.rs`,
`crates/core/tests/glm5_native_repairs.rs`,
`python/aisimulate/tests/unit/sdk/test_glm5_next_execution_profile_f4.py`,
`python/aisimulate/tests/unit/sdk/test_glm5_next_native_repairs.py`,
`python/aisimulate/tests/unit/sdk/test_glm5_next_quant_contract.py`,
`docs/glm5-next-adversarial-review.md`,
`docs/glm5_next_adversarial_repros.py`, this file.
