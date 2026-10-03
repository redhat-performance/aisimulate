# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""F4 permanent regressions. Native cases require the coordinator's rebuilt wheel.

No latency/weight goldens: identity is asserted directly, cache work is compared
with the declared no-reuse contract, and live-state reservation is tested once.
"""

import json
from dataclasses import asdict, replace

import pytest

import aisimulate_core
from aisimulate.compiler import prediction_to_replay_spec
from aisimulate.config import CorePredictionConfig
from aisimulate.config.engine import EngineRecommendationConfig
from aisimulate.runner import EngineReplayRunnerFactory
from aisimulate.sweeper.replay import BackendDeploymentSpec, ReplaySpec
from aisimulate_core.sdk import ForwardPassPerfModelConfig, RustForwardPassPerfModel, models
from aisimulate_core.sdk.config import ModelConfig
from aisimulate_core.sdk.engine import EngineHandle, build_engine_spec_json
from aisimulate_core.sdk.errors import InvalidEngineConfigurationError

pytestmark = pytest.mark.unit
MODEL = "zai-org/GLM-5.3-Flash"
PROFILE = "glm5_next_text_tp_v1"


def canonical(**changes):
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
            "database_mode": "SOL",
            "estimation_mode": "op_level",
            **changes,
        }
    )


def low_level(**changes):
    return ModelConfig(tp_size=4, moe_tp_size=4, moe_ep_size=1, **changes)


def normalize(config):
    return json.loads(aisimulate_core.RustForwardPassPerfModel.normalize_config(json.dumps(config.to_dict())))


def test_saved_resolved_config_has_explicit_profile_and_reloads(tmp_path):
    estimator = RustForwardPassPerfModel.best_available(canonical())
    saved = estimator.diagnostics()["provenance"]["config"]
    assert saved["execution_profile"] == PROFILE
    assert saved["model_family"] == "GLM5NEXT"
    path = tmp_path / "f4-config.json"
    path.write_text(json.dumps(saved))
    reloaded = RustForwardPassPerfModel.best_available(json.loads(path.read_text()))
    assert reloaded.diagnostics()["provenance"]["config"] == saved
    assert reloaded.static_phase_latency(batch_size=1, input_tokens=256, output_tokens=2, prefill=True) == (
        estimator.static_phase_latency(batch_size=1, input_tokens=256, output_tokens=2, prefill=True)
    )


def test_explicit_profile_dataclass_serialization_and_replacement():
    config = canonical(execution_profile=PROFILE, model_family="GLM5NEXT")
    restored = ForwardPassPerfModelConfig(**json.loads(json.dumps(config.to_dict())))
    assert restored.execution_profile == PROFILE
    assert restored.model_family == "GLM5NEXT"
    assert replace(restored, worker_type="prefill").execution_profile == PROFILE


def test_metadata_infers_profile_for_alternate_checkpoint(monkeypatch):
    from aisimulate_core.sdk import utils

    original = utils.get_model_config_from_model_path(MODEL)
    monkeypatch.setattr(utils, "get_model_config_from_model_path", lambda _: original)
    assert normalize(canonical(model="alternate/flash"))["execution_profile"] == PROFILE


@pytest.mark.parametrize("mode", ["fpm_regression", "auto"])
@pytest.mark.parametrize("policy", ["deny", "allow"])
def test_alternate_malformed_metadata_cannot_bypass_validation(tmp_path, monkeypatch, mode, policy):
    from aisimulate_core.sdk import engine, utils

    raw = json.loads(json.dumps(utils._load_model_config_from_model_path(MODEL)))
    raw["text_config"]["index_head_dim"] = 64
    (tmp_path / "config.json").write_text(json.dumps(raw))

    def unexpected_compilation(*args, **kwargs):
        pytest.fail("hard metadata errors must surface before any estimator candidate is compiled")

    monkeypatch.setattr(engine, "compile_engine", unexpected_compilation)
    with pytest.raises(ValueError, match="index_head_dim"):
        RustForwardPassPerfModel.best_available(
            canonical(model=str(tmp_path), estimation_mode=mode, fallback_policy=policy)
        )


def test_alternate_regression_reload_retains_identity_when_metadata_is_unavailable(monkeypatch):
    from aisimulate_core.sdk import utils

    original = utils.get_model_config_from_model_path(MODEL)
    monkeypatch.setattr(utils, "get_model_config_from_model_path", lambda _: original)
    model = RustForwardPassPerfModel.best_available(
        canonical(model="alternate/flash", estimation_mode="fpm_regression")
    )
    saved = json.loads(json.dumps(model.diagnostics()["provenance"]["config"]))
    assert saved["model_family"] == "GLM5NEXT" and saved["execution_profile"] == PROFILE
    # The public SDK dataclass must accept the complete resolved identity too.
    reloaded_config = ForwardPassPerfModelConfig(**saved)

    def unavailable(_):
        raise utils.HuggingFaceDownloadError("checkpoint metadata is offline")

    monkeypatch.setattr(utils, "get_model_config_from_model_path", unavailable)
    reloaded = RustForwardPassPerfModel.best_available(reloaded_config)
    assert reloaded.diagnostics()["provenance"]["config"] == saved
    assert reloaded.diagnostics()["readiness"] == "insufficient_data"


@pytest.mark.parametrize("mode", ["fpm_regression", "auto"])
@pytest.mark.parametrize("policy", ["deny", "allow"])
@pytest.mark.parametrize(
    "content,cause_type", [(b'{"architectures": [broken', json.JSONDecodeError), (b"\xff", UnicodeDecodeError)]
)
def test_downloaded_corrupt_json_is_hard_error_before_candidate_compilation(
    monkeypatch, mode, policy, content, cause_type
):
    from io import BytesIO

    from aisimulate_core.sdk import engine, utils

    requests = []

    def response(request, *, timeout):
        requests.append((request.full_url, timeout))
        return BytesIO(content)

    def unexpected_compilation(*args, **kwargs):
        pytest.fail("downloaded corrupt metadata must fail before candidate compilation")

    monkeypatch.setattr(utils, "_get_hf_auth_headers", lambda: {})
    monkeypatch.setattr(utils.urllib.request, "urlopen", response)
    monkeypatch.setattr(engine, "compile_engine", unexpected_compilation)
    # Exercise the real urllib -> json.load -> SDK wrapper boundary, not just
    # a synthetic wrapper instance. Exceptions are not cached by the loader.
    model_id = "f4-review/corrupt-http-metadata"
    with pytest.raises(utils.HuggingFaceDownloadError) as caught:
        utils._download_hf_json(model_id, "config.json")
    assert isinstance(caught.value.__cause__, cause_type)
    with pytest.raises(ValueError, match=cause_type.__name__):
        RustForwardPassPerfModel.best_available(canonical(model=model_id, estimation_mode=mode, fallback_policy=policy))
    assert requests and all(url.endswith("/config.json") and timeout == 30 for url, timeout in requests)


@pytest.mark.parametrize("kind", ["url", "http", "timeout", "incomplete_read"])
def test_real_download_network_failures_remain_metadata_unavailability(monkeypatch, kind):
    from email.message import Message
    from http.client import IncompleteRead
    from urllib.error import HTTPError, URLError

    from aisimulate_core.sdk import utils

    def unavailable(request, *, timeout):
        if kind == "url":
            raise URLError("offline")
        if kind == "http":
            raise HTTPError(request.full_url, 404, "missing metadata", Message(), None)
        if kind == "timeout":
            raise TimeoutError("timed out")
        raise IncompleteRead(b"partial response", 100)

    monkeypatch.setattr(utils, "_get_hf_auth_headers", lambda: {})
    monkeypatch.setattr(utils.urllib.request, "urlopen", unavailable)
    # Declared family identity must still survive a real wrapped network miss.
    model = RustForwardPassPerfModel.best_available(
        canonical(
            model=f"f4-review/unavailable-{kind}",
            estimation_mode="fpm_regression",
            model_family="GLM5NEXT",
            execution_profile=PROFILE,
        )
    )
    diagnostics = model.diagnostics()
    assert diagnostics["readiness"] == "insufficient_data"
    assert diagnostics["provenance"]["config"]["model_family"] == "GLM5NEXT"
    assert diagnostics["provenance"]["config"]["execution_profile"] == PROFILE


@pytest.mark.parametrize("chain_kind", ["cycle", "too_long", "implicit_parser", "suppressed_parser"])
def test_download_availability_exception_chain_is_bounded_and_respects_suppression(monkeypatch, chain_kind):
    from aisimulate_core.sdk import utils

    corruption = json.JSONDecodeError("corrupt metadata", "{", 1)
    wrapper = utils.HuggingFaceDownloadError("outer wrapper")
    if chain_kind == "cycle":
        wrapper.__cause__ = wrapper
    elif chain_kind == "too_long":
        for _ in range(40):
            parent = utils.HuggingFaceDownloadError("nested wrapper")
            parent.__cause__ = wrapper
            wrapper = parent
    else:
        wrapper.__context__ = corruption
        wrapper.__suppress_context__ = chain_kind == "suppressed_parser"

    def response(request, *, timeout):
        raise wrapper

    monkeypatch.setattr(utils, "_get_hf_auth_headers", lambda: {})
    monkeypatch.setattr(utils.urllib.request, "urlopen", response)
    request = canonical(model=f"f4-review/chain-{chain_kind}", estimation_mode="fpm_regression")
    if chain_kind == "suppressed_parser":
        assert RustForwardPassPerfModel.best_available(request).diagnostics()["readiness"] == "insufficient_data"
    else:
        expected = {"cycle": "cyclic", "too_long": "exceeds 32", "implicit_parser": "JSONDecodeError"}
        with pytest.raises(ValueError, match=expected[chain_kind]):
            RustForwardPassPerfModel.best_available(request)


@pytest.mark.parametrize("failure", [InvalidEngineConfigurationError, ValueError, RuntimeError])
def test_persisted_family_does_not_mask_invalid_or_unexpected_metadata(monkeypatch, failure):
    from aisimulate_core.sdk import utils

    def invalid(_):
        raise failure("malformed checkpoint metadata")

    monkeypatch.setattr(utils, "get_model_config_from_model_path", invalid)
    with pytest.raises(ValueError, match="malformed checkpoint metadata"):
        RustForwardPassPerfModel.best_available(
            canonical(
                model="alternate/flash",
                model_family="GLM5NEXT",
                execution_profile=PROFILE,
                estimation_mode="fpm_regression",
            )
        )


def test_persisted_family_is_revalidated_against_available_dense_metadata(monkeypatch):
    from aisimulate_core.sdk import utils

    dense = utils.get_model_config_from_model_path("Qwen/Qwen3-32B")
    monkeypatch.setattr(utils, "get_model_config_from_model_path", lambda _: dense)
    with pytest.raises(ValueError, match="requires GLM5NEXT"):
        RustForwardPassPerfModel.best_available(
            canonical(
                model="alternate/flash",
                model_family="GLM5NEXT",
                execution_profile=PROFILE,
                estimation_mode="fpm_regression",
            )
        )


def test_unsupported_architecture_cannot_mask_a_persisted_family_mismatch(monkeypatch):
    from aisimulate_core.sdk import utils

    def unsupported(_):
        raise ValueError("The model's architecture UnknownForCausalLM is not supported. Supported architectures: ...")

    monkeypatch.setattr(utils, "get_model_config_from_model_path", unsupported)
    with pytest.raises(ValueError, match="requires GLM5NEXT"):
        RustForwardPassPerfModel.best_available(
            canonical(
                model="alternate/flash",
                model_family="GLM5NEXT",
                execution_profile=PROFILE,
                estimation_mode="fpm_regression",
            )
        )
    # Undeclared unsupported families retain the existing regression contract.
    model = RustForwardPassPerfModel.best_available(
        canonical(model="alternate/unknown", estimation_mode="fpm_regression")
    )
    assert model.diagnostics()["readiness"] == "insufficient_data"
    assert model.diagnostics()["provenance"]["config"]["model_family"] is None


@pytest.mark.parametrize("changes", [{"moe_tp_size": 4}, {"moe_ep_size": 1}])
def test_supported_partial_low_level_moe_dimensions_are_preserved(changes):
    model = models.get_model(MODEL, ModelConfig(tp_size=4, **changes), "vllm")
    assert model.config.moe_tp_size == 4
    assert model.config.moe_ep_size == 1
    assert model.config.execution_profile == PROFILE


@pytest.mark.parametrize(
    "changes",
    [
        {"moe_tp_size": 0},
        {"moe_ep_size": 0},
        {"moe_tp_size": 3},
        {"moe_ep_size": 3},
        {"moe_tp_size": 2},
        {"moe_ep_size": 2},
    ],
)
def test_incompatible_partial_low_level_moe_dimensions_reject(changes):
    with pytest.raises(InvalidEngineConfigurationError, match="MoE|EP=1"):
        models.get_model(MODEL, ModelConfig(tp_size=4, **changes), "vllm")


@pytest.mark.parametrize("model", ["Qwen/Qwen3-32B", "Qwen/Qwen3-0.6B"])
@pytest.mark.parametrize("policy", ["deny", "allow"])
def test_profile_family_mismatch_never_falls_back(model, policy):
    with pytest.raises(ValueError, match="requires GLM5NEXT"):
        RustForwardPassPerfModel.best_available(
            canonical(model=model, execution_profile=PROFILE, estimation_mode="auto", fallback_policy=policy)
        )


@pytest.mark.parametrize("profile", ["glm5_next_text_tp_v2", "full", False])
def test_unknown_profile_identity_rejects(profile):
    with pytest.raises(ValueError, match="execution_profile"):
        normalize(canonical(execution_profile=profile))


@pytest.mark.parametrize(
    "changes",
    [
        {"pp": 2},
        {"attention_dp": 2},
        {"backend": "sglang"},
        {"nextn": 1},
        {"attention_backend": "fa3"},
        {"enable_eplb": True},
        {"gemm_quant_mode": "bfloat16"},
        {"kvcache_quant_mode": "fp8"},
        {"estimation_mode": "fpm_interpolation"},
    ],
)
def test_non_profile_canonical_inputs_reject_before_selection(changes):
    with pytest.raises(ValueError, match="glm5_next_text_tp_v1"):
        RustForwardPassPerfModel.best_available(canonical(fallback_policy="allow", **changes))


def test_low_level_explicit_false_is_not_reinterpreted():
    config = low_level(language_only=False)
    with pytest.raises(InvalidEngineConfigurationError, match="language_only=True"):
        models.get_model(MODEL, config, "vllm")
    assert config.language_only is False


def test_direct_glm_constructor_also_rejects_explicit_tower_residency():
    # Coordinator integration gate: get_model's shared boundary is not enough
    # for constructors used directly by SDK/embedded callers.
    model = models.get_model(MODEL, low_level(), "vllm")
    info = dict(models._get_model_info(MODEL))
    info.update(model_path=MODEL, model_family="GLM5NEXT")
    with pytest.raises(InvalidEngineConfigurationError, match="language_only=True"):
        type(model).create(info, replace(model.config, language_only=False), "vllm")


@pytest.mark.parametrize("language_only", [None, True])
def test_low_level_baseline_infers_tower_free_identity(language_only):
    model = models.get_model(MODEL, low_level(language_only=language_only), "vllm")
    assert model.config.language_only is True
    assert model.config.execution_profile == PROFILE
    assert model.encoder_ops == []
    restored = ModelConfig(**json.loads(json.dumps(asdict(model.config), default=lambda item: item.name)))
    assert restored.language_only is True and restored.execution_profile == PROFILE


def test_other_family_omission_still_means_tower_resident():
    config = ModelConfig()
    config.resolve_execution_profile("dense", "LLAMA", "vllm")
    assert config.language_only is False
    assert config.execution_profile is None


@pytest.mark.parametrize("changes", [{"cp_size": 2}, {"cp_style": "ring"}])
def test_low_level_sequence_parallel_requests_reject(changes):
    with pytest.raises(InvalidEngineConfigurationError, match="CP=1|sequence parallelism"):
        low_level(**changes).resolve_execution_profile(MODEL, "GLM5NEXT", "vllm")


def test_engine_spec_profile_identity_survives_bincode_reload():
    model = models.get_model(MODEL, low_level(), "vllm")
    spec_json = build_engine_spec_json(
        model,
        model_path=MODEL,
        system="h200_sxm",
        backend="vllm",
        backend_version="0.24.0",
        kv_block_size=64,
        systems_path=None,
        nextn=0,
        database_mode="SOL",
    )
    assert json.loads(spec_json)["engine"]["extra"]["execution_profile"] == PROFILE
    bytes_ = bytes(aisimulate_core.engine_spec_bincode_from_json(spec_json))
    reloaded = EngineHandle(bytes_)
    assert reloaded.spec_bytes == bytes_
    assert reloaded.run_static(batch_size=1, isl=256, osl=2)[2] > 0


def public_config(**cache):
    return CorePredictionConfig.model_validate(
        {
            "engine": {
                "model": MODEL,
                "hardware": "h200_sxm",
                "backend": "vllm",
                "backend_version": "0.24.0",
                "context_length": 2048,
                "database_mode": "SOL",
                "workers": {
                    "aggregated": {
                        "parallelism": {"tensor": 4, "moe_tensor": 4},
                        "scheduler": {"max_sequences": 2},
                        "kv_cache": cache,
                    }
                },
            },
            "traffic": {
                "source": {"type": "synthetic", "input_tokens": 256, "output_tokens": 2, "cached_prefix_tokens": 128},
                "load": {"type": "concurrency", "concurrency": 1},
                "stop": {"requests": 2},
            },
        }
    )


def test_compiler_rejects_implicit_prefix_caching_default():
    with pytest.raises(ValueError, match="prefix_caching=false"):
        prediction_to_replay_spec(public_config())


def test_compiler_saves_profile_with_prefix_caching_explicitly_disabled():
    spec = prediction_to_replay_spec(public_config(prefix_caching=False))
    args = spec.backend_deployment.agg_engine_args
    assert args["enable_prefix_caching"] is False
    assert args["timing_model"]["config"]["execution_profile"] == PROFILE
    assert spec.backend_deployment.performance_model_metadata["aggregated"]["config"]["execution_profile"] == PROFILE
    assert "state_cache" not in args  # automatic capacity already reserves live state


def test_dense_compiler_keeps_default_prefix_cache_and_absent_moe_overrides():
    data = public_config().model_dump(mode="json")
    data["engine"].update(model="Qwen/Qwen3-32B", database_mode="SILICON")
    worker = data["engine"]["workers"]["aggregated"]
    worker["parallelism"].update(tensor=2, moe_tensor=1, moe_expert=1)
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(data))
    args = spec.backend_deployment.agg_engine_args
    assert args["enable_prefix_caching"] is True
    assert "execution_profile" not in args["timing_model"]["config"]
    assert args["timing_model"]["config"]["moe_tp_size"] is None


def test_recommendation_cannot_silently_drop_an_explicit_profile():
    with pytest.raises(ValueError, match="profile-aware Sweeper"):
        EngineRecommendationConfig.model_validate(
            {
                "model": MODEL,
                "hardware": "h200_sxm",
                "backend": "vllm",
                "mode": "aggregated",
                "execution_profile": PROFILE,
                "workers": {"aggregated": {}},
            }
        )


@pytest.mark.parametrize("host,alignment", [(True, False), (False, True)])
def test_rust_profile_validator_rejects_retention_even_with_prefix_disabled(host, alignment):
    with pytest.raises(ValueError, match="prefix retention|host/G3"):
        aisimulate_core.RustForwardPassPerfModel.validate_replay_cache(
            json.dumps(canonical().to_dict()), False, host, alignment
        )


def replay_spec(**overrides):
    args = {
        "tensor_parallel_size": 4,
        "dp_size": 1,
        "block_size": 64,
        "max_num_seqs": 2,
        "max_num_batched_tokens": 8192,
        "timing_model": {"type": "external", "provider": "aic", "config": canonical().to_dict()},
        **overrides,
    }
    return ReplaySpec(
        backend_deployment=BackendDeploymentSpec(
            deployment_mode="agg",
            backend="vllm",
            backend_version="0.24.0",
            agg_engine_args=args,
            num_workers=1,
        ),
        workload={
            "source_type": "synthetic",
            "load_type": "concurrency",
            "isl": 256,
            "osl": 2,
            "cached_prefix_tokens": 128,
            "request_count": 2,
            "concurrency": 1,
        },
        goal={},
        execution_mode="offline",
        concurrency=1,
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"enable_prefix_caching": True},
        {
            "enable_prefix_caching": True,
            "native_host_offload": {"num_host_blocks": 8},
            "kv_cache_bytes_per_token": 11627,
        },
        {
            "enable_prefix_caching": True,
            "native_host_offload": {"num_host_blocks": 8},
            "g3_offload": {"scope": "worker_local", "num_g3_blocks": 8},
            "kv_cache_bytes_per_token": 11627,
        },
        {
            "enable_prefix_caching": False,
            "prefix_match_unit": 64,
            "num_gpu_blocks": 4096,
            "kv_cache_bytes_per_token": 11627,
            "state_cache": {"bytes_per_request": 36927488},
        },
    ],
)
def test_direct_replay_rejects_prefix_reuse_host_and_state_retention(overrides):
    from aisimulate import _runtime

    with pytest.raises((ValueError, RuntimeError), match="prefix|retention|offload"):
        EngineReplayRunnerFactory(runtime=_runtime).create(0).run(replay_spec(**overrides))


def test_direct_replay_prefix_false_computes_all_prompt_tokens_without_reuse():
    from aisimulate import _runtime

    result = EngineReplayRunnerFactory(runtime=_runtime).create(0).run(replay_spec(enable_prefix_caching=False))
    assert result.metrics["completed_requests"] == 2
    assert result.metrics["committed_prefill_tokens"] == 512
    assert result.metrics["prefix_cache_reused_ratio"] == 0


def test_fixed_small_token_pool_is_not_charged_for_live_state_twice():
    from aisimulate import _runtime

    # Withdrawn F1 scope control: the token pool is not the whole hybrid budget.
    # 2 blocks * 64 * 11627 = 1,488,256 token-pool bytes, less than one live state.
    # The scheduler-slot live state is already reserved by automatic capacity.
    spec = replay_spec(enable_prefix_caching=False, num_gpu_blocks=2)
    spec = replace(spec, workload={**spec.workload, "isl": 64, "osl": 1, "request_count": 1, "cached_prefix_tokens": 0})
    result = EngineReplayRunnerFactory(runtime=_runtime).create(0).run(spec)
    assert result.metrics["completed_requests"] == 1


def test_explicit_live_state_without_prefix_retention_remains_supported():
    from aisimulate import _runtime

    spec = replay_spec(
        enable_prefix_caching=False,
        num_gpu_blocks=256,
        kv_cache_bytes_per_token=11627,
        state_cache={"bytes_per_request": 36_927_488},
    )
    result = EngineReplayRunnerFactory(runtime=_runtime).create(0).run(spec)
    assert result.metrics["completed_requests"] == 2
    assert result.metrics["committed_prefill_tokens"] == 512
    assert result.metrics["prefix_cache_reused_ratio"] == 0


def test_automatic_capacity_preserves_single_fixed_state_reservation():
    model = models.get_model(MODEL, low_level(), "vllm")
    state = model._fixed_state_bytes()
    assert state == 36_927_488  # independent pinned logical-state inventory from F4 review
    assert model.get_kvcache_batch_capacity(state - 1, 1) == 0
    assert model.get_kvcache_batch_capacity(4 * state - 1, 4) == 0
    budget = 4 * state + 64 * model.get_kvcache_bytes_per_sequence(256)
    capacity = model.get_kvcache_batch_capacity(budget, 4)
    assert capacity > 0
    assert model.get_kvcache_batch_capacity(budget, 8) < capacity
