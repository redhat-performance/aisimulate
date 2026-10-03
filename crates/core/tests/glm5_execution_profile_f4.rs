// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! F4 identity/cache contract tests: no timing goldens or Python required.

use aisimulate_core::perfmodel::{ForwardPassPerfModelConfig, engine::spec::EngineSpec};
use serde_json::{Value, json};

const MODEL: &str = "zai-org/GLM-5.3-Flash";
const PROFILE: &str = "glm5_next_text_tp_v1";

#[cfg(feature = "python")]
static PYTHON_METADATA_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn config(changes: Value) -> ForwardPassPerfModelConfig {
    let mut value = json!({
        "model": MODEL, "system": "h200_sxm", "backend": "vllm",
        "worker_type": "aggregated", "tp": 4, "database_mode": "SOL"
    });
    value
        .as_object_mut()
        .unwrap()
        .extend(changes.as_object().unwrap().clone());
    serde_json::from_value(value).unwrap()
}

#[test]
fn omitted_glm_profile_is_resolved_and_saved_explicitly() {
    let mut config = config(json!({}));
    config.resolve_execution_profile(None).unwrap();
    assert_eq!(config.execution_profile.unwrap().as_str(), PROFILE);
    assert_eq!(config.moe_tp_size, Some(4));
    assert_eq!(config.moe_ep_size, Some(1));
    let saved = serde_json::to_value(&config).unwrap();
    assert_eq!(saved["execution_profile"], PROFILE);
    assert_eq!(saved["model_family"], "GLM5NEXT");
    let mut restored: ForwardPassPerfModelConfig = serde_json::from_value(saved).unwrap();
    restored.resolve_execution_profile(None).unwrap();
    assert_eq!(config, restored);
}

#[test]
fn metadata_infers_profile_for_an_alternate_checkpoint_name() {
    let mut config = config(json!({"model": "local/alternate-flash"}));
    config.resolve_execution_profile(Some("GLM5NEXT")).unwrap();
    assert_eq!(config.execution_profile.unwrap().as_str(), PROFILE);
    let mut restored: ForwardPassPerfModelConfig =
        serde_json::from_str(&serde_json::to_string(&config).unwrap()).unwrap();
    restored.resolve_execution_profile(None).unwrap();
    assert_eq!(restored, config);
}

#[test]
fn persisted_family_identity_is_cross_checked_against_known_metadata() {
    let mut config = config(json!({"model": "alternate/flash"}));
    config.resolve_execution_profile(Some("GLM5NEXT")).unwrap();
    let mut restored: ForwardPassPerfModelConfig =
        serde_json::from_str(&serde_json::to_string(&config).unwrap()).unwrap();
    assert!(
        restored
            .resolve_execution_profile(Some("QWEN3"))
            .unwrap_err()
            .to_string()
            .contains("requires GLM5NEXT")
    );
}

#[test]
fn profile_resolves_both_supported_partial_moe_dimensions() {
    for changes in [json!({"moe_tp_size": 4}), json!({"moe_ep_size": 1})] {
        let mut config = config(changes);
        config.resolve_execution_profile(None).unwrap();
        assert_eq!(config.moe_tp_size, Some(4));
        assert_eq!(config.moe_ep_size, Some(1));
    }
    for changes in [
        json!({"moe_tp_size": 0}),
        json!({"moe_ep_size": 0}),
        json!({"moe_tp_size": 3}),
        json!({"moe_ep_size": 3}),
        json!({"moe_tp_size": 2}),
        json!({"moe_ep_size": 2}),
        json!({"moe_tp_size": 4, "moe_ep_size": 2}),
        json!({"moe_tp_size": u32::MAX}),
        json!({"moe_ep_size": u32::MAX}),
    ] {
        let mut config = config(changes);
        assert!(config.resolve_execution_profile(None).is_err());
    }
}

#[test]
fn rust_callers_can_select_the_typed_profile_without_json() {
    let mut config = config(json!({}));
    config.execution_profile = Some(ForwardPassPerfModelConfig::GLM5_NEXT_TEXT_TP_V1);
    config.resolve_execution_profile(None).unwrap();
    assert_eq!(config.execution_profile.unwrap().as_str(), PROFILE);
}

#[test]
fn explicit_profile_rejects_other_families_and_authoritative_metadata() {
    for (name, family) in [("Qwen/Qwen3-32B", None), (MODEL, Some("LLAMA"))] {
        let mut config = config(json!({"model": name, "execution_profile": PROFILE}));
        let error = config.resolve_execution_profile(family).unwrap_err();
        assert!(error.to_string().contains("requires GLM5NEXT"));
    }
}

#[test]
fn unknown_profile_versions_and_untyped_runtime_flags_fail_closed() {
    for changes in [
        json!({"execution_profile": "glm5_next_text_tp_v2"}),
        json!({"execution_profile": "full"}),
        json!({"language_only": false}),
        json!({"sparse_mla_force_mqa": false}),
        json!({"use_sequence_parallel_moe": true}),
        json!({"prefix_caching": true}),
    ] {
        let mut value = serde_json::to_value(config(json!({}))).unwrap();
        value
            .as_object_mut()
            .unwrap()
            .extend(changes.as_object().unwrap().clone());
        assert!(serde_json::from_value::<ForwardPassPerfModelConfig>(value).is_err());
    }
}

#[test]
fn incompatible_canonical_controls_are_invalid_not_fallback_safe() {
    for changes in [
        json!({"backend": "sglang"}),
        json!({"tp": 3}),
        json!({"pp": 2}),
        json!({"attention_dp": 2}),
        json!({"moe_tp_size": 2, "moe_ep_size": 2}),
        json!({"nextn": 1}),
        json!({"decoder_replay": true}),
        json!({"enable_eplb": true}),
        json!({"wideep_num_slots": 288}),
        json!({"attention_backend": "fa3"}),
        json!({"moe_backend": "deepep_moe"}),
        json!({"gemm_quant_mode": "bfloat16"}),
        json!({"moe_quant_mode": "bfloat16"}),
        json!({"fmha_quant_mode": "fp8"}),
        json!({"kvcache_quant_mode": "fp8"}),
        json!({"comm_quant_mode": "int8"}),
        json!({"estimation_mode": "fpm_interpolation"}),
        json!({"speculation": {"kind": "ngram", "params": {"num_speculative_tokens": 2}}}),
    ] {
        let mut config = config(changes.clone());
        let error = config.resolve_execution_profile(None).unwrap_err();
        assert!(
            matches!(error, aisimulate_core::AicError::InvalidEngineConfig(_)),
            "{changes}: {error}"
        );
    }
}

#[test]
fn replay_rejects_reuse_retention_and_offload_but_allows_live_state() {
    let mut config = config(json!({}));
    config.resolve_execution_profile(None).unwrap();
    for (prefix, host, alignment) in [
        (true, false, false),
        (false, true, false),
        (false, false, true),
    ] {
        let error = config
            .validate_replay_cache(prefix, host, alignment)
            .unwrap_err();
        assert!(error.to_string().contains("prefix_caching=false"));
    }
    config.validate_replay_cache(false, false, false).unwrap();
}

#[test]
fn dense_default_retains_its_cache_contract() {
    let mut config = config(json!({"model": "Qwen/Qwen3-32B"}));
    config.resolve_execution_profile(Some("QWEN3")).unwrap();
    assert_eq!(config.execution_profile, None);
    config.validate_replay_cache(true, true, true).unwrap();
}

#[test]
fn engine_spec_extra_identity_survives_json_and_bincode_without_layout_change() {
    let spec: EngineSpec = serde_json::from_value(json!({
        "schema_version": aisimulate_core::ENGINE_SPEC_SCHEMA_VERSION,
        "engine": {
            "schema_version": aisimulate_core::ENGINE_CONFIG_SCHEMA_VERSION,
            "model_name": MODEL, "system_name": "h200_sxm", "backend": "vllm",
            "tp_size": 4, "pp_size": 1, "weight_dtype": "fp8_block",
            "activation_dtype": "bfloat16", "kv_cache_dtype": "bfloat16",
            "kv_block_size": 64, "extra": {"execution_profile": PROFILE}
        },
        "context_ops": [], "generation_ops": []
    }))
    .unwrap();
    let restored = EngineSpec::from_bincode(&spec.to_bincode().unwrap()).unwrap();
    assert_eq!(restored, spec);
    assert_eq!(restored.engine.extra["execution_profile"], PROFILE);
    let json_restored: EngineSpec =
        serde_json::from_str(&serde_json::to_string(&restored).unwrap()).unwrap();
    assert_eq!(json_restored, spec);
}

#[cfg(not(feature = "python"))]
#[test]
fn automatic_regression_fallback_keeps_the_profile_and_remains_unready() {
    let model = aisimulate_core::ForwardPassPerfModel::best_available(config(json!({}))).unwrap();
    let diagnostics = model.diagnostics();
    assert_eq!(
        diagnostics.readiness,
        aisimulate_core::ForwardPassPerfReadiness::UnsupportedConfig
    );
    assert_eq!(
        diagnostics
            .provenance
            .unwrap()
            .config
            .execution_profile
            .unwrap()
            .as_str(),
        PROFILE
    );
}

#[cfg(not(feature = "python"))]
#[test]
fn alternate_resolved_identity_reloads_through_pure_rust_regression() {
    let mut config =
        config(json!({"model": "alternate/flash", "estimation_mode": "fpm_regression"}));
    config.resolve_execution_profile(Some("GLM5NEXT")).unwrap();
    let restored = serde_json::from_str(&serde_json::to_string(&config).unwrap()).unwrap();
    let model = aisimulate_core::ForwardPassPerfModel::best_available(restored).unwrap();
    let provenance = model.provenance().unwrap();
    assert_eq!(provenance.config, config);
    assert_eq!(provenance.config.model_family.unwrap().as_str(), "GLM5NEXT");
    assert_eq!(
        provenance.config.execution_profile.unwrap().as_str(),
        PROFILE
    );
    assert_eq!(
        model.diagnostics().readiness,
        aisimulate_core::ForwardPassPerfReadiness::InsufficientData
    );
}

#[cfg(feature = "python")]
#[test]
fn embedded_metadata_errors_offline_reload_and_mismatch_are_validated() {
    use pyo3::{prelude::*, types::PyModule};
    use std::ffi::CString;

    let _guard = PYTHON_METADATA_TEST_LOCK.lock().unwrap();
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let utils = py.import("aisimulate_core.sdk.utils").unwrap();
        let original = utils.getattr("get_model_config_from_model_path").unwrap();
        let fixture = PyModule::from_code(
            py,
            &CString::new(
                r#"
from aisimulate_core.sdk.errors import InvalidEngineConfigurationError
from aisimulate_core.sdk.utils import HuggingFaceDownloadError
def invalid(path):
    raise InvalidEngineConfigurationError('GLM5NEXT index_head_dim=64 is invalid')
def unavailable(path):
    raise HuggingFaceDownloadError('offline checkpoint metadata')
def malformed(path):
    raise ValueError('malformed checkpoint geometry')
def known_glm(path):
    return {'architecture': 'Glm5NextForConditionalGeneration'}
def known_dense(path):
    return {'architecture': 'Qwen3ForCausalLM'}
"#,
            )
            .unwrap(),
            c"f4_metadata_fixture.py",
            c"f4_metadata_fixture",
        )
        .unwrap();

        // All monkeypatches are within one GIL-held test in this independent
        // integration-test process, and the original is restored before asserts.
        let results = (|| {
            let mut errors = Vec::new();
            for source in ["invalid", "malformed"] {
                utils
                    .setattr(
                        "get_model_config_from_model_path",
                        fixture.getattr(source).unwrap(),
                    )
                    .unwrap();
                for mode in ["fpm_regression", "auto"] {
                    let result = aisimulate_core::ForwardPassPerfModel::best_available(config(
                        json!({"model": "alternate/flash", "estimation_mode": mode}),
                    ));
                    errors.push(result.err());
                }
            }
            utils
                .setattr(
                    "get_model_config_from_model_path",
                    fixture.getattr("known_glm").unwrap(),
                )
                .unwrap();
            let resolved = aisimulate_core::ForwardPassPerfModel::best_available(config(
                json!({"model": "alternate/flash", "estimation_mode": "fpm_regression"}),
            ))
            .unwrap()
            .provenance()
            .unwrap()
            .config
            .clone();
            let saved = serde_json::to_string(&resolved).unwrap();
            utils
                .setattr(
                    "get_model_config_from_model_path",
                    fixture.getattr("unavailable").unwrap(),
                )
                .unwrap();
            let restored = aisimulate_core::ForwardPassPerfModel::best_available(
                serde_json::from_str(&saved).unwrap(),
            );
            utils
                .setattr(
                    "get_model_config_from_model_path",
                    fixture.getattr("known_dense").unwrap(),
                )
                .unwrap();
            let mismatch = aisimulate_core::ForwardPassPerfModel::best_available(
                serde_json::from_str(&saved).unwrap(),
            )
            .err();
            (errors, resolved, restored, mismatch)
        })();
        utils
            .setattr("get_model_config_from_model_path", original)
            .unwrap();
        let (errors, resolved, restored, mismatch) = results;
        for error in errors {
            assert!(matches!(
                error,
                Some(aisimulate_core::AicError::InvalidEngineConfig(_))
            ));
        }
        assert_eq!(restored.unwrap().provenance().unwrap().config, resolved);
        assert!(mismatch.unwrap().to_string().contains("requires GLM5NEXT"));
    });
}

#[cfg(feature = "python")]
#[test]
fn embedded_real_download_parser_corruption_and_transport_controls() {
    use pyo3::{prelude::*, types::PyModule};
    use std::ffi::CString;

    let _guard = PYTHON_METADATA_TEST_LOCK.lock().unwrap();
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let utils = py.import("aisimulate_core.sdk.utils").unwrap();
        let urllib = py.import("urllib.request").unwrap();
        let original_urlopen = urllib.getattr("urlopen").unwrap();
        let original_headers = utils.getattr("_get_hf_auth_headers").unwrap();
        let fixture = PyModule::from_code(
            py,
            &CString::new(
                r#"
from io import BytesIO
from urllib.error import URLError, HTTPError
from http.client import IncompleteRead
from aisimulate_core.sdk.utils import HuggingFaceDownloadError
import json
def headers():
    return {}
def corrupt_json(request, *, timeout):
    return BytesIO(b'{"architectures": [broken')
def corrupt_encoding(request, *, timeout):
    return BytesIO(b'\xff')
def network(request, *, timeout):
    raise URLError('offline')
def missing(request, *, timeout):
    raise HTTPError(request.full_url, 404, 'missing metadata', {}, None)
def timeout(request, **kwargs):
    raise TimeoutError('timed out')
def incomplete(request, *, timeout):
    raise IncompleteRead(b'partial', 100)
def cyclic(request, *, timeout):
    error = HuggingFaceDownloadError('cycle')
    error.__cause__ = error
    raise error
def too_long(request, *, timeout):
    error = HuggingFaceDownloadError('terminal')
    for _ in range(40):
        wrapper = HuggingFaceDownloadError('nested')
        wrapper.__cause__ = error
        error = wrapper
    raise error
def implicit_corruption(request, *, timeout):
    error = HuggingFaceDownloadError('implicit')
    error.__context__ = json.JSONDecodeError('corrupt metadata', '{', 1)
    raise error
def suppressed_corruption(request, *, timeout):
    error = HuggingFaceDownloadError('suppressed')
    error.__context__ = json.JSONDecodeError('ignored context', '{', 1)
    error.__suppress_context__ = True
    raise error
"#,
            )
            .unwrap(),
            c"f4_download_fixture.py",
            c"f4_download_fixture",
        )
        .unwrap();
        utils
            .setattr("_get_hf_auth_headers", fixture.getattr("headers").unwrap())
            .unwrap();
        let mut corrupt = Vec::new();
        for (source, expected) in [
            ("corrupt_json", "JSONDecodeError"),
            ("corrupt_encoding", "UnicodeDecodeError"),
        ] {
            urllib
                .setattr("urlopen", fixture.getattr(source).unwrap())
                .unwrap();
            for mode in ["fpm_regression", "auto"] {
                for policy in ["deny", "allow"] {
                    let result =
                        aisimulate_core::ForwardPassPerfModel::best_available(config(json!({
                            "model": "f4-review/embedded-corrupt-download", "estimation_mode": mode,
                            "fallback_policy": policy
                        })));
                    corrupt.push((expected, result.err()));
                }
            }
        }
        let mut unavailable = Vec::new();
        for source in [
            "network",
            "missing",
            "timeout",
            "incomplete",
            "suppressed_corruption",
        ] {
            urllib
                .setattr("urlopen", fixture.getattr(source).unwrap())
                .unwrap();
            unavailable.push(aisimulate_core::ForwardPassPerfModel::best_available(config(json!({
                "model": format!("f4-review/embedded-{source}"), "estimation_mode": "fpm_regression",
                "model_family": "GLM5NEXT", "execution_profile": PROFILE
            }))));
        }
        let mut bounded = Vec::new();
        for (source, expected) in [
            ("cyclic", "cyclic"),
            ("too_long", "exceeds 32"),
            ("implicit_corruption", "JSONDecodeError"),
        ] {
            urllib
                .setattr("urlopen", fixture.getattr(source).unwrap())
                .unwrap();
            let result = aisimulate_core::ForwardPassPerfModel::best_available(config(json!({
                "model": format!("f4-review/embedded-{source}"), "estimation_mode": "fpm_regression"
            })));
            bounded.push((expected, result.err()));
        }
        urllib.setattr("urlopen", original_urlopen).unwrap();
        utils
            .setattr("_get_hf_auth_headers", original_headers)
            .unwrap();
        for (expected, error) in corrupt.into_iter().chain(bounded) {
            let error =
                error.expect("corruption/invalid exception chains must reject, never fall back");
            assert!(matches!(
                error,
                aisimulate_core::AicError::InvalidEngineConfig(_)
            ));
            assert!(error.to_string().contains(expected), "{error}");
        }
        for result in unavailable {
            let model = result.expect("genuine transport unavailability must remain fallback-safe");
            assert_eq!(
                model.diagnostics().readiness,
                aisimulate_core::ForwardPassPerfReadiness::InsufficientData
            );
            assert_eq!(
                model
                    .provenance()
                    .unwrap()
                    .config
                    .model_family
                    .unwrap()
                    .as_str(),
                "GLM5NEXT"
            );
            assert_eq!(
                model
                    .provenance()
                    .unwrap()
                    .config
                    .execution_profile
                    .unwrap()
                    .as_str(),
                PROFILE
            );
        }
    });
}
