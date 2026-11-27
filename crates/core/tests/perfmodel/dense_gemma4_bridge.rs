// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dense Gemma 4 rejection across the Rust → Python boundary.
//!
//! `py.rs` maps the SDK's `InvalidEngineConfigurationError` to
//! `AicError::InvalidEngineConfig`, the native failure `best_available` does not
//! fall back from. The in-crate unit test
//! (`dense_gemma4_expert_parallelism_is_not_fallback_safe`) pins that predicate,
//! but it constructs the `AicError` directly, so no Python exception ever crosses
//! the boundary. These tests raise the real Python error and assert the
//! observable behavior:
//!
//! * [`dense_gemma4_rejection_survives_the_bridge_with_fallback_allowed`] drives
//!   `ForwardPassPerfModel::best_available` with
//!   `ForwardPassFallbackPolicy::Allow`: a dense checkpoint with `moe_ep_size=2`
//!   must return the typed rejection, not a regression estimator.
//! * [`dense_gemma4_capacity_rejects_instead_of_estimating_naively`] calls the
//!   KV-cache capacity entry point with `allow_naive_fallback: true`: it must
//!   return `KvCacheEstimateError::BadConfig` carrying the dense diagnostic,
//!   never a fallback-eligible `Unsupported` or a naive estimate built from the
//!   same checkpoint metadata.
//! * [`dense_gemma4_num_gpu_blocks_rejects_instead_of_returning_a_count`] calls
//!   the `aisim_core.sdk.memory.estimate_num_gpu_blocks` entry point that
//!   `python.rs::estimate_aic_num_gpu_blocks` invokes, and must raise the dense
//!   diagnostic instead of returning a scheduler block count.
//! * [`routed_gemma4_is_not_rejected_by_the_dense_guard`] is the control: the
//!   routed checkpoint keeps expert parallelism, so the dense diagnostic must not
//!   appear for it.
//!
//! ## Run requirements
//!
//! The tests embed a Python interpreter (PyO3 `auto-initialize`) and import the
//! SDK, which needs the maturin-built extension plus the SDK sources on the
//! embedded interpreter's path:
//!
//! ```text
//! uv run maturin develop -m crates/core/Cargo.toml --release --features extension-module
//! AIC_REQUIRE_DENSE_GEMMA4_BRIDGE=1 \
//!   PYO3_PYTHON="$PWD/python/aisim/.venv/bin/python" \
//!   PYTHONPATH="$PWD/python/aisim/src:$PWD/python/aisim/.venv/lib/python3.13/site-packages" \
//!   cargo test -p aisim-core --features embed-python \
//!     --test perfmodel_dense_gemma4_bridge -- --nocapture
//! ```
//!
//! `PYO3_PYTHON` matters when the host default `python3` is a different minor
//! version (for example Homebrew 3.15): PyO3 links the embedded interpreter it
//! resolves at build time, and a mismatched interpreter cannot import the venv's
//! compiled extensions (`numpy`), which turns the SDK import into a fallback-safe
//! `UnsupportedModel` instead of exercising the dense rejection.
//!
//! ## Honest skip vs. enforced run
//!
//! On a bare `cargo test` the SDK imports are usually unavailable, so each test
//! prints a visible `SKIP` line and returns; it does not silently read as
//! coverage. Set `AIC_REQUIRE_DENSE_GEMMA4_BRIDGE=1` to make an import failure a
//! hard failure, so the test cannot false-pass. The `rust-feature-modes` CI job
//! installs the Python package and sets that variable in its embedded `cargo
//! test` step, so an import regression fails CI instead of skipping silently.

#![cfg(feature = "embed-python")]

use std::collections::BTreeMap;

use aisim_core::perfmodel::{
    AicError, BackendKind, ENGINE_CONFIG_SCHEMA_VERSION, EngineConfig, ForwardPassFallbackPolicy,
    ForwardPassPerfModel, ForwardPassPerfModelConfig, ForwardPassWorkerType, KvCacheEstimateError,
    KvCacheEstimateOptions, KvCacheEstimateRequest, KvCacheMemoryFraction, ParallelMapping,
    QuantizationConfig, estimate_kv_cache,
};
use pyo3::prelude::*;

/// Dense Gemma 4 (31B): `num_experts` and `top_k_experts` are both null.
const DENSE_MODEL: &str = "nvidia/Gemma-4-31B-IT-NVFP4";
/// Routed Gemma 4 (26B-A4B): `num_experts=128`, `top_k_experts=8`.
const ROUTED_MODEL: &str = "nvidia/Gemma-4-26B-A4B-NVFP4";
const SYSTEM: &str = "gb300";
const DENSE_REJECTION: &str = "dense Gemma 4 variants require moe_ep_size=1";

/// True when the embedded interpreter can import the SDK model resolver, which
/// transitively imports the maturin-built extension.
fn python_sdk_importable() -> bool {
    Python::attach(|py| match py.import("aisim_core.sdk.models") {
        Ok(_) => true,
        Err(error) => {
            let executable: String = py
                .import("sys")
                .and_then(|module| module.getattr("executable"))
                .and_then(|value| value.extract())
                .unwrap_or_default();
            eprintln!("dense_gemma4_bridge: import failed (sys.executable={executable}): {error}");
            false
        }
    })
}

/// Returns false when the test must skip. Panics instead when the run is
/// enforced, so an unavailable SDK cannot read as passing coverage.
fn sdk_available_or_skip() -> bool {
    if python_sdk_importable() {
        return true;
    }
    assert!(
        std::env::var_os("AIC_REQUIRE_DENSE_GEMMA4_BRIDGE").is_none(),
        "dense_gemma4_bridge: AIC_REQUIRE_DENSE_GEMMA4_BRIDGE is set but the SDK is not \
         importable — run after `maturin develop` with PYTHONPATH including \
         python/aisim/src and the venv site-packages."
    );
    eprintln!(
        "dense_gemma4_bridge: SKIP (SDK not importable; set AIC_REQUIRE_DENSE_GEMMA4_BRIDGE=1 to enforce)"
    );
    false
}

fn model_config(model: &str, moe_ep_size: u32) -> ForwardPassPerfModelConfig {
    let mut config = ForwardPassPerfModelConfig::new(
        model,
        SYSTEM,
        BackendKind::Vllm,
        ForwardPassWorkerType::Aggregated,
    );
    // The config layer requires both MoE widths together and pins
    // tp * attention_dp * cp_size == moe_tp_size * moe_ep_size, so widen tp with
    // the expert-parallel degree. The dense guard is reached regardless.
    config.tp = moe_ep_size;
    config.moe_tp_size = Some(1);
    config.moe_ep_size = Some(moe_ep_size);
    config.fallback_policy = ForwardPassFallbackPolicy::Allow;
    config
}

fn kv_request(model: &str, moe_ep_size: u32) -> KvCacheEstimateRequest {
    KvCacheEstimateRequest {
        engine: EngineConfig {
            schema_version: ENGINE_CONFIG_SCHEMA_VERSION,
            model_name: model.to_string(),
            system_name: SYSTEM.to_string(),
            systems_path: None,
            backend: BackendKind::Vllm,
            backend_version: None,
            forward_model: None,
            fpm_parquet_path: None,
            decoder_replay: false,
            moe_kernel_source: None,
            kv_block_size: None,
            parallel: ParallelMapping {
                tp_size: moe_ep_size,
                pp_size: 1,
                attention_dp_size: Some(1),
                moe_tp_size: Some(1),
                moe_ep_size: Some(moe_ep_size),
                cp_size: None,
                dcp_size: None,
            },
            quantization: QuantizationConfig {
                weight_dtype: None,
                moe_dtype: None,
                activation_dtype: None,
                fpm_fmha_dtype: None,
                kv_cache_dtype: None,
            },
            speculative: None,
            enable_shared_layer: None,
            strict_provenance: false,
            tolerate_dirless_version: false,
            database_mode: Default::default(),
            transfer_policy: None,
            extra: BTreeMap::new(),
        },
        max_num_tokens: 8192,
        max_batch_size: 256,
        kv_cache_memory_fraction: KvCacheMemoryFraction::OfTotal(0.8),
        // The naive fallback requires a capacity override, so supply one: this
        // request is fallback-eligible in every respect except the metadata the
        // dense guard rejects.
        gpu_memory_capacity_bytes_override: Some(180 * 1024 * 1024 * 1024),
        cuda_graph_reserved_bytes: 0,
        tolerance_fraction: None,
        options: KvCacheEstimateOptions {
            allow_naive_fallback: true,
            allow_hf_config_download: false,
            naive_kv_reservation: 0.8,
        },
    }
}

#[test]
fn dense_gemma4_rejection_survives_the_bridge_with_fallback_allowed() {
    if !sdk_available_or_skip() {
        return;
    }

    let error = ForwardPassPerfModel::best_available(model_config(DENSE_MODEL, 2))
        .err()
        .expect("dense Gemma 4 with moe_ep_size=2 must not select any estimator");
    assert!(
        matches!(error, AicError::InvalidEngineConfig(_)),
        "the SDK's InvalidEngineConfigurationError must map to InvalidEngineConfig, got {error:?}"
    );
    // The `compile_engine:` prefix proves the error came back through the
    // `py.rs` mapping rather than from an earlier Rust-side config check.
    let message = error.to_string();
    assert!(
        message.contains("compile_engine:") && message.contains(DENSE_REJECTION),
        "the dense diagnostic must cross the bridge verbatim, got {message}"
    );
}

#[test]
fn dense_gemma4_capacity_rejects_instead_of_estimating_naively() {
    if !sdk_available_or_skip() {
        return;
    }

    let error = estimate_kv_cache(kv_request(DENSE_MODEL, 2))
        .err()
        .expect("dense Gemma 4 with moe_ep_size=2 must not produce a capacity estimate");
    assert!(
        matches!(&error, KvCacheEstimateError::BadConfig { field, .. } if field == "engine"),
        "the SDK's InvalidEngineConfigurationError must map to non-fallback BadConfig, got {error:?}"
    );
    // `estimate_kv_cache:` is the `memory.rs` bridge context; the dense message
    // proves the naive fallback did not re-estimate the same metadata.
    let message = error.to_string();
    assert!(
        message.contains("estimate_kv_cache:") && message.contains(DENSE_REJECTION),
        "capacity must report the dense diagnostic, not a generic fallback, got {message}"
    );
}

#[test]
fn unsupported_capacity_keeps_the_fallback_eligible_variant() {
    if !sdk_available_or_skip() {
        return;
    }

    // Valid dense metadata and parallelism, but no perf database for this
    // system. This is a coverage miss rather than a configuration rejection.
    let mut request = kv_request(DENSE_MODEL, 1);
    request.engine.system_name = "unavailable_gemma4_bridge_system".into();
    request.options.allow_naive_fallback = false;
    let error = estimate_kv_cache(request)
        .err()
        .expect("an unavailable system must not produce a native estimate");
    assert!(
        matches!(&error, KvCacheEstimateError::Unsupported { model, backend, gpu_sku, reason }
            if model == DENSE_MODEL && *backend == BackendKind::Vllm
                && gpu_sku == "unavailable_gemma4_bridge_system"
                && reason.contains("estimate_kv_cache:")
                && reason.contains("unsupported model/backend/GPU")),
        "ordinary unsupported requests must retain their identity and diagnostic, got {error:?}"
    );
}

/// The scheduler-block estimator the replay path installs.
///
/// `python.rs::materialize_aic_capacity` calls `estimate_aic_num_gpu_blocks`,
/// which imports `aisim_core.sdk.memory.estimate_num_gpu_blocks`, passes the
/// kwargs `aic_capacity_kwargs` builds, and wraps the `PyErr` in `anyhow!`.
/// That wrapper has no other behavior, so calling the same Python entry point
/// with the same dense request proves the third consumer rejects the checkpoint
/// instead of converting it into a block count.
#[test]
fn dense_gemma4_num_gpu_blocks_rejects_instead_of_returning_a_count() {
    if !sdk_available_or_skip() {
        return;
    }

    let message = Python::attach(|py| -> String {
        let memory = py
            .import("aisim_core.sdk.memory")
            .expect("import aisim_core.sdk.memory");
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("scheduler_block_size", 16).unwrap();
        kwargs.set_item("max_num_tokens", 8192).unwrap();
        kwargs.set_item("max_batch_size", 256).unwrap();
        kwargs.set_item("memory_fraction_kind", "of_total").unwrap();
        kwargs.set_item("memory_fraction_value", 0.8).unwrap();
        kwargs.set_item("tp_size", 2).unwrap();
        kwargs.set_item("moe_tp_size", 1).unwrap();
        kwargs.set_item("moe_ep_size", 2).unwrap();
        kwargs.set_item("allow_naive_fallback", true).unwrap();
        kwargs
            .set_item(
                "gpu_memory_capacity_bytes_override",
                180_u64 * 1024 * 1024 * 1024,
            )
            .unwrap();
        match memory
            .getattr("estimate_num_gpu_blocks")
            .and_then(|function| function.call((DENSE_MODEL, SYSTEM, "vllm"), Some(&kwargs)))
        {
            Ok(blocks) => panic!("dense Gemma 4 must not produce a block count, got {blocks}"),
            Err(error) => error.to_string(),
        }
    });
    assert!(
        message.contains(DENSE_REJECTION),
        "num_gpu_blocks must report the dense diagnostic, not a generic fallback, got {message}"
    );
}

#[test]
fn routed_gemma4_is_not_rejected_by_the_dense_guard() {
    if !sdk_available_or_skip() {
        return;
    }

    // The routed checkpoint keeps moe_ep_size=2; it may still fail for an
    // unrelated reason (perf data, version resolution), but never with the dense
    // diagnostic.
    if let Err(error) = ForwardPassPerfModel::best_available(model_config(ROUTED_MODEL, 2)) {
        assert!(
            !error.to_string().contains(DENSE_REJECTION),
            "routed Gemma 4 must not hit the dense guard, got {error}"
        );
    }
}
