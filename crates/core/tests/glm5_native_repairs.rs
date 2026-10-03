// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Opt-in SDK integration, without rebuilding/installing a Python extension.
//! Set PYO3_PYTHON to the dev venv's interpreter and PYTHONPATH to SDK src and
//! that venv's site-packages. Run cargo test --features embed-python
//! --test glm5_native_repairs -- --ignored --test-threads=1 --nocapture.

#![cfg(feature = "embed-python")]

use pyo3::prelude::*;

#[test]
#[ignore = "requires SDK dev venv and embedded Python; no installed extension rebuild"]
fn glm5_native_python_repairs() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let module = PyModule::new(py, "aisimulate._runtime").unwrap();
        aisimulate_core::perfmodel::register_python(&module).unwrap();
        let modules = py.import("sys").unwrap().getattr("modules").unwrap();
        // Process-local binding of the current Rust library, not the installed
        // extension. All operation/estimator values still have ONE Rust oracle.
        modules.set_item("aisimulate._runtime", &module).unwrap();
        let tests = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../python/aisimulate/tests/unit/sdk/test_glm5_next_native_repairs.py");
        let code: i32 = py
            .import("pytest")
            .unwrap()
            .call_method1(
                "main",
                (vec![
                    tests.to_str().unwrap(),
                    "-q",
                    "-o",
                    "addopts=",
                    "-p",
                    "no:cacheprovider",
                ],),
            )
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(code, 0, "native Python repair regressions failed");
    });
}
