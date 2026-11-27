# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for llm-d deployment target support."""

import pytest
import yaml

from aisim.generator.rendering.engine import render_backend_templates
from aisim.generator.rendering.schemas import apply_defaults


@pytest.mark.unit
def test_llmd_values_template_renders():
    """Test that llm-d values template renders correctly for disaggregated mode."""
    # Set up context exactly as the template expects
    params = {
        "ServiceConfig": {
            "model_path": "meta-llama/Llama-3.1-8B",
            "served_model_name": "llama-3.1-8b",
            "port": 8000,
        },
        "DynConfig": {"mode": "disagg"},
        "SlaConfig": {"isl": 4000, "osl": 1000},
        "LlmdConfig": {
            "vllm_image": "vllm/vllm-openai:v0.6.0",
            "model_cache_size": "50Gi",
            "routing_proxy_enabled": True,
            "multinode": False,
        },
        "WorkerConfig": {
            "prefill_workers": 2,
            "decode_workers": 4,
            "prefill_gpus_per_worker": 2,
            "decode_gpus_per_worker": 4,
        },
        "params": {
            "prefill": {"gpus_per_worker": 2},
            "decode": {"gpus_per_worker": 4},
        },
        # Parallelism settings (used directly by template)
        "prefill_tensor_parallel_size": 2,
        "prefill_data_parallel_size": 1,
        "decode_tensor_parallel_size": 4,
        "decode_data_parallel_size": 1,
        # CLI args (used directly by template)
        "prefill_cli_args_list": ["--tensor-parallel-size", "2", "--max-num-seqs", "1"],
        "decode_cli_args_list": ["--tensor-parallel-size", "4", "--max-num-seqs", "128"],
    }

    artifacts = render_backend_templates(param_values=params, backend="vllm", deployment_target="llm-d-helm")

    assert "llm-d-values.yaml" in artifacts
    assert "k8s_deploy.yaml" not in artifacts  # Should not render Dynamo manifest

    values_content = artifacts["llm-d-values.yaml"]

    # Verify key sections are present
    assert "modelArtifacts:" in values_content
    assert "name: meta-llama/Llama-3.1-8B" in values_content
    assert "uri: hf://meta-llama/Llama-3.1-8B" in values_content
    assert "size: 50Gi" in values_content

    # Verify disaggregated mode
    assert "multinode: False" in values_content or "multinode: false" in values_content
    assert "decode:" in values_content
    assert "prefill:" in values_content
    assert "replicas: 2" in values_content  # prefill replicas
    assert "replicas: 4" in values_content  # decode replicas

    # Verify parallelism
    assert "tensor: 2" in values_content  # prefill TP
    assert "tensor: 4" in values_content  # decode TP

    # Verify vLLM image
    assert "image: vllm/vllm-openai:v0.6.0" in values_content


@pytest.mark.unit
def test_llmd_values_template_aggregated_mode():
    """Test that llm-d values template renders correctly for aggregated mode."""
    params = {
        "ServiceConfig": {
            "model_path": "Qwen/Qwen3-8B",
            "served_model_name": "qwen3-8b",
            "port": 8000,
        },
        "DynConfig": {"mode": "agg"},
        "LlmdConfig": {},
        "WorkerConfig": {
            "agg_workers": 4,
            "agg_gpus_per_worker": 2,
        },
        "params": {
            "agg": {"gpus_per_worker": 2},
        },
        # Parallelism settings
        "agg_tensor_parallel_size": 2,
        "agg_data_parallel_size": 1,
        # CLI args
        "agg_cli_args_list": ["--tensor-parallel-size", "2"],
    }

    artifacts = render_backend_templates(param_values=params, backend="vllm", deployment_target="llm-d-helm")

    values_content = artifacts["llm-d-values.yaml"]

    # Verify aggregated mode
    assert "multinode: false" in values_content
    assert "decode:" in values_content
    assert "create: true" in values_content
    assert "prefill:" in values_content
    assert "create: false" in values_content  # prefill should be disabled in agg mode
    assert "replicas: 4" in values_content  # agg workers
    assert "tensor: 2" in values_content  # TP


@pytest.mark.unit
def test_llmd_kustomize_vllm_disagg_mode():
    """Test that vLLM can render llm-d Kustomize overlay patches."""
    params = {
        "ServiceConfig": {
            "model_path": "Qwen/Qwen3-32B",
            "served_model_name": "qwen3-32b",
            "port": 8000,
        },
        "DynConfig": {"mode": "disagg"},
        "SlaConfig": {"isl": 4000, "osl": 1000},
        "LlmdConfig": {
            "vllm_image": "vllm/vllm-openai:v0.19.0",
            "kustomize_base_path": "/repo/llm-d/guides/pd-disaggregation/modelserver/gpu/vllm/base",
        },
        "WorkerConfig": {
            "prefill_workers": 14,
            "decode_workers": 9,
            "prefill_gpus_per_worker": 1,
            "decode_gpus_per_worker": 2,
        },
        "params": {
            "prefill": {"gpus_per_worker": 1},
            "decode": {"gpus_per_worker": 2},
        },
        "prefill_tensor_parallel_size": 1,
        "prefill_data_parallel_size": 1,
        "decode_tensor_parallel_size": 2,
        "decode_data_parallel_size": 1,
        "prefill_gpu": 1,
        "decode_gpu": 2,
        "prefill_cli_args_list": ["--tensor-parallel-size", "1", "--max-num-batched-tokens", "5500"],
        "decode_cli_args_list": ["--tensor-parallel-size", "2", "--max-num-seqs", "512"],
    }

    artifacts = render_backend_templates(param_values=params, backend="vllm", deployment_target="llm-d-kustomize")

    assert set(artifacts) >= {"kustomization.yaml", "patch-decode.yaml", "patch-prefill.yaml"}
    assert "llm-d-values.yaml" not in artifacts
    assert "k8s_deploy.yaml" not in artifacts

    kustomization = yaml.safe_load(artifacts["kustomization.yaml"])
    assert kustomization["resources"] == ["/repo/llm-d/guides/pd-disaggregation/modelserver/gpu/vllm/base"]
    assert kustomization["patches"] == [{"path": "patch-decode.yaml"}, {"path": "patch-prefill.yaml"}]

    decode = yaml.safe_load(artifacts["patch-decode.yaml"])
    decode_container = decode["spec"]["template"]["spec"]["containers"][0]
    assert decode["spec"]["replicas"] == 9
    assert decode_container["image"] == "vllm/vllm-openai:v0.19.0"
    assert decode_container["command"] == ["vllm", "serve"]
    assert decode_container["resources"]["limits"]["nvidia.com/gpu"] == "2"
    assert "--kv-transfer-config" in decode_container["args"]
    assert '{"kv_connector":"NixlConnector", "kv_role":"kv_both"}' in decode_container["args"]
    assert "--port=8200" in decode_container["args"]

    prefill = yaml.safe_load(artifacts["patch-prefill.yaml"])
    prefill_container = prefill["spec"]["template"]["spec"]["containers"][0]
    assert prefill["spec"]["replicas"] == 14
    assert prefill_container["resources"]["limits"]["nvidia.com/gpu"] == "1"
    assert "--max-num-batched-tokens" in prefill_container["args"]
    # every token budget renders 64-aligned (owner decision 2026-09-24): 5500 -> 5504
    assert "5504" in prefill_container["args"]


@pytest.mark.unit
def test_llmd_kustomize_vllm_agg_b60_mode():
    """Test that agg mode renders a single llm-d Kustomize modelserver patch."""
    params = {
        "ServiceConfig": {
            "model_path": "Qwen/Qwen3-8B",
            "served_model_name": "Qwen/Qwen3-8B",
            "port": 8000,
        },
        "DynConfig": {"mode": "agg"},
        "SlaConfig": {"isl": 4500, "osl": 1000},
        "NodeConfig": {"system_name": "b60"},
        "LlmdConfig": {
            "vllm_image": "vllm/vllm-openai:v0.20.0",
        },
        "WorkerConfig": {
            "agg_workers": 1,
            "agg_gpus_per_worker": 4,
        },
        "params": {
            "agg": {"gpus_per_worker": 4},
        },
        "agg_tensor_parallel_size": 4,
        "agg_pipeline_parallel_size": 1,
        "agg_data_parallel_size": 1,
        "agg_cli_args_list": ["--tensor-parallel-size", "4", "--max-model-len", "7000"],
    }

    artifacts = render_backend_templates(param_values=params, backend="vllm", deployment_target="llm-d-kustomize")

    assert set(artifacts) >= {"kustomization.yaml", "patch-vllm.yaml"}
    assert "patch-decode.yaml" not in artifacts
    assert "patch-prefill.yaml" not in artifacts

    kustomization = yaml.safe_load(artifacts["kustomization.yaml"])
    assert kustomization["resources"] == ["guides/optimized-baseline/modelserver/xpu/vllm"]
    assert kustomization["patches"] == [{"path": "patch-vllm.yaml"}]

    decode = yaml.safe_load(artifacts["patch-vllm.yaml"])
    decode_container = decode["spec"]["template"]["spec"]["containers"][0]
    assert decode["spec"]["replicas"] == 1
    assert decode_container["image"] == "vllm/vllm-openai:v0.20.0"
    assert "resources" not in decode_container
    assert any('"kv_buffer_device":"cpu"' in arg for arg in decode_container["args"])


@pytest.mark.unit
def test_llmd_sglang_disagg_mode():
    """Test that sglang backend works with llm-d deployment target in disagg mode."""
    params = {
        "ServiceConfig": {
            "model_path": "Qwen/Qwen3-32B",
            "served_model_name": "qwen3-32b",
            "port": 8000,
        },
        "DynConfig": {"mode": "disagg"},
        "LlmdConfig": {
            "sglang_image": "lmsysorg/sglang:v0.3.0",
            "model_cache_size": "200Gi",
            "routing_proxy_enabled": True,
            "multinode": False,
        },
        "WorkerConfig": {
            "prefill_workers": 1,
            "decode_workers": 2,
            "prefill_gpus_per_worker": 4,
            "decode_gpus_per_worker": 8,
        },
        "params": {
            "prefill": {"gpus_per_worker": 4},
            "decode": {"gpus_per_worker": 8},
        },
        "prefill_tensor_parallel_size": 4,
        "prefill_data_parallel_size": 1,
        "decode_tensor_parallel_size": 8,
        "decode_data_parallel_size": 1,
        "prefill_cli_args_list": ["--tp-size", "4", "--mem-fraction-static", "0.9"],
        "decode_cli_args_list": ["--tp-size", "8", "--mem-fraction-static", "0.9"],
    }

    artifacts = render_backend_templates(param_values=params, backend="sglang", deployment_target="llm-d-helm")

    assert "llm-d-values.yaml" in artifacts
    assert "k8s_deploy.yaml" not in artifacts  # Should not render Dynamo manifest

    values_content = artifacts["llm-d-values.yaml"]

    # Verify sglang-specific content
    assert "image: lmsysorg/sglang:v0.3.0" in values_content
    assert "modelCommand: sglangServe" in values_content
    assert "size: 200Gi" in values_content
    assert "name: sglang" in values_content

    # Verify disaggregated mode structure
    assert "multinode: False" in values_content or "multinode: false" in values_content
    assert "prefill:" in values_content
    assert "decode:" in values_content
    assert "create: true" in values_content
    assert "replicas: 1" in values_content  # prefill
    assert "replicas: 2" in values_content  # decode

    # Verify parallelism
    assert "tensor: 4" in values_content  # prefill TP
    assert "tensor: 8" in values_content  # decode TP

    # Verify model configuration
    assert "model-path" in values_content
    assert "Qwen/Qwen3-32B" in values_content


@pytest.mark.unit
def test_llmd_sglang_agg_mode():
    """Test that sglang backend works with llm-d deployment target in agg mode."""
    params = {
        "ServiceConfig": {
            "model_path": "meta-llama/Llama-3.1-70B",
            "served_model_name": "llama-3.1-70b",
            "port": 8000,
        },
        "DynConfig": {"mode": "agg"},
        "LlmdConfig": {
            "sglang_image": "lmsysorg/sglang:latest",
            "model_cache_size": "300Gi",
        },
        "WorkerConfig": {
            "agg_workers": 2,
            "agg_gpus_per_worker": 8,
        },
        "params": {
            "agg": {"gpus_per_worker": 8},
        },
        "agg_tensor_parallel_size": 8,
        "agg_data_parallel_size": 1,
        "agg_cli_args_list": ["--tp-size", "8", "--context-length", "8192"],
    }

    artifacts = render_backend_templates(param_values=params, backend="sglang", deployment_target="llm-d-helm")

    assert "llm-d-values.yaml" in artifacts
    values_content = artifacts["llm-d-values.yaml"]

    # Verify aggregated mode
    assert "multinode: false" in values_content
    assert "decode:" in values_content
    assert "create: true" in values_content
    assert "replicas: 2" in values_content  # agg workers
    assert "prefill:" in values_content
    assert "create: false" in values_content  # prefill should be disabled in agg mode

    # Verify parallelism
    assert "tensor: 8" in values_content

    # Verify sglang-specific content
    assert "image: lmsysorg/sglang:latest" in values_content
    assert "modelCommand: sglangServe" in values_content
    assert "model-path" in values_content
    assert "meta-llama/Llama-3.1-70B" in values_content


@pytest.mark.unit
def test_llmd_sglang_default_image():
    """Test that sglang uses default image when not specified."""
    params = {
        "ServiceConfig": {
            "model_path": "Qwen/Qwen3-8B",
            "port": 8000,
        },
        "DynConfig": {"mode": "agg"},
        "LlmdConfig": {},  # No sglang_image specified
        "WorkerConfig": {
            "agg_workers": 1,
            "agg_gpus_per_worker": 2,
        },
        "params": {
            "agg": {"gpus_per_worker": 2},
        },
        "agg_tensor_parallel_size": 2,
        "agg_data_parallel_size": 1,
        "agg_cli_args_list": [],
    }

    artifacts = render_backend_templates(param_values=params, backend="sglang", deployment_target="llm-d-helm")

    values_content = artifacts["llm-d-values.yaml"]
    # Should use default sglang image
    assert "image: lmsysorg/sglang:latest" in values_content


@pytest.mark.unit
@pytest.mark.parametrize("backend", ["vllm", "sglang"])
def test_llmd_hf_token_secret(backend):
    """Test that hf_token_secret renders as authSecretName for both backends."""
    params = {
        "ServiceConfig": {
            "model_path": "meta-llama/Llama-3.1-8B",
            "served_model_name": "llama-3.1-8b",
            "port": 8000,
        },
        "DynConfig": {"mode": "agg"},
        "LlmdConfig": {"hf_token_secret": "my-hf-secret"},
        "WorkerConfig": {"agg_workers": 1, "agg_gpus_per_worker": 1},
        "params": {"agg": {"gpus_per_worker": 1}},
        "agg_tensor_parallel_size": 1,
        "agg_data_parallel_size": 1,
        "agg_cli_args_list": [],
    }

    artifacts = render_backend_templates(param_values=params, backend=backend, deployment_target="llm-d-helm")
    values_content = artifacts["llm-d-values.yaml"]
    assert "authSecretName: my-hf-secret" in values_content


@pytest.mark.unit
@pytest.mark.parametrize("backend", ["vllm", "sglang"])
def test_llmd_hf_token_secret_default_empty(backend):
    """Test that authSecretName defaults to empty when hf_token_secret is not set."""
    params = {
        "ServiceConfig": {
            "model_path": "meta-llama/Llama-3.1-8B",
            "port": 8000,
        },
        "DynConfig": {"mode": "agg"},
        "LlmdConfig": {},
        "WorkerConfig": {"agg_workers": 1, "agg_gpus_per_worker": 1},
        "params": {"agg": {"gpus_per_worker": 1}},
        "agg_tensor_parallel_size": 1,
        "agg_data_parallel_size": 1,
        "agg_cli_args_list": [],
    }

    artifacts = render_backend_templates(param_values=params, backend=backend, deployment_target="llm-d-helm")
    values_content = artifacts["llm-d-values.yaml"]
    assert "authSecretName:" in values_content
    assert "authSecretName: my-hf-secret" not in values_content


@pytest.mark.unit
def test_dynamo_deployment_still_works():
    """Test that Dynamo deployment target still works (default behavior)."""
    params = {
        "ServiceConfig": {
            "model_path": "meta-llama/Llama-3.1-8B",
            "served_model_name": "llama-3.1-8b",
            "port": 8000,
        },
        "DynConfig": {"mode": "disagg"},
        "K8sConfig": {},
        "WorkerConfig": {
            "prefill_workers": 2,
            "decode_workers": 4,
            "prefill_gpus_per_worker": 2,
            "decode_gpus_per_worker": 4,
        },
        "params": {
            "prefill": {"gpus_per_worker": 2},
            "decode": {"gpus_per_worker": 4},
        },
        "prefill_cli_args_list": [],
        "decode_cli_args_list": [],
        "name": "test-deployment",
    }

    artifacts = render_backend_templates(param_values=params, backend="vllm", deployment_target="dynamo-j2")

    # Should render Dynamo K8s manifest, not llm-d values
    assert "k8s_deploy.yaml" in artifacts
    assert "llm-d-values.yaml" not in artifacts

    k8s_content = artifacts["k8s_deploy.yaml"]
    assert "DynamoGraphDeployment" in k8s_content


@pytest.mark.unit
def test_dynamo_python_deployment():
    """Test that dynamo-python deployment target attempts to use Python config modifiers."""
    # First check if dynamo package is available
    try:
        import dynamo.profiler.utils.config_modifiers  # noqa: F401

        dynamo_available = True
    except ImportError:
        dynamo_available = False

    params = {
        "ServiceConfig": {
            "model_path": "meta-llama/Llama-3.1-8B",
            "served_model_name": "llama-3.1-8b",
            "port": 8000,
        },
        "DynConfig": {"mode": "disagg"},
        "K8sConfig": {
            "k8s_image": "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.0.0",
            "k8s_namespace": "default",
        },
        "WorkerConfig": {
            "prefill_workers": 2,
            "decode_workers": 4,
            "prefill_gpus_per_worker": 2,
            "decode_gpus_per_worker": 4,
        },
        "params": {
            "prefill": {"gpus_per_worker": 2},
            "decode": {"gpus_per_worker": 4},
        },
        "prefill_cli_args_list": ["--tensor-parallel-size", "2"],
        "decode_cli_args_list": ["--tensor-parallel-size", "4"],
    }

    artifacts = render_backend_templates(param_values=params, backend="vllm", deployment_target="dynamo-python")

    if dynamo_available:
        # If dynamo is available, should generate k8s_deploy.yaml via Python config modifiers
        assert "k8s_deploy.yaml" in artifacts
        k8s_content = artifacts["k8s_deploy.yaml"]
        assert len(k8s_content) > 0  # Should have content
    else:
        # If dynamo is not available, generation should fail gracefully (logged as warning)
        # but other artifacts should still be generated (run scripts, etc)
        assert "k8s_deploy.yaml" not in artifacts or artifacts["k8s_deploy.yaml"] == ""
        # Should still have other artifacts like run scripts
        assert any(k.startswith("run_") for k in artifacts)


def _disaggregatedset_params(mode: str = "disagg") -> dict:
    """Parameters shared by the DisaggregatedSet layout tests."""
    return {
        "ServiceConfig": {
            "model_path": "Qwen/Qwen3-32B",
            "served_model_name": "qwen3-32b",
            "port": 8000,
        },
        "DynConfig": {"mode": mode},
        "SlaConfig": {"isl": 4000, "osl": 1000},
        "LlmdConfig": {
            "vllm_image": "vllm/vllm-openai@sha256:abc123",
            "kustomize_layout": "disaggregatedset",
        },
        "WorkerConfig": {
            "prefill_workers": 3,
            "decode_workers": 9,
            "prefill_gpus_per_worker": 1,
            "decode_gpus_per_worker": 2,
        },
        "params": {
            "prefill": {
                "gpus_per_worker": 1,
                "tensor_parallel_size": 1,
                "pipeline_parallel_size": 1,
                "data_parallel_size": 1,
            },
            "decode": {
                "gpus_per_worker": 2,
                "tensor_parallel_size": 2,
                "pipeline_parallel_size": 1,
                "data_parallel_size": 1,
            },
        },
        "prefill_tensor_parallel_size": 1,
        "prefill_pipeline_parallel_size": 1,
        "prefill_data_parallel_size": 1,
        "decode_tensor_parallel_size": 2,
        "decode_pipeline_parallel_size": 1,
        "decode_data_parallel_size": 1,
        "prefill_cli_args_list": ["--tensor-parallel-size", "1", "--max-num-batched-tokens", "5500"],
        "decode_cli_args_list": ["--tensor-parallel-size", "2", "--max-num-seqs", "512"],
    }


@pytest.mark.unit
def test_llmd_kustomize_disaggregatedset_layout():
    """The v1.0+ layout patches one DisaggregatedSet instead of two Deployments."""
    artifacts = render_backend_templates(
        param_values=_disaggregatedset_params(),
        backend="vllm",
        deployment_target="llm-d-kustomize",
    )

    assert set(artifacts) >= {
        "kustomization.yaml",
        "name-reference.yaml",
        "patch-decode.yaml",
        "patch-prefill.yaml",
    }
    assert "patch-vllm.yaml" not in artifacts
    assert "llm-d-values.yaml" not in artifacts

    kustomization = yaml.safe_load(artifacts["kustomization.yaml"])
    assert kustomization["resources"] == ["guides/pd-disaggregation/modelserver/gpu/vllm/base"]
    assert kustomization["configurations"] == ["name-reference.yaml"]
    assert [patch["path"] for patch in kustomization["patches"]] == [
        "patch-decode.yaml",
        "patch-prefill.yaml",
    ]
    target = kustomization["patches"][0]["target"]
    assert target == {
        "group": "disaggregatedset.x-k8s.io",
        "version": "v1",
        "kind": "DisaggregatedSet",
        "name": "pd-disagg-vllm",
    }

    decode = yaml.safe_load(artifacts["patch-decode.yaml"])
    assert isinstance(decode, list)
    # The guard op makes a reordered base fail the build instead of patching
    # the wrong role.
    assert decode[0] == {"op": "test", "path": "/spec/roles/1/name", "value": "decode"}
    decode_ops = {op["path"]: op["value"] for op in decode if op["op"] == "replace"}
    decode_container = "/spec/roles/1/spec/leaderWorkerTemplate/workerTemplate/spec/containers/0"
    assert decode_ops["/spec/roles/1/spec/replicas"] == 9
    assert decode_ops[f"{decode_container}/image"] == "vllm/vllm-openai@sha256:abc123"
    decode_args = decode_ops[f"{decode_container}/args"]
    assert "--port=8200" in decode_args
    assert any('"kv_role":"kv_consumer"' in arg for arg in decode_args)
    # The GPU entries are patched in place, so the base's cpu/memory stay.
    decode_gpu = {op["path"]: op["value"] for op in decode if op["op"] == "add" and "nvidia.com~1gpu" in op["path"]}
    assert decode_gpu == {
        f"{decode_container}/resources/limits/nvidia.com~1gpu": "2",
        f"{decode_container}/resources/requests/nvidia.com~1gpu": "2",
    }
    assert f"{decode_container}/resources" not in decode_ops

    prefill = yaml.safe_load(artifacts["patch-prefill.yaml"])
    assert prefill[0] == {"op": "test", "path": "/spec/roles/0/name", "value": "prefill"}
    prefill_ops = {op["path"]: op["value"] for op in prefill if op["op"] == "replace"}
    prefill_container = "/spec/roles/0/spec/leaderWorkerTemplate/workerTemplate/spec/containers/0"
    assert prefill_ops["/spec/roles/0/spec/replicas"] == 3
    prefill_args = prefill_ops[f"{prefill_container}/args"]
    assert any('"kv_role":"kv_producer"' in arg for arg in prefill_args)
    # every token budget renders 64-aligned (owner decision 2026-09-24)
    assert "5504" in prefill_args
    prefill_gpu = {op["path"]: op["value"] for op in prefill if op["op"] == "add" and "nvidia.com~1gpu" in op["path"]}
    assert prefill_gpu == {
        f"{prefill_container}/resources/limits/nvidia.com~1gpu": "1",
        f"{prefill_container}/resources/requests/nvidia.com~1gpu": "1",
    }


@pytest.mark.unit
def test_llmd_kustomize_disaggregatedset_layout_is_disagg_only():
    """Aggregated mode has no P/D roles, so it keeps the Deployment overlay."""
    params = {
        "ServiceConfig": {
            "model_path": "Qwen/Qwen3-8B",
            "served_model_name": "qwen3-8b",
            "port": 8000,
        },
        "DynConfig": {"mode": "agg"},
        "LlmdConfig": {
            "vllm_image": "vllm/vllm-openai:v0.24.0",
            "kustomize_layout": "disaggregatedset",
        },
        "WorkerConfig": {"agg_workers": 2, "agg_gpus_per_worker": 4},
        "params": {"agg": {"gpus_per_worker": 4}},
        "agg_tensor_parallel_size": 4,
        "agg_pipeline_parallel_size": 1,
        "agg_data_parallel_size": 1,
        "agg_cli_args_list": ["--tensor-parallel-size", "4"],
    }

    artifacts = render_backend_templates(
        param_values=params,
        backend="vllm",
        deployment_target="llm-d-kustomize",
    )

    assert set(artifacts) >= {"kustomization.yaml", "patch-vllm.yaml"}
    assert "patch-decode.yaml" not in artifacts
    assert "patch-prefill.yaml" not in artifacts
    kustomization = yaml.safe_load(artifacts["kustomization.yaml"])
    assert kustomization["patches"] == [{"path": "patch-vllm.yaml"}]
    patch = yaml.safe_load(artifacts["patch-vllm.yaml"])
    assert patch["kind"] == "Deployment"


@pytest.mark.unit
def test_llmd_kustomize_rejects_unknown_layout():
    """An unknown layout must fail closed rather than render the wrong shape."""
    params = _disaggregatedset_params()
    params["LlmdConfig"]["kustomize_layout"] = "statefulset"

    with pytest.raises(ValueError, match="kustomize_layout"):
        render_backend_templates(
            param_values=params,
            backend="vllm",
            deployment_target="llm-d-kustomize",
        )


def _render_disaggregatedset(**llmd_overrides) -> dict:
    """Render the DisaggregatedSet overlay with LlmdConfig overrides."""
    params = _disaggregatedset_params()
    params["LlmdConfig"].update(llmd_overrides)
    return render_backend_templates(
        param_values=params,
        backend="vllm",
        deployment_target="llm-d-kustomize",
    )


@pytest.mark.unit
def test_llmd_kustomize_disaggregatedset_honours_explicit_resource_name():
    """An explicit name wins, and is asserted against the base at build time."""
    artifacts = _render_disaggregatedset(kustomize_disaggregatedset_name="team-pd")

    kustomization = yaml.safe_load(artifacts["kustomization.yaml"])
    path_patches = [patch for patch in kustomization["patches"] if "path" in patch]
    assert [patch["target"]["name"] for patch in path_patches] == ["team-pd", "team-pd"]

    # kustomize silently skips an unmatched target, so the explicit name is
    # tested against the base DisaggregatedSet's metadata name first.
    guards = [patch for patch in kustomization["patches"] if "patch" in patch]
    assert len(guards) == 1
    assert guards[0]["target"] == {
        "group": "disaggregatedset.x-k8s.io",
        "version": "v1",
        "kind": "DisaggregatedSet",
    }
    assert yaml.safe_load(guards[0]["patch"]) == [{"op": "test", "path": "/metadata/name", "value": "team-pd"}]


@pytest.mark.unit
def test_llmd_kustomize_disaggregatedset_default_name_needs_no_build_guard():
    """The resolved default matches its base, so no extra guard is emitted."""
    kustomization = yaml.safe_load(_render_disaggregatedset()["kustomization.yaml"])

    assert all("path" in patch for patch in kustomization["patches"])


@pytest.mark.unit
def test_llmd_kustomize_disaggregatedset_prefix_keeps_the_service_account_binding():
    """A prefix renames the account, so the reference mapping must ship too."""
    artifacts = _render_disaggregatedset(kustomize_name_prefix="configiq-")

    kustomization = yaml.safe_load(artifacts["kustomization.yaml"])
    assert kustomization["namePrefix"] == "configiq-"
    assert kustomization["configurations"] == ["name-reference.yaml"]

    references = yaml.safe_load(artifacts["name-reference.yaml"])
    assert references == {
        "nameReference": [
            {
                "kind": "ServiceAccount",
                "version": "v1",
                "fieldSpecs": [
                    {
                        "path": ("spec/roles/spec/leaderWorkerTemplate/workerTemplate/spec/serviceAccountName"),
                        "kind": "DisaggregatedSet",
                        "group": "disaggregatedset.x-k8s.io",
                        "version": "v1",
                    }
                ],
            }
        ]
    }


def _claim_patches(kustomization: dict) -> dict[str, list[dict]]:
    """Inline ResourceClaimTemplate patches, keyed by target name."""
    return {
        patch["target"]["name"]: yaml.safe_load(patch["patch"])
        for patch in kustomization["patches"]
        if patch.get("target", {}).get("kind") == "ResourceClaimTemplate"
    }


@pytest.mark.unit
def test_llmd_kustomize_disaggregatedset_xpu_follows_the_xpu_guide():
    """XPU selects its own base, resource name, claim counts, and KV buffer."""
    params = _disaggregatedset_params()
    params["LlmdConfig"].update({"accelerator_type": "xpu", "xpu_kv_buffer_device": "cpu"})
    # TP=1/DP=2 prefill and TP=2/DP=2 decode give the two roles distinct counts.
    params["params"]["prefill"].update({"tensor_parallel_size": 1, "data_parallel_size": 2})
    params["params"]["decode"].update({"tensor_parallel_size": 2, "data_parallel_size": 2})
    params["prefill_data_parallel_size"] = 2
    params["decode_data_parallel_size"] = 2
    artifacts = render_backend_templates(
        param_values=params,
        backend="vllm",
        deployment_target="llm-d-kustomize",
    )

    kustomization = yaml.safe_load(artifacts["kustomization.yaml"])
    assert kustomization["resources"] == ["guides/pd-disaggregation/modelserver/xpu/vllm"]
    assert [patch["target"]["name"] for patch in kustomization["patches"][:2]] == [
        "pd-disagg-xpu",
        "pd-disagg-xpu",
    ]

    for artifact in ("patch-decode.yaml", "patch-prefill.yaml"):
        ops = yaml.safe_load(artifacts[artifact])
        # XPU shapes devices with ResourceClaimTemplates, so no GPU op applies.
        assert not any("resources" in op["path"] for op in ops)
        args = next(op["value"] for op in ops if op["path"].endswith("/args"))
        assert any('"kv_buffer_device":"cpu"' in arg for arg in args)

    claims = _claim_patches(kustomization)
    assert claims == {
        "xpu-decode-claim-xpu": [
            {"op": "test", "path": "/spec/spec/devices/requests/0/name", "value": "intel"},
            {
                "op": "replace",
                "path": "/spec/spec/devices/requests/0/exactly/count",
                "value": 4,
            },
        ],
        "xpu-prefill-claim-xpu": [
            {"op": "test", "path": "/spec/spec/devices/requests/0/name", "value": "intel"},
            {
                "op": "replace",
                "path": "/spec/spec/devices/requests/0/exactly/count",
                "value": 2,
            },
        ],
    }

    rdma_params = _disaggregatedset_params()
    rdma_params["LlmdConfig"].update({"accelerator_type": "xpu", "xpu_rdma_enabled": True})
    rdma_params["params"]["decode"].update({"tensor_parallel_size": 2, "data_parallel_size": 2})
    rdma_params["params"]["prefill"].update({"tensor_parallel_size": 1, "data_parallel_size": 2})
    rdma_params["decode_data_parallel_size"] = 2
    rdma_params["prefill_data_parallel_size"] = 2
    rdma = yaml.safe_load(
        render_backend_templates(
            param_values=rdma_params,
            backend="vllm",
            deployment_target="llm-d-kustomize",
        )["kustomization.yaml"]
    )
    assert rdma["resources"] == ["guides/pd-disaggregation/modelserver/xpu/vllm/rdma"]
    rdma_claims = _claim_patches(rdma)
    # The RDMA override renames the accelerator request; the `nic` request that
    # follows it must not be touched.
    assert [ops[0]["value"] for ops in rdma_claims.values()] == ["gpu", "gpu"]
    assert all(ops[1]["path"].endswith("requests/0/exactly/count") for ops in rdma_claims.values())
    assert [ops[1]["value"] for ops in rdma_claims.values()] == [4, 2]


@pytest.mark.unit
def test_llmd_kustomize_disaggregatedset_uses_base_path_override():
    """An explicit base path replaces the accelerator default."""
    artifacts = _render_disaggregatedset(kustomize_base_path="guides/custom/pd/base")

    kustomization = yaml.safe_load(artifacts["kustomization.yaml"])
    assert kustomization["resources"] == ["guides/custom/pd/base"]


@pytest.mark.unit
def test_llmd_kustomize_explicit_deployment_layout_still_renders_deployments():
    """An explicit deployment layout in disaggregated mode is not overridden."""
    artifacts = _render_disaggregatedset(kustomize_layout="deployment")

    assert "name-reference.yaml" not in artifacts
    kustomization = yaml.safe_load(artifacts["kustomization.yaml"])
    assert kustomization["patches"] == [
        {"path": "patch-decode.yaml"},
        {"path": "patch-prefill.yaml"},
    ]
    assert yaml.safe_load(artifacts["patch-decode.yaml"])["kind"] == "Deployment"


@pytest.mark.unit
def test_llmd_kustomize_disaggregatedset_counts_data_parallel_ranks_in_gpus():
    """Each DP rank runs its own TP x PP GPU set, and the args request them."""
    params = _disaggregatedset_params()
    params["params"]["decode"].update({"tensor_parallel_size": 2, "data_parallel_size": 2})
    params["params"]["prefill"].update({"tensor_parallel_size": 1, "data_parallel_size": 2})
    # The planner states each parallelism input once; keep the top-level
    # compatibility keys in step with the worker params, as it does.
    params["decode_data_parallel_size"] = 2
    params["prefill_data_parallel_size"] = 2

    artifacts = render_backend_templates(
        param_values=params,
        backend="vllm",
        deployment_target="llm-d-kustomize",
    )

    decode = yaml.safe_load(artifacts["patch-decode.yaml"])
    decode_paths = {op["path"]: op["value"] for op in decode if op["op"] == "replace"}
    decode_container = "/spec/roles/1/spec/leaderWorkerTemplate/workerTemplate/spec/containers/0"
    decode_args = decode_paths[f"{decode_container}/args"]
    dp_index = decode_args.index("--data-parallel-size")
    assert decode_args[dp_index + 1] == "2"

    decode_gpu = {
        op["path"].rsplit("/", 1)[-1]: op["value"]
        for op in decode
        # TP=2, PP=1, DP=2 -> two ranks of two GPUs each.
        if op["op"] == "add" and "nvidia.com~1gpu" in op["path"]
    }
    assert decode_gpu == {"nvidia.com~1gpu": "4"}

    prefill = yaml.safe_load(artifacts["patch-prefill.yaml"])
    prefill_gpu = [op["value"] for op in prefill if op["op"] == "add" and "nvidia.com~1gpu" in op["path"]]
    assert prefill_gpu == ["2", "2"]


@pytest.mark.unit
def test_llmd_kustomize_disaggregatedset_counts_prefill_context_ranks_in_gpus():
    """Prefill CP expands the worker, so the pod request must grow with it."""
    params = _disaggregatedset_params()
    params["params"]["prefill"].update({"tensor_parallel_size": 2, "context_parallel_size": 2})
    params["prefill_tensor_parallel_size"] = 2

    artifacts = render_backend_templates(
        param_values=params,
        backend="vllm",
        # PCP is rendered from the versioned cli-args template.
        version="0.20.1",
        deployment_target="llm-d-kustomize",
    )

    prefill = yaml.safe_load(artifacts["patch-prefill.yaml"])
    prefill_container = "/spec/roles/0/spec/leaderWorkerTemplate/workerTemplate/spec/containers/0"
    args = next(op["value"] for op in prefill if op["path"] == f"{prefill_container}/args")
    pcp_index = args.index("--prefill-context-parallel-size")
    assert args[pcp_index + 1] == "2"

    prefill_gpu = [op["value"] for op in prefill if op["op"] == "add" and "nvidia.com~1gpu" in op["path"]]
    # TP=2, PP=1, DP=1, CP=2 -> a four-GPU world.
    assert prefill_gpu == ["4", "4"]

    decode = yaml.safe_load(artifacts["patch-decode.yaml"])
    decode_gpu = [op["value"] for op in decode if op["op"] == "add" and "nvidia.com~1gpu" in op["path"]]
    # Decode has no CP here, so its count is unchanged.
    assert decode_gpu == ["2", "2"]


@pytest.mark.unit
def test_llmd_kustomize_layout_defaults_to_deployment_only_when_unset():
    """Absent or null selects the default; a non-string value is an error."""
    assert apply_defaults("LlmdConfig", {})["kustomize_layout"] == "deployment"

    params = _disaggregatedset_params()
    params["LlmdConfig"]["kustomize_layout"] = None
    artifacts = render_backend_templates(
        param_values=params,
        backend="vllm",
        deployment_target="llm-d-kustomize",
    )
    kustomization = yaml.safe_load(artifacts["kustomization.yaml"])
    assert kustomization["patches"] == [
        {"path": "patch-decode.yaml"},
        {"path": "patch-prefill.yaml"},
    ]

    params = _disaggregatedset_params()
    params["LlmdConfig"]["kustomize_layout"] = False
    with pytest.raises(ValueError, match="kustomize_layout"):
        render_backend_templates(
            param_values=params,
            backend="vllm",
            deployment_target="llm-d-kustomize",
        )
