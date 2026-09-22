# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aisimulate.compiler import _worker_engine_args
from aisimulate.config.cli import CoreRecommendationConfig
from aisimulate.config.engine import EnginePredictionConfig
from aisimulate.recommend import recommendation_to_sweeper
from aisimulate.sweeper.deploy import _engine_args_payload
from aisimulate.sweeper.parallel_enum import DisaggParallelConfig, ParallelShape, ReplicaParallelConfig
from aisimulate.sweeper.sample import unroll_sample

pytestmark = pytest.mark.unit


def _recommendation_config():
    return CoreRecommendationConfig.model_validate(
        {
            "engine": {
                "model": "Qwen/Qwen3-32B",
                "hardware": "h200_sxm",
                "mode": "disaggregated",
                "context_length": 1_000_000,
                "workers": {
                    "prefill": {"context_length": 64_000},
                    "decode": {"context_length": 128_000},
                },
            },
            "optimization": {
                "target": "throughput_per_gpu",
                "constraints": {"max_candidate_gpus": 8},
            },
        }
    )


def test_recommendation_context_limits_reach_search_space():
    search = recommendation_to_sweeper(_recommendation_config())

    assert search.search_space.context_length == 1_000_000
    assert search.search_space.prefill_context_length == 64_000
    assert search.search_space.decode_context_length == 128_000


def test_aggregate_worker_context_limit_must_use_engine_field():
    config = _recommendation_config().model_dump(mode="python")
    config["engine"]["workers"]["aggregated"] = {"context_length": 64_000}

    with pytest.raises(ValueError, match="engine.context_length"):
        CoreRecommendationConfig.model_validate(config)


def test_worker_context_limit_overrides_engine_context_for_prediction():
    engine = EnginePredictionConfig(
        model="Qwen/Qwen3-32B",
        hardware="h200_sxm",
        backend="trtllm",
        context_length=1_000_000,
        workers={"aggregated": {}},
    )
    worker = engine.workers.aggregated
    assert worker is not None
    worker.context_length = 64_000

    payload = _worker_engine_args(engine, worker, "agg", transfer_bytes_per_token=None)

    assert payload["max_model_len"] == 64_000


def test_disagg_sample_preserves_role_context_limits():
    search = recommendation_to_sweeper(_recommendation_config()).search_space
    shape = ParallelShape(tp=1, pp=1, dp=1, moe_tp=1, moe_ep=1)
    parallel = DisaggParallelConfig(
        prefill=ReplicaParallelConfig(shape=shape, replicas=1),
        decode=ReplicaParallelConfig(shape=shape, replicas=1),
    )
    selection = {
        "deployment_mode": "disagg",
        "backend": "vllm",
        "prefill_max_num_batched_tokens": 8192,
        "prefill_max_num_seqs": 1,
        "decode_max_num_batched_tokens": 8192,
        "decode_max_num_seqs": 256,
    }

    sample = unroll_sample(search_space=search, selection=selection, parallel_config=parallel)

    assert sample["prefill_context_length"] == 64_000
    assert sample["decode_context_length"] == 128_000

    payload = _engine_args_payload(sample, "prefill", backend_version="0.24.0")
    assert payload["max_model_len"] == 64_000
