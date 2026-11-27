# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The FPE generator must not plan expert parallelism for a dense Gemma 4.

``check_is_moe`` resolves the ``GEMMA4MIX`` family from the parsed expert
configuration, so the op-level FPE generator plans no ``moe_ep>1`` topology for
a dense checkpoint and records no dense diagnostic for it. This mirrors the CLI
guard in ``test_support_matrix_dense_gemma4.py`` at the generator boundary.

The checked-in op-level FPE rows are one campaign's output under a single source
identity (``SourceSHA``), so a row that predates the classifier fix stays until
that campaign is re-run; issue #1006 tracks the stale rows. This check covers
the combinations the FPE matrix still enumerates.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from aisim_core.sdk.models import check_is_moe
from tools.support_matrix.fpe_support_matrix import ParallelTopology, build_probe_plans
from tools.support_matrix.support_matrix import SupportMatrix, _get_test_constraints

pytestmark = pytest.mark.unit

DENSE_REJECTION = "dense Gemma 4 variants require moe_ep_size=1"
DENSE_CHECKPOINT = "nvidia/Gemma-4-31B-IT-NVFP4"

FPE_MATRIX_DIR = Path(__file__).resolve().parents[3] / "src" / "aisim_core" / "systems" / "fpe_support_matrix"


def _fpe_rows():
    for path in sorted(FPE_MATRIX_DIR.glob("*.csv")):
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                yield path.name, row


def _current_dense_combinations(checkpoint: str) -> set[tuple[str, str, str, str]]:
    current = set(SupportMatrix().generate_combinations())
    combinations = set()
    for _path, row in _fpe_rows():
        if row["HuggingFaceID"] != checkpoint:
            continue
        key = (row["HuggingFaceID"], row["System"], row["Backend"], row["Version"])
        if key in current:
            combinations.add(key)
    return combinations


def _dense_expert_parallelism_offenders(combinations):
    """Plans or diagnostics that contradict a dense classification."""
    offenders = []
    for model, system, backend, version in sorted(combinations):
        plans = build_probe_plans(
            models={model},
            systems={system},
            backends={backend},
            backend_versions={version},
        )
        for plan in plans:
            if plan.topology.moe_ep_size > 1:
                offenders.append(f"{system}/{backend}/{version}: plans moe_ep_size={plan.topology.moe_ep_size}")
            reasons = (*plan.unrepresentable_reasons, plan.planning_error_message)
            if any(DENSE_REJECTION in reason for reason in reasons):
                offenders.append(f"{system}/{backend}/{version}: records the dense rejection")
    return offenders


def _enumerated_choices(combinations):
    """The topologies the generator offers for each combination, before builds."""
    matrix = SupportMatrix()
    for model, system, backend, version in sorted(combinations):
        task = matrix._create_task(
            mode="agg",
            model=model,
            system=system,
            backend=backend,
            version=version,
            constraints=_get_test_constraints(model),
            database_mode="SILICON",
        )
        yield (system, backend, version), sorted({tuple(choice) for choice in task.iter_parallel("agg")})


def test_fpe_generator_plans_no_expert_parallelism_for_a_dense_gemma4():
    assert not check_is_moe(DENSE_CHECKPOINT)
    combinations = _current_dense_combinations(DENSE_CHECKPOINT)
    assert combinations, "no currently enumerated FPE row covers the dense Gemma 4 checkpoint"
    offenders = []
    for (system, backend, version), choices in _enumerated_choices(combinations):
        # The enumeration is the pre-build signal: a dense checkpoint must not
        # offer expert parallelism at all, whether or not a build succeeds.
        assert choices, f"no topology enumerated for {system}/{backend}/{version}"
        offenders.extend(f"{system}/{backend}/{version}: enumerates {choice}" for choice in choices if choice[4] > 1)
    offenders.extend(_dense_expert_parallelism_offenders(combinations))
    assert not offenders, (
        "The FPE generator contradicts the dense Gemma 4 classifier; the checkpoint must not be planned with "
        "expert parallelism.\n" + "\n".join(offenders)
    )


def test_fpe_dense_gemma4_check_flags_a_planned_expert_parallel_topology(monkeypatch):
    # Guard against the check above passing because the generator planned
    # nothing: a single moe_ep>1 plan must be reported.
    plan = SimpleNamespace(
        topology=ParallelTopology(4, 1, 1, 1, 2, 1),
        unrepresentable_reasons=(),
        planning_error_message="",
    )
    monkeypatch.setattr(sys.modules[__name__], "build_probe_plans", lambda **kwargs: [plan])
    offenders = _dense_expert_parallelism_offenders({(DENSE_CHECKPOINT, "a100_sxm", "sglang", "0.5.10")})
    assert offenders == ["a100_sxm/sglang/0.5.10: plans moe_ep_size=2"]


def test_fpe_dense_gemma4_check_flags_a_recorded_dense_rejection(monkeypatch):
    plan = SimpleNamespace(
        topology=ParallelTopology(4, 1, 1, 1, 1, 1),
        unrepresentable_reasons=(f"{DENSE_REJECTION}, got 16",),
        planning_error_message="",
    )
    monkeypatch.setattr(sys.modules[__name__], "build_probe_plans", lambda **kwargs: [plan])
    offenders = _dense_expert_parallelism_offenders({(DENSE_CHECKPOINT, "a100_sxm", "sglang", "0.5.10")})
    assert offenders == ["a100_sxm/sglang/0.5.10: records the dense rejection"]
