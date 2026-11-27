# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contracts that keep migrated CI active at the repository root."""

from __future__ import annotations

import ast
import base64
import csv
import hashlib
import importlib.metadata
import io
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

import pytest
import yaml

from scripts.ci import check_python_licenses as python_licenses
from scripts.ci import generate_license_evidence as license_evidence
from scripts.ci import readme_rc_wheels as rc_wheels
from scripts.ci import select_full_ci as full_ci_selector
from scripts.ci.check_application_test_inventory import Inventory, assignment
from scripts.ci.require_fast_ci import REQUIRED_JOBS, GateError, latest_run, require_fast_ci, verify_jobs
from scripts.ci.select_full_ci import COMPONENTS, select_components
from scripts.performance import select_forward_perf as forward_perf
from scripts.release import build_manylinux_wheel as manylinux_builder
from scripts.release import build_release_artifacts as release_artifacts
from scripts.release.build_manylinux_wheel import manylinux_platform

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_ROOT = REPOSITORY_ROOT / ".github" / "workflows"
ACTION_ROOT = REPOSITORY_ROOT / ".github" / "actions"


@pytest.mark.parametrize("pool", ["amd64", "arm64"])
@pytest.mark.parametrize("failed_probe", [None, "machine", "workspace", "scratch"])
def test_runner_canary_probes_survive_container_home_override(tmp_path, pool, failed_probe):
    job = _workflow("runner-canary.yml")["jobs"]["canary"]
    entry = next(item for item in job["strategy"]["matrix"]["include"] if item["pool"] == pool)
    step = next(item for item in job["steps"] if "run" in item)
    assert job["container"]["env"]["USER"] == "runner"
    workspace = tmp_path / "workspace"
    scratch = tmp_path / "scratch"
    workspace.mkdir()
    scratch.mkdir()
    log = tmp_path / "probes.log"
    log.touch()
    # Model the runner replacing container HOME, then applying the job and step
    # environments. Only host paths and architecture are controlled here; the
    # exact canary run block still performs its HOME and source assertions.
    env = {
        **os.environ,
        **job["container"]["env"],
        "HOME": "/github/home",
        **job.get("env", {}),
        "EXPECTED_MACHINE": entry["machine"],
        "REPORTED_MACHINE": "incompatible-machine" if failed_probe == "machine" else entry["machine"],
        "EXPECT_SCRATCH": str(entry["scratch"]).lower(),
        "GITHUB_REPOSITORY": "ai-dynamo/aisim",
        "GITHUB_SHA": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True).strip(),
        "RUNNER_NAME": f"controlled-{pool}",
        "PROBE_WORKSPACE": str(workspace),
        "PROBE_SCRATCH": str(scratch),
        "PROBE_WORKSPACE_WRITABLE": str(failed_probe != "workspace").lower(),
        "PROBE_SCRATCH_WRITABLE": str(failed_probe != "scratch").lower(),
        "PROBE_LOG": str(log),
    }
    controls = r"""
pwd() { printf '/__w/aisim/aisim\n'; }
uname() { printf 'architecture\n' >> "$PROBE_LOG"; printf '%s\n' "$REPORTED_MACHINE"; }
test() {
  if [[ "$1" == -w && "$2" == /__w/aisim/aisim ]]; then
    [[ "$PROBE_WORKSPACE_WRITABLE" == true ]] && builtin test -w "$PROBE_WORKSPACE"
  elif [[ "$1" == -w && "$2" == /scratch ]]; then
    [[ "$PROBE_SCRATCH_WRITABLE" == true ]] && builtin test -w "$PROBE_SCRATCH"
  else
    builtin test "$@"
  fi
}
mktemp() {
  printf '%s\n' "$1" >> "$PROBE_LOG"
  case "$1" in
    /__w/aisim/aisim/*) command mktemp "$PROBE_WORKSPACE/${1##*/}" ;;
    /scratch/*) command mktemp "$PROBE_SCRATCH/${1##*/}" ;;
    *) return 1 ;;
  esac
}
"""
    result = subprocess.run(
        ["bash", "-euc", controls + step["run"]],
        cwd=REPOSITORY_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    probes = [
        "architecture",
        "/__w/aisim/aisim/.runner-canary.XXXXXX",
        "/scratch/runner-canary.XXXXXX",
    ]
    if failed_probe is None:
        assert result.returncode == 0, result.stderr
        assert f"({entry['machine']}) with writable HOME=/__w/aisim/aisim" in result.stdout
        assert log.read_text().splitlines() == probes
    else:
        assert result.returncode != 0
        assert "Verified source" not in result.stdout
        completed = {"workspace": 0, "machine": 1, "scratch": 2}[failed_probe]
        assert log.read_text().splitlines() == probes[:completed]
    assert not list(workspace.iterdir())
    assert not list(scratch.iterdir())


def test_documented_active_runner_labels_match_workflow_selectors():
    labels = set()
    for path in WORKFLOW_ROOT.glob("*.yml"):
        for job in _workflow(path.name)["jobs"].values():
            selector = job.get("runs-on", "")
            matrix = job.get("strategy", {}).get("matrix", {})
            if selector == "${{ matrix.runner }}":
                selectors = [entry["runner"] for entry in matrix["include"]]
            elif isinstance(selector, str) and "${{ matrix.arch }}" in selector:
                arches = matrix["arch"] if "arch" in matrix else [entry["arch"] for entry in matrix["include"]]
                selectors = [selector.replace("${{ matrix.arch }}", arch) for arch in arches]
            else:
                selectors = selector if isinstance(selector, list) else [selector]
            labels.update(label for label in selectors if isinstance(label, str) and label.startswith("prod-aisim"))
    assert labels
    documentation = (REPOSITORY_ROOT / "docs/ci/README.md").read_text()
    assert set(re.findall(r"`(prod-aisim[^`]+)`", documentation)) == labels
    lint_labels = yaml.safe_load((REPOSITORY_ROOT / ".github/actionlint.yaml").read_text())
    assert {label for label in lint_labels["self-hosted-runner"]["labels"] if label.startswith("prod-aisim")} == labels


def test_stable_release_migrations_require_reviewed_clearance(tmp_path):
    from scripts.release.check_release_migrations import GATES, require_completed_migrations

    with pytest.raises(RuntimeError, match="dynamo/pull/14065"):
        require_completed_migrations(GATES)
    path = tmp_path / "gates.json"
    path.write_text(json.dumps({"pending_migrations": []}))
    require_completed_migrations(path)
    for invalid in ({}, {"pending_migrations": None}, {"pending_migrations": [{}]}):
        path.write_text(json.dumps(invalid))
        with pytest.raises(ValueError):
            require_completed_migrations(path)
    path.unlink()
    with pytest.raises(FileNotFoundError):
        require_completed_migrations(path)


def test_nightly_can_publish_the_wheel_needed_by_pending_downstream_migrations():
    workflow = _workflow("nightly-ci.yml")
    serialized_workflow = json.dumps(workflow)
    assert "check_release_migrations.py" not in serialized_workflow
    assert "release-gates.json" not in serialized_workflow
    jobs = workflow["jobs"]
    guard = jobs["changes-guard"]
    commands = "\n".join(_run_commands(job) for job in jobs.values() if "steps" in job)
    assert "check_release_migrations.py" not in commands
    assert "release-gates.json" not in commands
    assert not any(step.get("uses", "").startswith("actions/checkout@") for step in guard["steps"])
    steps = [step.get("id") for step in guard["steps"]]
    assert steps.index("target") < steps.index("version") < steps.index("decide")
    assert guard["outputs"]["dev-version"] == "${{ steps.version.outputs.dev-version }}"
    build = jobs["build-artifacts"]
    assert "scripts/release/apply_dev_version.py" in _run_commands(build)
    assert {"changes-guard", "manual-approval", "python-compliance"} <= set(build["needs"])
    assert "needs.changes-guard.outputs.should-build == 'true'" in build["if"]
    publish = jobs["trigger-gitlab-security"]
    assert {"build-artifacts", "manual-approval", "fpe-support-matrix", "license-evidence"} <= set(publish["needs"])
    for name in ("build-artifacts", "fpe-support-matrix", "license-evidence"):
        assert f"needs.{name}.result == 'success'" in publish["if"]


@pytest.mark.parametrize(
    "current,target,expected",
    [
        ("clear", "clear", 0),
        ("clear", "pending", 1),
        ("pending", "clear", 1),
        ("clear", "missing", 1),
        ("clear", "malformed", 1),
    ],
)
def test_stable_publication_checks_current_policy_and_selected_target(tmp_path, monkeypatch, current, target, expected):
    from scripts.release import check_release_migrations as checker

    paths = {}
    for name, state in (("current", current), ("target", target)):
        paths[name] = tmp_path / f"{name}.json"
        if state == "missing":
            continue
        pending = [] if state == "clear" else [{"pull_request": "migration/pr/1", "requirement": "Migrate consumer"}]
        document = {} if state == "malformed" else {"pending_migrations": pending}
        paths[name].write_text(json.dumps(document))
    monkeypatch.setattr(checker, "GATES", paths["current"])
    assert checker.main(["--target-gates", str(paths["target"])]) == expected


def test_forward_perf_selects_before_allocating_the_benchmark_runner():
    workflow = _workflow("performance.yml")
    assert workflow["on"]["push"] == {"branches": ["pull-request/*"]}
    assert set(workflow["on"]) == {"push", "workflow_dispatch"}
    selector = workflow["jobs"]["select"]
    compare = workflow["jobs"]["compare"]
    assert selector["runs-on"] == "ubuntu-24.04"
    # Draft and stacked-base skips are selector decisions with a reported reason.
    assert "if" not in selector
    assert "python scripts/performance/select_forward_perf.py" in _run_commands(selector)
    assert compare["needs"] == "select"
    assert compare["if"] == "needs.select.outputs.run_comparison == 'true'"
    assert set(selector["outputs"]) == {
        "number",
        "head_sha",
        "base_ref",
        "run_comparison",
    }
    checkout = next(step for step in compare["steps"] if step.get("uses", "").startswith("actions/checkout@"))
    assert checkout["with"]["ref"] == "${{ needs.select.outputs.head_sha }}"
    assert "${BASE_SRC}/python/aisim/tools/forward_perf_gate/run.py" in _run_commands(compare)


def test_forward_perf_validates_the_pr_controller_without_replacing_the_base_comparison():
    steps = _workflow("performance.yml")["jobs"]["compare"]["steps"]
    revisions = next(step for step in steps if step.get("id") == "revisions")
    assert "validate_head_controller=true" in revisions["run"]
    assert "validate_head_controller=false" in revisions["run"]
    base = next(step for step in steps if step.get("name") == "Run paired benchmark")
    head = next(step for step in steps if step.get("name") == "Validate PR benchmark controller")
    assert "!cancelled()" in head["if"]
    assert "steps.build.outcome == 'success'" in head["if"]
    assert "steps.revisions.outputs.validate_head_controller == 'true'" in head["if"]
    build = next(step for step in steps if step.get("id") == "build")
    assert build["name"] == "Build and install both revisions"
    for destination in ("${BASE_DIST}", "${HEAD_DIST}"):
        assert f'python -m maturin build --release --locked --out "{destination}"' in build["run"]
    expected = base["run"].replace('"${BASE_VENV}/bin/python"', '"${HEAD_VENV}/bin/python"', 1)
    expected = expected.replace(
        "${BASE_SRC}/python/aisim/tools/forward_perf_gate/run.py",
        "${HEAD_SRC}/python/aisim/tools/forward_perf_gate/run.py",
    )
    expected = expected.replace('--output-dir "${RESULTS_DIR}"', '--output-dir "${RESULTS_DIR}/head-controller"')
    assert head["run"] == expected
    publish = next(step for step in steps if step.get("name") == "Publish PR controller validation")
    assert publish["if"].startswith("always()")
    assert "head-controller/summary.md" in publish["run"]
    assert "head-controller/annotations.txt" in publish["run"]


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("python/aisim/tools/forward_perf_gate/cases.py", True),
        ("python/aisim/tools/forward_perf_gate/measurement.py", True),
        ("python/aisim/tools/prediction_regression_gate/grid.py", True),
        ("python/aisim/tools/forward_perf_gate/README.md", False),
    ],
)
def test_forward_perf_controller_change_detection(tmp_path, path, expected):
    steps = _workflow("performance.yml")["jobs"]["compare"]["steps"]
    revisions = next(step for step in steps if step.get("id") == "revisions")["run"]
    detection = "controller_changes=" + revisions.split("controller_changes=", 1)[1].split("git worktree prune", 1)[0]
    _git(tmp_path, "init", "--quiet")
    base = _commit_file(tmp_path, "base", "base\n")
    (tmp_path / path).parent.mkdir(parents=True)
    head = _commit_file(tmp_path, path, "changed\n")
    output = tmp_path / "output"
    subprocess.run(
        ["bash", "-euc", detection],
        cwd=tmp_path,
        env={
            **os.environ,
            "base_sha": base,
            "PR_HEAD_SHA": head,
            "gate_path": "python/aisim/tools/forward_perf_gate",
            "GITHUB_OUTPUT": str(output),
        },
        check=True,
    )
    assert output.read_text() == f"validate_head_controller={str(expected).lower()}\n"
    assert forward_perf.matches_path(path) is expected


@pytest.mark.parametrize("base_protocol", [1, 2])
def test_forward_perf_skips_incompatible_data_policies(tmp_path, base_protocol):
    steps = _workflow("performance.yml")["jobs"]["compare"]["steps"]
    revisions = next(step for step in steps if step.get("id") == "revisions")["run"]
    protocol_check = revisions.split('if [[ -z "${skip_reason}" ]]; then', 1)[1]
    protocol_check = 'if [[ -z "${skip_reason}" ]]; then' + protocol_check.split("controller_changes=", 1)[0]
    gate_path = "python/aisim/tools/forward_perf_gate"
    _git(tmp_path, "init", "--quiet")
    (tmp_path / gate_path).mkdir(parents=True)
    base = _commit_file(tmp_path, f"{gate_path}/__init__.py", f"PROTOCOL_VERSION = {base_protocol}\n")
    head = _commit_file(tmp_path, f"{gate_path}/__init__.py", "# Shared-layer reuse enabled.\nPROTOCOL_VERSION = 2\n")
    output = tmp_path / "output"
    summary = tmp_path / "summary"
    subprocess.run(
        ["bash", "-euc", protocol_check],
        cwd=tmp_path,
        env={
            **os.environ,
            "base_sha": base,
            "PR_HEAD_SHA": head,
            "gate_path": gate_path,
            "skip_reason": "",
            "GITHUB_OUTPUT": str(output),
            "GITHUB_STEP_SUMMARY": str(summary),
        },
        check=True,
    )
    if base_protocol == 1:
        assert output.read_text() == "run_comparison=false\n"
        assert "protocol versions differ: base is 1, head is 2" in summary.read_text()
    else:
        assert not output.exists()
        assert not summary.exists()


def _forward_api(pages, *, count=None, after=None, canonical="a" * 40, draft=False, base_ref="main"):
    pull = {
        "head": {"sha": "a" * 40},
        "base": {"sha": "b" * 40, "ref": base_ref},
        "draft": draft,
        "changed_files": sum(map(len, pages)) if count is None else count,
    }
    calls = []

    def api(endpoint):
        calls.append(endpoint)
        if "/files?" in endpoint:
            return pages
        if "/git/ref/" in endpoint:
            return [{"object": {"sha": canonical}}]
        return [after if after is not None and calls.count(endpoint) > 1 else pull]

    return api, calls


@pytest.mark.parametrize(
    ("pages", "count", "expected"),
    [
        ([[{"filename": "crates/core/src/python.rs"}]], None, "true"),
        ([[{"filename": "docs/ci.md"}]], None, "false"),
        ([[{"filename": "python/aisim/tools/forward_perf_gate/README.md"}]], None, "false"),
        # Full PR files still include the code change after a later docs-only push.
        (
            [[{"filename": "docs/ci.md"}], [{"filename": "crates/core/src/python.rs"}]],
            None,
            "true",
        ),
        (
            [
                [
                    {
                        "filename": "archive/old.py",
                        "previous_filename": "python/aisim/src/aisim_core/foo.py",
                    }
                ]
            ],
            None,
            "true",
        ),
        ([[]], 0, "true"),
        ([[{"filename": "docs/ci.md"}]], 2, "true"),
        ([[]], 3001, "true"),
    ],
)
def test_forward_perf_uses_complete_pr_files(pages, count, expected):
    api, calls = _forward_api(pages, count=count)
    # Selection intentionally has no dependency on push.commits or push.before.
    result = forward_perf.select_comparison("owner/repo", "push", "refs/heads/pull-request/236", "a" * 40, api=api)
    assert result["run_comparison"] == expected
    assert result["head_sha"] == "a" * 40
    assert result["number"] == "236"
    assert result["base_ref"] == "main"
    assert calls[0] == calls[-1] == "repos/owner/repo/pulls/236"
    assert any("/files?per_page=100" in call for call in calls) is (count != 3001)


@pytest.mark.parametrize("selector", ["select_forward_perf", "select_simulation_perf"])
@pytest.mark.parametrize(
    ("draft", "base_ref", "reason"),
    [
        (False, "main", None),
        (False, "release/0.3", None),
        (True, "main", "PR is a draft"),
        (True, "feature/base", "PR is a draft"),
        (False, "feature/base", "PR targets `feature/base`"),
        (False, "release/0.3/backport", "PR targets `release/0.3/backport`"),
        (False, "mainline", "PR targets `mainline`"),
    ],
)
def test_perf_pushes_skip_drafts_and_stacked_bases(selector, draft, base_ref, reason):
    module = importlib.import_module(f"scripts.performance.{selector}")
    api, calls = _forward_api([[{"filename": "crates/core/src/python.rs"}]], draft=draft, base_ref=base_ref)
    result = module.select_comparison("owner/repo", "push", "refs/heads/pull-request/236", "a" * 40, api=api)
    assert result["number"] == "236"
    assert result["head_sha"] == "a" * 40
    assert result["base_ref"] == base_ref
    assert calls[0] == calls[-1] == "repos/owner/repo/pulls/236"
    if reason is None:
        assert result["run_comparison"] == "true"
        assert any("/files?per_page=100" in call for call in calls)
    else:
        # Skips need no file list, but still re-read the PR for a stable revision.
        assert result["run_comparison"] == "false"
        assert result["reason"].startswith(reason)
        assert "Dispatch manually with pr_number" in result["reason"]
        assert calls == ["repos/owner/repo/pulls/236"] * 2


@pytest.mark.parametrize("draft", [None, "false"])
def test_perf_selection_rejects_an_unknown_draft_state(draft):
    api, _ = _forward_api([[{"filename": "crates/core/src/python.rs"}]], draft=draft)
    with pytest.raises(ValueError, match="Invalid draft state for PR #236"):
        forward_perf.select_comparison("owner/repo", "push", "refs/heads/pull-request/236", "a" * 40, api=api)


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("scripts/performance/select_forward_perf.py", True),
        ("Cargo.toml.bak", False),
        ("crates/core/src/engine/nested/predict.rs", True),
        ("crates/core/src/engine-other/predict.rs", False),
        ("python/aisim/src/aisim_core/example.py", True),
        ("python/aisim/src/aisim_core/unrelated/example.py", False),
        ("python/aisim/src/aisim_core/systems/h100_sxm.yaml", True),
        (
            "python/aisim/src/aisim_core/systems/unrelated/nested.yaml",
            False,
        ),
    ],
)
def test_forward_perf_path_matching_preserves_directory_boundaries(path, expected):
    assert forward_perf.matches_path(path) is expected


@pytest.mark.parametrize(
    ("draft", "base_ref"),
    [(False, "main"), (True, "main"), (False, "feature/base"), (True, "feature/base")],
)
def test_forward_perf_manual_dispatch_forces_comparison_of_the_trusted_copy(draft, base_ref):
    api, calls = _forward_api([[{"filename": "docs/ci.md"}]], draft=draft, base_ref=base_ref)
    result = forward_perf.select_comparison(
        "owner/repo", "workflow_dispatch", "refs/heads/main", "c" * 40, "236", api=api
    )
    assert result["run_comparison"] == "true"
    assert result["reason"] == "Manual dispatch requests a comparison."
    assert result["head_sha"] == "a" * 40
    assert result["base_ref"] == base_ref
    assert not any("/files?" in call for call in calls)


@pytest.mark.parametrize("change", ["head", "base_sha", "base_ref", "stale_push", "stale_manual"])
def test_forward_perf_rejects_changed_or_stale_revisions(change):
    after = {"head": {"sha": "a" * 40}, "base": {"sha": "b" * 40, "ref": "main"}}
    if change == "head":
        after["head"]["sha"] = "d" * 40
    elif change == "base_sha":
        after["base"]["sha"] = "d" * 40
    elif change == "base_ref":
        after["base"]["ref"] = "release/test"
    api, _ = _forward_api(
        [[{"filename": "docs/ci.md"}]],
        after=after,
        canonical="d" * 40 if change == "stale_manual" else "a" * 40,
    )
    with pytest.raises(ValueError, match="does not match|head or base changed"):
        forward_perf.select_comparison(
            "owner/repo",
            "workflow_dispatch" if change == "stale_manual" else "push",
            "refs/heads/pull-request/236",
            "d" * 40 if change == "stale_push" else "a" * 40,
            "236",
            api=api,
        )


@pytest.mark.parametrize(
    ("scenario", "reason"),
    [
        ("unrelated", "No PR files affect forward prediction."),
        ("draft", "PR is a draft"),
        ("stacked", "PR targets `feature/base`"),
        ("api_failure", None),
    ],
)
def test_forward_perf_cli_reports_skip_or_api_failure(monkeypatch, tmp_path, scenario, reason):
    fail_api = scenario == "api_failure"
    api, _ = _forward_api(
        [[{"filename": "docs/ci.md"}]],
        draft=scenario == "draft",
        base_ref="feature/base" if scenario == "stacked" else "main",
    )

    def run(command, **kwargs):
        assert command[:4] == ["gh", "api", "--paginate", "--slurp"]
        assert kwargs["check"] is True
        if fail_api and "/files?" in command[-1]:
            raise subprocess.CalledProcessError(1, command)
        return SimpleNamespace(stdout=json.dumps(api(command[-1])))

    monkeypatch.setattr(forward_perf.subprocess, "run", run)
    for name, value in {
        "GITHUB_REPOSITORY": "owner/repo",
        "GITHUB_EVENT_NAME": "push",
        "GITHUB_REF": "refs/heads/pull-request/236",
        "GITHUB_SHA": "a" * 40,
        "GITHUB_OUTPUT": str(tmp_path / "output"),
        "GITHUB_STEP_SUMMARY": str(tmp_path / "summary"),
    }.items():
        monkeypatch.setenv(name, value)
    assert forward_perf.main() == int(fail_api)
    if fail_api:
        assert not (tmp_path / "output").exists()
        assert not (tmp_path / "summary").exists()
    else:
        assert "run_comparison=false" in (tmp_path / "output").read_text()
        assert f"**SKIPPED** — {reason}" in (tmp_path / "summary").read_text()


def _fast_run(**overrides):
    return {
        "id": 10,
        "path": ".github/workflows/fast-ci.yml",
        "head_sha": "a" * 40,
        "head_branch": "main",
        "head_repository": {"full_name": "ai-dynamo/aisim"},
        "event": "push",
        "status": "completed",
        "conclusion": "success",
        "run_attempt": 1,
        "html_url": "https://github.com/ai-dynamo/aisim/actions/runs/10",
        **overrides,
    }


def _fast_jobs():
    return [
        {"name": name, "head_sha": "a" * 40, "status": "completed", "conclusion": "success"}
        for name in sorted(REQUIRED_JOBS)
    ]


def _require_fast(api, **kwargs):
    return require_fast_ci("ai-dynamo/aisim", "a" * 40, "refs/heads/main", "push", api=api, **kwargs)


def test_fast_prerequisite_accepts_all_jobs_from_the_latest_attempt_across_pages():
    calls = []

    def api(endpoint):
        calls.append(endpoint)
        if "/jobs?" in endpoint:
            assert "/attempts/2/" in endpoint
            return [{"jobs": _fast_jobs()[:2]}, {"jobs": _fast_jobs()[2:]}]
        return [
            {"workflow_runs": [_fast_run(id=9, conclusion="failure")]},
            {"workflow_runs": [_fast_run(run_attempt=2)]},
        ]

    assert _require_fast(api)["run_attempt"] == 2
    assert len(calls) == 3


@pytest.mark.parametrize(
    "override",
    [
        {"head_sha": "b" * 40},
        {"head_branch": "release/old"},
        {"head_repository": {"full_name": "someone/aisim"}},
        {"path": ".github/workflows/ci.yml"},
        {"event": "pull_request"},
        {"event": "workflow_dispatch"},
    ],
)
def test_fast_prerequisite_rejects_wrong_commit_branch_origin_workflow_and_event(override):
    with pytest.raises(GateError, match="No complete Fast CI evidence"):
        _require_fast(lambda _: [{"workflow_runs": [_fast_run(**override)]}], timeout=0)


def test_manual_full_ci_can_reuse_a_same_branch_push_or_manual_fast_run():
    for event in ("push", "workflow_dispatch"):
        run = _fast_run(event=event)
        assert latest_run([{"workflow_runs": [run]}], "a" * 40, "main", "workflow_dispatch", "ai-dynamo/aisim") == run


@pytest.mark.parametrize("conclusion", ["failure", "cancelled", "skipped", "neutral", "timed_out", None])
def test_fast_prerequisite_cannot_fall_back_to_an_older_success(conclusion):
    pages = [{"workflow_runs": [_fast_run(), _fast_run(id=11, conclusion=conclusion)]}]
    with pytest.raises(GateError, match="Latest Fast CI run 11"):
        _require_fast(lambda _: pages)


@pytest.mark.parametrize(
    "override",
    [{"head_sha": "b" * 40}, {"status": "queued"}, {"conclusion": "skipped"}, {"conclusion": "failure"}],
)
def test_fast_prerequisite_rejects_partial_or_wrong_commit_job_evidence(override):
    jobs = _fast_jobs()
    jobs[0].update(override)
    with pytest.raises(GateError, match="wrong-commit job"):
        verify_jobs([{"jobs": jobs}], "a" * 40)


@pytest.mark.parametrize("kind", ["missing", "duplicate", "aggregate-only"])
def test_fast_prerequisite_requires_substantive_jobs_not_just_a_green_aggregate(kind):
    jobs = _fast_jobs()
    jobs = jobs[1:] if kind == "missing" else jobs + [jobs[0]] if kind == "duplicate" else [jobs[0]]
    with pytest.raises(GateError, match="every required job"):
        verify_jobs([{"jobs": jobs}], "a" * 40)


@pytest.mark.parametrize("status", ["queued", "in_progress", "waiting"])
def test_fast_prerequisite_waits_with_a_bounded_timeout(status):
    now = [0]

    def sleep(seconds):
        now[0] += seconds

    with pytest.raises(GateError, match="within 20s"):
        _require_fast(
            lambda _: [{"workflow_runs": [_fast_run(status=status, conclusion=None)]}],
            timeout=20,
            interval=15,
            clock=lambda: now[0],
            sleep=sleep,
        )
    assert now[0] == 20


def test_fast_prerequisite_rechecks_a_rerun_started_during_job_inspection():
    reads = [0]

    def api(endpoint):
        if "/jobs?" in endpoint:
            return [{"jobs": _fast_jobs()}]
        reads[0] += 1
        run = _fast_run() if reads[0] == 1 else _fast_run(run_attempt=2, conclusion="failure")
        return [{"workflow_runs": [run]}]

    with pytest.raises(GateError, match="Latest Fast CI run 10"):
        _require_fast(api, sleep=lambda _: None)


def test_fast_prerequisite_api_failure_never_becomes_success():
    def api(_):
        raise GateError("API unavailable")

    with pytest.raises(GateError, match="API unavailable"):
        _require_fast(api)


def _run_fast_prerequisite_cli(tmp_path, *, overrides=False, failure=None):
    run = _fast_run(head_branch="codex/manual", event="workflow_dispatch")
    jobs = _fast_jobs()
    if failure == "wrong-job-sha":
        jobs[0]["head_sha"] = "b" * 40
    fixture = tmp_path / "api.json"
    fixture.write_text(json.dumps({"run": run, "jobs": jobs}))
    gh = tmp_path / "gh"
    gh.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, sys\n"
        "assert sys.argv[1:4] == ['api', '--paginate', '--slurp']\n"
        "if os.environ['TEST_API_FAILURE'] == 'true': sys.exit(1)\n"
        "data = json.loads(pathlib.Path(os.environ['TEST_API_FIXTURE']).read_text())\n"
        "with open(os.environ['TEST_API_CALLS'], 'a') as stream: stream.write(sys.argv[-1] + '\\n')\n"
        "print(json.dumps([{'jobs': data['jobs']} if '/jobs?' in sys.argv[-1] "
        "else {'workflow_runs': [data['run']]}]))\n"
    )
    gh.chmod(0o755)
    summary = tmp_path / "summary.md"
    summary.write_text("Existing evidence\n")
    env = {
        **os.environ,
        "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
        "GITHUB_REPOSITORY": "ai-dynamo/aisim",
        "GITHUB_SHA": "a" * 40,
        "GITHUB_REF": "refs/heads/codex/manual",
        "GITHUB_EVENT_NAME": "workflow_dispatch",
        "GITHUB_STEP_SUMMARY": str(summary),
        "TEST_API_FIXTURE": str(fixture),
        "TEST_API_CALLS": str(tmp_path / "calls"),
        "TEST_API_FAILURE": str(failure == "api").lower(),
    }
    args = [sys.executable, str(REPOSITORY_ROOT / "scripts/ci/require_fast_ci.py"), "--timeout", "0"]
    if overrides:
        for option, variable in (
            ("--repository", "GITHUB_REPOSITORY"),
            ("--sha", "GITHUB_SHA"),
            ("--ref", "GITHUB_REF"),
            ("--event", "GITHUB_EVENT_NAME"),
        ):
            args.extend([option, env[variable]])
            env[variable] = "invalid-environment-default"
    result = subprocess.run(args, env=env, capture_output=True, text=True, check=False, timeout=10)
    return result, summary


@pytest.mark.parametrize("overrides", [False, True], ids=["environment-defaults", "cli-overrides"])
def test_fast_prerequisite_cli_verifies_manual_run_and_appends_summary(tmp_path, overrides):
    result, summary = _run_fast_prerequisite_cli(tmp_path, overrides=overrides)
    assert result.returncode == 0, result.stdout + result.stderr
    assert summary.read_text().startswith("Existing evidence\n### Standalone Fast CI verified")
    for evidence in ("a" * 40, "https://github.com/ai-dynamo/aisim/actions/runs/10", "attempt 1"):
        assert evidence in result.stdout
        assert evidence in summary.read_text()
    calls = (tmp_path / "calls").read_text().splitlines()
    assert calls[0] == calls[2]
    assert calls[0].startswith("repos/ai-dynamo/aisim/actions/workflows/fast-ci.yml/runs?")
    assert calls[1] == "repos/ai-dynamo/aisim/actions/runs/10/attempts/1/jobs?per_page=100"


@pytest.mark.parametrize("failure", ["api", "wrong-job-sha"])
def test_fast_prerequisite_cli_errors_leave_success_summary_unwritten(tmp_path, failure):
    result, summary = _run_fast_prerequisite_cli(tmp_path, failure=failure)
    assert result.returncode != 0
    assert "::error::" in result.stderr
    assert "Standalone Fast CI verified" not in result.stdout
    assert summary.read_text() == "Existing evidence\n"


@pytest.mark.parametrize(
    ("path", "markers", "expected"),
    [
        ("tests/unit/generator/test_unmarked.py", set(), "unit"),
        ("tests/unit/sdk/test_mixed.py", {"integration"}, "unit"),
        ("tests/golden/generator/test_contract.py", set(), "unit"),
        ("tests/integration/test_new.py", set(), "integration-build"),
        ("tests/cross_package/test_public_api.py", set(), "contracts"),
        ("tests/e2e/cli/test_new.py", {"build"}, "cli-build"),
        ("tests/e2e/support_matrix/test_new.py", {"build"}, "integration-build"),
        ("tests/e2e/tools/test_new.py", {"build"}, "integration-build"),
    ],
)
def test_collected_inventory_assigns_cases_without_requiring_unit_markers(path, markers, expected):
    assert assignment(path, markers, {})[0] == expected
    # Inventory shards name the Full CI legs that run the case.
    shards = _workflow("ci.yml")["jobs"]["application-tests"]["strategy"]["matrix"]["shard"]
    assert expected in {shard["suite"] for shard in shards}


def test_collected_inventory_requires_explicit_manual_exceptions():
    path = "tests/e2e/cli/test_full_sweep.py"
    with pytest.raises(ValueError, match="unassigned collected test"):
        assignment(path, {"e2e", "sweep"}, {})
    assert assignment(path, {"e2e", "sweep"}, {path: "manual compatibility sweep"}) == (
        "manual",
        "manual compatibility sweep",
    )
    with pytest.raises(ValueError, match="unassigned collected test"):
        assignment("tests/new_category/test_new.py", {"unit"}, {})


def test_collected_inventory_reports_unexpected_skips_and_collection_failures():
    inventory = Inventory(
        {"manual_suites": {}, "optional_collection_skips": {"tests/unit/test_tensor.py": "real torch"}}
    )
    inventory.pytest_collectreport(
        SimpleNamespace(failed=False, skipped=True, nodeid="tests/unit/test_tensor.py", longrepr="real torch required")
    )
    assert not inventory.errors
    inventory.pytest_collectreport(
        SimpleNamespace(
            failed=False,
            skipped=True,
            nodeid="tests/integration/test_native.py",
            longrepr="native extension unavailable",
        )
    )
    inventory.pytest_collectreport(
        SimpleNamespace(failed=True, skipped=False, nodeid="tests/unit/test_bad.py", longrepr="import failed")
    )
    assert len(inventory.errors) == 2
    assert "unexpected collection skip" in inventory.errors[0]
    assert "collection failed" in inventory.errors[1]


def _workflow(name: str) -> dict:
    with (WORKFLOW_ROOT / name).open(encoding="utf-8") as handle:
        return yaml.load(handle, Loader=yaml.BaseLoader)


def test_commit_authorship_rechecks_retargeted_history(tmp_path: Path) -> None:
    workflow = _workflow("commit-authorship.yml")
    assert set(workflow["on"]["pull_request"]["types"]) == {"opened", "synchronize", "reopened", "edited"}
    job = workflow["jobs"]["commit-authorship"]
    assert job["env"]["BASE_SHA"] == (
        "${{ github.event.pull_request.base.sha || github.event.merge_group.base_sha || github.event.before }}"
    )
    assert job["env"]["HEAD_SHA"] == (
        "${{ github.event.pull_request.head.sha || github.event.merge_group.head_sha || github.sha }}"
    )
    command = next(step["run"] for step in job["steps"] if step.get("name") == "Check new commits")
    checker = tmp_path / ".github" / "commit-policy" / "check.py"
    checker.parent.mkdir(parents=True)
    shutil.copyfile(REPOSITORY_ROOT / ".github" / "commit-policy" / "check.py", checker)
    _git(tmp_path, "init", "--quiet")
    main = _commit_file(tmp_path, "main.txt", "main\n")
    _git(tmp_path, "commit", "--quiet", "--allow-empty", "--author=Codex <codex@openai.com>", "-m", "parent PR")
    parent = _git(tmp_path, "rev-parse", "HEAD").stdout.strip()
    head = _commit_file(tmp_path, "head.txt", "human child PR\n")
    # Retarget the same head from the parent PR to main. The parent commit
    # becomes newly included, so the workflow must run again with the new base.
    for base, expected in ((parent, 0), (main, 1)):
        result = subprocess.run(
            ["bash", "-eo", "pipefail", "-c", command],
            cwd=tmp_path,
            env={**os.environ, "BASE_SHA": base, "HEAD_SHA": head},
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == expected, result.stdout + result.stderr
        if expected:
            assert f"{parent}: forbidden AI attribution: author: Codex <codex@openai.com>" in result.stderr
        else:
            assert "Checked 1 commit(s): no known AI authorship found." in result.stdout


def test_nvskills_request_events():
    assert _workflow("request-nvskills-ci.yml")["on"] == {
        "issue_comment": {"types": ["created"]},
        "pull_request": {"types": ["opened", "reopened", "synchronize", "ready_for_review"]},
        "push": "",
    }


def test_nvskills_request_token_permissions():
    workflow = _workflow("request-nvskills-ci.yml")
    assert workflow["permissions"] == {}
    assert workflow["jobs"]["request"]["permissions"] == {
        "contents": "read",
        "pull-requests": "read",
        "statuses": "read",
    }


def test_nvskills_request_pins_the_external_callee():
    request = _workflow("request-nvskills-ci.yml")["jobs"]["request"]
    assert request["uses"] == (
        "NVIDIA/skills/.github/workflows/team-request.yml@bdc6bf92f4f76c741f0d499415f75c2ee7ceab6f"
    )


def test_nvskills_request_passes_only_the_dispatch_secret():
    jobs = _workflow("request-nvskills-ci.yml")["jobs"]
    assert set(jobs) == {"request"}
    assert jobs["request"]["secrets"] == {"NVSKILLS_CI_DISPATCH_TOKEN": "${{ secrets.NVSKILLS_CI_DISPATCH_TOKEN }}"}
    assert "steps" not in jobs["request"]


def _nvskills_comment(body, *, is_pr=True):
    issue = {"pull_request": {"url": "https://api.github.com/repos/ai-dynamo/aisim/pulls/936"}} if is_pr else {}
    return {"event_name": "issue_comment", "event": {"issue": issue, "comment": {"body": body}}}


def _nvskills_push(actor="nv-skills-ci[bot]", message="Attach NVSkills validation signatures"):
    return {"event_name": "push", "actor": actor, "event": {"head_commit": {"message": message}}}


@pytest.mark.parametrize(
    ("github", "variables", "expected"),
    [
        *[
            pytest.param({"event_name": "pull_request", "event": {"action": action}}, {}, True, id=f"pr-{action}")
            for action in ("opened", "reopened", "synchronize", "ready_for_review")
        ],
        pytest.param(_nvskills_comment("/nvskills-ci"), {}, True, id="pr-command"),
        pytest.param(_nvskills_comment("/nvskills-ci --force"), {}, True, id="command-arguments"),
        pytest.param(_nvskills_comment("/nvskills-ci\nmore context"), {}, True, id="multiline-comment"),
        pytest.param(_nvskills_comment("/NVSKILLS-CI"), {}, True, id="case-insensitive-command"),
        # startsWith deliberately accepts suffixes; this is not an exact command parser.
        pytest.param(_nvskills_comment("/nvskills-ci-extra"), {}, True, id="command-prefix"),
        pytest.param(_nvskills_comment("ordinary review comment"), {}, False, id="ordinary-comment"),
        pytest.param(_nvskills_comment(""), {}, False, id="empty-comment"),
        pytest.param(_nvskills_comment(" /nvskills-ci"), {}, False, id="leading-space-comment"),
        pytest.param(_nvskills_comment("/nvskills-ci", is_pr=False), {}, False, id="non-pr-issue"),
        pytest.param({"event_name": "issue_comment", "event": {}}, {}, False, id="missing-comment-fields"),
        pytest.param(_nvskills_comment(None), {}, False, id="null-comment"),
        pytest.param(_nvskills_push(), {}, True, id="signature-push"),
        pytest.param(
            _nvskills_push(message="Attach NVSkills validation signatures\n\nSigned-off-by: Bot"),
            {},
            True,
            id="multiline-commit",
        ),
        pytest.param(
            _nvskills_push("NV-SKILLS-CI[BOT]", "attach nvskills validation signatures"),
            {},
            True,
            id="case-insensitive-push",
        ),
        pytest.param(
            _nvskills_push(message="Attach NVSkills validation signatures for PR 936"), {}, True, id="title-prefix"
        ),
        pytest.param(_nvskills_push(actor="contributor"), {}, False, id="wrong-push-actor"),
        pytest.param(_nvskills_push(message="Update skills"), {}, False, id="wrong-commit-title"),
        pytest.param(_nvskills_push(actor=""), {}, False, id="empty-push-actor"),
        pytest.param({"event_name": "push", "actor": "nv-skills-ci[bot]"}, {}, False, id="missing-head-commit"),
        pytest.param(_nvskills_push(message=None), {}, False, id="null-commit-message"),
        pytest.param(
            _nvskills_push("custom[bot]", "Sign skills for PR 936"),
            {"NVSKILLS_SIGNATURE_PUSH_ACTOR": "custom[bot]", "NVSKILLS_SIGNATURE_COMMIT_TITLE": "Sign skills"},
            True,
            id="configured-actor-and-title",
        ),
        pytest.param(
            _nvskills_push(),
            {"NVSKILLS_SIGNATURE_PUSH_ACTOR": "custom[bot]"},
            False,
            id="override-rejects-default-actor",
        ),
        pytest.param(
            _nvskills_push(),
            {"NVSKILLS_SIGNATURE_COMMIT_TITLE": "Sign skills"},
            False,
            id="override-rejects-default-title",
        ),
        pytest.param(
            _nvskills_push(),
            {"NVSKILLS_SIGNATURE_PUSH_ACTOR": "", "NVSKILLS_SIGNATURE_COMMIT_TITLE": ""},
            True,
            id="empty-overrides-use-defaults",
        ),
        pytest.param({"event_name": "workflow_dispatch"}, {}, False, id="manual-event"),
        pytest.param({"event_name": "schedule"}, {}, False, id="scheduled-event"),
    ],
)
def test_nvskills_request_gate(github, variables, expected):
    expression = _workflow("request-nvskills-ci.yml")["jobs"]["request"]["if"]
    contexts = {"github": github, "vars": variables}

    def substitute(match):
        token = match[0]
        if token.startswith("'"):
            value = token[1:-1].replace("''", "'")
        else:
            value = contexts
            for part in token.split("."):
                value = value.get(part, "") if isinstance(value, dict) else ""
        # GitHub treats absent properties as empty strings, compares strings
        # without case, and casts null to an empty string in startsWith.
        return repr(value.casefold() if isinstance(value, str) else "" if value is None else value)

    # Evaluate the actual caller expression, using only the operators/functions
    # it needs. actionlint independently validates the GitHub expression syntax.
    expression = " ".join(expression.splitlines()).replace("&&", " and ").replace("||", " or ")
    expression = re.sub(r"'(?:[^']|'')*'|(?:github|vars)(?:\.[\w_]+)+", substitute, expression)
    assert bool(eval(expression, {"__builtins__": {}}, {"startsWith": str.startswith})) is expected


def _run_commands(job: dict) -> str:
    return "\n".join(step.get("run", "") for step in job["steps"])


def _run_resolve_step(script: str, event_name: str, old_ref: str, tmp_path: Path) -> dict[str, str]:
    output_path = tmp_path / f"{event_name}.out"
    output_path.unlink(missing_ok=True)
    subprocess.run(
        ["bash", "-c", script],
        cwd=REPOSITORY_ROOT,
        env={
            **os.environ,
            "GITHUB_EVENT_NAME": event_name,
            "GITHUB_OUTPUT": str(output_path),
            "OLD_REF_INPUT": old_ref,
        },
        text=True,
        capture_output=True,
        check=True,
    )
    return dict(line.split("=", 1) for line in output_path.read_text().splitlines())


def test_platform_wheels_require_the_installed_fpe_exercise() -> None:
    action = yaml.safe_load((ACTION_ROOT / "build-platform-wheel" / "action.yml").read_text())
    verification = next(
        step for step in action["runs"]["steps"] if step.get("name") == "Verify installed unified wheel"
    )
    assert "if" not in verification
    assert "--exercise-engine --exercise-fpe" in verification["run"]


def test_restored_workflows_are_active_at_repository_root() -> None:
    expected = {
        "collector-check.yml",
        "prediction-regression-gate.yml",
        "validate-platform-wheels.yml",
    }

    assert expected.issubset({path.name for path in WORKFLOW_ROOT.glob("*.yml")})
    assert not (WORKFLOW_ROOT / "build-platform-wheels-copied-pr.yml").exists()


def test_runner_canary_checks_both_production_architectures_with_step_home() -> None:
    workflow = _workflow("runner-canary.yml")
    assert set(workflow["on"]) == {"workflow_dispatch", "push"}
    assert workflow["on"]["push"]["branches"] == ["pull-request/*"]
    assert workflow["permissions"] == {"contents": "read"}
    production = workflow["jobs"]["canary"]
    pools = production["strategy"]["matrix"]["include"]
    assert {(pool["runner"], pool["machine"], pool["scratch"]) for pool in pools} == {
        ("prod-aisim-default-amd64-v1", "x86_64", "true"),
        ("prod-aisim-default-arm64-v1", "aarch64", "true"),
    }
    assert production["runs-on"] == "${{ matrix.runner }}"
    # container.env alone is overwritten by the runner's /github/home default.
    assert production["env"]["HOME"] == "/__w/aisim/aisim"
    commands = _run_commands(production)
    assert 'test "${HOME}" = /__w/aisim/aisim' in commands
    assert 'test "$(git rev-parse HEAD)" = "${GITHUB_SHA}"' in commands
    assert 'test "$(uname -m)" = "${EXPECTED_MACHINE}"' in commands
    assert "test -w /scratch" in commands
    shared = workflow["jobs"]["shared"]
    assert shared["runs-on"] == "dynamo-cpu-4gi-amd64"
    assert "container" not in shared  # Shared admission does not permit container pods.


def test_workflow_runner_selectors_match_registered_pools_and_runbook() -> None:
    labels = yaml.safe_load((REPOSITORY_ROOT / ".github/actionlint.yaml").read_text())["self-hosted-runner"]["labels"]
    for name in ("prod-aisim-default-amd64-v1", "prod-aisim-default-arm64-v1", "dynamo-cpu-4gi-amd64"):
        assert name in labels
        assert f"`{name}`" in (REPOSITORY_ROOT / "docs/ci/README.md").read_text()
    for path in WORKFLOW_ROOT.glob("*.yml"):
        assert "prod-aisimulate-default-" not in path.read_text(), path


# Jobs that need nothing from the NVIDIA network stay off the capped self-hosted
# pools. Values map each matrix architecture to its GitHub-hosted label.
GITHUB_HOSTED_JOBS = {
    ("ci.yml", "cargo-deny"): {"amd64": "ubuntu-24.04", "arm64": "ubuntu-24.04-arm"},
    ("ci.yml", "rust"): {"amd64": "ubuntu-24.04", "arm64": "ubuntu-24.04-arm"},
    ("ci.yml", "public-api-rust"): {"amd64": "ubuntu-24.04", "arm64": "ubuntu-24.04-arm"},
    ("ci.yml", "python-compliance"): {"amd64": "ubuntu-24.04"},
    ("ci.yml", "python-compatibility"): {"amd64": "ubuntu-24.04"},
    ("ci.yml", "rust-feature-modes"): {"amd64": "ubuntu-24.04"},
    ("collector-check.yml", "check"): {"amd64": "ubuntu-24.04"},
    ("collector-check.yml", "cross-backend-consistency"): {"amd64": "ubuntu-24.04"},
}


@pytest.mark.parametrize(("workflow_name", "job_name"), sorted(GITHUB_HOSTED_JOBS))
def test_public_network_jobs_run_on_github_hosted_runners(workflow_name: str, job_name: str) -> None:
    job = _workflow(workflow_name)["jobs"][job_name]
    expected = GITHUB_HOSTED_JOBS[(workflow_name, job_name)]
    if job["runs-on"] == "${{ matrix.runner }}":
        include = job["strategy"]["matrix"]["include"]
        assert "arch" not in job["strategy"]["matrix"]
        runners = {entry["arch"]: entry["runner"] for entry in include}
        assert len(runners) == len(include)
        assert "${{ matrix.arch }}" in job["name"]
    else:
        assert "strategy" not in job or "arch" not in job["strategy"]["matrix"]
        runners = {"amd64": job["runs-on"]}
    assert runners == expected
    assert int(job["timeout-minutes"]) > 0
    # Hosted runners cannot resolve Artifactory and get no internal job image.
    assert "container" not in job
    assert "environment" not in job
    serialized = json.dumps(job)
    for internal in ("wheel-storage", "WHEEL_STORAGE", "ARTIFACTORY", "secrets.", "vars.", "/__w/"):
        assert internal not in serialized, internal
    if job_name == "cargo-deny":
        targets = {entry["arch"]: entry["cargo_deny_target"] for entry in job["strategy"]["matrix"]["include"]}
        assert targets == {"amd64": "x86_64-unknown-linux-musl", "arm64": "aarch64-unknown-linux-musl"}


def test_full_ci_owns_migrated_expensive_suites() -> None:
    jobs = _workflow("ci.yml")["jobs"]

    assert {
        "rust-feature-modes",
        "python-compatibility",
        "engine-golden-regression",
        "platform-wheels",
        "collector-data",
        "prediction-regression",
        "readiness",
    }.issubset(jobs)

    assert jobs["platform-wheels"]["uses"] == "./.github/workflows/validate-platform-wheels.yml"
    assert jobs["collector-data"]["uses"] == "./.github/workflows/collector-check.yml"
    assert jobs["prediction-regression"]["uses"] == "./.github/workflows/prediction-regression-gate.yml"

    application_test_wheel = jobs["application-test-wheel"]
    assert application_test_wheel["timeout-minutes"] == "${{ matrix.build_timeout_minutes }}"
    assert {
        entry["arch"]: entry["build_timeout_minutes"]
        for entry in application_test_wheel["strategy"]["matrix"]["include"]
    } == {"amd64": "15", "arm64": "15"}
    assert {"fast-ci", "select-full-ci"}.issubset(application_test_wheel["needs"])
    assert "application-test-wheel" in jobs["application-tests"]["needs"]
    assert {shard["suite"] for shard in jobs["application-tests"]["strategy"]["matrix"]["shard"]} == {
        "contracts",
        "unit",
        "integration-build",
        "cli-build",
    }
    application_commands = _run_commands(jobs["application-tests"])
    compatibility_commands = _run_commands(jobs["python-compatibility"])
    assert "python/aisim/tests/cross_package" in application_commands
    assert "python/aisim/tests/cross_package" in compatibility_commands
    assert "tests/unit tests/golden" in application_commands
    integration_build_steps = [
        step for step in jobs["application-tests"]["steps"] if "'integration-build'" in step.get("if", "")
    ]
    # The merged leg keeps each suite's own selection, workers and timeout.
    assert [shlex.split(step["run"])[3:] for step in integration_build_steps] == [
        ["-q", "-ra", "tests/integration", "-n", "2", "--timeout=600"],
        ["-q", "tests/e2e/support_matrix", "-m", "build", "-n", "auto", "--timeout=180"],
        ["-q", "tests/e2e/tools", "-m", "build", "-n", "auto", "--timeout=600"],
    ]
    assert all(step["working-directory"] == "python/aisim" for step in integration_build_steps)
    assert "scripts/ci/check_application_test_inventory.py" in application_commands
    assert "tests/e2e/cli" in application_commands
    assert "tests/e2e/support_matrix" in application_commands
    assert "tests/e2e/tools" in application_commands
    assert "test_core_public_api.py" not in application_commands
    assert "test_core_public_api.py" not in compatibility_commands

    contract_steps = [
        step for step in jobs["application-tests"]["steps"] if step.get("if") == "matrix.shard.suite == 'contracts'"
    ]
    assert len(contract_steps) == 1
    assert 'echo "AISIM_TEST_WHEEL=${GITHUB_WORKSPACE}/${wheels[0]}" >> "${GITHUB_ENV}"' in application_commands
    contract_command = contract_steps[0]["run"]
    assert "--group scripts/pyproject.toml:e2e-accuracy" in contract_command
    assert "--ignore=tests/fpm_accuracy" not in contract_command
    assert "--ignore=tests/test_ci_workflow_contracts.py" in contract_command

    recommendation_path = "tests/e2e/cli/test_cli_recommend.py"
    recommendation_steps = [
        step for step in jobs["application-tests"]["steps"] if recommendation_path in step.get("run", "")
    ]
    assert len(recommendation_steps) == 2
    assert "-n auto" not in recommendation_steps[0]["run"]
    assert f"--ignore={recommendation_path}" in recommendation_steps[1]["run"]

    build_marker_files = {
        path.relative_to(REPOSITORY_ROOT / "python" / "aisim").as_posix()
        for path in (REPOSITORY_ROOT / "python" / "aisim" / "tests").rglob("test_*.py")
        if "pytest.mark.build" in path.read_text(encoding="utf-8")
    }
    assert build_marker_files
    assert all(
        path.startswith(("tests/e2e/cli/", "tests/e2e/support_matrix/", "tests/e2e/tools/"))
        for path in build_marker_files
    )

    regression = jobs["engine-golden-regression"]
    regression_commands = _run_commands(regression)
    assert regression["name"] == "Engine Golden Regression"
    assert "test_engine_step_parity.py" in regression_commands
    assert "test_compile_engine_parity.py" in regression_commands
    assert regression_commands.count("-c python/aisim/pytest.ini") == 2
    assert "check_prediction_numerics.py --output" in regression_commands
    # The sentinel values need no extra commit: no job fetches one.
    for workflow in ("ci.yml", "fast-ci.yml"):
        assert "--fetch-baseline-only" not in (WORKFLOW_ROOT / workflow).read_text()

    feature_mode_commands = _run_commands(jobs["rust-feature-modes"])
    assert "cargo test --workspace --features embed-python,replay-bench" in feature_mode_commands
    assert "--all-features" not in feature_mode_commands
    assert "--no-default-features" not in feature_mode_commands
    assert "PYTHONPATH" not in feature_mode_commands

    required_by_aggregate = set(jobs["readiness"]["needs"])
    assert {
        "rust-feature-modes",
        "python-compatibility",
        "engine-golden-regression",
        "platform-wheels",
        "collector-data",
        "prediction-regression",
        "application-test-wheel",
    }.issubset(required_by_aggregate)
    aggregate = jobs["readiness"]
    assert aggregate["steps"][0]["env"]["NEEDS_JSON"] == "${{ toJSON(needs) }}"

    assert set(jobs["stage-application-wheel"]["needs"]) == {"readiness", "application-wheel"}
    application_wheel_commands = _run_commands(jobs["application-wheel"])
    assert "maturin build" not in application_wheel_commands
    assert any(
        step.get("uses", "").startswith("dtolnay/rust-toolchain@") for step in jobs["application-wheel"]["steps"]
    )
    assert any(
        step.get("with", {}).get("name") == "application-test-wheel-${{ matrix.arch }}"
        for step in jobs["application-wheel"]["steps"]
    )


def test_readme_checks_remain_in_full_ci_without_standalone_daily_jobs() -> None:
    jobs = _workflow("ci.yml")["jobs"]
    assert "readme_commands" in COMPONENTS
    assert jobs["readme-commands"]["uses"] == "./.github/workflows/readme-checks.yml"
    assert "readme-commands" in jobs["readiness"]["needs"]
    assert set(_workflow("readme-checks.yml")["on"]) == {"workflow_call"}
    for name in ("readme-daily.yml", "readme-report.yml"):
        assert not (WORKFLOW_ROOT / name).exists()
        assert all(name not in path.read_text() for path in WORKFLOW_ROOT.glob("*.yml"))


def _rc_producer():
    artifact = {
        "name": "bundle-manifest",
        "expired": False,
        "workflow_run": {
            "id": 10,
            "repository_id": 123,
            "head_repository_id": 123,
            "head_sha": "a" * 40,
        },
    }
    run = {
        "id": 10,
        "status": "completed",
        "conclusion": "success",
        "path": ".github/workflows/ci.yml",
        "event": "push",
        "head_branch": "main",
        "head_sha": "a" * 40,
        "repository": {"id": 123},
        "head_repository": {"id": 123},
    }
    env = {
        "GITHUB_REPOSITORY": "ai-dynamo/aisim",
        "GITHUB_REPOSITORY_ID": "123",
        "GITHUB_RUN_ID": "20",
        "GITHUB_REF_NAME": "pull-request/42",
        "GITHUB_SHA": "b" * 40,
        "RC_PR_NUMBER": "42",
    }
    return artifact, run, env


def test_rc_wheels_reuse_only_successful_main_or_exact_current_pr():
    artifact, run, env = _rc_producer()

    def request(endpoint):
        return {"artifacts": [artifact]} if "/artifacts?" in endpoint else run

    assert rc_wheels.find_producer("bundle", env, request) == {
        "run-id": "10",
        "scope": "preview",
        "pr-number": "",
    }
    run["head_branch"] = env["GITHUB_REF_NAME"]
    run["head_sha"] = artifact["workflow_run"]["head_sha"] = env["GITHUB_SHA"]
    assert rc_wheels.find_producer("bundle", env, request) == {
        "run-id": "10",
        "scope": "pr",
        "pr-number": "42",
    }
    run["head_sha"] = artifact["workflow_run"]["head_sha"] = "c" * 40
    assert rc_wheels.find_producer("bundle", env, request)["run-id"] == ""


@pytest.mark.parametrize(
    ("target", "key", "value"),
    [
        ("artifact", "expired", True),
        ("artifact", "name", "other-manifest"),
        ("producer", "head_repository_id", 999),
        ("producer", "repository_id", 999),
        ("producer", "id", 20),
        ("run", "status", "in_progress"),
        ("run", "conclusion", "failure"),
        ("run", "path", ".github/workflows/untrusted.yml"),
        ("run", "event", "pull_request"),
        ("run", "head_branch", "pull-request/99"),
        ("run", "head_sha", "d" * 40),
        ("run", "repository", {"id": 999}),
        ("run", "head_repository", {"id": 999}),
    ],
)
def test_rc_wheels_reject_untrusted_incomplete_or_superseded_producers(target, key, value):
    artifact, run, env = _rc_producer()
    {"artifact": artifact, "producer": artifact["workflow_run"], "run": run}[target][key] = value

    def request(endpoint):
        return {"artifacts": [artifact]} if "/artifacts?" in endpoint else run

    assert rc_wheels.find_producer("bundle", env, request)["run-id"] == ""


def test_rc_wheels_discovery_paginates_past_expired_manifests():
    artifact, run, env = _rc_producer()
    calls = []

    def request(endpoint):
        calls.append(endpoint)
        if parse_qs(urlsplit(endpoint).query).get("page") == ["1"]:
            return {"artifacts": [{**artifact, "expired": True}] * 100}
        return {"artifacts": [artifact]} if "/artifacts?" in endpoint else run

    assert rc_wheels.find_producer("bundle", env, request)["run-id"] == "10"
    assert any("page=2" in call for call in calls)


RC_OS_RELEASE = (
    'PRETTY_NAME="Ubuntu 24.04.3 LTS"\nID=ubuntu\nVERSION_ID="24.04"\nVERSION="24.04.3 LTS (Noble Numbat)"\n'
)


def _rc_dpkg_database():
    """A trimmed noble runner database: package -> (version, Pre-Depends/Depends, Provides, status)."""
    return {
        "build-essential": ("12.10ubuntu1", "libc6-dev | libc-dev, gcc (>= 4:12.3), g++, make, dpkg-dev", ""),
        "gcc": ("4:13.2.0-7ubuntu1", "cpp (= 4:13.2.0-7ubuntu1), gcc-13 (>= 13.2.0-11~)", "c-compiler"),
        "g++": ("4:13.2.0-7ubuntu1", "gcc (= 4:13.2.0-7ubuntu1), g++-13 (>= 13.2.0-11~)", "c++-compiler"),
        "cpp": ("4:13.2.0-7ubuntu1", "cpp-13 (>= 13.2.0-11~)", ""),
        "cpp-13": ("13.3.0-6ubuntu2~24.04.1", "gcc-13-base (= 13.3.0-6ubuntu2~24.04.1)", ""),
        "gcc-13": ("13.3.0-6ubuntu2~24.04.1", "cpp-13 (= 13.3.0-6ubuntu2~24.04.1), binutils (>= 2.42), libc6", ""),
        "gcc-13-base": ("13.3.0-6ubuntu2~24.04.1", "", ""),
        "g++-13": ("13.3.0-6ubuntu2~24.04.1", "gcc-13 (= 13.3.0-6ubuntu2~24.04.1), libstdc++-13-dev", ""),
        "libstdc++-13-dev": ("13.3.0-6ubuntu2~24.04.1", "libc6-dev (>= 2.13-0ubuntu6), libstdc++6", ""),
        "libstdc++6": ("14.2.0-4ubuntu2~24.04.1", "libgcc-s1, libc6 (>= 2.36)", ""),
        "libgcc-s1": ("14.2.0-4ubuntu2~24.04.1", "libc6 (>= 2.35)", ""),
        "binutils": ("2.42-4ubuntu2.10", "libc6 (>= 2.38)", ""),
        "make": ("4.3-4.1build2", "libc6 (>= 2.34)", ""),
        "dpkg-dev": ("1.22.6ubuntu6.6", "perl:any, make, binutils", ""),
        "libc6": ("2.39-0ubuntu8.12", "libgcc-s1", ""),
        "libc6-dev": ("2.39-0ubuntu8.12", "libc6 (= 2.39-0ubuntu8.12), libc-dev-bin, linux-libc-dev", "libc-dev"),
        "libc-dev-bin": ("2.39-0ubuntu8.11", "libc6", "", "rc "),
        "linux-libc-dev": ("6.8.0-149.149", "", ""),
        "cmake": ("3.28.3-1build7", "cmake-data (= 3.28.3-1build7), libc6, libstdc++6", ""),
        "cmake-data": ("3.28.3-1build7", "", ""),
        "libclang-dev": ("1:18.0-59~exp2", "libclang-18-dev", ""),
        "libclang-18-dev": ("1:18.1.3-1ubuntu1", "libclang1-18 (= 1:18.1.3-1ubuntu1)", ""),
        "libclang1-18": ("1:18.1.3-1ubuntu1", "libc6, libstdc++6", ""),
        "libhwloc-dev": ("2.10.0-1build1", "libhwloc15 (= 2.10.0-1build1)", ""),
        "libhwloc15": ("2.10.0-1build1", "libc6, libudev1 (>= 183)", ""),
        "libudev-dev": ("255.4-1ubuntu8.18", "libudev1 (= 255.4-1ubuntu8.18)", ""),
        "libudev1": ("255.4-1ubuntu8.18", "libc6", ""),
        "pkg-config": ("1.8.1-2build1", "pkgconf", ""),
        "pkgconf": ("1.8.1-2build1", "libc6", "pkg-config"),
        "protobuf-compiler": ("3.21.12-8.2ubuntu0.4", "libprotoc32t64 (= 3.21.12-8.2ubuntu0.4)", ""),
        "libprotoc32t64": ("3.21.12-8.2ubuntu0.4", "libc6, libstdc++6", ""),
        # Installed in the job but unreachable from the build prerequisites.
        "gh": ("2.45.0-1ubuntu0.3", "libc6 (>= 2.34)", ""),
        "systemd": ("255.4-1ubuntu8.18", "libudev1, libc6", ""),
        "tzdata": ("2025b-0ubuntu0.24.04.1", "debconf", ""),
    }


def _fake_rc_runner(tmp_path, monkeypatch, packages):
    for key in ("AISIM_RC_REVISION", "DYNAMO_RC_REVISION"):
        monkeypatch.setenv(key, "a" * 40)
    monkeypatch.setenv("RC_JOB_IMAGE", "runner:2.336.0")
    monkeypatch.setenv("RC_STORAGE_REPO", "sw-dynamo-aisimulate-pr-pypi-local")
    monkeypatch.setenv("RC_STORAGE_URL", "https://artifactory.invalid/artifactory")
    os_release = tmp_path / "os-release"
    os_release.write_text(RC_OS_RELEASE)
    monkeypatch.setattr(rc_wheels, "OS_RELEASE", os_release)
    monkeypatch.setattr(rc_wheels.platform, "system", lambda: "Linux")
    monkeypatch.setattr(rc_wheels.platform, "machine", lambda: "x86_64")

    def check_output(command, text=False):
        if command[0] != "dpkg-query":
            return "pinned tool version"
        if command[2] == f"--showformat={rc_wheels.INVENTORY_FORMAT}":
            return "".join(f"{name}={entry[0]}\n" for name, entry in sorted(packages.items())).encode()
        assert command[2] == f"--showformat={rc_wheels.DEPENDENCY_FORMAT}" and text
        return "".join(
            f"{(entry + ('ii ',))[3]}\t{name}\t{entry[0]}\t{entry[2]}\t, {entry[1]}\n"
            for name, entry in sorted(packages.items())
        )

    monkeypatch.setattr(rc_wheels.subprocess, "check_output", check_output)
    for relative in rc_wheels.INPUTS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("original")
    return os_release


def _rc_key(tmp_path):
    return rc_wheels.handoff_name(rc_wheels.identity(tmp_path))


@pytest.mark.parametrize("changed_input", rc_wheels.INPUTS)
def test_rc_wheels_build_recipe_changes_invalidate_reuse(tmp_path, monkeypatch, changed_input):
    os_release = _fake_rc_runner(tmp_path, monkeypatch, _rc_dpkg_database())
    original = _rc_key(tmp_path)
    (tmp_path / changed_input).write_text("changed")
    assert _rc_key(tmp_path) != original
    (tmp_path / changed_input).write_text("original")
    monkeypatch.setenv("RC_JOB_IMAGE", "runner:new-version")
    assert _rc_key(tmp_path) != original
    monkeypatch.setenv("RC_JOB_IMAGE", "runner:2.336.0")
    os_release.write_text(RC_OS_RELEASE.replace('VERSION_ID="24.04"', 'VERSION_ID="26.04"'))
    assert _rc_key(tmp_path) != original
    os_release.write_text(RC_OS_RELEASE)
    monkeypatch.setenv("RC_STORAGE_REPO", "sw-dynamo-aisimulate-pypi-local")
    assert _rc_key(tmp_path) != original


@pytest.mark.parametrize(
    "changes",
    [
        # Packages the build cannot reach, and a removed package's leftover record.
        {"gh": "2.83.0-1"},
        {"systemd": "255.4-1ubuntu8.19"},
        {"tzdata": "2026a-0ubuntu0.24.04.1"},
        {"libc-dev-bin": "2.40-1ubuntu1"},
    ],
    ids=["gh", "systemd", "tzdata", "removed"],
)
def test_rc_wheels_key_ignores_unrelated_packages(tmp_path, monkeypatch, changes):
    packages = _rc_dpkg_database()
    os_release = _fake_rc_runner(tmp_path, monkeypatch, packages)
    original, evidence = _rc_key(tmp_path), rc_wheels.environment()
    for name, version in changes.items():
        packages[name] = (version, *packages[name][1:])
    assert _rc_key(tmp_path) == original
    # The full inventory still records the drift, outside the key.
    changed = rc_wheels.environment()
    assert changed["packages_sha256"] != evidence["packages_sha256"]
    assert {f"{name}={version}" for name, version in changes.items()} <= set(changed["packages"])
    os_release.write_text(RC_OS_RELEASE.replace("24.04.3", "24.04.4"))
    assert _rc_key(tmp_path) == original
    assert rc_wheels.environment()["os_release_sha256"] != evidence["os_release_sha256"]


def test_rc_wheels_kernel_header_revision_pair_invalidates_cached_build(tmp_path, monkeypatch):
    packages = _rc_dpkg_database()
    _fake_rc_runner(tmp_path, monkeypatch, packages)
    original = rc_wheels.identity(tmp_path)
    assert packages["linux-libc-dev"][0] == "6.8.0-149.149"
    (tmp_path / "rc-build-inputs.json").write_text(json.dumps(original))

    # This Ubuntu revision pair changes public headers despite the same upstream version.
    packages["linux-libc-dev"] = ("6.8.0-151.151", *packages["linux-libc-dev"][1:])
    changed = rc_wheels.identity(tmp_path)
    assert rc_wheels.handoff_name(changed) != rc_wheels.handoff_name(original)
    assert original["image"]["build_packages"]["linux-libc-dev"] == "6.8.0-149.149"
    assert changed["image"]["build_packages"]["linux-libc-dev"] == "6.8.0-151.151"
    with pytest.raises(ValueError, match="build inputs"):
        rc_wheels.verify(tmp_path, changed)


@pytest.mark.parametrize(
    ("package", "version"),
    [
        ("libudev-dev", "255.4-1ubuntu8.19"),
        ("libudev1", "255.4-1ubuntu8.19"),
        ("libc6", "2.39-0ubuntu8.13"),
        ("libc6-dev", "2.39-0ubuntu8.13"),
        ("gcc-13", "13.3.0-6ubuntu2~24.04.2"),
        ("libstdc++-13-dev", "13.3.0-6ubuntu2~24.04.2"),
        ("binutils", "2.42-4ubuntu2.11"),
        ("dpkg-dev", "1.22.6ubuntu6.7"),
        ("build-essential", "12.10ubuntu1.1"),
        ("libclang-18-dev", "1:18.1.3-1ubuntu2"),
        ("libprotoc32t64", "3.21.12-8.2ubuntu0.5"),
        ("cmake", "3.28.3-1build8"),
        ("pkgconf", "1.8.1-2build2"),
        ("libhwloc15", "2.10.0-1build2"),
    ],
)
def test_rc_wheels_native_package_revision_changes_invalidate_reuse(tmp_path, monkeypatch, package, version):
    packages = _rc_dpkg_database()
    _fake_rc_runner(tmp_path, monkeypatch, packages)
    original = _rc_key(tmp_path)
    packages[package] = (version, *packages[package][1:])
    assert _rc_key(tmp_path) != original


@pytest.mark.parametrize(
    ("package", "version"),
    [
        ("gcc-13", "13.4.0-1ubuntu1~24.04"),  # C compiler, behind the gcc meta package
        ("libstdc++-13-dev", "14.2.0-4ubuntu2~24.04"),  # C++ headers
        ("binutils", "2.43-1ubuntu1"),  # linker
        ("libc6", "2.40-1ubuntu1"),  # glibc
        ("linux-libc-dev", "6.11.0-8.8"),  # kernel headers
        ("libclang-18-dev", "1:18.1.8-1ubuntu1"),  # bindgen, behind the libclang-dev meta package
        ("libprotoc32t64", "3.25.3-1ubuntu1"),  # protoc
        ("cmake", "3.30.3-1"),
        ("pkgconf", "2.1.0-1"),  # provider behind pkg-config
        ("libhwloc15", "2.11.0-1"),
        ("libudev-dev", "256.1-1ubuntu1"),
    ],
)
def test_rc_wheels_toolchain_and_library_changes_invalidate_reuse(tmp_path, monkeypatch, package, version):
    packages = _rc_dpkg_database()
    _fake_rc_runner(tmp_path, monkeypatch, packages)
    original = _rc_key(tmp_path)
    packages[package] = (version, *packages[package][1:])
    assert _rc_key(tmp_path) != original


def test_rc_wheels_key_follows_new_build_dependencies(tmp_path, monkeypatch):
    packages = _rc_dpkg_database()
    _fake_rc_runner(tmp_path, monkeypatch, packages)
    original = _rc_key(tmp_path)
    assert "tzdata" not in rc_wheels.build_packages()
    packages["libhwloc15"] = (packages["libhwloc15"][0], "libc6, libudev1 (>= 183), tzdata", "")
    assert "tzdata" in rc_wheels.build_packages()
    assert _rc_key(tmp_path) != original


@pytest.mark.parametrize("removed", [True, False], ids=["removed", "never-installed"])
def test_rc_wheels_require_every_build_prerequisite(tmp_path, monkeypatch, removed):
    packages = _rc_dpkg_database()
    _fake_rc_runner(tmp_path, monkeypatch, packages)
    if removed:
        packages["libudev-dev"] = (*packages["libudev-dev"][:3], "rc ")
    else:
        del packages["libudev-dev"]
    with pytest.raises(ValueError, match="libudev-dev"):
        rc_wheels.identity(tmp_path)


def test_rc_wheels_key_covers_every_native_prerequisite_except_gh():
    job = _workflow("readme-checks.yml")["jobs"]["dynamo"]
    install = re.search(r"apt-get install -y ([^\n]+)", _run_commands(job))
    assert install is not None
    installed = set(install[1].split())
    assert set(rc_wheels.BUILD_PACKAGES) <= installed
    assert installed - set(rc_wheels.BUILD_PACKAGES) == {"gh"}


def test_main_rc_producers_share_test_storage_with_pr_consumers():
    workflow = _workflow("readme-checks.yml")
    job = workflow["jobs"]["dynamo"]
    assert job["environment"] == "pr-wheel-staging"
    prepare = next(step for step in job["steps"] if step.get("id") == "rc-wheels")
    assert prepare["env"]["RC_STORAGE_REPO"] == "${{ vars.WHEEL_STORAGE_REPO }}"
    upload = next(
        step
        for step in job["steps"]
        if step.get("uses", "").endswith("/wheel-storage") and step["with"]["mode"] == "upload"
    )
    assert upload["with"]["scope"] == "${{ inputs.pr_number != '' && 'pr' || 'preview' }}"
    assert job["runs-on"] == "prod-aisim-default-amd64-v1"
    assert job["container"] == "${{ vars.CI_JOB_CONTAINER_IMAGE }}"
    commands = _run_commands(job)
    assert commands.index("socket.getaddrinfo") < commands.index("readme_rc_wheels.py prepare")
    assert commands.index("apt-get install") < commands.index("readme_rc_wheels.py prepare")


@pytest.mark.parametrize(
    "producer",
    [
        {"run-id": "", "scope": "", "pr-number": ""},
        {"run-id": "10", "scope": "preview", "pr-number": ""},
        {"run-id": "11", "scope": "pr", "pr-number": "42"},
    ],
    ids=["cold", "trusted-main", "exact-pr"],
)
def test_rc_prepare_cli_outputs_reach_workflow_restore_inputs(tmp_path, monkeypatch, producer):
    expected = {"verified": "build identity"}
    inventory = {"packages_sha256": "full runner inventory"}
    monkeypatch.setattr(rc_wheels, "identity", lambda: expected)
    monkeypatch.setattr(rc_wheels, "environment", lambda: inventory)
    monkeypatch.setattr(rc_wheels, "find_producer", lambda name, env: producer)
    output = tmp_path / "github-output"
    directory = tmp_path / "wheels"
    evidence = tmp_path / "rc-build-environment.json"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.setattr(sys, "argv", ["readme_rc_wheels.py", "prepare", "--directory", str(directory)])
    with pytest.raises(SystemExit):
        rc_wheels.main()
    assert not output.exists()
    monkeypatch.setattr(sys, "argv", [*sys.argv, "--evidence", str(evidence)])
    rc_wheels.main()
    values = dict(line.split("=", 1) for line in output.read_text().splitlines())
    assert values == {"name": rc_wheels.handoff_name(expected), **producer}
    assert json.loads((directory / "rc-build-inputs.json").read_text()) == expected
    # The inventory stays out of the wheelhouse, which is the Artifactory handoff.
    assert json.loads(evidence.read_text()) == inventory
    assert [path.name for path in directory.iterdir()] == ["rc-build-inputs.json"]

    job = _workflow("readme-checks.yml")["jobs"]["dynamo"]
    prepare = next(step for step in job["steps"] if step.get("id") == "rc-wheels")
    assert 'prepare --directory "$wheelhouse" --evidence "$evidence"' in prepare["run"]
    assert 'evidence="${RUNNER_TEMP}/rc-build-environment.json"' in prepare["run"]
    assert 'echo "AISIM_RC_EVIDENCE=${evidence}" >> "${GITHUB_ENV}"' in prepare["run"]
    preserve = next(step for step in job["steps"] if step.get("name") == "Preserve RC build-input and runner evidence")
    assert "${{ env.AISIM_RC_EVIDENCE }}" in preserve["with"]["path"].splitlines()
    assert job["env"]["RC_PR_NUMBER"] == "${{ inputs.pr_number }}"
    restore = next(step for step in job["steps"] if step.get("name") == "Restore verified pinned Dynamo RC wheels")
    inputs = {}
    for name in ("name", "run-id", "scope", "pr-number"):
        binding = re.fullmatch(r"\$\{\{ steps\.rc-wheels\.outputs\.([^ ]+) \}\}", restore["with"][name])
        assert binding is not None, f"restore input {name} must consume the RC prepare outputs"
        inputs[name] = values[binding[1]]
    assert inputs == {"name": rc_wheels.handoff_name(expected), **producer}


def test_rc_wheels_require_all_three_packages_and_matching_build_inputs(tmp_path):
    expected = {"pinned": "build inputs"}
    (tmp_path / "rc-build-inputs.json").write_text(json.dumps(expected))
    for name in ("aisim", "ai_dynamo", "ai_dynamo_runtime"):
        with zipfile.ZipFile(tmp_path / f"{name}.whl", "w") as archive:
            archive.writestr(f"{name}.dist-info/METADATA", f"Name: {name}\nVersion: 1.0\n")
    rc_wheels.verify(tmp_path, expected)
    with pytest.raises(ValueError, match="build inputs"):
        rc_wheels.verify(tmp_path, {"pinned": "different"})
    (tmp_path / "ai_dynamo.whl").unlink()
    with pytest.raises(ValueError, match="exactly the matching"):
        rc_wheels.verify(tmp_path, expected)


def test_pinned_rc_build_executes_all_three_locked_feature_builds(tmp_path):
    job = _workflow("readme-checks.yml")["jobs"]["dynamo"]
    log = tmp_path / "commands.jsonl"
    executables = tmp_path / "bin"
    executables.mkdir()
    for name in ("git", "uv"):
        executable = executables / name
        executable.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            "with open(os.environ['COMMAND_LOG'], 'a') as output:\n"
            "    output.write(json.dumps([os.path.basename(sys.argv[0]), *sys.argv[1:]]) + '\\n')\n"
        )
        executable.chmod(0o755)
    wheelhouse = tmp_path / "wheelhouse"
    subprocess.run(
        ["bash", str(REPOSITORY_ROOT / "scripts/ci/build_readme_rc_wheels.sh")],
        check=True,
        env={
            **os.environ,
            "PATH": str(executables) + os.pathsep + os.environ["PATH"],
            "RUNNER_TEMP": str(tmp_path),
            "COMMAND_LOG": str(log),
            "AISIM_RC_WHEELHOUSE": str(wheelhouse),
            "AISIM_RC_REVISION": job["env"]["AISIM_RC_REVISION"],
            "DYNAMO_RC_REVISION": job["env"]["DYNAMO_RC_REVISION"],
        },
    )
    commands = [json.loads(line) for line in log.read_text().splitlines()]
    expected = []
    for name, revision in (
        ("aisim", "77d82652b2f83fd67afefc3141d69a5f06e630be"),
        ("dynamo", "9653d05441b37b6a004837651fd991e44158d755"),
    ):
        checkout = str(tmp_path / f"{name}-rc")
        expected.extend(
            [
                ["git", "init", checkout],
                ["git", "-C", checkout, "fetch", "--depth", "1", f"https://github.com/ai-dynamo/{name}.git", revision],
                ["git", "-C", checkout, "checkout", "--detach", "FETCH_HEAD"],
            ]
        )
    build = ["uv", "build", "--wheel", "--out-dir", str(wheelhouse)]
    expected.extend(
        [
            [*build, "--config-setting=build-args=--locked", str(tmp_path / "aisim-rc/python/aisim")],
            [*build, str(tmp_path / "dynamo-rc")],
            [
                *build,
                "--config-setting=build-args=--locked --features ais-forward-pass",
                str(tmp_path / "dynamo-rc/lib/bindings/python"),
            ],
        ]
    )
    assert commands == expected


@pytest.mark.parametrize("filename,exists", [("missing.whl", False), ("not-a-wheel.txt", True)])
def test_release_build_rejects_invalid_prebuilt_wheel_before_building(tmp_path, monkeypatch, filename, exists):
    wheel = tmp_path / filename
    if exists:
        wheel.write_bytes(b"not a wheel")
    calls = []
    monkeypatch.setattr(release_artifacts, "_run", lambda *args, **kwargs: calls.append(args))
    output = tmp_path / "dist"
    with pytest.raises(ValueError, match="existing .whl file"):
        release_artifacts.build(output, release_artifacts.VERSION, release_artifacts.VERSION, wheel)
    assert calls == []
    assert list(output.iterdir()) == []


def test_release_main_forwards_prebuilt_wheel_argument(tmp_path, monkeypatch):
    wheel = tmp_path / "application.whl"
    output = tmp_path / "dist"
    monkeypatch.setattr(
        sys, "argv", ["build_release_artifacts.py", "--output-dir", str(output), "--prebuilt-wheel", str(wheel)]
    )
    monkeypatch.setattr(release_artifacts, "check_manifests", lambda: ("1.2.3", "1.2.3"))
    # main supports historical source roots; restore its derived globals after this invocation.
    for name in ("EXPECTED_PYTHON_PROJECTS", "EXPECTED_CRATE"):
        monkeypatch.setattr(release_artifacts, name, getattr(release_artifacts, name))
    calls = []
    monkeypatch.setattr(release_artifacts, "build", lambda *args: calls.append(args))
    release_artifacts.main()
    assert calls == [(output.resolve(), "1.2.3", "1.2.3", wheel)]


@pytest.mark.parametrize("reuse", [False, True])
def test_release_build_reuses_exact_wheel_but_still_builds_and_verifies_crate(tmp_path, monkeypatch, reuse):
    version = release_artifacts.VERSION
    dist_info = f"aisim-{version}.dist-info"
    wheel = tmp_path / f"aisim-{version}-cp311-abi3-manylinux_2_28_x86_64.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(f"{dist_info}/METADATA", f"Name: aisim\nVersion: {version}\n")
        scripts = "\n".join(f"{key} = {value}" for key, value in release_artifacts._expected_scripts().items())
        archive.writestr(f"{dist_info}/entry_points.txt", "[console_scripts]\n" + scripts)
        for name in release_artifacts.LEGAL_FILES:
            archive.writestr(f"{dist_info}/licenses/{name}", (REPOSITORY_ROOT / name).read_bytes())
    calls = []

    def run(*command, **kwargs):
        calls.append(command)
        if command[:2] == ("cargo", "package"):
            target = Path(command[command.index("--target-dir") + 1]) / "package"
            target.mkdir(parents=True)
            (target / f"aisim-core-{version}.crate").write_bytes(b"crate output")
        else:
            assert "build_manylinux_wheel.py" in command[1]
            shutil.copy2(wheel, Path(command[-1]) / wheel.name)

    monkeypatch.setattr(release_artifacts, "_run", run)
    monkeypatch.setattr(release_artifacts.sys, "platform", "linux")
    output = tmp_path / "dist"
    release_artifacts.build(output, version, version, wheel if reuse else None)
    assert (output / wheel.name).read_bytes() == wheel.read_bytes()
    assert (output / f"aisim-core-{version}.crate").is_file()
    assert len(calls) == (1 if reuse else 2)
    assert calls[-1][:2] == ("cargo", "package")


def test_release_contract_reuses_the_same_source_architecture_wheel():
    jobs = _workflow("ci.yml")["jobs"]
    job = jobs["release-artifact-contract"]
    assert "application-test-wheel" in job["needs"]
    download = next(step for step in job["steps"] if step.get("uses", "").endswith("/wheel-storage"))
    assert download["with"]["source-sha"] == "${{ github.sha }}"
    assert download["with"]["architecture"] == "${{ matrix.arch }}"
    assert download["with"]["name"] == "application-test-wheel-${{ matrix.arch }}"
    assert "--prebuilt-wheel application-test-dist/*.whl" in _run_commands(job)
    cases = yaml.safe_load((REPOSITORY_ROOT / ".github/full-ci-selection-cases.yml").read_text())
    for case in cases["cases"]:
        components = select_components(case["paths"])["components"]
        if components["release_artifact_contract"]:
            assert components["application_test_wheel"], case


def test_full_ci_aggregate_checks_every_declared_dependency() -> None:
    jobs = _workflow("ci.yml")["jobs"]
    aggregate = jobs["readiness"]
    commands = _run_commands(aggregate)

    assert "select-full-ci" in aggregate["needs"]
    assert "stage-application-wheel" not in aggregate["needs"]
    assert set(jobs["stage-application-wheel"]["needs"]) == {"readiness", "application-wheel"}
    assert set(aggregate["needs"]) == set(jobs) - {"readiness", "stage-application-wheel"}
    assert aggregate["steps"][0]["env"]["NEEDS_JSON"] == "${{ toJSON(needs) }}"
    assert aggregate["steps"][0]["env"]["PLAN_JSON"] == ("${{ toJSON(needs.select-full-ci.outputs) }}")
    assert 'selected not in {"true", "false"}' in commands
    assert '"success" if selected == "true" else "skipped"' in commands
    assert 'payload["result"]' in commands
    assert "Full CI did not pass" in commands


def test_fast_ci_owns_static_and_workflow_contract_checks() -> None:
    jobs = _workflow("fast-ci.yml")["jobs"]

    policy_commands = _run_commands(jobs["policy"])
    static_commands = _run_commands(jobs["python-static"])
    assert "tests/test_ci_workflow_contracts.py" in policy_commands
    assert "test_cli_recommend.py" in static_commands
    assert "ruff format --check" in static_commands
    assert "python -m compileall" in static_commands
    assert set(jobs["readiness"]["needs"]) == {
        "policy",
        "python-static",
        "rust-format",
    }
    assert jobs["readiness"]["if"] == "${{ always() }}"


def test_expensive_workflows_are_reusable_full_ci_components() -> None:
    for filename in (
        "collector-check.yml",
        "prediction-regression-gate.yml",
        "validate-platform-wheels.yml",
    ):
        triggers = _workflow(filename)["on"]
        assert "workflow_call" in triggers
        assert "workflow_dispatch" in triggers
        assert "push" not in triggers
        assert "pull_request" not in triggers


def test_full_ci_propagates_the_exact_sha_to_reusable_gates() -> None:
    jobs = _workflow("ci.yml")["jobs"]

    for job_name in ("platform-wheels", "collector-data", "prediction-regression"):
        assert jobs[job_name]["with"]["expected_sha"] == "${{ github.sha }}"

    for filename in (
        "collector-check.yml",
        "prediction-regression-gate.yml",
        "validate-platform-wheels.yml",
    ):
        inputs = _workflow(filename)["on"]["workflow_call"]["inputs"]
        assert "expected_sha" in inputs


def test_full_ci_checkouts_do_not_persist_credentials() -> None:
    jobs = _workflow("ci.yml")["jobs"]

    for job in jobs.values():
        for step in job.get("steps", []):
            if step.get("uses", "").startswith("actions/checkout@"):
                assert step.get("with", {}).get("persist-credentials") == "false"


def test_migrated_workflows_keep_reviewed_safety_fixes() -> None:
    collector = _workflow("collector-check.yml")
    prediction = _workflow("prediction-regression-gate.yml")
    platform_wheels = _workflow("validate-platform-wheels.yml")

    collector_commands = _run_commands(collector["jobs"]["check"])
    assert "--depth=1" not in collector_commands
    assert "github.run_id" in collector["concurrency"]["group"]
    assert "cancel-in-progress" not in collector["concurrency"]

    collect_commands = _run_commands(prediction["jobs"]["collect"])
    report_commands = _run_commands(prediction["jobs"]["report"])
    assert "NO_HARNESS" in collect_commands
    assert "pyyaml==6.0.3" in report_commands
    for workflow in (prediction, platform_wheels):
        concurrency_group = workflow["concurrency"]["group"]
        assert "github.event_name == 'push'" in concurrency_group
        assert "startsWith(github.ref, 'refs/heads/pull-request/')" in concurrency_group
        assert "&& github.ref || github.run_id" in concurrency_group
    fast_group = _workflow("fast-ci.yml")["concurrency"]["group"]
    assert "github.event_name == 'pull_request'" in fast_group
    assert "github.event_name == 'push' && startsWith(github.ref, 'refs/heads/pull-request/')" in fast_group
    assert "&& github.ref || github.run_id" in fast_group


def test_migrated_workflows_pin_actions_and_do_not_persist_checkout_credentials() -> None:
    for filename in (
        "collector-check.yml",
        "prediction-regression-gate.yml",
        "validate-platform-wheels.yml",
    ):
        workflow = _workflow(filename)
        source = (WORKFLOW_ROOT / filename).read_text(encoding="utf-8")
        assert "https://github.com/ai-dynamo/AIConfigurator/tree/" in source

        for uses in re.findall(r"^\s*-?\s*uses:\s+([^\s#]+)", source, re.MULTILINE):
            if uses.startswith("./"):
                continue
            assert re.search(r"@[0-9a-f]{40}$", uses), f"{filename}: {uses} is not pinned"

        for job in workflow["jobs"].values():
            for step in job.get("steps", []):
                if step.get("uses", "").startswith("actions/checkout@"):
                    assert step.get("with", {}).get("persist-credentials") == "false"


def test_exact_target_checks_bind_expressions_through_step_environments() -> None:
    expression = "${{ inputs.expected_sha || github.sha }}"
    for filename in (
        "collector-check.yml",
        "prediction-regression-gate.yml",
        "validate-platform-wheels.yml",
    ):
        workflow = _workflow(filename)
        for job in workflow["jobs"].values():
            for step in job.get("steps", []):
                assert expression not in step.get("run", "")


def test_platform_wheel_build_and_verifiers_cover_collector_payload() -> None:
    dockerfile = (REPOSITORY_ROOT / "python" / "aisim" / "docker" / "Dockerfile").read_text()
    release_verifier = (REPOSITORY_ROOT / "python" / "aisim" / "tools" / "verify_release_wheels.py").read_text()
    installed_verifier = (
        REPOSITORY_ROOT / "python" / "aisim" / "tools" / "verify_installed_package_layers.py"
    ).read_text()

    assert "COPY python/aisim/collector/ /workspace/python/aisim/collector/" in dockerfile
    assert "ln -s ../src" not in dockerfile
    assert "test -d /workspace/python/aisim/src/aisim_core/model_configs" in dockerfile
    assert "test -d /workspace/python/aisim/src/aisim_core/systems" in dockerfile
    assert '"cases/**/*.yaml"' in release_verifier
    assert '"fpm_forward/**/*.py"' in release_verifier
    assert '"collector/fpm_forward/runtime/fpm_exec.sh"' in installed_verifier
    assert 'importlib.import_module("collector.fpm_forward")' in installed_verifier


def test_linux_release_wheels_are_repaired_for_manylinux_2_28() -> None:
    expected_images = {
        "amd64": "quay.io/pypa/manylinux_2_28_x86_64@sha256:",
        "arm64": "quay.io/pypa/manylinux_2_28_aarch64@sha256:",
    }
    full_ci = _workflow("ci.yml")["jobs"]
    for job_name in ("application-test-wheel", "release-artifact-contract"):
        job = full_ci[job_name]
        assert job["container"]["image"] == "${{ matrix.container_image }}"
        images = {entry["arch"]: entry["container_image"] for entry in job["strategy"]["matrix"]["include"]}
        assert images.keys() == expected_images.keys()
        for arch, prefix in expected_images.items():
            assert images[arch].startswith(prefix)
            assert len(images[arch].removeprefix(prefix)) == 64

    application_commands = _run_commands(full_ci["application-test-wheel"])
    assert "scripts/release/build_manylinux_wheel.py" in application_commands
    assert "maturin build" not in application_commands

    fpe_prepare = _workflow("fpe-support-matrix.yml")["jobs"]["prepare-wheel"]
    assert fpe_prepare["container"]["image"].startswith(expected_images["amd64"])
    assert "scripts/release/build_manylinux_wheel.py" in _run_commands(fpe_prepare)

    release_builder = (REPOSITORY_ROOT / "scripts/release/build_release_artifacts.py").read_text()
    assert 'sys.platform.startswith("linux")' in release_builder
    assert '"scripts/release/build_manylinux_wheel.py"' in release_builder

    dockerfile = (REPOSITORY_ROOT / "python" / "aisim" / "docker" / "Dockerfile").read_text()
    assert 'MATURIN_PEP517_ARGS="--auditwheel skip"' in dockerfile
    assert "auditwheel repair" in dockerfile
    assert "auditwheel show /workspace/dist/*.whl" in dockerfile
    assert "--compatibility manylinux_2_28" not in dockerfile


@pytest.mark.parametrize(
    ("machine", "expected"),
    [
        ("x86_64", "manylinux_2_28_x86_64"),
        ("amd64", "manylinux_2_28_x86_64"),
        ("aarch64", "manylinux_2_28_aarch64"),
        ("arm64", "manylinux_2_28_aarch64"),
    ],
)
def test_manylinux_platform_maps_native_architectures(machine: str, expected: str) -> None:
    assert manylinux_platform(machine) == expected


def test_manylinux_platform_rejects_unknown_architecture() -> None:
    with pytest.raises(SystemExit, match="unsupported wheel architecture"):
        manylinux_platform("riscv64")


def _manylinux_build_case(
    tmp_path, monkeypatch, *, machine="x86_64", raw_count=1, repaired_count=1, repaired_tag=None, fail_repair=False
):
    monkeypatch.setattr(manylinux_builder.sys, "platform", "linux")
    monkeypatch.setattr(manylinux_builder.platform, "machine", lambda: machine)
    monkeypatch.setattr(manylinux_builder.shutil, "which", lambda _: "/usr/bin/auditwheel")
    calls = []
    output = tmp_path / "dist"

    def run(*command, cwd=REPOSITORY_ROOT):
        calls.append((command, cwd))
        if command[:3] == (sys.executable, "-m", "maturin"):
            raw_output = Path(command[command.index("--out") + 1])
            for index in range(raw_count):
                (raw_output / f"aisim-0.12.{index}-cp311-abi3-linux_{machine}.whl").touch()
        elif command[:2] == ("auditwheel", "repair"):
            if fail_repair:
                raise subprocess.CalledProcessError(1, command)
            tag = repaired_tag or f"manylinux_2_28_{machine}"
            for index in range(repaired_count):
                (output / f"aisim-0.12.{index}-cp311-abi3-{tag}.whl").touch()
        else:
            assert command[:2] == ("auditwheel", "show")

    monkeypatch.setattr(manylinux_builder, "_run", run)
    return output, calls


@pytest.mark.parametrize("machine", ["x86_64", "aarch64"])
def test_manylinux_build_repairs_the_exact_raw_wheel_then_audits_it(tmp_path, monkeypatch, machine):
    output, calls = _manylinux_build_case(tmp_path, monkeypatch, machine=machine)
    repaired = manylinux_builder.build(output)
    raw_output = calls[0][0][-1]
    raw_wheel = str(Path(raw_output) / f"aisim-0.12.0-cp311-abi3-linux_{machine}.whl")
    assert repaired == output / f"aisim-0.12.0-cp311-abi3-manylinux_2_28_{machine}.whl"
    assert list(output.iterdir()) == [repaired]
    assert calls == [
        (
            (sys.executable, "-m", "maturin", "build", "--release", "--auditwheel", "skip", "--out", raw_output),
            manylinux_builder.PYTHON_PROJECT,
        ),
        (
            ("auditwheel", "repair", "--plat", f"manylinux_2_28_{machine}", "--wheel-dir", str(output), raw_wheel),
            REPOSITORY_ROOT,
        ),
        (("auditwheel", "show", str(repaired)), REPOSITORY_ROOT),
    ]


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"raw_count": 0}, "exactly one unrepaired"),
        ({"raw_count": 2}, "exactly one unrepaired"),
        ({"repaired_count": 0}, "exactly one repaired"),
        ({"repaired_count": 2}, "exactly one repaired"),
        ({"repaired_tag": "manylinux_2_17_x86_64"}, "required tag manylinux_2_28_x86_64"),
    ],
)
def test_manylinux_build_rejects_missing_duplicate_or_wrong_policy_wheels(tmp_path, monkeypatch, options, message):
    output, calls = _manylinux_build_case(tmp_path, monkeypatch, **options)
    with pytest.raises(SystemExit, match=message):
        manylinux_builder.build(output)
    assert not any(command[:2] == ("auditwheel", "show") for command, _ in calls)


def test_manylinux_build_propagates_repair_failure(tmp_path, monkeypatch):
    output, calls = _manylinux_build_case(tmp_path, monkeypatch, fail_repair=True)
    with pytest.raises(subprocess.CalledProcessError):
        manylinux_builder.build(output)
    assert not any(command[:2] == ("auditwheel", "show") for command, _ in calls)


def test_prediction_gate_owns_report_dependencies_and_pre_harness_fallback() -> None:
    jobs = _workflow("prediction-regression-gate.yml")["jobs"]
    report_commands = _run_commands(jobs["report"])
    collect_commands = _run_commands(jobs["collect"])

    assert "python -m pip install pyyaml==6.0.3" in report_commands
    assert "NO_HARNESS.txt" in collect_commands
    assert "predates the prediction-regression harness" in collect_commands


def test_prediction_gate_resolves_base_for_push_and_manual_callers(tmp_path: Path) -> None:
    resolve_step = _workflow("prediction-regression-gate.yml")["jobs"]["refs"]["steps"][0]
    script = resolve_step["run"]

    assert _run_resolve_step(script, "push", "main", tmp_path) == {"old": "main"}
    assert _run_resolve_step(script, "push", "", tmp_path) == {"old": "main"}
    assert _run_resolve_step(script, "workflow_dispatch", "release/0.12", tmp_path) == {"old": "release/0.12"}


@pytest.mark.parametrize(
    ("event", "ref", "before", "uses_before"),
    [
        ("push", "refs/heads/main", "a" * 40, True),
        ("push", "refs/heads/release/0.12.0", "b" * 40, True),
        ("workflow_dispatch", "refs/heads/main", "a" * 40, False),
        ("workflow_dispatch", "refs/heads/codex/restore-migrated-ci", "", False),
        ("push", "refs/heads/release/0.12.0", "0" * 40, False),
    ],
)
def test_full_ci_comparison_base_uses_previous_lifecycle_commit(
    tmp_path: Path, event: str, ref: str, before: str, uses_before: bool
) -> None:
    result, output = _run_comparison_base(tmp_path, event, ref, before)
    assert result.returncode == 0, result.stderr
    assert output == {"sha": before if uses_before else "c" * 40}
    assert (tmp_path / "gh-called").exists() is not uses_before


@pytest.mark.parametrize(("before", "api_sha", "api_status"), [("bad", "c" * 40, 0), ("", "main", 0), ("", "", 1)])
def test_full_ci_comparison_base_fails_closed(tmp_path: Path, before: str, api_sha: str, api_status: int) -> None:
    result, output = _run_comparison_base(tmp_path, "push", "refs/heads/main", before, api_sha, api_status)
    assert result.returncode != 0
    assert output == {}


def _run_comparison_base(
    tmp_path: Path, event: str, ref: str, before: str, api_sha: str = "c" * 40, api_status: int = 0, pr_base: str = ""
) -> tuple[subprocess.CompletedProcess, dict[str, str]]:
    jobs = _workflow("ci.yml")["jobs"]
    assert jobs["verify-target"]["outputs"]["comparison-base"] == "${{ steps.comparison-base.outputs.sha }}"
    # The selector passes this base through, except for a full-scope main push,
    # which compares with the last main commit that ran the full suite.
    plan_step = next(step for step in jobs["select-full-ci"]["steps"] if step.get("id") == "plan")
    assert plan_step["env"]["COMPARISON_BASE"] == "${{ needs.verify-target.outputs.comparison-base }}"
    assert jobs["select-full-ci"]["outputs"]["comparison-base"] == "${{ steps.plan.outputs.comparison_base }}"
    assert "select-full-ci" in jobs["prediction-regression"]["needs"]
    assert jobs["prediction-regression"]["with"]["old-ref"] == "${{ needs.select-full-ci.outputs.comparison-base }}"
    step = next(step for step in jobs["verify-target"]["steps"] if step.get("id") == "comparison-base")
    assert step["env"]["PR_BASE_SHA"] == "${{ steps.pr-target.outputs.base }}"
    gh = tmp_path / "gh"
    gh.write_text(
        '#!/bin/bash\n[[ "$*" == "api -X GET repos/ai-dynamo/aisim/commits/main --jq .sha" ]] || exit 9\n'
        'touch "${GH_CALLED}"\nprintf "%s\\n" "${TEST_API_SHA}"\nexit "${TEST_API_STATUS}"\n'
    )
    gh.chmod(0o755)
    output_path = tmp_path / "outputs"
    result = subprocess.run(
        ["bash", "-c", step["run"]],
        env={
            **os.environ,
            "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
            "GITHUB_EVENT_NAME": event,
            "GITHUB_REF": ref,
            "BEFORE_SHA": before,
            "PR_BASE_SHA": pr_base,
            "REPOSITORY": "ai-dynamo/aisim",
            "GITHUB_OUTPUT": str(output_path),
            "GH_CALLED": str(tmp_path / "gh-called"),
            "TEST_API_SHA": api_sha,
            "TEST_API_STATUS": str(api_status),
        },
        capture_output=True,
        text=True,
    )
    output = dict(line.split("=", 1) for line in output_path.read_text().splitlines()) if output_path.exists() else {}
    return result, output


def _full_ci_concurrency_group(event: str, ref: str, run_id: str) -> str:
    configuration = _workflow("ci.yml")["concurrency"]
    assert configuration["cancel-in-progress"] == "true"
    prefix, expression = configuration["group"].split("${{", 1)
    expression = expression.rsplit("}}", 1)[0]
    values = {
        "github.event_name": event,
        "github.ref": ref,
        "github.run_id": run_id,
    }
    expression = re.sub(r"github\.[\w_]+", lambda match: repr(values[match[0]]), expression)
    expression = expression.replace("&&", " and ").replace("||", " or ").replace("!startsWith", "not startsWith")
    return prefix + str(
        eval(
            " ".join(expression.splitlines()),
            {"__builtins__": {}},
            {"startsWith": str.startswith},
        )
    )


def test_full_ci_replaces_same_pr_across_trusted_and_manual_runs():
    group = _full_ci_concurrency_group("push", "refs/heads/pull-request/159", "101")
    assert group == _full_ci_concurrency_group("push", "refs/heads/pull-request/159", "102")
    assert group == _full_ci_concurrency_group("workflow_dispatch", "refs/heads/pull-request/159", "103")
    assert group != _full_ci_concurrency_group("push", "refs/heads/pull-request/160", "105")


def test_full_ci_manual_source_branches_replace_only_their_branch():
    group = _full_ci_concurrency_group("workflow_dispatch", "refs/heads/feature", "101")
    assert group == _full_ci_concurrency_group("workflow_dispatch", "refs/heads/feature", "102")
    assert group != _full_ci_concurrency_group("workflow_dispatch", "refs/heads/other-feature", "103")
    assert group != _full_ci_concurrency_group("push", "refs/heads/pull-request/159", "104")


@pytest.mark.parametrize("event", ["push", "workflow_dispatch"])
@pytest.mark.parametrize("ref", ["refs/heads/main", "refs/heads/release/0.12.0", "refs/tags/v0.12.0"])
def test_full_ci_preserves_every_lifecycle_and_tag_run(event, ref):
    # Main batching narrows a superseded run's scope; it never cancels the run,
    # which must still stage its own wheel.
    assert _full_ci_concurrency_group(event, ref, "101") == "full-ci-101"
    assert _full_ci_concurrency_group(event, ref, "101") != _full_ci_concurrency_group(event, ref, "102")


def test_copy_pr_bot_syncs_ready_prs_but_not_drafts():
    # Every trusted-copy push starts Full CI and the advisory performance
    # workflows on the self-hosted pools; drafts keep Fast CI only until they
    # are marked ready or a maintainer comments `/ok to test <sha>`.
    config = yaml.safe_load((REPOSITORY_ROOT / ".github/copy-pr-bot.yaml").read_text())
    assert config == {"enabled": True, "auto_sync_draft": False, "auto_sync_ready": True}


def test_fast_ci_is_standalone_with_an_exact_commit_prerequisite() -> None:
    fast_ci = _workflow("fast-ci.yml")
    assert "workflow_call" not in fast_ci["on"]
    assert set(fast_ci["on"]["push"]["branches"]) == {"main", "pull-request/*", "release/*"}
    assert fast_ci["on"]["workflow_dispatch"]["inputs"]["expected_sha"]["required"] == "true"
    gate = _workflow("ci.yml")["jobs"]["fast-ci"]
    assert gate["name"] == "Require Fast CI"
    assert "uses" not in gate
    assert gate["permissions"] == {"contents": "read", "actions": "read"}
    assert "scripts/ci/require_fast_ci.py" in _run_commands(gate)
    assert "workflow_run" not in _workflow("ci.yml")["on"]
    whitespace = _run_commands(fast_ci["jobs"]["python-static"])
    assert "[.head.sha, .base.sha] | @tsv" in whitespace
    assert fast_ci["jobs"]["python-static"]["env"]["BASE_SHA"] == (
        "${{ inputs.base_sha || github.event.pull_request.base.sha || github.event.before }}"
    )


def test_fast_ci_missing_base_fetch_uses_temporary_checkout_authentication() -> None:
    workflow = _workflow("fast-ci.yml")
    job = workflow["jobs"]["python-static"]
    checkout = next(step for step in job["steps"] if step.get("uses", "").startswith("actions/checkout@"))
    whitespace = next(step for step in job["steps"] if step.get("name") == "Check changed-line whitespace")

    assert workflow["permissions"]["contents"] == "read"
    assert checkout["with"]["persist-credentials"] == "false"
    assert whitespace["env"]["GH_TOKEN"] == "${{ github.token }}"
    assert "printf 'x-access-token:%s' \"${GH_TOKEN}\"" in whitespace["run"]
    assert (
        'git -c "http.https://github.com/.extraheader=AUTHORIZATION: basic ${basic_token}" fetch' in whitespace["run"]
    )
    assert '--no-tags origin "${BASE_SHA}"' in whitespace["run"]


def _run_pr_target(
    tmp_path: Path, head: str, base: str, api_status: int = 0
) -> tuple[subprocess.CompletedProcess[str], dict[str, str]]:
    step = next(step for step in _workflow("ci.yml")["jobs"]["verify-target"]["steps"] if step.get("id") == "pr-target")
    gh = tmp_path / "gh"
    gh.write_text(
        '#!/bin/bash\nprintf "%s\\n" "$*" >> "${GH_CALLED}"\n'
        'if [[ "${TEST_API_STATUS}" != 0 ]]; then exit "${TEST_API_STATUS}"; fi\n'
        'printf "%s\\t%s\\n" "${TEST_PR_HEAD}" "${TEST_PR_BASE}"\n'
    )
    gh.chmod(0o755)
    output_path = tmp_path / "pr-output"
    calls_path = tmp_path / "gh-called"
    result = _run_workflow_script(
        step["run"],
        {
            "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
            "GITHUB_EVENT_NAME": "push",
            "GITHUB_REF": "refs/heads/pull-request/121",
            "RUN_SHA": "a" * 40,
            "REPOSITORY": "ai-dynamo/aisim",
            "GITHUB_OUTPUT": str(output_path),
            "GH_CALLED": str(calls_path),
            "TEST_PR_HEAD": head,
            "TEST_PR_BASE": base,
            "TEST_API_STATUS": str(api_status),
        },
    )
    assert calls_path.read_text().splitlines() == [
        "api -X GET repos/ai-dynamo/aisim/pulls/121 --jq [.head.sha, .base.sha] | @tsv"
    ]
    output = dict(line.split("=", 1) for line in output_path.read_text().splitlines()) if output_path.exists() else {}
    return result, output


def test_full_ci_trusted_copy_validates_the_stacked_pr_base(tmp_path: Path) -> None:
    result, output = _run_pr_target(tmp_path, "a" * 40, "d" * 40)
    assert result.returncode == 0, result.stderr
    assert output == {"base": "d" * 40, "number": "121"}


@pytest.mark.parametrize("pr_base", ["d" * 40, "", "main", "d" * 39])
def test_full_ci_stacked_comparison_uses_only_the_validated_pr_base(tmp_path: Path, pr_base: str) -> None:
    result, output = _run_comparison_base(tmp_path, "push", "refs/heads/pull-request/235", "a" * 40, pr_base=pr_base)
    if pr_base == "d" * 40:
        assert result.returncode == 0, result.stderr
        assert output == {"sha": pr_base}
    else:
        assert result.returncode != 0
        assert output == {}
    assert not (tmp_path / "gh-called").exists()


@pytest.mark.parametrize(
    ("head", "base", "api_status"),
    [
        pytest.param("b" * 40, "d" * 40, 0, id="head-changed"),
        pytest.param("a" * 40, "main", 0, id="mutable-base"),
        pytest.param("a" * 40, "d" * 39, 0, id="short-base"),
        pytest.param("a" * 40, "", 0, id="missing-base"),
        pytest.param("a" * 40, "d" * 40, 1, id="api-failure"),
    ],
)
def test_full_ci_trusted_copy_base_fails_closed(tmp_path: Path, head: str, base: str, api_status: int) -> None:
    result, output = _run_pr_target(tmp_path, head, base, api_status)
    assert result.returncode != 0
    assert output == {}


def _git(repository: Path, *arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "git",
            "-c",
            "user.name=CI contract test",
            "-c",
            "user.email=ci-test@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "core.hooksPath=/dev/null",
            "-C",
            str(repository),
            *arguments,
        ],
        env={**os.environ, "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull},
        capture_output=True,
        text=True,
        check=check,
    )


def _commit_file(repository: Path, name: str, contents: str) -> str:
    (repository / name).write_text(contents)
    _git(repository, "add", name)
    _git(repository, "commit", "--quiet", "-m", name)
    return _git(repository, "rev-parse", "HEAD").stdout.strip()


def _run_whitespace_step(
    repository: Path,
    base: str,
    target: str,
    ref: str = "refs/heads/pull-request/121",
    event: str = "push",
) -> subprocess.CompletedProcess[str]:
    job = _workflow("fast-ci.yml")["jobs"]["python-static"]
    script = next(step["run"] for step in job["steps"] if step.get("name") == "Check changed-line whitespace")
    gh = repository / "gh"
    gh.write_text('#!/bin/bash\nprintf "%s\\t%s\\n" "$TARGET_SHA" "$TEST_PR_BASE"\n')
    gh.chmod(0o755)
    return subprocess.run(
        ["bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", script],
        cwd=repository,
        env={
            **os.environ,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GITHUB_EVENT_NAME": event,
            "GITHUB_REF": ref,
            "BEFORE_SHA": "f" * 40,
            "BASE_SHA": base,
            "TARGET_SHA": target,
            "GH_TOKEN": "ci-contract-dummy-token",
            "PATH": f"{repository}{os.pathsep}{os.environ['PATH']}",
            "REPOSITORY": "ai-dynamo/aisim",
            "TEST_PR_BASE": base,
        },
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("case", ["matching", "missing-expected", "wrong-run-sha", "wrong-checkout"])
def test_fast_ci_manual_exact_target_guard(tmp_path: Path, case: str) -> None:
    _git(tmp_path, "init", "--quiet")
    head = _commit_file(tmp_path, "root.txt", "root\n")
    job = _workflow("fast-ci.yml")["jobs"]["policy"]
    assert job["env"]["EXPECTED_SHA"] == "${{ inputs.expected_sha }}"
    assert job["env"]["TARGET_SHA"] == "${{ inputs.expected_sha || github.event.pull_request.head.sha || github.sha }}"
    step = next(step for step in job["steps"] if step.get("name") == "Verify exact target")
    expected = "" if case == "missing-expected" else "b" * 40 if case == "wrong-checkout" else head
    target = expected or head
    run_sha = "b" * 40 if case in {"wrong-run-sha", "wrong-checkout"} else head
    result = subprocess.run(
        ["bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", step["run"]],
        cwd=tmp_path,
        env={
            **os.environ,
            "GITHUB_EVENT_NAME": "workflow_dispatch",
            "EXPECTED_SHA": expected,
            "TARGET_SHA": target,
            "GITHUB_SHA": run_sha,
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert (result.returncode == 0) is (case == "matching"), result.stdout + result.stderr


@pytest.mark.parametrize("supplied_base", [False, True], ids=["default-last-commit", "explicit-whole-change"])
def test_fast_ci_manual_base_selects_the_whitespace_range(tmp_path: Path, supplied_base: bool) -> None:
    _git(tmp_path, "init", "--quiet")
    base = _commit_file(tmp_path, "root.txt", "root\n")
    _commit_file(tmp_path, "first.txt", "earlier whitespace error \n")
    target = _commit_file(tmp_path, "last.txt", "clean final commit\n")
    workflow = _workflow("fast-ci.yml")
    base_input = workflow["on"]["workflow_dispatch"]["inputs"]["base_sha"]
    assert base_input["required"] == "false"
    assert base_input["type"] == "string"
    assert base_input["default"] == ""
    # A manual event has neither pull_request.base.sha nor before; the workflow
    # expression selects this input or its empty default.
    assert workflow["jobs"]["python-static"]["env"]["BASE_SHA"] == (
        "${{ inputs.base_sha || github.event.pull_request.base.sha || github.event.before }}"
    )
    result = _run_whitespace_step(
        tmp_path,
        base if supplied_base else base_input["default"],
        target,
        ref="refs/heads/codex/manual",
        event="workflow_dispatch",
    )
    assert (result.returncode != 0) is supplied_base, result.stdout + result.stderr
    assert ("first.txt" in result.stdout) is supplied_base


@pytest.mark.parametrize("earlier_whitespace", [False, True], ids=["clean-rebased-copy", "earlier-pr-commit"])
def test_fast_ci_whitespace_checks_the_whole_pr_above_its_stacked_base(
    tmp_path: Path, earlier_whitespace: bool
) -> None:
    _git(tmp_path, "init", "--quiet")
    _commit_file(tmp_path, "root.txt", "root\n")
    base = _commit_file(tmp_path, "parent-pr.txt", "outside this PR \n")
    _commit_file(tmp_path, "first-pr.txt", "first change \n" if earlier_whitespace else "first change\n")
    target = _commit_file(tmp_path, "last-pr.txt", "last change\n")
    assert _git(tmp_path, "cat-file", "-e", f"{'f' * 40}^{{commit}}", check=False).returncode != 0
    assert _git(tmp_path, "diff", "--check", f"{target}^", target).returncode == 0

    result = _run_whitespace_step(tmp_path, base, target)

    assert (result.returncode != 0) is earlier_whitespace, result.stdout + result.stderr
    assert "parent-pr.txt" not in result.stdout
    if earlier_whitespace:
        assert "first-pr.txt" in result.stdout
        assert "trailing whitespace" in result.stdout


@pytest.mark.parametrize("base", ["", "0" * 40], ids=["manual-run", "new-release-branch"])
def test_fast_ci_without_a_base_preserves_the_last_commit_range(tmp_path: Path, base: str) -> None:
    _git(tmp_path, "init", "--quiet")
    _commit_file(tmp_path, "earlier.txt", "outside the last commit \n")
    target = _commit_file(tmp_path, "last.txt", "last change\n")

    result = _run_whitespace_step(tmp_path, base, target, ref="refs/heads/release/new")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "earlier.txt" not in result.stdout


@pytest.mark.parametrize("available", [True, False], ids=["fetch-base-history", "missing-base-fails-closed"])
def test_fast_ci_whitespace_fetches_an_absent_base_without_truncating_history(tmp_path: Path, available: bool) -> None:
    origin = tmp_path / "origin"
    checkout = tmp_path / "checkout"
    origin.mkdir()
    checkout.mkdir()
    _git(origin, "init", "--quiet")
    common = _commit_file(origin, "root.txt", "root\n")
    target = _commit_file(origin, "pr.txt", "PR change\n")
    _git(origin, "checkout", "--quiet", "-b", "base", common)
    _commit_file(origin, "base-first.txt", "first base change\n")
    base = _commit_file(origin, "base-last.txt", "last base change\n")
    _git(origin, "tag", "not-needed-for-whitespace", base)
    _git(checkout, "init", "--quiet")
    _git(checkout, "remote", "add", "origin", origin.as_uri())
    _git(checkout, "fetch", "--no-tags", "origin", target)
    _git(checkout, "checkout", "--quiet", "--detach", target)
    assert _git(checkout, "cat-file", "-e", f"{base}^{{commit}}", check=False).returncode != 0

    result = _run_whitespace_step(checkout, base if available else "f" * 40, target)

    if available:
        assert result.returncode == 0, result.stdout + result.stderr
        assert _git(checkout, "merge-base", base, target).stdout.strip() == common
        assert _git(checkout, "rev-parse", "--is-shallow-repository").stdout.strip() == "false"
        assert not _git(checkout, "tag", "--list").stdout.strip()
        assert _git(checkout, "config", "--local", "--get-regexp", "extraheader", check=False).returncode == 1
    else:
        assert result.returncode != 0
        assert "upload-pack: not our ref" in result.stderr
        assert "Invalid symmetric difference expression" not in result.stderr


def test_fpe_job_uses_required_container_without_legacy_lfs_data() -> None:
    generate = _workflow("fpe-support-matrix.yml")["jobs"]["generate"]
    checkout = next(step for step in generate["steps"] if step.get("uses", "").startswith("actions/checkout@"))
    tracked = subprocess.check_output(
        [
            "git",
            "-C",
            str(REPOSITORY_ROOT),
            "ls-files",
            "python/aisim/src/aisim_core/systems/**/*.txt",
        ],
        text=True,
    ).splitlines()

    assert generate["container"]["image"] == "${{ vars.CI_JOB_CONTAINER_IMAGE }}"
    assert checkout.get("with", {}).get("lfs") == "true" or not tracked, (
        "FPE checkout must enable LFS before a legacy systems/**/*.txt payload is tracked"
    )


def test_shared_python_rust_setup_is_used_by_same_revision_jobs() -> None:
    action_path = "./.github/actions/setup-python-rust"
    full_ci = _workflow("ci.yml")["jobs"]
    for job_name in (
        "rust",
        "rust-feature-modes",
        "public-api-rust",
        "python-compatibility",
        "engine-golden-regression",
    ):
        assert any(step.get("uses") == action_path for step in full_ci[job_name]["steps"])

    assert any(step.get("uses") == action_path for step in _workflow("collector-check.yml")["jobs"]["check"]["steps"])


def test_manylinux_jobs_select_bundled_python_without_setup_python() -> None:
    action_path = "./.github/actions/setup-manylinux-python-rust"
    for workflow_name, job_name in (
        ("ci.yml", "application-test-wheel"),
        ("ci.yml", "release-artifact-contract"),
        ("fpe-support-matrix.yml", "prepare-wheel"),
    ):
        steps = _workflow(workflow_name)["jobs"][job_name]["steps"]
        assert any(step.get("uses") == action_path for step in steps)
        assert not any(
            step.get("uses", "").startswith(("actions/setup-python@", "./.github/actions/setup-python-rust"))
            for step in steps
        )
    action = yaml.safe_load((ACTION_ROOT / "setup-manylinux-python-rust" / "action.yml").read_text())
    assert action["runs"]["steps"][0]["env"]["MANYLINUX_PYTHON_ROOT"] == "/opt/python/cp312-cp312"
    assert not any(step.get("uses", "").startswith("actions/setup-python@") for step in action["runs"]["steps"])
    rust = next(step for step in action["runs"]["steps"] if step.get("uses", "").startswith("dtolnay/rust-toolchain@"))
    assert rust["with"]["toolchain"] == _workflow("nightly-ci.yml")["jobs"]["build-artifacts"]["env"]["RUST_TOOLCHAIN"]


@pytest.mark.parametrize(
    "missing", [None, "python", "cc", "c++", "make", "auditwheel", "patchelf", "auditwheel-broken", "patchelf-broken"]
)
def test_manylinux_bootstrap_exports_working_python_or_fails_before_build(tmp_path: Path, missing: str | None) -> None:
    action = yaml.safe_load((ACTION_ROOT / "setup-manylinux-python-rust" / "action.yml").read_text())
    bootstrap = action["runs"]["steps"][0]["run"]
    python_root = tmp_path / "bundled-python"
    binaries = python_root / "bin"
    binaries.mkdir(parents=True)
    if missing != "python":
        (binaries / "python").symlink_to(sys.executable)
    tools = tmp_path / "tools"
    tools.mkdir()
    tool_log = tmp_path / "tool-log"
    for name in ("cc", "c++", "make", "auditwheel", "patchelf"):
        if name != missing:
            binary = tools / name
            if name in ("auditwheel", "patchelf"):
                binary.write_text(
                    f'#!/bin/bash\necho "{name} $*" >> "$AUDIT_TOOL_LOG"\n'
                    + ("exit 1\n" if missing == f"{name}-broken" else 'test "$1" = --version\n')
                )
                binary.chmod(0o755)
            else:
                binary.symlink_to("/usr/bin/true")
    env_file = tmp_path / "github-env"
    path_file = tmp_path / "github-path"
    env_file.touch()
    path_file.touch()
    runner_temp = tmp_path / "runner-temp"
    result = subprocess.run(
        ["/bin/bash", "-c", bootstrap],
        env={
            **os.environ,
            "PATH": str(tools),
            "MANYLINUX_PYTHON_ROOT": str(python_root),
            "RUNNER_TEMP": str(runner_temp),
            "GITHUB_ENV": str(env_file),
            "GITHUB_PATH": str(path_file),
            "AUDIT_TOOL_LOG": str(tool_log),
        },
        capture_output=True,
        text=True,
        timeout=10,
    )
    if missing:
        assert result.returncode != 0
        assert env_file.read_text() == path_file.read_text() == ""
        return
    assert result.returncode == 0, result.stdout + result.stderr
    assert tool_log.read_text().splitlines() == ["auditwheel --version", "patchelf --version"]
    exported = dict(line.split("=", 1) for line in env_file.read_text().splitlines())
    assert exported == {
        "PYO3_PYTHON": str(binaries / "python"),
        "LIBRARY_PATH": str(python_root / "lib"),
        "CARGO_HOME": str(runner_temp / "cargo"),
        "RUSTUP_HOME": str(runner_temp / "rustup"),
    }
    assert path_file.read_text().splitlines() == [str(binaries)]
    subprocess.run(
        ["python", "-c", "import sys; assert sys.version_info[:2] == (3, 12)"],
        env={**os.environ, **exported, "PATH": path_file.read_text().strip()},
        check=True,
        timeout=10,
    )


def test_macos_wheel_environment_seeds_pip_for_shared_verification() -> None:
    with (ACTION_ROOT / "build-platform-wheel" / "action.yml").open(encoding="utf-8") as handle:
        action = yaml.load(handle, Loader=yaml.BaseLoader)

    commands = _run_commands(action["runs"])
    assert 'uv venv --seed --python 3.13 "${RUNNER_TEMP}/aisim-wheel-venv"' in commands
    assert "python -m pip install --quiet wheelhouse/*.whl" in commands


def test_omitted_wheel_base_preserves_dockerfile_default() -> None:
    with (ACTION_ROOT / "build-platform-wheel" / "action.yml").open(encoding="utf-8") as handle:
        action = yaml.load(handle, Loader=yaml.BaseLoader)
    dockerfile = (REPOSITORY_ROOT / "python/aisim/docker/Dockerfile").read_text()
    docker_default = re.search(r"^ARG WHEEL_BUILD_BASE=(.+)$", dockerfile, re.MULTILINE).group(1)
    wheel_base = action["inputs"]["wheel_base"]
    assert wheel_base["required"] == "false"
    assert wheel_base["default"] == docker_default
    build = next(step for step in action["runs"]["steps"] if step["name"] == "Build Linux wheel")
    omitted_input_arguments = build["with"]["build-args"].replace("${{ inputs.wheel_base }}", wheel_base["default"])
    assert omitted_input_arguments == f"WHEEL_BUILD_BASE={docker_default}"


def test_containerized_workflows_do_not_require_git_lfs_during_checkout() -> None:
    jobs = (
        ("ci.yml", "engine-golden-regression"),
        ("collector-check.yml", "cross-backend-consistency"),
        ("prediction-regression-gate.yml", "collect"),
    )

    for workflow_name, job_name in jobs:
        job = _workflow(workflow_name)["jobs"][job_name]
        checkout = next(step for step in job["steps"] if step.get("uses", "").startswith("actions/checkout@"))
        assert checkout.get("with", {}).get("lfs") != "true"


def test_numerics_jobs_run_on_github_hosted_runners() -> None:
    # Both build AISim from source and use no NVIDIA-internal service. Old and
    # new snapshots may come from different hosted CPUs; the report blocks only
    # on status regressions and reports numerical drift.
    golden = _workflow("ci.yml")["jobs"]["engine-golden-regression"]
    prediction = _workflow("prediction-regression-gate.yml")["jobs"]
    collect = prediction["collect"]

    assert golden["name"] == "Engine Golden Regression"
    assert collect["name"] == "Collect prediction snapshot (${{ matrix.side }})"
    for job in (golden, collect, prediction["report"]):
        assert job["runs-on"] == "ubuntu-24.04"
        assert "container" not in job
        assert "wheel-storage" not in json.dumps(job)
        assert "WHEEL_STORAGE_URL" not in json.dumps(job)

    # Sentinels use checked-in expected values and need no historical commit.
    golden_checkout = next(step for step in golden["steps"] if step.get("uses", "").startswith("actions/checkout@"))
    assert "fetch-depth" not in golden_checkout["with"]
    assert "--fetch-baseline-only" not in _run_commands(golden)
    assert "check_prediction_numerics.py --output numerical-sentinels.json" in _run_commands(golden)
    # The old side reads setup helpers from the workflow commit, so it keeps full history.
    collect_checkout = next(step for step in collect["steps"] if step.get("uses", "").startswith("actions/checkout@"))
    assert collect_checkout["with"]["fetch-depth"] == "0"
    assert 'git show "${GITHUB_SHA}:scripts/ci/ci_install_build_tools.sh"' in _run_commands(collect)


def test_numerics_jobs_keep_single_run_artifact_contracts() -> None:
    golden = _workflow("ci.yml")["jobs"]["engine-golden-regression"]
    prediction = _workflow("prediction-regression-gate.yml")["jobs"]
    collect = prediction["collect"]

    assert "strategy" not in golden
    assert collect["strategy"]["matrix"] == {"side": ["old", "new"]}
    golden_upload = next(
        step for step in golden["steps"] if step.get("uses", "").startswith("actions/upload-artifact@")
    )
    assert golden_upload["with"]["name"] == "native-prediction-numerics"
    assert golden_upload["with"]["path"] == "numerical-sentinels.json"
    collect_upload = next(
        step for step in collect["steps"] if step.get("uses", "").startswith("actions/upload-artifact@")
    )
    assert collect_upload["with"]["name"] == "prediction-gate-${{ matrix.side }}"
    assert "--old /tmp/prediction-snapshots/prediction-gate-old" in _run_commands(prediction["report"])
    assert "--new /tmp/prediction-snapshots/prediction-gate-new" in _run_commands(prediction["report"])
    for job in (golden, collect, prediction["report"]):
        assert "sample" not in json.dumps(job)
        assert "runner-cpu.txt" not in json.dumps(job)


def test_active_workflows_do_not_call_nested_inert_github_assets() -> None:
    for path in WORKFLOW_ROOT.glob("*.yml"):
        assert "./python/aisim/.github/" not in path.read_text(encoding="utf-8")


DATA_READER_JOBS = (
    "rust",
    "rust-feature-modes",
    "public-api-rust",
    "application-tests",
    "python-compatibility",
    "engine-golden-regression",
    "collector-data",
    "readme-commands",
)


def _op_latency_data_result(plan: dict[str, str]) -> str:
    """The data job runs exactly when a job that reads operation tables is selected."""
    return "success" if any(plan.get(job.replace("-", "_")) == "true" for job in DATA_READER_JOBS) else "skipped"


def _run_full_ci_aggregate(
    results: dict[str, str],
    plan: dict[str, str],
    summary: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    aggregate = _workflow("ci.yml")["jobs"]["readiness"]
    script = aggregate["steps"][0]["run"]
    results = {"op-latency-data": _op_latency_data_result(plan), **results}
    needs = {name: {"result": result, "outputs": {}} for name, result in results.items()}
    return subprocess.run(
        ["bash", "-c", script],
        cwd=REPOSITORY_ROOT,
        env={
            **os.environ,
            "NEEDS_JSON": json.dumps(needs),
            "PLAN_JSON": json.dumps(plan),
            **({"GITHUB_STEP_SUMMARY": str(summary)} if summary else {}),
        },
        capture_output=True,
        text=True,
        check=False,
    )


def _run_workflow_script(
    script: str,
    env: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", script],
        cwd=REPOSITORY_ROOT,
        env={**os.environ, **env},
        capture_output=True,
        text=True,
        check=False,
    )


def test_selective_full_ci_keeps_the_aggregate_fail_closed() -> None:
    jobs = _workflow("ci.yml")["jobs"]
    aggregate_needs = set(jobs["readiness"]["needs"])
    component_jobs = {component.replace("_", "-") for component in COMPONENTS}
    assert component_jobs.issubset(aggregate_needs)

    for component in COMPONENTS:
        job = jobs[component.replace("_", "-")]
        assert "select-full-ci" in job["needs"]
        assert f"needs.select-full-ci.outputs.{component} == 'true'" in job["if"]
        assert jobs["select-full-ci"]["outputs"][component] == f"${{{{ steps.plan.outputs.{component} }}}}"

    # The wheel build and the test legs are separate components, so a superseded
    # main commit can build its wheel without running the legs.
    readiness_script = _run_commands(jobs["readiness"])
    component_outputs = ast.literal_eval(re.search(r"component_outputs = (\{.*?\})", readiness_script, re.S)[1])
    assert component_outputs == {component.replace("_", "-"): component for component in COMPONENTS}
    for consumer in ("application-wheel", "application-tests", "release-artifact-contract"):
        assert "application-test-wheel" in jobs[consumer]["needs"]

    wheel_condition = jobs["application-wheel"]["if"]
    assert "needs.fast-ci.result == 'success'" in wheel_condition
    assert "needs.select-full-ci.result == 'success'" in wheel_condition


def test_full_ci_selector_uses_the_complete_exact_head_pr_change_set() -> None:
    selector = _workflow("ci.yml")["jobs"]["select-full-ci"]
    commands = _run_commands(selector)

    assert "pulls/${pr_number}/files?per_page=100" in commands
    assert "--paginate" in commands
    assert ".previous_filename" in commands
    assert "@base64" in commands
    assert "changed_files > 3000" in commands
    assert '"${pr_head}" != "${GITHUB_SHA}"' in commands
    assert "workflow_dispatch:*|push:refs/heads/main|push:refs/heads/release/*" in commands
    assert "force_all=true" in commands


def test_full_ci_scope_resolver_handles_copy_manual_and_race_cases(
    tmp_path: Path,
) -> None:
    selector = _workflow("ci.yml")["jobs"]["select-full-ci"]
    scope_script = next(
        step["run"] for step in selector["steps"] if step.get("name") == "Resolve the trusted change set"
    )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_gh = fake_bin / "gh"
    fake_gh.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'if [[ ${FAKE_GH_EXIT:-0} != 0 ]]; then exit "${FAKE_GH_EXIT}"; fi\n'
        "args=$*\n"
        "if [[ ${args} == *'.changed_files'* ]]; then\n"
        "  printf '%s\\n' \"${FAKE_CHANGED_FILES:-2}\"\n"
        "elif [[ ${args} == *'/files?per_page=100'* ]]; then\n"
        "  if [[ ${FAKE_FAIL_FILES:-false} == true ]]; then exit 1; fi\n"
        "  printf '%s\\n' \"${FAKE_FILES:-README.md}\"\n"
        "elif [[ ${args} == *'.head.sha'* ]]; then\n"
        "  printf '%s\\n' \"${FAKE_PR_HEAD:-}\"\n"
        "else\n"
        "  exit 2\n"
        "fi\n",
        encoding="utf-8",
    )
    fake_gh.chmod(0o755)

    output = tmp_path / "copy-output"
    target_sha = "0123456789abcdef"
    copy_env = {
        "GITHUB_EVENT_NAME": "push",
        "GITHUB_REF": "refs/heads/pull-request/136",
        "GITHUB_SHA": target_sha,
        "GITHUB_OUTPUT": str(output),
        "RUNNER_TEMP": str(tmp_path),
        "REPOSITORY": "ai-dynamo/aisim",
        "FAKE_PR_HEAD": target_sha,
        "FAKE_FILES": "\n".join(
            base64.b64encode(path.encode()).decode() for path in ("README.md", "crates/core/src/replay/event.rs")
        ),
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
    }
    copy_result = _run_workflow_script(scope_script, copy_env)
    assert copy_result.returncode == 0, copy_result.stdout + copy_result.stderr
    assert "force_all=false" in output.read_text(encoding="utf-8")
    encoded_paths = (tmp_path / "full-ci-paths.b64").read_text(encoding="ascii").splitlines()
    assert [base64.b64decode(path).decode() for path in encoded_paths] == [
        "README.md",
        "crates/core/src/replay/event.rs",
    ]

    output.unlink()
    truncated = _run_workflow_script(
        scope_script,
        {**copy_env, "FAKE_FAIL_FILES": "true"},
    )
    assert truncated.returncode != 0

    oversized = _run_workflow_script(
        scope_script,
        {
            **copy_env,
            "FAKE_CHANGED_FILES": "3001",
            "FAKE_FAIL_FILES": "true",
        },
    )
    assert oversized.returncode == 0, oversized.stdout + oversized.stderr
    assert "force_all=true" in output.read_text(encoding="utf-8")

    raced = _run_workflow_script(
        scope_script,
        {**copy_env, "FAKE_PR_HEAD": "fedcba9876543210"},
    )
    assert raced.returncode != 0
    assert "PR head changed" in raced.stdout

    api_failure = _run_workflow_script(
        scope_script,
        {**copy_env, "FAKE_GH_EXIT": "1"},
    )
    assert api_failure.returncode != 0

    invalid_ref = _run_workflow_script(
        scope_script,
        {**copy_env, "GITHUB_REF": "refs/heads/pull-request/not-a-number"},
    )
    assert invalid_ref.returncode != 0
    assert "invalid trusted PR copy ref" in invalid_ref.stdout

    manual_output = tmp_path / "manual-output"
    manual = _run_workflow_script(
        scope_script,
        {
            **copy_env,
            "GITHUB_EVENT_NAME": "workflow_dispatch",
            "GITHUB_REF": "refs/heads/codex/topic",
            "GITHUB_OUTPUT": str(manual_output),
            "FAKE_GH_EXIT": "1",
        },
    )
    assert manual.returncode == 0, manual.stdout + manual.stderr
    assert "force_all=true" in manual_output.read_text(encoding="utf-8")


def test_full_ci_aggregate_accepts_only_explicit_na_results() -> None:
    plan = dict.fromkeys(COMPONENTS, "false")
    plan["application_test_wheel"] = "true"
    plan["application_tests"] = "true"
    results = {
        "verify-target": "success",
        "select-full-ci": "success",
        "fast-ci": "success",
        "python-compliance": "success",
        "license-crate-inventory": "success",
        "license-evidence": "success",
        **{
            component.replace("_", "-"): ("success" if plan[component] == "true" else "skipped")
            for component in COMPONENTS
        },
    }

    passed = _run_full_ci_aggregate(results, plan)
    assert passed.returncode == 0, passed.stdout + passed.stderr

    results["application-tests"] = "skipped"
    missing_selected_job = _run_full_ci_aggregate(results, plan)
    assert missing_selected_job.returncode != 0
    assert "application-tests=skipped, expected success" in missing_selected_job.stdout

    results["application-tests"] = "success"
    results["rust"] = "success"
    unexpected_unselected_job = _run_full_ci_aggregate(results, plan)
    assert unexpected_unselected_job.returncode != 0
    assert "rust=success, expected skipped" in unexpected_unselected_job.stdout


def test_full_ci_aggregate_requires_the_data_job_exactly_when_a_reader_is_selected() -> None:
    plan = dict.fromkeys(COMPONENTS, "false")
    plan["application_test_wheel"] = "true"
    plan["application_tests"] = "true"
    results = {
        "verify-target": "success",
        "select-full-ci": "success",
        "fast-ci": "success",
        "python-compliance": "success",
        "license-crate-inventory": "success",
        "license-evidence": "success",
        **{c.replace("_", "-"): ("success" if plan[c] == "true" else "skipped") for c in COMPONENTS},
    }
    assert _run_full_ci_aggregate(results, plan).returncode == 0
    skipped = _run_full_ci_aggregate({**results, "op-latency-data": "skipped"}, plan)
    assert skipped.returncode != 0 and "op-latency-data=skipped, expected success" in skipped.stdout

    nothing_selected = dict.fromkeys(COMPONENTS, "false")
    idle = {**results, "application-tests": "skipped", "application-test-wheel": "skipped"}
    assert _run_full_ci_aggregate(idle, nothing_selected).returncode == 0
    unexpected = _run_full_ci_aggregate({**idle, "op-latency-data": "success"}, nothing_selected)
    assert unexpected.returncode != 0 and "op-latency-data=success, expected skipped" in unexpected.stdout


def test_jobs_that_read_operation_data_wait_for_the_single_download() -> None:
    """Parallel jobs that each download the dataset get rate limited by Hugging Face and time out."""
    jobs = _workflow("ci.yml")["jobs"]
    readers = {
        name
        for name, job in jobs.items()
        if "prefetch-op-latency-data" in json.dumps(job) or "compose_systems_root" in json.dumps(job)
    } | {"collector-data", "readme-commands"}  # reusable workflows that compose the dataset
    assert readers == set(DATA_READER_JOBS)
    producer = jobs["op-latency-data"]
    for name in readers:
        assert "op-latency-data" in jobs[name]["needs"], name
        assert f"needs.select-full-ci.outputs.{name.replace('-', '_')} == 'true'" in producer["if"], name
    assert "op-latency-data" in jobs["readiness"]["needs"]

    # The artifact the consumers restore is the one the producer uploads, under the name and file they expect.
    (upload,) = [s for s in producer["steps"] if s.get("uses", "").startswith("actions/upload-artifact@")]
    assert upload["with"]["name"] == "op-latency-data"
    assert upload["with"]["path"] == "${{ runner.temp }}/op-latency-data.tar"
    action = yaml.safe_load((ACTION_ROOT / "prefetch-op-latency-data/action.yml").read_text())
    restore, compose = action["runs"]["steps"]
    assert restore["uses"].startswith("actions/download-artifact@") and restore["with"]["name"] == "op-latency-data"
    assert restore["continue-on-error"] is True  # a standalone run falls back to downloading the dataset itself
    assert "op-latency-data/op-latency-data.tar" in compose["run"]
    assert "op_latency_handoff.py unpack" in compose["run"] and "compose_systems_root.py" in compose["run"]
    assert "op_latency_handoff.py pack" in _run_commands(producer)


def test_full_ci_aggregate_rejects_missing_selection_output() -> None:
    plan = dict.fromkeys(COMPONENTS, "false")
    del plan["collector_data"]
    results = {
        "verify-target": "success",
        "select-full-ci": "success",
        "fast-ci": "success",
        "python-compliance": "success",
        "license-crate-inventory": "success",
        "license-evidence": "success",
        **{component.replace("_", "-"): "skipped" for component in COMPONENTS},
        "application-test-wheel": "skipped",
    }

    result = _run_full_ci_aggregate(results, plan)
    assert result.returncode != 0
    assert "invalid selection outputs: collector_data=None" in result.stdout


def test_full_ci_aggregate_rejects_missing_dependency() -> None:
    plan = dict.fromkeys(COMPONENTS, "false")
    results = {
        "verify-target": "success",
        "select-full-ci": "success",
        "fast-ci": "success",
        "python-compliance": "success",
        "license-crate-inventory": "success",
        "license-evidence": "success",
        **{component.replace("_", "-"): "skipped" for component in COMPONENTS},
        "application-test-wheel": "skipped",
    }
    del results["collector-data"]

    result = _run_full_ci_aggregate(results, plan)
    assert result.returncode != 0
    assert "missing dependencies: collector-data" in result.stdout


def test_full_ci_selector_skips_heavy_jobs_for_documentation() -> None:
    plan = select_components(["README.md", "docs/architecture.md"])

    assert plan["run_all"] is False
    assert {name for name, value in plan["components"].items() if value} == {"readme_commands"}


def test_full_ci_selector_maps_python_rust_and_data_boundaries() -> None:
    python_plan = select_components(["python/aisim/src/aisim/traffic.py"])
    assert {component for component, selected in python_plan["components"].items() if selected} == {
        "readme_commands",
        "platform_wheels",
        "application_test_wheel",
        "application_wheel",
        "application_tests",
        "python_compatibility",
        "release_artifact_contract",
    }

    rust_plan = select_components(["crates/core/src/replay/event.rs"])
    assert rust_plan["components"]["rust"] is True
    assert rust_plan["components"]["prediction_regression"] is True
    assert rust_plan["components"]["collector_data"] is True
    assert rust_plan["components"]["cargo_deny"] is False

    data_plan = select_components(["python/aisim/src/aisim_core/systems/data/b200/op.parquet"])
    assert data_plan["components"]["collector_data"] is True
    assert data_plan["components"]["prediction_regression"] is True
    assert data_plan["components"]["engine_golden_regression"] is True


@pytest.mark.parametrize(
    "paths",
    [
        [],
        ["future/unclassified.file"],
        [".github/workflows/ci.yml"],
        ["scripts/ci/select_full_ci.py"],
    ],
)
def test_full_ci_selector_defaults_unknown_or_contract_changes_to_all(
    paths: list[str],
) -> None:
    plan = select_components(paths)

    assert plan["run_all"] is True
    assert all(plan["components"].values())


def test_full_ci_selector_force_all_and_path_validation() -> None:
    forced = select_components(["README.md"], force_all=True)
    assert forced["run_all"] is True
    assert all(forced["components"].values())

    with pytest.raises(ValueError, match="repository-relative"):
        select_components(["../outside.py"])

    whitespace_name = select_components([" README.md"])
    assert whitespace_name["run_all"] is True
    backslash_name = select_components([r"docs\architecture.md"])
    assert backslash_name["run_all"] is True
    with pytest.raises(ValueError, match="repository-relative"):
        select_components(["docs/readme.md\nREADME.md"])


def test_full_ci_selector_cli_decodes_paths_and_writes_outputs(tmp_path: Path) -> None:
    encoded = tmp_path / "paths.b64"
    encoded.write_text(
        base64.b64encode(b"README.md").decode() + "\n",
        encoding="ascii",
    )
    output = tmp_path / "github-output"
    summary = tmp_path / "summary.md"
    command = [
        "python3",
        "scripts/ci/select_full_ci.py",
        "--base64-paths-file",
        str(encoded),
        "--github-output",
        str(output),
        "--summary",
        str(summary),
    ]

    result = subprocess.run(
        [*command, "--comparison-base", "b" * 40],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "run_all=false" in output.read_text(encoding="utf-8")
    assert "application_tests=false" in output.read_text(encoding="utf-8")
    assert "superseded=false" in output.read_text(encoding="utf-8")
    assert f"comparison_base={'b' * 40}" in output.read_text(encoding="utf-8")
    assert "Explicitly N/A" in summary.read_text(encoding="utf-8")
    assert "Main batching" not in summary.read_text(encoding="utf-8")

    # An empty comparison output would make downstream checkouts compare the
    # target with itself.
    output.unlink()
    for base in ([], ["--comparison-base", "main"]):
        missing = subprocess.run([*command, *base], cwd=REPOSITORY_ROOT, capture_output=True, text=True, check=False)
        assert missing.returncode != 0
        assert "comparison base must be a full commit SHA" in missing.stdout + missing.stderr
        assert not output.exists()


def test_python_dependency_changes_run_the_complete_matrix() -> None:
    for path in (
        "python/aisim/pyproject.toml",
        "python/aisim/uv.lock",
        "python/aisim/pytest.ini",
    ):
        plan = select_components([path])

        assert plan["run_all"] is True
        assert all(plan["components"].values())


def test_full_ci_selector_matches_the_independent_mapping_oracle() -> None:
    oracle_path = REPOSITORY_ROOT / ".github" / "full-ci-selection-cases.yml"
    oracle = yaml.safe_load(oracle_path.read_text(encoding="utf-8"))
    assert oracle["schema_version"] == 1
    cases = oracle["cases"]
    assert len({case["id"] for case in cases}) == len(cases)

    observed_components: set[str] = set()
    for case in cases:
        plan = select_components(
            case["paths"],
            force_all=case.get("force_all", False),
            superseded=case.get("superseded", False),
        )
        actual = {component for component, is_selected in plan["components"].items() if is_selected}
        expected = set(COMPONENTS) if case["run_all"] else set(case["selected"])
        assert plan["run_all"] is case["run_all"], case["id"]
        assert actual == expected, case["id"]
        if not case["run_all"]:
            observed_components.update(actual)

    assert observed_components == set(COMPONENTS)


MAIN_TARGET = "a" * 40
MAIN_PARENT = "b" * 40
MAIN_NEWER = "e" * 40
MAIN_COMPARISON_JOBS = (
    "Prediction Regression / Prediction Regression Gate",
    "Python Dependency Licenses",
    "Cargo Dependency Inventories",
    "License compliance evidence",
)


def _main_run(run_id: int, sha: str, **overrides: object) -> dict:
    return {
        "id": run_id,
        "head_sha": sha,
        "path": ".github/workflows/ci.yml",
        "event": "push",
        "head_branch": "main",
        **overrides,
    }


def _scope_jobs(marker: str | None = "skipped", *, conclusion: str | None = "success", plan: str = "success") -> dict:
    """Jobs page of an earlier main run; ``marker=None`` is a run from before main batching."""
    steps = [{"name": full_ci_selector.PLAN_STEP, "status": "completed", "conclusion": plan}]
    if marker is not None:
        steps.append({"name": full_ci_selector.SUPERSEDED_STEP, "status": "completed", "conclusion": marker})
    return {
        "jobs": [
            {"name": "Verify Exact Target", "status": "completed", "conclusion": "success", "steps": []},
            {
                "name": full_ci_selector.SCOPE_JOB,
                "status": "completed" if conclusion else "in_progress",
                "conclusion": conclusion,
                "steps": steps,
            },
            *[{"name": name, "status": "completed", "conclusion": "success"} for name in MAIN_COMPARISON_JOBS],
        ]
    }


def _main_batch_api(
    main_sha: str,
    *,
    compare: dict[str, str] | None = None,
    runs: tuple[dict, ...] = (),
    successor: dict | Exception | None = None,
    jobs: dict[int, dict | list[dict]] | None = None,
    failing: tuple[str, ...] = (),
) -> tuple[object, list]:
    calls: list = []

    def api(endpoint: str) -> dict:
        calls.append(endpoint)
        path = urlsplit(endpoint).path
        if any(fragment in path for fragment in failing):
            raise full_ci_selector.GitHubError("unavailable")
        if path == "repos/ai-dynamo/aisim/commits/main":
            return {"sha": main_sha}
        if path.startswith("repos/ai-dynamo/aisim/compare/"):
            assert parse_qs(urlsplit(endpoint).query) == {"per_page": ["1"]}
            return {"status": (compare or {})[path.rsplit("/", 1)[1]]}
        if path == "repos/ai-dynamo/aisim/actions/workflows/ci.yml/runs":
            query = parse_qs(urlsplit(endpoint).query)
            if "head_sha" in query:
                assert query == {"branch": ["main"], "event": ["push"], "head_sha": [main_sha], "per_page": ["1"]}
                if isinstance(successor, Exception):
                    raise successor
                return {"workflow_runs": [successor] if successor else []}
            page = int(query.pop("page", ["1"])[0])
            assert query == {
                "branch": ["main"],
                "event": ["push"],
                "per_page": [str(full_ci_selector.MAIN_RUNS_PER_PAGE)],
            }
            start = (page - 1) * full_ci_selector.MAIN_RUNS_PER_PAGE
            return {"workflow_runs": list(runs[start : start + full_ci_selector.MAIN_RUNS_PER_PAGE])}
        if path.endswith("/jobs"):
            query = parse_qs(urlsplit(endpoint).query)
            page = int(query.pop("page", ["1"])[0])
            assert query == {"filter": ["latest"], "per_page": ["100"]}
            response = (jobs or {})[int(path.split("/")[-2])]
            if isinstance(response, list):
                return response[page - 1]
            assert page == 1
            return response
        raise AssertionError(endpoint)

    return api, calls


@pytest.mark.parametrize(("status", "conclusion"), [("queued", None), ("in_progress", None), ("completed", "success")])
def test_main_push_waits_then_defers_its_tests_when_main_moved_on(status: str, conclusion: str | None) -> None:
    api, calls = _main_batch_api(
        MAIN_NEWER,
        compare={f"{MAIN_TARGET}...{MAIN_NEWER}": "ahead"},
        successor=_main_run(201, MAIN_NEWER, status=status, conclusion=conclusion),
    )

    batch = full_ci_selector.resolve_main_push(
        "ai-dynamo/aisim", MAIN_TARGET, 200, MAIN_PARENT, api=api, sleep=lambda seconds: calls.append(seconds)
    )

    assert full_ci_selector.MAIN_DEBOUNCE_SECONDS == 600
    # Wait first, then confirm ancestry and an available run for the successor.
    assert calls == [
        600,
        "repos/ai-dynamo/aisim/commits/main",
        f"repos/ai-dynamo/aisim/compare/{MAIN_TARGET}...{MAIN_NEWER}?per_page=1",
        f"repos/ai-dynamo/aisim/actions/workflows/ci.yml/runs?branch=main&event=push&head_sha={MAIN_NEWER}&per_page=1",
    ]
    assert batch["superseded"] is True
    assert batch["main_sha"] == MAIN_NEWER
    assert batch["comparison_base"] == MAIN_PARENT
    plan = select_components([], force_all=True, superseded=True)
    assert plan["run_all"] is False
    assert {name for name, selected in plan["components"].items() if selected} == {
        "application_test_wheel",
        "application_wheel",
    }


@pytest.mark.parametrize(
    "successor",
    [
        None,
        _main_run(201, MAIN_NEWER, status="completed", conclusion="skipped"),
        _main_run(201, MAIN_NEWER, status="completed", conclusion="cancelled"),
        _main_run(201, MAIN_NEWER, status="completed", conclusion="failure"),
        _main_run(201, MAIN_TARGET, status="queued", conclusion=None),
        full_ci_selector.GitHubError("unavailable"),
    ],
)
def test_main_push_keeps_full_scope_without_an_eligible_successor(successor: dict | Exception | None) -> None:
    api, _ = _main_batch_api(
        MAIN_NEWER,
        compare={f"{MAIN_TARGET}...{MAIN_NEWER}": "ahead", f"{MAIN_PARENT}...{MAIN_TARGET}": "ahead"},
        successor=successor,
        runs=(_main_run(199, MAIN_PARENT),),
        jobs={199: _scope_jobs()},
    )

    batch = full_ci_selector.resolve_main_push(
        "ai-dynamo/aisim", MAIN_TARGET, 200, MAIN_PARENT, api=api, sleep=lambda _: None
    )

    assert batch["superseded"] is False
    assert batch["comparison_base"] == MAIN_PARENT
    assert batch["notes"][0][0] == "warning"
    assert all(select_components([], force_all=True, superseded=batch["superseded"])["components"].values())


def test_main_push_keeps_full_scope_and_compares_with_the_last_full_scope_commit() -> None:
    deferred, undecided, validated = MAIN_PARENT, "c" * 40, "d" * 40
    api, calls = _main_batch_api(
        MAIN_TARGET,
        compare={f"{validated}...{MAIN_TARGET}": "ahead"},
        runs=(
            _main_run(201, MAIN_NEWER),
            _main_run(200, MAIN_TARGET),
            _main_run(199, deferred),
            _main_run(198, undecided),
            _main_run(197, validated),
            _main_run(196, "f" * 40),
        ),
        jobs={199: _scope_jobs("success"), 198: _scope_jobs(conclusion=None), 197: _scope_jobs("skipped")},
    )

    batch = full_ci_selector.resolve_main_push(
        "ai-dynamo/aisim", MAIN_TARGET, 200, MAIN_PARENT, api=api, sleep=lambda _: None
    )

    assert batch["superseded"] is False
    assert batch["comparison_base"] == validated
    assert [level for level, _ in batch["notes"]] == ["notice", "notice"]
    # The surviving run names every commit whose test suite it now covers.
    assert deferred in batch["notes"][1][1] and undecided in batch["notes"][1][1]
    inspected = [call for call in calls if call.endswith("/jobs?filter=latest&per_page=100")]
    assert [call.split("/")[-2] for call in inspected] == ["199", "198", "197"]


@pytest.mark.parametrize("superseded_count", [29, 59])
@pytest.mark.parametrize("newer_count", [0, 30])
def test_main_push_recovers_a_validated_base_beyond_the_first_history_page(
    superseded_count: int, newer_count: int
) -> None:
    validated = "d" * 40
    newer = [_main_run(230 - index, MAIN_NEWER) for index in range(newer_count)]
    superseded = [_main_run(199 - index, f"{index + 1:040x}") for index in range(superseded_count)]
    validated_run_id = 199 - superseded_count
    api, calls = _main_batch_api(
        MAIN_TARGET,
        compare={f"{validated}...{MAIN_TARGET}": "ahead"},
        runs=(*newer, _main_run(200, MAIN_TARGET), *superseded, _main_run(validated_run_id, validated)),
        jobs={
            **{run["id"]: _scope_jobs("success") for run in superseded},
            validated_run_id: _scope_jobs("skipped"),
        },
    )

    batch = full_ci_selector.resolve_main_push(
        "ai-dynamo/aisim", MAIN_TARGET, 200, MAIN_PARENT, api=api, sleep=lambda _: None
    )

    assert batch["superseded"] is False
    assert batch["comparison_base"] == validated
    assert all(select_components([], force_all=True, superseded=batch["superseded"])["components"].values())
    assert all(run["head_sha"] in batch["notes"][1][1] for run in superseded)
    history_pages = [call for call in calls if "/actions/workflows/ci.yml/runs?" in call]
    assert len(history_pages) == (newer_count + superseded_count + 1) // 30 + 1
    assert parse_qs(urlsplit(history_pages[-1]).query)["page"] == [str(len(history_pages))]


@pytest.mark.parametrize("marker", ["skipped", None])
@pytest.mark.parametrize("run_status", ["in_progress", "failure", "success"])
def test_main_comparison_base_is_the_parent_after_successful_comparisons(marker: str | None, run_status: str) -> None:
    api, _ = _main_batch_api(
        MAIN_TARGET,
        compare={f"{MAIN_PARENT}...{MAIN_TARGET}": "ahead"},
        runs=(
            _main_run(
                199,
                MAIN_PARENT,
                status="in_progress" if run_status == "in_progress" else "completed",
                conclusion=None if run_status == "in_progress" else run_status,
            ),
            _main_run(198, "c" * 40),
        ),
        jobs={199: _scope_jobs(marker)},
    )

    base, notes = full_ci_selector.last_full_scope_commit("ai-dynamo/aisim", MAIN_TARGET, 200, MAIN_PARENT, api=api)

    assert base == MAIN_PARENT
    assert notes == [
        (
            "notice",
            f"Comparison base `{MAIN_PARENT}`: the last full-scope main commit with successful comparisons (run 199).",
        )
    ]


@pytest.mark.parametrize("marker", ["skipped", None])
@pytest.mark.parametrize("job_name", MAIN_COMPARISON_JOBS)
@pytest.mark.parametrize("result", ["missing", "queued", "in_progress", "cancelled", "failure", "skipped", "duplicate"])
def test_main_comparison_base_skips_unfinished_or_unsuccessful_comparisons(
    marker: str | None, job_name: str, result: str
) -> None:
    validated = "d" * 40
    incomplete = _scope_jobs(marker)
    job = next(job for job in incomplete["jobs"] if job["name"] == job_name)
    if result == "missing":
        incomplete["jobs"].remove(job)
    elif result == "duplicate":
        incomplete["jobs"].append(dict(job))
    else:
        job.update(
            status=result if result in {"queued", "in_progress"} else "completed",
            conclusion=None if result in {"queued", "in_progress"} else result,
        )
    api, _ = _main_batch_api(
        MAIN_TARGET,
        compare={f"{MAIN_PARENT}...{MAIN_TARGET}": "ahead", f"{validated}...{MAIN_TARGET}": "ahead"},
        runs=(_main_run(199, MAIN_PARENT), _main_run(198, validated)),
        jobs={199: incomplete, 198: _scope_jobs()},
    )

    base, _ = full_ci_selector.last_full_scope_commit("ai-dynamo/aisim", MAIN_TARGET, 200, MAIN_PARENT, api=api)

    assert base == validated


@pytest.mark.parametrize("comparison_result", ["success", "failure", "missing"])
def test_main_comparison_base_reads_all_job_pages(comparison_result: str) -> None:
    validated = "d" * 40
    scope_jobs = _scope_jobs()["jobs"]
    # Scope selection is on the first page; comparison evidence is on the next.
    first_page = scope_jobs[:2] + [{"name": f"other job {number}"} for number in range(98)]
    comparisons = scope_jobs[2:]
    if comparison_result == "missing":
        comparisons.pop()
    else:
        comparisons[-1]["conclusion"] = comparison_result
    api, calls = _main_batch_api(
        MAIN_TARGET,
        compare={f"{MAIN_PARENT}...{MAIN_TARGET}": "ahead", f"{validated}...{MAIN_TARGET}": "ahead"},
        runs=(_main_run(199, MAIN_PARENT), _main_run(198, validated)),
        jobs={199: [{"jobs": first_page}, {"jobs": comparisons}], 198: _scope_jobs()},
    )

    base, _ = full_ci_selector.last_full_scope_commit("ai-dynamo/aisim", MAIN_TARGET, 200, MAIN_PARENT, api=api)

    assert base == (MAIN_PARENT if comparison_result == "success" else validated)
    assert "repos/ai-dynamo/aisim/actions/runs/199/jobs?filter=latest&per_page=100&page=2" in calls


def test_main_comparison_base_requires_readable_job_pages() -> None:
    api, _ = _main_batch_api(
        MAIN_TARGET,
        runs=(_main_run(199, MAIN_PARENT),),
        jobs={199: {}},
    )

    with pytest.raises(full_ci_selector.GitHubError, match="no Full CI jobs"):
        full_ci_selector.last_full_scope_commit("ai-dynamo/aisim", MAIN_TARGET, 200, MAIN_PARENT, api=api)


def test_main_push_retry_keeps_full_scope_when_main_advances_between_attempts() -> None:
    history = {
        "compare": {f"{MAIN_PARENT}...{MAIN_TARGET}": "ahead", f"{MAIN_TARGET}...{MAIN_NEWER}": "ahead"},
        "runs": (_main_run(199, MAIN_PARENT),),
        "jobs": {199: _scope_jobs()},
    }
    initial_api, _ = _main_batch_api(MAIN_TARGET, **history)
    retry_api, _ = _main_batch_api(MAIN_NEWER, **history)

    for attempt, api in ((1, initial_api), (2, retry_api)):
        batch = full_ci_selector.resolve_main_push(
            "ai-dynamo/aisim", MAIN_TARGET, 200, MAIN_PARENT, run_attempt=attempt, api=api, sleep=lambda _: None
        )
        plan = select_components([], force_all=True, superseded=batch["superseded"])
        # A failed application test must still be selected on the retry of this
        # same run and SHA, even though main now contains a newer commit.
        assert batch["superseded"] is False
        assert plan["run_all"] is True
        assert plan["components"]["application_tests"] is True
        assert all(plan["components"].values())


@pytest.mark.parametrize(
    ("jobs_page", "expected"),
    [
        (_scope_jobs("skipped"), "full"),
        (_scope_jobs(None), "full"),
        (_scope_jobs("success"), "superseded"),
        (_scope_jobs("skipped", conclusion="failure"), "undecided"),
        (_scope_jobs("success", conclusion="cancelled"), "undecided"),
        (_scope_jobs(None, conclusion=None), "undecided"),
        (_scope_jobs(None, plan="failure"), "undecided"),
        ({"jobs": []}, "undecided"),
        ({"jobs": [{**_scope_jobs()["jobs"][1], "steps": []}]}, "undecided"),
        ({"jobs": _scope_jobs()["jobs"] + _scope_jobs()["jobs"]}, "undecided"),
    ],
)
def test_main_scope_decision_reads_only_completed_selection_evidence(jobs_page: dict, expected: str) -> None:
    assert full_ci_selector.main_scope_decision(jobs_page) == expected


@pytest.mark.parametrize("parent_evidence", ["absent", "superseded", "undecided", "comparisons failed"])
def test_main_comparison_base_rejects_an_unvalidated_parent(parent_evidence: str) -> None:
    other = "c" * 40
    jobs = _scope_jobs("success" if parent_evidence == "superseded" else "skipped")
    if parent_evidence == "undecided":
        jobs = _scope_jobs(conclusion=None)
    elif parent_evidence == "comparisons failed":
        jobs["jobs"][-1]["conclusion"] = "failure"
    no_full_scope, _ = _main_batch_api(
        MAIN_TARGET,
        runs=(
            *(() if parent_evidence == "absent" else (_main_run(199, MAIN_PARENT),)),
            _main_run(198, other, head_branch="release/0.12.0"),
            _main_run(197, other, event="workflow_dispatch"),
            _main_run(196, other, path=".github/workflows/fast-ci.yml"),
        ),
        jobs={199: jobs},
    )

    # In particular, a superseded parent may already contain a regression.
    # Selecting every test cannot recover it if both comparison snapshots do.
    with pytest.raises(full_ci_selector.GitHubError, match="No full-scope Full CI run with successful comparisons"):
        full_ci_selector.resolve_main_push(
            "ai-dynamo/aisim", MAIN_TARGET, 200, MAIN_PARENT, api=no_full_scope, sleep=lambda _: None
        )


@pytest.mark.parametrize("status", ["diverged", "behind"])
def test_main_comparison_base_rejects_a_non_ancestor_instead_of_using_a_superseded_parent(status: str) -> None:
    other = "c" * 40
    rewritten, _ = _main_batch_api(
        MAIN_TARGET,
        compare={f"{other}...{MAIN_TARGET}": status},
        runs=(_main_run(199, MAIN_PARENT), _main_run(198, other)),
        jobs={199: _scope_jobs("success"), 198: _scope_jobs("skipped")},
    )

    with pytest.raises(full_ci_selector.GitHubError, match="not an ancestor"):
        full_ci_selector.resolve_main_push(
            "ai-dynamo/aisim", MAIN_TARGET, 200, MAIN_PARENT, api=rewritten, sleep=lambda _: None
        )


def test_main_batching_fails_closed_to_full_scope_or_an_error() -> None:
    history = {
        "compare": {f"{MAIN_PARENT}...{MAIN_TARGET}": "ahead"},
        "runs": (_main_run(199, MAIN_PARENT),),
        "jobs": {199: _scope_jobs("skipped")},
    }
    unreadable_main, _ = _main_batch_api(MAIN_NEWER, failing=("/commits/main",), **history)
    invalid_main, _ = _main_batch_api("main", **history)
    diverged, _ = _main_batch_api(
        MAIN_NEWER, **{**history, "compare": {**history["compare"], f"{MAIN_TARGET}...{MAIN_NEWER}": "diverged"}}
    )
    uncomparable, _ = _main_batch_api(MAIN_NEWER, failing=(f"/compare/{MAIN_TARGET}",), **history)
    for api in (unreadable_main, invalid_main, diverged, uncomparable):
        batch = full_ci_selector.resolve_main_push(
            "ai-dynamo/aisim", MAIN_TARGET, 200, MAIN_PARENT, api=api, sleep=lambda _: None
        )
        assert batch["superseded"] is False
        assert batch["comparison_base"] == MAIN_PARENT
        assert batch["notes"][0][0] == "warning"

    # A full-scope run must not silently narrow its comparison when the run
    # history cannot be read.
    no_history, _ = _main_batch_api(MAIN_TARGET, failing=("/actions/",))
    with pytest.raises(full_ci_selector.GitHubError):
        full_ci_selector.resolve_main_push(
            "ai-dynamo/aisim", MAIN_TARGET, 200, MAIN_PARENT, api=no_history, sleep=lambda _: None
        )
    for repository, sha, run_id, parent in (
        ("ai-dynamo/aisim", MAIN_TARGET, 200, ""),
        ("ai-dynamo/aisim", "main", 200, MAIN_PARENT),
        ("ai-dynamo/aisim", MAIN_TARGET, 0, MAIN_PARENT),
        ("", MAIN_TARGET, 200, MAIN_PARENT),
    ):
        with pytest.raises(ValueError):
            full_ci_selector.resolve_main_push(repository, sha, run_id, parent, api=no_history, sleep=lambda _: None)
    for attempt in (0, -1):
        with pytest.raises(ValueError):
            full_ci_selector.resolve_main_push(
                "ai-dynamo/aisim",
                MAIN_TARGET,
                200,
                MAIN_PARENT,
                run_attempt=attempt,
                api=no_history,
                sleep=lambda _: None,
            )


@pytest.mark.parametrize(
    ("main_advanced", "run_attempt", "superseded"),
    [(True, 1, True), (False, 1, False), (True, 2, False), (True, 3, False)],
)
def test_main_batching_cli_writes_scope_and_comparison_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    main_advanced: bool,
    run_attempt: int,
    superseded: bool,
) -> None:
    validated = "d" * 40
    api, calls = _main_batch_api(
        MAIN_NEWER if main_advanced else MAIN_TARGET,
        compare={f"{MAIN_TARGET}...{MAIN_NEWER}": "ahead", f"{validated}...{MAIN_TARGET}": "ahead"},
        runs=(_main_run(199, MAIN_PARENT), _main_run(198, validated)),
        successor=_main_run(201, MAIN_NEWER, status="queued", conclusion=None),
        jobs={199: _scope_jobs("success"), 198: _scope_jobs("skipped")},
    )
    monkeypatch.setattr(full_ci_selector, "github_api", api)
    monkeypatch.setattr(full_ci_selector.time, "sleep", calls.append)
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", str(run_attempt))
    output, summary = tmp_path / "github-output", tmp_path / "summary.md"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "select_full_ci.py",
            "--force-all",
            "--comparison-base",
            MAIN_PARENT,
            "--main-push",
            "--repository",
            "ai-dynamo/aisim",
            "--sha",
            MAIN_TARGET,
            "--run-id",
            "200",
            "--github-output",
            str(output),
            "--summary",
            str(summary),
        ],
    )

    full_ci_selector.main()

    outputs = dict(line.split("=", 1) for line in output.read_text(encoding="utf-8").splitlines())
    selected = {component for component in COMPONENTS if outputs[component] == "true"}
    assert calls[0] == full_ci_selector.MAIN_DEBOUNCE_SECONDS
    assert outputs["superseded"] == str(superseded).lower()
    assert outputs["run_all"] == str(not superseded).lower()
    assert selected == ({"application_test_wheel", "application_wheel"} if superseded else set(COMPONENTS))
    assert outputs["comparison_base"] == (MAIN_PARENT if superseded else validated)
    assert "### Main batching" in summary.read_text(encoding="utf-8")
    assert ("::notice::main moved to" in capsys.readouterr().out) is superseded


@pytest.mark.parametrize(
    "failure",
    [
        "api",
        "empty history",
        "superseded parent",
        "non-ancestor",
        "history exhausted",
        "history API error",
        "history malformed",
        "history non-ancestor",
    ],
)
def test_main_batching_cli_errors_leave_outputs_unwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    other = "c" * 40
    runs = () if failure in {"api", "empty history"} else (_main_run(199, MAIN_PARENT),)
    if failure == "non-ancestor":
        runs += (_main_run(198, other),)
    if failure.startswith("history "):
        runs += tuple(_main_run(198 - index, f"{index + 1:040x}") for index in range(29))
        if failure == "history non-ancestor":
            runs += (_main_run(169, other),)
    jobs = {run["id"]: _scope_jobs("success") for run in runs}
    if failure in {"non-ancestor", "history non-ancestor"}:
        jobs[runs[-1]["id"]] = _scope_jobs("skipped")
    api, calls = _main_batch_api(
        MAIN_TARGET,
        failing=("/actions/",) if failure == "api" else (),
        runs=runs,
        jobs=jobs,
        compare={f"{other}...{MAIN_TARGET}": "diverged"},
    )

    def read_history(endpoint: str) -> dict:
        if "/actions/workflows/ci.yml/runs?" in endpoint and "page=2" in endpoint:
            if failure == "history API error":
                calls.append(endpoint)
                raise full_ci_selector.GitHubError("unavailable history page")
            if failure == "history malformed":
                calls.append(endpoint)
                return {"workflow_runs": None}
        return api(endpoint)

    monkeypatch.setattr(full_ci_selector, "github_api", read_history)
    monkeypatch.setattr(full_ci_selector.time, "sleep", lambda _: None)
    output, summary = tmp_path / "github-output", tmp_path / "summary.md"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "select_full_ci.py",
            "--force-all",
            "--comparison-base",
            MAIN_PARENT,
            "--main-push",
            "--repository",
            "ai-dynamo/aisim",
            "--sha",
            MAIN_TARGET,
            "--run-id",
            "200",
            "--github-output",
            str(output),
            "--summary",
            str(summary),
        ],
    )

    with pytest.raises(SystemExit, match="::error::"):
        full_ci_selector.main()
    assert not output.exists()
    assert not summary.exists()
    if failure.startswith("history "):
        assert any("/actions/workflows/ci.yml/runs?" in call and "page=2" in call for call in calls)


def test_main_batching_is_wired_only_into_main_pushes(tmp_path: Path) -> None:
    jobs = _workflow("ci.yml")["jobs"]
    selector = jobs["select-full-ci"]
    assert selector["name"] == full_ci_selector.SCOPE_JOB
    assert selector["permissions"] == {"contents": "read", "pull-requests": "read", "actions": "read"}
    assert int(selector["timeout-minutes"]) * 60 >= full_ci_selector.MAIN_DEBOUNCE_SECONDS + 300
    names = [step.get("name") for step in selector["steps"]]
    plan_index = names.index(full_ci_selector.PLAN_STEP)
    plan_step = selector["steps"][plan_index]
    marker = selector["steps"][plan_index + 1]
    assert plan_step["id"] == "plan"
    # Later main runs classify this run from the marker step's conclusion.
    assert marker["name"] == full_ci_selector.SUPERSEDED_STEP
    assert marker["if"] == "steps.plan.outputs.superseded == 'true'"
    assert "continue-on-error" not in marker
    assert selector["outputs"]["superseded"] == "${{ steps.plan.outputs.superseded }}"
    assert full_ci_selector.WORKFLOW_PATH == ".github/workflows/ci.yml"
    prediction_jobs = _workflow("prediction-regression-gate.yml")["jobs"]
    assert (
        full_ci_selector.COMPARISON_JOBS
        == set(MAIN_COMPARISON_JOBS)
        == {
            f"{jobs['prediction-regression']['name']} / {prediction_jobs['report']['name']}",
            jobs["python-compliance"]["name"],
            jobs["license-crate-inventory"]["name"],
            jobs["license-evidence"]["name"],
        }
    )

    # Only the selector reads the exact-target base; every comparison uses the selected base.
    workflow_text = (WORKFLOW_ROOT / "ci.yml").read_text(encoding="utf-8")
    assert workflow_text.count("needs.verify-target.outputs.comparison-base") == 1
    selected_base = "${{ needs.select-full-ci.outputs.comparison-base }}"
    assert jobs["prediction-regression"]["with"]["old-ref"] == selected_base
    for job_name in ("python-compliance", "license-crate-inventory"):
        baselines = [
            step["with"]["ref"]
            for step in jobs[job_name]["steps"]
            if step.get("with", {}).get("path") == "license-baseline-source"
        ]
        assert baselines == [selected_base]
        assert "select-full-ci" in jobs[job_name]["needs"]
    evidence = next(step for step in jobs["license-evidence"]["steps"] if "BASE_SHA" in step.get("env", {}))
    assert evidence["env"]["BASE_SHA"] == selected_base
    assert "select-full-ci" in jobs["license-evidence"]["needs"]

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python"
    fake_python.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "${ARGS_FILE}"\n', encoding="utf-8")
    fake_python.chmod(0o755)
    args_file = tmp_path / "args"
    for event, ref, attempt, batched in (
        ("push", "refs/heads/main", "1", True),
        ("push", "refs/heads/main", "2", True),
        ("push", "refs/heads/release/0.12.0", "1", False),
        ("push", "refs/heads/pull-request/136", "1", False),
        ("workflow_dispatch", "refs/heads/main", "1", False),
    ):
        result = _run_workflow_script(
            plan_step["run"],
            {
                "GITHUB_EVENT_NAME": event,
                "GITHUB_REF": ref,
                "GITHUB_SHA": MAIN_TARGET,
                "GITHUB_RUN_ID": "200",
                "GITHUB_RUN_ATTEMPT": attempt,
                "GITHUB_OUTPUT": str(tmp_path / "github-output"),
                "GITHUB_STEP_SUMMARY": str(tmp_path / "summary.md"),
                "FORCE_ALL": "true",
                "PATHS_FILE": str(tmp_path / "paths.b64"),
                "COMPARISON_BASE": MAIN_PARENT,
                "REPOSITORY": "ai-dynamo/aisim",
                "ARGS_FILE": str(args_file),
                "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
            },
        )
        assert result.returncode == 0, result.stdout + result.stderr
        args = args_file.read_text(encoding="utf-8").splitlines()
        assert args[0] == "scripts/ci/select_full_ci.py"
        assert args[args.index("--comparison-base") + 1] == MAIN_PARENT
        assert "--force-all" in args
        if batched:
            assert args[args.index("--main-push") :] == [
                "--main-push",
                "--repository",
                "ai-dynamo/aisim",
                "--sha",
                MAIN_TARGET,
                "--run-id",
                "200",
                "--run-attempt",
                attempt,
            ]
        else:
            assert "--main-push" not in args


def test_full_ci_aggregate_accepts_the_superseded_main_scope(tmp_path: Path) -> None:
    plan = {
        component: str(selected).lower()
        for component, selected in select_components([], force_all=True, superseded=True)["components"].items()
    }
    plan["superseded"] = "true"
    results = {
        "verify-target": "success",
        "select-full-ci": "success",
        "fast-ci": "success",
        "python-compliance": "success",
        "license-crate-inventory": "success",
        "license-evidence": "success",
        **{
            component.replace("_", "-"): ("success" if plan[component] == "true" else "skipped")
            for component in COMPONENTS
        },
    }
    summary = tmp_path / "summary.md"

    passed = _run_full_ci_aggregate(results, plan, summary)
    assert passed.returncode == 0, passed.stdout + passed.stderr
    assert "application-tests: skipped (expected skipped)" in passed.stdout
    assert "application-wheel: success (expected success)" in passed.stdout
    assert "Superseded main commit" in summary.read_text(encoding="utf-8")

    for job, result in (
        ("application-test-wheel", "skipped"),
        ("application-wheel", "failure"),
        ("application-tests", "success"),
        ("prediction-regression", "success"),
        ("license-evidence", "skipped"),
    ):
        failed = _run_full_ci_aggregate({**results, job: result}, plan)
        assert failed.returncode != 0, job


def test_parallel_test_matrix_has_no_missing_or_duplicate_partitions():
    jobs = _workflow("ci.yml")["jobs"]
    shards = jobs["application-tests"]["strategy"]["matrix"]["shard"]
    for suite in {entry["suite"] for entry in shards}:
        entries = [entry for entry in shards if entry["suite"] == suite]
        counts = {int(entry["groups"]) for entry in entries}
        assert len(counts) == 1
        assert sorted(int(entry["group"]) for entry in entries) == list(range(1, counts.pop() + 1))
    assert "application-tests" not in jobs["application-wheel"]["needs"]
    assert {
        "application-tests",
        "application-wheel",
        "platform-wheels",
        "collector-data",
        "prediction-regression",
    }.issubset(jobs["readiness"]["needs"])
    assert set(jobs["stage-application-wheel"]["needs"]) == {"readiness", "application-wheel"}
    split_steps = [step for step in jobs["application-tests"]["steps"] if "--splits" in step.get("run", "")]
    assert [re.findall(r"matrix\.shard\.suite == '([\w-]+)'", step["if"]) for step in split_steps] == [
        ["unit"],
        ["cli-build"],
    ]
    for step in split_steps:
        command = step["run"]
        assert command.count("--splits ") == 1
        assert command.count("--group ") == 1
        assert command.count("--splitting-algorithm ") == 1
        assert "--splits ${{ matrix.shard.groups }}" in command
        assert "--group ${{ matrix.shard.group }}" in command
        assert "--splitting-algorithm least_duration" in command


def _unwrap_expression(expression: str) -> str:
    return expression.strip().removeprefix("${{").removesuffix("}}").strip()


def _application_expression(expression: str, shard: dict, *, outcomes: dict, failed: bool, cancelled: bool):
    # Evaluate the workflow's own expression with GitHub's operand-returning
    # `&&`/`||`, which Python's `and`/`or` share. actionlint checks the syntax.
    def substitute(match):
        token = match[0]
        if token.startswith("'"):
            return token
        if token.startswith("steps."):
            return repr(outcomes.get(token.split(".")[1], ""))
        return repr(int(shard["group"]) if token == "matrix.shard.group" else shard[token.rsplit(".", 1)[1]])

    expression = _unwrap_expression(expression)
    expression = re.sub(r"'[^']*'|steps\.[\w-]+\.outcome|matrix\.shard\.\w+", substitute, expression)
    for function, value in (
        ("!cancelled()", not cancelled),
        ("cancelled()", cancelled),
        ("always()", True),
        ("success()", not failed and not cancelled),
        ("failure()", failed),
    ):
        expression = expression.replace(function, repr(value))
    return eval(expression.replace("&&", " and ").replace("||", " or "), {"__builtins__": {}}, {})


def _application_leg_steps(job: dict, shard: dict, *, failing: str = "", cancelled: bool = False) -> list[str]:
    ran, outcomes, failed = [], {}, False
    for step in job["steps"]:
        name = step.get("name") or step["uses"]
        condition = _unwrap_expression(step.get("if", "success()"))
        if not re.search(r"\b(?:success|failure|always|cancelled)\(\)", condition):
            condition = f"success() && ({condition})"
        runs = _application_expression(condition, shard, outcomes=outcomes, failed=failed, cancelled=cancelled)
        outcome = "skipped" if not runs else "failure" if name == failing else "success"
        outcomes[step.get("id", "")] = outcome
        failed = failed or outcome == "failure"
        if runs:
            ran.append(name)
    return ran


def test_application_test_legs_still_report_every_suite_after_an_earlier_failure():
    job = _workflow("ci.yml")["jobs"]["application-tests"]
    install = "Install the prebuilt application and test dependencies"
    suite_steps = {
        "contracts": ["Run repository and package contracts"],
        "unit": ["Run unit tests"],
        "cli-build": ["Run recommendation E2E without parallel resource contention", "Run remaining CLI build tests"],
        "integration-build": [
            "Run migrated integration tests against the installed wheel",
            "Run support-matrix build tests",
            "Run tool build tests",
        ],
    }
    test_steps = {name for names in suite_steps.values() for name in names}
    assert test_steps == {step["name"] for step in job["steps"] if "pytest" in step.get("run", "")}
    assert job["timeout-minutes"] == "15"
    checkout = next(step for step in job["steps"] if step.get("uses", "").startswith("actions/checkout@"))
    depths = {}
    for shard in job["strategy"]["matrix"]["shard"]:
        expected = suite_steps[shard["suite"]]

        def tests_run(**kwargs) -> list[str]:
            return [name for name in _application_leg_steps(job, shard, **kwargs) if name in test_steps]

        assert tests_run() == expected
        for failing in expected:
            assert tests_run(failing=failing) == expected
        assert tests_run(failing=install) == []
        assert tests_run(cancelled=True) == []
        depths[shard["suite"]] = _application_expression(
            checkout["with"]["fetch-depth"], shard, outcomes={}, failed=False, cancelled=False
        )
    # Repository-root contracts read Git history; the application suites do not.
    assert depths == {"contracts": 0, "unit": 1, "cli-build": 1, "integration-build": 1}


@pytest.mark.parametrize("failed", ["package", "rust", "neither"])
def test_parallel_native_preparation_propagates_either_build_failure(tmp_path, failed):
    job = _workflow("ci.yml")["jobs"]["rust-feature-modes"]
    script = next(step["run"] for step in job["steps"] if step.get("name", "").endswith("concurrently"))
    for executable, component in (("python", "package"), ("cargo", "rust")):
        path = tmp_path / executable
        path.write_text(
            f"#!/bin/sh\necho {component} >> '{tmp_path}/completed'\nexit {1 if failed == component else 0}\n"
        )
        path.chmod(0o755)
    result = subprocess.run(
        ["bash", "-c", script],
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "RUNNER_TEMP": str(tmp_path),
            "pythonLocation": str(tmp_path),
        },
        capture_output=True,
        text=True,
    )
    assert (result.returncode == 0) == (failed == "neither")
    assert set((tmp_path / "completed").read_text().splitlines()) == {"package", "rust"}


def test_pages_release_completion_still_executes_main_checkout():
    from pathlib import Path

    import yaml

    root = Path(__file__).resolve().parents[1]
    workflow = yaml.safe_load((root / ".github/workflows/pages.yml").read_text())
    trigger = workflow.get("on", workflow.get(True))
    assert trigger["workflow_run"]["branches"] == ["main", "release/**"]
    checkout = workflow["jobs"]["build"]["steps"][0]
    assert checkout["with"]["ref"] == "${{ github.event_name == 'pull_request' && github.ref || 'main' }}"
    assert checkout["with"]["fetch-depth"] == 0
    assert checkout["with"]["persist-credentials"] is False
    watched_workflows = trigger["workflow_run"]["workflows"]
    for producer in ("fpe-support-matrix.yml", "nightly-ci.yml", "release-nightly-ci.yml"):
        assert _workflow(producer)["name"] in watched_workflows
    assert "Nightly CI" in watched_workflows  # Runs started before the main workflow rename.


def test_release_nightly_discovers_versions_and_bounds_total_concurrency():
    workflow = _workflow("release-nightly-ci.yml")
    trigger = workflow.get("on", workflow.get(True))
    assert trigger["schedule"] == [{"cron": "23 9 * * *"}]
    assert "workflow_dispatch" in trigger
    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["concurrency"] == {"group": "release-nightly-ci", "cancel-in-progress": "false"}
    discover = workflow["jobs"]["discover"]
    assert discover["if"] == "github.ref == 'refs/heads/main'"
    checkout = discover["steps"][0]["with"]
    assert checkout == {"ref": "${{ github.sha }}", "fetch-depth": "0", "persist-credentials": "false"}
    assert "run_release_fpe.py list-releases" in _run_commands(discover)
    releases = workflow["jobs"]["releases"]
    assert releases["needs"] == "discover"
    assert releases["if"] == "needs.discover.outputs.releases != '[]'"
    assert releases["strategy"] == {
        "fail-fast": "false",
        "max-parallel": "1",
        "matrix": {"release": "${{ fromJSON(needs.discover.outputs.releases) }}"},
    }
    assert releases["uses"] == "./.github/workflows/fpe-release-qualify.yml"
    assert releases["with"] == {
        "release": "${{ matrix.release.version }}",
        "source_sha": "${{ matrix.release.source_sha }}",
    }


def test_release_qualification_pins_source_and_tooling_across_all_jobs():
    workflow = _workflow("fpe-release-qualify.yml")
    assert set(workflow["on"]) == {"workflow_call"}
    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["env"] == {
        "FPE_BRANCH": "release/${{ inputs.release }}",
        "FPE_SOURCE_SHA": "${{ inputs.source_sha }}",
        "FPE_TOOLING_SHA": "${{ github.sha }}",
    }
    jobs = workflow["jobs"]
    assert jobs["prepare"]["if"] == "github.ref == 'refs/heads/main'"
    assert jobs["generate"]["needs"] == "prepare"
    assert set(jobs["qualify"]["needs"]) == {"prepare", "generate"}
    for job in jobs.values():
        checkouts = [step["with"] for step in job["steps"] if step.get("uses", "").startswith("actions/checkout@")]
        assert len(checkouts) == 2
        assert checkouts[0]["ref"] == "${{ github.sha }}"
        assert checkouts[1]["ref"] == "${{ inputs.source_sha }}"
        assert checkouts[1]["path"] == "release-source"
        assert all(step["persist-credentials"] == "false" for step in checkouts)
    prepare = _run_commands(jobs["prepare"])
    assert "maturin build --release --locked" in prepare
    assert "run_release_fpe.py record-wheel" in prepare
    assert "run_release_fpe.py discover" in prepare
    assert 'project_directory(Path("release-source"))' in prepare
    assert 'uv sync --project "${project}" --locked --no-install-project --no-dev' in prepare
    generate = _run_commands(jobs["generate"])
    assert 'project_directory(Path("release-source"))' in generate
    assert 'uv sync --project "${project}" --locked --no-install-project --no-dev' in generate
    assert 'echo "RELEASE_PROJECT=${project}" >> "${GITHUB_ENV}"' in generate
    assert '"${RELEASE_PROJECT}/.venv/bin/python" scripts/fpe/run_release_fpe.py probe' in generate
    assert "run_release_fpe.py probe" in generate
    assert "--max-workers 8" in generate
    assert "--max-topologies-per-role" not in generate
    assert jobs["generate"]["strategy"]["max-parallel"] == "20"
    assert "run_release_fpe.py package" in _run_commands(jobs["qualify"])
    upload = jobs["qualify"]["steps"][-1]
    assert upload["with"]["name"] == "fpe-support-matrix-web-release-${{ inputs.release }}"
    assert upload["with"]["retention-days"] == "90"
    assert "if" not in upload  # Never publish a partially failed qualification.


def test_release_artifact_handoffs_cannot_mix_versions():
    import fnmatch

    jobs = _workflow("fpe-release-qualify.yml")["jobs"]
    prepared_artifact = jobs["prepare"]["steps"][-1]["with"]
    wheel = prepared_artifact["name"]
    assert set(prepared_artifact["path"].splitlines()) == {
        "fpe-release-wheel",
        "fpe-release-shards.json",
        "fpe-release-probes.json",
    }
    for name in ("generate", "qualify"):
        download = next(
            s
            for s in jobs[name]["steps"]
            if s.get("uses", "").endswith("/wheel-storage") and s["with"]["mode"] == "download"
        )
        assert download["with"]["name"] == wheel
    assert wheel == "fpe-release-wheel-${{ inputs.release }}"
    upload = jobs["generate"]["steps"][-1]["with"]["name"]
    pattern = next(s["with"]["pattern"] for s in jobs["qualify"]["steps"] if "pattern" in s.get("with", {}))
    versions = ["0.12.0", "0.13.0", "0.13.0-rc1", "0.13.0--preview"]
    for selected in versions:
        selected_pattern = pattern.replace("${{ inputs.release }}", selected)
        for version in versions:
            name = (
                upload.replace("${{ inputs.release }}", version)
                .replace("${{ matrix.shard.system }}", "h200_sxm")
                .replace("${{ matrix.shard.backend }}", "vllm")
            )
            assert fnmatch.fnmatchcase(name, selected_pattern) == (version == selected)


def test_release_reproduction_documents_manifest_artifact_instead_of_wheel_payload():
    jobs = _workflow("fpe-release-qualify.yml")["jobs"]
    handoff = jobs["prepare"]["steps"][-1]["with"]
    action = yaml.safe_load((ACTION_ROOT / "wheel-storage/action.yml").read_text())
    upload = next(
        step["with"] for step in action["runs"]["steps"] if "actions/upload-artifact@" in step.get("uses", "")
    )
    artifact = upload["name"].replace("${{ inputs.name }}", handoff["name"])
    artifact = artifact.replace("${{ inputs.release }}", "<version>")
    documentation = (REPOSITORY_ROOT / "docs/ci/accuracy.md").read_text()
    reproduction = documentation.split("To reproduce a release result,", 1)[1].split("\nThe deployed", 1)[0]
    assert f"`{artifact}`" in reproduction
    assert "`fpe-release-wheel-<version>`" not in reproduction
    assert "`wheel-manifest.json`" in reproduction


def test_main_fpe_preserves_discovered_probe_plan_when_packaging(tmp_path):
    jobs = _workflow("fpe-support-matrix.yml")["jobs"]
    assert jobs["discover-shards"]["outputs"]["required-probes"] == "${{ steps.shards.outputs.required_probes }}"
    package = next(
        step for step in jobs["build-web-matrix"]["steps"] if "run_release_fpe.py package" in step.get("run", "")
    )
    assert package["env"]["EXPECTED_PROBES"] == "${{ needs.discover-shards.outputs.required-probes }}"
    shards = json.dumps([{"system": "b200_sxm", "backend": "trtllm"}])
    plan = json.dumps(
        {
            "source_sha": "a" * 40,
            "tooling_sha": "b" * 40,
            "wheel_sha256": "c" * 64,
            "probes": [{"model": "literal/$(touch substituted)`touch substituted`", "backend_version": "1.3.0rc29"}],
        }
    )
    package_stub = tmp_path / "scripts/fpe/run_release_fpe.py"
    package_stub.parent.mkdir(parents=True)
    package_stub.write_text(
        "import os, pathlib, sys\n"
        "assert sys.argv[1] == 'package'\n"
        "for name, suffix in [('EXPECTED_SHARDS', 'shards'), ('EXPECTED_PROBES', 'probes')]:\n"
        "    assert pathlib.Path(f'fpe-release-{suffix}.json').read_text() == os.environ[name] + '\\n'\n"
        "pathlib.Path('packaged').touch()\n"
    )
    result = subprocess.run(
        ["bash", "-euc", package["run"]],
        cwd=tmp_path,
        env={**os.environ, "EXPECTED_SHARDS": shards, "EXPECTED_PROBES": plan},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "packaged").is_file()
    assert not (tmp_path / "substituted").exists()


@pytest.mark.parametrize("manifest_version", ["0.13.0", "0.13.0.dev20260918"])
@pytest.mark.parametrize("installed_matches", [True, False])
def test_release_smoke_checks_manifest_version(tmp_path, monkeypatch, manifest_version, installed_matches):
    import importlib.metadata

    manifest = tmp_path / "python/aisim/pyproject.toml"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(f'[project]\nversion = "{manifest_version}"\n')
    monkeypatch.chdir(tmp_path)
    installed_version = manifest_version if installed_matches else "0.12.0"
    for name in ("aisim", "aisim_core"):
        monkeypatch.setitem(sys.modules, name, SimpleNamespace(__version__=installed_version))
    monkeypatch.setattr(importlib.metadata, "version", lambda name: installed_version)
    monkeypatch.setattr(importlib.metadata, "distributions", lambda: [])
    smoke = next(
        step["run"]
        for step in _workflow("ci.yml")["jobs"]["release-artifact-contract"]["steps"]
        if "import importlib.metadata" in step.get("run", "")
    )
    command = shlex.split(smoke)
    assert command[:2] == ["release-smoke/bin/python", "-c"]
    if installed_matches:
        exec(command[2], {})
    else:
        with pytest.raises(AssertionError):
            exec(command[2], {})


def _nightly_condition(job: str, *, cancelled: bool = False, **overrides) -> bool:
    """Evaluate the checked-in predicate with explicit Actions context values."""
    values = {
        "github.event_name": "schedule",
        "github.ref": "refs/heads/main",
        "github.run_attempt": "1",
        "needs.manual-approval.result": "skipped",
        "needs.manual-approval.outputs.approved-attempt": "",
        "needs.changes-guard.outputs.should-build": "true",
        "vars.GITLAB_SECURITY_TRIGGER_ENABLED": "true",
        **{
            f"needs.{name}.result": "success"
            for name in ("build-artifacts", "python-compliance", "fpe-support-matrix", "license-evidence")
        },
        **overrides,
    }
    expression = _workflow("nightly-ci.yml")["jobs"][job]["if"]
    expression = re.sub(r"(?:github|needs|vars)\.[\w.-]+", lambda match: repr(values[match[0]]), expression)
    expression = expression.replace("!cancelled()", repr(not cancelled)).replace("&&", " and ").replace("||", " or ")
    return eval(expression, {"__builtins__": {}})


@pytest.mark.parametrize("job", ["license-evidence", "fpe-support-matrix"])
def test_nightly_validation_survives_skipped_approval_but_requires_successful_inputs(job):
    configuration = _workflow("nightly-ci.yml")["jobs"][job]
    # A status function overrides Actions' implicit success(), which would
    # otherwise reject the skipped approval ancestor of a scheduled first run.
    assert "!cancelled()" in configuration["if"]
    assert _nightly_condition(job)
    assert not _nightly_condition(job, cancelled=True)
    for dependency in configuration["needs"]:
        if dependency == "changes-guard":
            assert not _nightly_condition(job, **{"needs.changes-guard.outputs.should-build": "false"})
            continue
        for result in ("failure", "skipped", "cancelled"):
            assert not _nightly_condition(job, **{f"needs.{dependency}.result": result})


@pytest.mark.parametrize(
    "job", ["python-compliance", "build-artifacts", "slack-thread-start", "trigger-gitlab-security"]
)
def test_nightly_retries_require_approval_from_the_current_attempt(job):
    assert _nightly_condition(job)
    for attempt in ("2", "3"):
        assert not _nightly_condition(job, **{"github.run_attempt": attempt})
        assert not _nightly_condition(
            job,
            **{
                "github.run_attempt": attempt,
                "needs.manual-approval.result": "success",
                "needs.manual-approval.outputs.approved-attempt": str(int(attempt) - 1),
            },
        )
        assert _nightly_condition(
            job,
            **{
                "github.run_attempt": attempt,
                "needs.manual-approval.result": "success",
                "needs.manual-approval.outputs.approved-attempt": attempt,
            },
        )
    push = {"github.event_name": "push", "github.ref": "refs/heads/pull-request/59"}
    assert not _nightly_condition(job, **push)
    assert _nightly_condition(
        job,
        **push,
        **{
            "needs.manual-approval.result": "success",
            "needs.manual-approval.outputs.approved-attempt": "1",
        },
    ) == (job in {"python-compliance", "build-artifacts"})


def test_nightly_slack_thread_mirrors_dynamo_and_forwards_the_root_ts():
    jobs = _workflow("nightly-ci.yml")["jobs"]
    start = jobs["slack-thread-start"]
    assert start["environment"] == "automated-release"
    assert start["outputs"]["thread-ts"] == "${{ steps.post.outputs.thread-ts }}"
    assert not re.search(r"^\s*[^#\s].*::add-mask::", _run_commands(start), re.MULTILINE)
    assert not _nightly_condition("slack-thread-start", **{"vars.GITLAB_SECURITY_TRIGGER_ENABLED": "false"})
    assert not _nightly_condition("slack-thread-start", **{"needs.changes-guard.outputs.should-build": "false"})
    thread = "${{ needs.slack-thread-start.outputs.thread-ts }}"
    assert jobs["slack-staging-reply"]["steps"][0]["env"]["SLACK_THREAD_TS"] == thread
    publish = jobs["trigger-gitlab-security"]
    assert "slack-thread-start" in publish["needs"]
    names = [step.get("name") for step in publish["steps"]]
    assert "Start the Slack thread" not in names
    trigger = publish["steps"][names.index("Trigger internal security scan")]
    assert trigger["env"]["SLACK_THREAD_TS"] == thread
    assert names.index("Trigger internal security scan") < names.index("Post GitLab trigger reply on Slack thread")
    stopped = jobs["slack-thread-stopped"]
    assert {"slack-thread-start", "trigger-gitlab-security"} <= set(stopped["needs"])
    assert "needs.trigger-gitlab-security.result != 'success'" in stopped["if"]
    assert {"slack-thread-start", "slack-staging-reply", "slack-thread-stopped"} <= set(jobs["notify-slack"]["needs"])


@pytest.mark.parametrize("gate", ["build-artifacts", "fpe-support-matrix", "license-evidence"])
@pytest.mark.parametrize("result", ["failure", "skipped", "cancelled"])
def test_nightly_failed_validation_cannot_publish(gate, result):
    assert not _nightly_condition("trigger-gitlab-security", **{f"needs.{gate}.result": result})


@pytest.mark.parametrize("result", ["failure", "skipped", "cancelled"])
def test_nightly_failed_compliance_cannot_stage(result):
    assert not _nightly_condition("build-artifacts", **{"needs.python-compliance.result": result})


@pytest.mark.parametrize("attempt", ["1", "2"])
def test_manual_nightly_requires_current_approval_to_publish(attempt):
    context = {"github.event_name": "workflow_dispatch", "github.run_attempt": attempt}
    assert _nightly_condition("manual-approval", **context)
    assert not _nightly_condition("python-compliance", **context)
    assert not _nightly_condition("build-artifacts", **context)
    assert not _nightly_condition("trigger-gitlab-security", **context)
    context.update(
        {
            "needs.manual-approval.result": "success",
            "needs.manual-approval.outputs.approved-attempt": attempt,
        }
    )
    assert _nightly_condition("build-artifacts", **context)
    assert _nightly_condition("python-compliance", **context)
    assert _nightly_condition("license-evidence", **context)
    assert _nightly_condition("fpe-support-matrix", **context)
    assert _nightly_condition("trigger-gitlab-security", **context)
    context["needs.manual-approval.outputs.approved-attempt"] = str(int(attempt) - 1)
    assert not _nightly_condition("build-artifacts", **context)
    assert not _nightly_condition("python-compliance", **context)
    assert not _nightly_condition("trigger-gitlab-security", **context)


def test_nightly_compliance_requires_approval_dependency_and_respects_cancellation():
    compliance = _workflow("nightly-ci.yml")["jobs"]["python-compliance"]
    assert set(compliance["needs"]) == {"changes-guard", "manual-approval"}
    assert "!cancelled()" in compliance["if"]
    assert not _nightly_condition("python-compliance", cancelled=True)
    assert not _nightly_condition("python-compliance", **{"needs.changes-guard.outputs.should-build": "false"})


def test_manual_nightly_checks_selected_source_with_current_license_tooling():
    workflow = _workflow("nightly-ci.yml")
    jobs = workflow["jobs"]
    target = "${{ needs.changes-guard.outputs.target-sha }}"
    compliance = jobs["python-compliance"]
    checkouts = [s["with"] for s in compliance["steps"] if "actions/checkout@" in s.get("uses", "")]
    assert checkouts == [
        {"ref": "${{ github.sha }}", "persist-credentials": "false"},
        {"ref": target, "path": "source", "persist-credentials": "false"},
    ]
    assert "--pyproject source/python/aisim/pyproject.toml" in _run_commands(compliance)
    build = jobs["build-artifacts"]
    checkout = next(s for s in build["steps"] if "actions/checkout@" in s.get("uses", ""))
    assert checkout["with"]["ref"] == target
    provenance = next(s for s in build["steps"] if "GH_SHA" in s.get("env", {}))
    assert provenance["env"]["GH_SHA"] == target
    assert provenance["env"]["GH_REF"] == "${{ needs.changes-guard.outputs.target-ref }}"
    assert jobs["changes-guard"]["outputs"]["target-ref"] == "${{ steps.target.outputs.ref }}"
    assert jobs["fpe-support-matrix"]["with"]["expected_sha"] == target
    assert workflow["concurrency"]["group"] == "nightly-ci-${{ github.event_name }}"


def test_nightly_dependency_execution_cannot_modify_staged_artifacts_or_inherit_deploy_secrets():
    jobs = _workflow("nightly-ci.yml")["jobs"]
    compliance = jobs["python-compliance"]
    assert "environment" not in compliance
    assert not re.search(r"\$\{\{\s*secrets\.", json.dumps(compliance))
    assert "scripts/ci/check_python_licenses.py" in _run_commands(compliance)
    assert "--inventory" in _run_commands(compliance)
    assert not any("download-artifact@" in s.get("uses", "") for s in compliance["steps"])
    assert all(
        s["with"]["path"].endswith("/*.csv") for s in compliance["steps"] if "upload-artifact@" in s.get("uses", "")
    )
    build = jobs["build-artifacts"]
    assert "python-compliance" in build["needs"]
    assert "pip-licenses" not in _run_commands(build)
    steps = build["steps"]
    stage_index = next(i for i, s in enumerate(steps) if s.get("name") == "Stage to Artifactory")
    fetch_index = next(
        i for i, s in enumerate(steps) if s.get("name") == "Fetch the staged wheel back from Artifactory"
    )
    smoke_index = next(
        i for i, s in enumerate(steps) if s.get("name") == "Smoke-test the staged wheel on every supported Python"
    )
    assert stage_index < fetch_index < smoke_index
    for step in steps[:stage_index]:
        if "pip install" in step.get("run", ""):
            assert "--require-hashes" in step["run"]
    assert not re.search(r"\$\{\{\s*secrets\.", json.dumps(steps[smoke_index:]))
    assert "ARTIFACTORY_TOKEN" not in steps[smoke_index].get("env", {})
    evidence = jobs["license-evidence"]
    assert "environment" not in evidence
    assert "pip install" not in _run_commands(evidence)
    assert set(evidence["needs"]) == {"changes-guard", "build-artifacts", "python-compliance"}


def _nightly_license_report(
    tmp_path,
    crates,
    prior=None,
    lookup_error=None,
    workspace_members=(),
    manual_prior=None,
    artifact_pages=None,
    preceding_runs=0,
    baseline_csv=None,
    failure_stage=None,
    missing_deps=False,
    bundled=None,
    bundled_arches=("amd64", "arm64"),
):
    inventories = tmp_path / "python"
    inventories.mkdir(exist_ok=True)
    # Same runtime dependency in multiple Python/architecture inventories must
    # be deduplicated, while concurrent versions of a crate must survive.
    for arch in ("amd64", "arm64"):
        (inventories / f"{arch}-3.12.csv").write_text("Name,Version,License\nprettytable,3.16.0,BSD-3-Clause\n")
    bundled_root = tmp_path / "bundled"
    if bundled_root.exists():
        shutil.rmtree(bundled_root)
    if bundled is None:
        bundled = [
            {
                "dependency_type": "bundled_model_config",
                "name": "aisimulate_core/model_configs/example.json",
                "version": "sha256:" + "a" * 64,
                "spdx_license": "LicenseRef-Example",
            }
        ]
    for arch in bundled_arches:
        directory = bundled_root / f"bundled-configs-nightly-{arch}"
        directory.mkdir(parents=True)
        with (directory / "bundled-deps.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["dependency_type", "name", "version", "spdx_license"])
            writer.writeheader()
            writer.writerows(bundled)
    (tmp_path / "cargo-metadata.json").write_text(
        json.dumps(
            {
                "workspace_members": list(workspace_members),
                "packages": [
                    {"id": f"getrandom@{version}", "name": "getrandom", "version": version, "license": spdx}
                    for version, spdx in crates
                ],
            }
        )
    )

    def archived(rows, include_deps=True):
        prior_csv = io.StringIO()
        writer = csv.DictWriter(prior_csv, fieldnames=["dependency_type", "name", "version", "spdx_license"])
        writer.writeheader()
        writer.writerows(rows or [])
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w") as zipped:
            if include_deps:
                zipped.writestr("deps.csv", prior_csv.getvalue())
            else:
                zipped.writestr("unrelated.csv", "name\nfixture\n")
        return archive.getvalue()

    archives = {
        "https://fixture/archive": archived(prior, include_deps=not missing_deps),
        "https://fixture/manual-archive": archived(manual_prior),
    }
    if baseline_csv is not None:
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w") as zipped:
            zipped.writestr("deps.csv", baseline_csv)
        archives["https://fixture/archive"] = archive.getvalue()

    responses = []

    def urlopen(request, timeout):
        assert timeout == 30
        assert request.get_header("Authorization") == "Bearer fixture"
        if lookup_error is not None and failure_stage is None:
            raise lookup_error
        if "/runs?" in request.full_url:
            query = parse_qs(urlsplit(request.full_url).query)
            assert query["branch"] == ["main"]
            assert query["status"] == ["success"]
            assert query["per_page"] == ["10"]
            runs = [{"id": 1, "artifacts_url": "https://fixture/artifacts"}] if prior else []
            runs = [
                {"id": i + 10, "artifacts_url": "https://fixture/empty-artifacts"} for i in range(preceding_runs)
            ] + runs
            if manual_prior and query.get("event") != ["schedule"]:
                # A newer successful dispatch shares the main workflow ref,
                # but its selected release source has unrelated dependencies.
                runs.insert(0, {"id": 3, "artifacts_url": "https://fixture/manual-artifacts"})
            page = int(query.get("page", ["1"])[0])
            payload = {"workflow_runs": runs[(page - 1) * 10 : page * 10]}
        elif request.full_url.startswith("https://fixture/empty-artifacts?"):
            payload = {"artifacts": []}
        elif request.full_url.startswith("https://fixture/artifacts?"):
            if failure_stage == "artifacts":
                raise lookup_error
            query = parse_qs(urlsplit(request.full_url).query)
            assert query["per_page"] == ["100"]
            if artifact_pages is not None:
                payload = {"artifacts": artifact_pages[int(query["page"][0]) - 1]}
            else:
                payload = {
                    "artifacts": [{"name": "license-artifacts", "archive_download_url": "https://fixture/archive"}]
                }
        elif request.full_url.startswith("https://fixture/manual-artifacts?"):
            payload = {
                "artifacts": [{"name": "license-artifacts", "archive_download_url": "https://fixture/manual-archive"}]
            }
        elif request.full_url in archives:
            if failure_stage == "archive":
                raise lookup_error
            # Exercise urllib's actual redirect handling, which used to carry
            # the GitHub bearer token to the signed storage URL and cause 401.
            from urllib.request import HTTPRedirectHandler

            redirected = HTTPRedirectHandler().redirect_request(
                request, None, 302, "Found", {}, "https://signed.fixture/archive"
            )
            assert redirected.get_header("Authorization") is None
            response = io.BytesIO(archives[request.full_url])
            responses.append(response)
            return response
        else:
            raise AssertionError(request.full_url)
        response = io.BytesIO(json.dumps(payload).encode())
        responses.append(response)
        return response

    with (
        patch.dict(os.environ, {"GH_API_TOKEN": "fixture", "GITHUB_REPOSITORY": "owner/repo", "GITHUB_RUN_ID": "2"}),
        patch("urllib.request.urlopen", side_effect=urlopen),
    ):
        assert (
            license_evidence.main(
                [str(tmp_path), "--previous-nightly", "--source-sha", "a" * 40, "--source-ref", "refs/heads/main"]
            )
            == 0
        )
    assert all(response.closed for response in responses)
    with (tmp_path / "deps.csv").open() as inventory, (tmp_path / "deps-diff.csv").open() as difference:
        return list(csv.DictReader(inventory)), list(csv.DictReader(difference))


def test_nightly_license_diff_preserves_concurrent_versions_and_license_changes(tmp_path):
    original = [("0.2.17", "MIT"), ("0.3.4", "MIT"), ("0.4.3", "MIT")]
    inventory, difference = _nightly_license_report(tmp_path, original)
    assert len(inventory) == len(difference) == 5
    assert {r["version"] for r in difference if r["dependency_type"] == "crate"} == {"0.2.17", "0.3.4", "0.4.3"}
    assert all(r["change"] == "added" for r in difference)
    assert _nightly_license_report(tmp_path, original, inventory)[1] == []
    removed = _nightly_license_report(tmp_path, original[1:], inventory)[1]
    assert len(removed) == 1
    assert (removed[0]["change"], removed[0]["prior_version"]) == ("removed", "0.2.17")
    changed = _nightly_license_report(tmp_path, [("0.2.17", "Apache-2.0"), *original[1:]], inventory)[1]
    assert len(changed) == 1
    assert (changed[0]["change"], changed[0]["version"], changed[0]["prior_spdx_license"]) == (
        "changed",
        "0.2.17",
        "MIT",
    )


def test_nightly_license_baseline_ignores_newer_manual_staging(tmp_path):
    crates = [("0.2.17", "MIT")]
    scheduled, _ = _nightly_license_report(tmp_path, crates)
    manual = [{**row, "version": "99.0.0"} for row in scheduled]
    _, difference = _nightly_license_report(tmp_path, crates, scheduled, manual_prior=manual)
    assert difference == []


def test_nightly_license_first_run_uses_empty_baseline(tmp_path, capsys):
    inventory, difference = _nightly_license_report(tmp_path, [("0.2.17", "MIT")])
    assert len(inventory) == len(difference) == 3
    assert all(row["change"] == "added" for row in difference)
    assert "No prior scheduled license evidence" in capsys.readouterr().out


@pytest.mark.parametrize("has_baseline", [False, True])
def test_nightly_license_report_records_source_and_baseline_provenance(tmp_path, has_baseline):
    crates = [("0.2.17", "MIT")]
    prior, _ = _nightly_license_report(tmp_path, crates)
    _nightly_license_report(tmp_path, crates, prior if has_baseline else None)
    evidence = json.loads((tmp_path / "evidence.json").read_text())
    assert evidence["source_sha"] == "a" * 40
    assert evidence["source_ref"] == "refs/heads/main"
    assert evidence["base_sha"] == ""
    assert evidence["baseline_status"] == ("previous_scheduled_nightly" if has_baseline else "unavailable")


@pytest.mark.parametrize("preceding_runs", [10, 20])
def test_nightly_license_baseline_finds_evidence_after_artifact_free_runs(tmp_path, preceding_runs):
    crates = [("0.2.17", "MIT")]
    prior, _ = _nightly_license_report(tmp_path, crates)
    assert _nightly_license_report(tmp_path, crates, prior, preceding_runs=preceding_runs)[1] == []
    assert json.loads((tmp_path / "evidence.json").read_text())["baseline_status"] == "previous_scheduled_nightly"


def test_nightly_license_baseline_exhausts_artifact_free_run_pages(tmp_path):
    inventory, difference = _nightly_license_report(tmp_path, [("0.2.17", "MIT")], preceding_runs=10)
    assert len(inventory) == len(difference) == 3
    assert json.loads((tmp_path / "evidence.json").read_text())["baseline_status"] == "unavailable"


@pytest.mark.parametrize(
    "baseline_csv",
    [
        "",
        "dependency_type,name,version,spdx_license\n",
        "dependency_type,name,version\npython,prettytable,3.16.0\n",
        "dependency_type,name,version,spdx_license,extra\npython,prettytable,3.16.0,BSD-3-Clause,x\n",
        "dependency_type,name,version,spdx_license\npython,prettytable,3.16.0\n",
        "dependency_type,name,version,spdx_license\npython,prettytable,,BSD-3-Clause\n",
    ],
)
def test_nightly_license_baseline_rejects_invalid_inventory_before_publishing(tmp_path, baseline_csv):
    prior = [{"dependency_type": "python", "name": "prettytable", "version": "3.16.0", "spdx_license": "BSD-3-Clause"}]
    with pytest.raises(ValueError, match="invalid prior license inventory"):
        _nightly_license_report(tmp_path, [("0.2.17", "MIT")], prior, baseline_csv=baseline_csv)
    for filename in ("deps.csv", "deps-diff.csv", "evidence.json"):
        assert not (tmp_path / filename).exists()


@pytest.mark.parametrize("licenses", [("GPL-3.0-only", "MIT"), ("MIT", "GPL-3.0-only")])
def test_nightly_license_baseline_rejects_conflicting_duplicates_before_publishing(tmp_path, licenses):
    prior = [
        {"dependency_type": "crate", "name": "getrandom", "version": "0.2.17", "spdx_license": spdx}
        for spdx in licenses
    ]
    with pytest.raises(ValueError, match="conflicting license metadata"):
        _nightly_license_report(tmp_path, [("0.2.17", "MIT")], prior)
    for filename in ("deps.csv", "deps-diff.csv", "evidence.json"):
        assert not (tmp_path / filename).exists()


def test_nightly_license_baseline_accepts_identical_duplicates(tmp_path):
    crates = [("0.2.17", "MIT")]
    prior, _ = _nightly_license_report(tmp_path, crates)
    assert _nightly_license_report(tmp_path, crates, prior * 2)[1] == []


@pytest.mark.parametrize(
    "bundled",
    [[], [{"dependency_type": "python", "name": "example", "version": "1.0", "spdx_license": "Apache-2.0"}]],
)
def test_nightly_license_inventory_rejects_empty_or_wrong_type_bundled_rows(tmp_path, bundled):
    with pytest.raises(SystemExit, match="invalid bundled configuration inventory"):
        _nightly_license_report(tmp_path, [], bundled=bundled)
    assert not (tmp_path / "deps-diff.csv").exists()


def test_nightly_license_baseline_lookup_failure_does_not_publish_diff(tmp_path):
    with pytest.raises(OSError, match="baseline API unavailable"):
        _nightly_license_report(tmp_path, [("0.2.17", "MIT")], lookup_error=OSError("baseline API unavailable"))
    assert not (tmp_path / "deps-diff.csv").exists()


def test_nightly_license_baseline_rejects_archive_without_inventory():
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("evidence.json", "{}")
    responses = [
        io.BytesIO(json.dumps({"workflow_runs": [{"id": 1, "artifacts_url": "https://fixture/artifacts"}]}).encode()),
        io.BytesIO(
            json.dumps(
                {"artifacts": [{"name": "license-artifacts", "archive_download_url": "https://fixture/archive"}]}
            ).encode()
        ),
        io.BytesIO(archive.getvalue()),
    ]
    with (
        patch.dict(os.environ, {"GH_API_TOKEN": "fixture", "GITHUB_REPOSITORY": "owner/repo", "GITHUB_RUN_ID": "2"}),
        patch("urllib.request.urlopen", side_effect=responses),
        pytest.raises(RuntimeError, match="missing deps.csv"),
    ):
        license_evidence.previous_nightly()
    assert all(response.closed for response in responses)


@pytest.mark.parametrize("failure_stage", ["artifacts", "archive"])
def test_nightly_license_post_selection_failure_does_not_publish_diff(tmp_path, failure_stage):
    prior = [{"dependency_type": "crate", "name": "getrandom", "version": "0.2.17", "spdx_license": "MIT"}]
    with pytest.raises(OSError, match=f"{failure_stage} unavailable"):
        _nightly_license_report(
            tmp_path,
            [("0.2.17", "MIT")],
            prior,
            lookup_error=OSError(f"{failure_stage} unavailable"),
            failure_stage=failure_stage,
        )
    assert not (tmp_path / "deps-diff.csv").exists()


def test_nightly_license_missing_deps_csv_does_not_publish_diff(tmp_path):
    prior = [{"dependency_type": "crate", "name": "getrandom", "version": "0.2.17", "spdx_license": "MIT"}]
    with pytest.raises(RuntimeError, match="missing deps.csv"):
        _nightly_license_report(tmp_path, [("0.2.17", "MIT")], prior, missing_deps=True)
    assert not (tmp_path / "deps-diff.csv").exists()


@pytest.mark.parametrize("preceding_artifacts", [35, 100])
def test_nightly_license_baseline_finds_evidence_after_matrix_artifacts(tmp_path, preceding_artifacts):
    crates = [("0.2.17", "MIT")]
    inventory, _ = _nightly_license_report(tmp_path, crates)
    artifacts = [{"name": f"matrix-{i}"} for i in range(preceding_artifacts)]
    artifacts.append({"name": "license-artifacts", "archive_download_url": "https://fixture/archive"})
    pages = [artifacts[i : i + 100] for i in range(0, len(artifacts), 100)]
    assert _nightly_license_report(tmp_path, crates, inventory, artifact_pages=pages)[1] == []


def test_nightly_license_inventory_excludes_workspace_packages(tmp_path):
    rows, _ = _nightly_license_report(tmp_path, [("0.2.17", "MIT")], workspace_members=["getrandom@0.2.17"])
    assert all(row["dependency_type"] != "crate" for row in rows)


def test_nightly_license_inventory_rejects_conflicting_metadata(tmp_path):
    with pytest.raises(ValueError, match="conflicting license metadata"):
        _nightly_license_report(tmp_path, [("0.2.17", "MIT"), ("0.2.17", "GPL-3.0-only")])


def test_ci_license_diff_uses_exact_local_baseline_and_records_commit_identity(tmp_path):
    head = tmp_path / "head"
    base = tmp_path / "base"
    head.mkdir()
    base.mkdir()
    _nightly_license_report(head, [("0.3.4", "MIT"), ("0.4.3", "Apache-2.0")])
    _nightly_license_report(base, [("0.2.17", "MIT"), ("0.3.4", "MIT")])
    # Full CI collects dependencies without building wheels or extracting their
    # bundled configuration inventories. Nightly alone requires those inputs.
    for directory in (head, base):
        shutil.rmtree(directory / "bundled")
    with patch("urllib.request.urlopen", side_effect=AssertionError("CI must not select a nightly baseline")):
        assert (
            license_evidence.main(
                [
                    str(head),
                    "--base-dir",
                    str(base),
                    "--source-sha",
                    "a" * 40,
                    "--base-sha",
                    "b" * 40,
                    "--source-ref",
                    "refs/heads/pull-request/123",
                ]
            )
            == 0
        )
    with (head / "deps-diff.csv").open() as stream:
        changes = list(csv.DictReader(stream))
    assert {(row["change"], row["version"], row["prior_version"]) for row in changes} == {
        ("added", "0.4.3", ""),
        ("removed", "", "0.2.17"),
    }
    evidence = json.loads((head / "evidence.json").read_text())
    assert evidence["source_sha"] == "a" * 40
    assert evidence["base_sha"] == "b" * 40
    assert evidence["baseline_status"] == "exact_commit"


def test_ci_license_evidence_requires_complete_baseline_inventories(tmp_path):
    head = tmp_path / "head"
    head.mkdir()
    _nightly_license_report(head, [("0.2.17", "MIT")])
    (head / "deps-diff.csv").unlink()
    with pytest.raises(ValueError, match="missing Python dependency inventories"):
        license_evidence.main([str(head), "--base-dir", str(tmp_path / "missing-base")])
    assert not (head / "deps-diff.csv").exists()


@pytest.mark.parametrize("input_side", ["head", "base"])
@pytest.mark.parametrize(
    "python_row",
    [
        ",1.0,MIT",
        " ,1.0,MIT",
        "example,,MIT",
        "example, ,MIT",
        "example,1.0,",
        "example,1.0, \t",
        "example,1.0",
    ],
)
def test_ci_license_evidence_rejects_incomplete_local_rows_before_publishing(tmp_path, input_side, python_row):
    for side in ("head", "base"):
        directory = tmp_path / side
        (directory / "python").mkdir(parents=True)
        (directory / "python" / "inventory.csv").write_text("Name,Version,License\nexample,1.0,MIT\n")
        (directory / "cargo-metadata.json").write_text('{"packages": []}')
    (tmp_path / input_side / "python" / "inventory.csv").write_text(f"Name,Version,License\n{python_row}\n")

    with pytest.raises(ValueError, match="missing or empty required fields"):
        license_evidence.main([str(tmp_path / "head"), "--base-dir", str(tmp_path / "base")])
    for filename in ("deps.csv", "deps-diff.csv", "evidence.json"):
        assert not (tmp_path / "head" / filename).exists()


def test_full_ci_evidence_runs_independently_of_build_scope_and_uses_shared_generator():
    jobs = _workflow("ci.yml")["jobs"]
    python = jobs["python-compliance"]
    assert "outputs.application_wheel" not in python["if"]
    head = next(step for step in python["steps"] if step.get("name") == "Check Python dependency licenses")
    assert "--inventory" in head["run"] and "--inventory-only" not in head["run"]
    assert "--inventory-only" in _run_commands(python)
    for job_name in ("python-compliance", "license-crate-inventory"):
        assert "select-full-ci" in jobs[job_name]["needs"]
        assert any(
            step.get("with", {}).get("ref") == "${{ needs.select-full-ci.outputs.comparison-base }}"
            for step in jobs[job_name]["steps"]
        )
    evidence = jobs["license-evidence"]
    assert set(evidence["needs"]) == {
        "fast-ci",
        "select-full-ci",
        "verify-target",
        "python-compliance",
        "license-crate-inventory",
    }
    commands = _run_commands(evidence)
    assert "--base-dir" in commands and "--previous-nightly" not in commands
    assert "pip install" not in commands and "cargo " not in commands
    for job in (python, jobs["license-crate-inventory"], evidence):
        assert "environment" not in job
        assert "secrets." not in json.dumps(job)
    for workflow in ("ci.yml", "nightly-ci.yml"):
        steps = _workflow(workflow)["jobs"]["license-evidence"]["steps"]
        assert "scripts/ci/generate_license_evidence.py" in _run_commands({"steps": steps})
        upload = next(step for step in steps if "upload-artifact@" in step.get("uses", ""))
        assert upload["with"]["name"] == "license-artifacts"
        assert upload["with"]["overwrite"] == "true"
        directory = "license-inventories/head" if workflow == "ci.yml" else "license-artifacts"
        assert "${{ runner.temp }}/" + directory + "/evidence.json" in upload["with"]["path"]
    python_commands = _run_commands(python)
    assert "python/aisim/pyproject.toml" in python_commands
    assert "python/aisimulate/" not in python_commands
    nightly_report = next(
        step
        for step in _workflow("nightly-ci.yml")["jobs"]["license-evidence"]["steps"]
        if step.get("name") == "Generate license compliance evidence"
    )
    assert nightly_report["env"]["SOURCE_SHA"] == "${{ needs.changes-guard.outputs.target-sha }}"
    assert nightly_report["env"]["SOURCE_REF"] == "${{ needs.changes-guard.outputs.target-ref }}"
    assert '--source-sha "${SOURCE_SHA}"' in nightly_report["run"]
    assert '--source-ref "${SOURCE_REF}"' in nightly_report["run"]
    nightly_checkout = next(
        step
        for step in _workflow("nightly-ci.yml")["jobs"]["license-evidence"]["steps"]
        if "checkout@" in step.get("uses", "")
    )
    assert nightly_checkout["with"]["ref"] == "${{ github.workflow_sha }}"


@pytest.mark.parametrize("job", ["python-compliance", "license-crate-inventory", "license-evidence"])
def test_missing_license_evidence_blocks_even_documentation_only_full_ci(job):
    jobs = _workflow("ci.yml")["jobs"]
    plan = dict.fromkeys(COMPONENTS, "false")
    results = dict.fromkeys(jobs["readiness"]["needs"], "success")
    for component in COMPONENTS:
        results[component.replace("_", "-")] = "skipped"
    results["application-test-wheel"] = "skipped"
    results["op-latency-data"] = "skipped"  # no job that reads operation tables is selected
    assert _run_full_ci_aggregate(results, plan).returncode == 0
    results[job] = "skipped"
    failed = _run_full_ci_aggregate(results, plan)
    assert failed.returncode != 0
    assert f"{job}=skipped, expected success" in failed.stdout


def test_nightly_osrb_reports_include_bundled_configs_once_and_diff_byte_changes(tmp_path):
    before, _ = _nightly_license_report(tmp_path, [])
    bundled = [row for row in before if row["dependency_type"] == "bundled_model_config"]
    assert len(bundled) == 1
    assert bundled[0]["spdx_license"] == "LicenseRef-Example"
    updated = [{**bundled[0], "version": "sha256:" + "b" * 64}]
    _, difference = _nightly_license_report(tmp_path, [], before, bundled=updated)
    assert {row["change"] for row in difference} == {"added", "removed"}
    assert all(row["dependency_type"] == "bundled_model_config" for row in difference)
    unknown = [{**bundled[0], "spdx_license": "UNKNOWN"}]
    inventory, _ = _nightly_license_report(tmp_path, [], bundled=unknown)
    assert (
        next(row for row in inventory if row["dependency_type"] == "bundled_model_config")["spdx_license"] == "UNKNOWN"
    )


@pytest.mark.parametrize("arches", [(), ("amd64",), ("arm64",), ("amd64", "other")])
def test_nightly_osrb_reports_require_both_architecture_inventories(tmp_path, arches):
    with pytest.raises(SystemExit, match="architecture-specific bundled configuration"):
        _nightly_license_report(tmp_path, [], bundled_arches=arches)


def test_nightly_osrb_reports_reject_empty_bundled_inventory(tmp_path):
    with pytest.raises(SystemExit, match="invalid bundled configuration inventory"):
        _nightly_license_report(tmp_path, [], bundled=[])


@pytest.mark.parametrize("package", ["aisim", "aisimulate"])
@pytest.mark.parametrize("kind", ["application", "platform", "nightly"])
def test_bundled_config_gate_selects_current_and_historical_wheels(tmp_path, package, kind):
    if kind == "platform":
        action = yaml.load((ACTION_ROOT / "build-platform-wheel/action.yml").read_text(), Loader=yaml.BaseLoader)
        steps = action["runs"]["steps"]
        directory = "wheelhouse"
    else:
        workflow, job, directory = (
            ("ci.yml", "application-test-wheel", "application-test-dist")
            if kind == "application"
            else ("nightly-ci.yml", "build-artifacts", "nightly-dist")
        )
        steps = _workflow(workflow)["jobs"][job]["steps"]
    command = next(s["run"] for s in steps if s.get("name") == "Check exact wheel configuration provenance")
    wheel_dir = tmp_path / directory
    wheel_dir.mkdir()
    (wheel_dir / f"{package}-0.14.0-cp311-abi3-manylinux_2_28_x86_64.whl").touch()
    capture = """
python() {
    while [ "$#" -gt 0 ]; do
        if [ "$1" = "--wheel" ]; then
            test -f "$2"
            return
        fi
        shift
    done
    return 1
}
"""
    tools = tmp_path / "tools/venv-build/bin"
    tools.mkdir(parents=True)
    executable = tools / "python"
    executable.write_text("#!/bin/bash\n" + capture + '\npython "$@"\n')
    executable.chmod(0o755)
    result = subprocess.run(
        ["bash", "-e", "-c", capture + command],
        cwd=tmp_path,
        env={**os.environ, "DIST_DIR": directory, "TOOLS_DIR": str(tmp_path / "tools"), "RUNNER_TEMP": str(tmp_path)},
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr


def test_bundled_config_gates_run_before_wheel_publication_or_installation(tmp_path):
    policy = _workflow("fast-ci.yml")["jobs"]["policy"]
    assert "scripts/ci/check_bundled_configs.py" in _run_commands(policy)
    assert "tests/test_bundled_configs.py" in _run_commands(policy)
    application = _workflow("ci.yml")["jobs"]["application-test-wheel"]["steps"]
    check = next(
        i for i, step in enumerate(application) if step.get("name") == "Check exact wheel configuration provenance"
    )
    upload = next(
        i
        for i, step in enumerate(application)
        if step.get("with", {}).get("name", "").startswith("application-test-wheel-")
    )
    assert check < upload
    assert "check_bundled_configs.py extract --check" in application[check]["run"]
    action = yaml.load((ACTION_ROOT / "build-platform-wheel/action.yml").read_text(), Loader=yaml.BaseLoader)
    steps = action["runs"]["steps"]
    check = next(i for i, step in enumerate(steps) if step.get("name") == "Check exact wheel configuration provenance")
    install = next(i for i, step in enumerate(steps) if step.get("name") == "Verify installed unified wheel")
    assert check < install
    assert "check_bundled_configs.py extract --check" in steps[check]["run"]
    assert "if" not in steps[check]  # Both Linux architectures and native macOS.
    steps = _workflow("nightly-ci.yml")["jobs"]["build-artifacts"]["steps"]
    check = next(i for i, step in enumerate(steps) if step.get("name") == "Check exact wheel configuration provenance")
    stage = next(i for i, step in enumerate(steps) if step.get("name") == "Stage to Artifactory")
    assert check < stage
    assert "check_bundled_configs.py extract --check" in steps[check]["run"]
    assert "release-tooling/scripts/ci/check_bundled_configs.py" in steps[check]["run"]
    for kind, collection in zip(
        ("application", "platform", "nightly"), (application, action["runs"]["steps"], steps), strict=True
    ):
        upload_report = next(
            step for step in collection if step.get("name") == "Preserve wheel configuration inventory"
        )
        pattern = upload_report["with"]["path"]
        assert len(pattern.splitlines()) == 1
        assert pattern.startswith("${{ runner.temp }}/")
        root = tmp_path / f"bundled-configs-{kind}"
        payload = root / "files/model_configs"
        payload.mkdir(parents=True)
        reports = {"bundled-deps.csv", "bundled-provenance.json", "extracted-files.json"}
        for name in reports:
            (root / name).touch()
        for name in ("nested.csv", "nested.json"):
            (payload / name).touch()
        # Check the upload path's intended file selection against a fixture tree.
        assert set(tmp_path.glob(pattern.removeprefix("${{ runner.temp }}/"))) == {root / name for name in reports}
    tooling = next(step for step in steps if step.get("with", {}).get("path") == "release-tooling")
    for file in ["check_bundled_configs.py", "bundled_configs.json", "bundled_config_legacy.json"]:
        assert "/scripts/ci/" + file in tooling["with"]["sparse-checkout"]


@pytest.mark.parametrize(
    "conclusion,expected", [("failure", True), ("timed_out", True), ("success", False), ("skipped", False)]
)
def test_nightly_alert_collector_includes_timeouts(tmp_path, conclusion, expected):
    step = next(s for s in _workflow("nightly-ci.yml")["jobs"]["notify-slack"]["steps"] if s.get("id") == "failed")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_curl = fake_bin / "curl"
    fake_curl.write_text('#!/bin/sh\nprintf "%s\\n" "$FIXTURE_JOBS"\n')
    fake_curl.chmod(0o755)
    output = tmp_path / "output"
    result = subprocess.run(
        ["bash", "-e", "-c", step["run"]],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "GH_TOKEN": "fixture",
            "GITHUB_REPOSITORY": "owner/repo",
            "GITHUB_RUN_ID": "1",
            "GITHUB_ENV": str(tmp_path / "env"),
            "GITHUB_OUTPUT": str(output),
            "MENTION_IDS": "",
            "FIXTURE_JOBS": json.dumps({"jobs": [{"name": "Build artifacts", "conclusion": conclusion}]}),
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert output.read_text().strip() == f"has_failures={str(expected).lower()}"


def test_full_ci_license_failure_blocks_readiness_and_staging():
    jobs = _workflow("ci.yml")["jobs"]
    assert "python-compliance" in jobs["readiness"]["needs"]
    assert "python-compliance" in jobs["application-wheel"]["needs"]
    assert "readiness" in jobs["stage-application-wheel"]["needs"]
    plan = dict.fromkeys(COMPONENTS, "true")
    results = dict.fromkeys(jobs["readiness"]["needs"], "success")
    passed = _run_full_ci_aggregate(results, plan)
    assert passed.returncode == 0, passed.stdout + passed.stderr
    results["python-compliance"] = "failure"
    failed = _run_full_ci_aggregate(results, plan)
    assert failed.returncode != 0
    assert "python-compliance=failure, expected success" in failed.stdout


def test_python_compliance_workflows_use_the_same_policy():
    for workflow in ("ci.yml", "nightly-ci.yml"):
        commands = _run_commands(_workflow(workflow)["jobs"]["python-compliance"])
        assert "python scripts/ci/check_python_licenses.py" in commands
        assert "--allow-only" not in commands


def test_nightly_license_matrix_covers_the_smoked_python_versions():
    jobs = _workflow("nightly-ci.yml")["jobs"]
    compliance = jobs["python-compliance"]
    matrix = compliance["strategy"]["matrix"]
    assert set(matrix["python-version"]) == set(jobs["build-artifacts"]["env"]["PYTHON_SERIES_LIST"].split())
    assert set(matrix["arch"]) == {entry["arch"] for entry in jobs["build-artifacts"]["strategy"]["matrix"]["include"]}
    setup = next(step for step in compliance["steps"] if "setup-python@" in step.get("uses", ""))
    assert setup["with"]["python-version"] == "${{ matrix.python-version }}"
    upload = next(step for step in compliance["steps"] if "upload-artifact@" in step.get("uses", ""))
    assert upload["with"]["name"] == "nightly-python-inventory-${{ matrix.arch }}-${{ matrix.python-version }}"


@pytest.mark.parametrize(
    "metadata_field,spdx,allowed",
    [
        ("License", "MIT", True),
        ("License", "GPL-3.0-only", False),
        ("License-Expression", "BSD-3-Clause AND ISC", True),
        ("License-Expression", "BSD-3-Clause AND GPL-3.0-only", False),
    ],
)
def test_real_pip_licenses_enforces_the_policy(tmp_path, metadata_field, spdx, allowed):
    assert importlib.metadata.version("pip-licenses") == "5.5.5"
    metadata = tmp_path / "license_policy_fixture-1.0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(
        f"Metadata-Version: 2.4\nName: license-policy-fixture\nVersion: 1.0\n{metadata_field}: {spdx}\n"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "piplicenses",
            "--with-system",
            "--packages",
            "license-policy-fixture",
            "--allow-only",
            python_licenses.ALLOWED_LICENSES,
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(tmp_path)},
        text=True,
        capture_output=True,
    )
    assert (result.returncode == 0) == allowed, result.stdout + result.stderr
    assert "license-policy-fixture" in result.stdout + result.stderr


@pytest.mark.parametrize("license_result", [0, 1])
@pytest.mark.parametrize("manifest_selection", ["default", "explicit_cli"])
def test_python_license_gate_checks_target_environment_before_export(
    tmp_path, monkeypatch, capsys, license_result, manifest_selection
):
    manifest = tmp_path / "pyproject.toml"
    manifest.write_text('[project]\ndependencies = ["prettytable>=3", "wcwidth", "aisim-core==1.0"]\n')
    # A historical source needs only its manifest; the policy/tool lives in
    # the workflow checkout. A missing default detects accidental fallback.
    monkeypatch.setattr(
        python_licenses, "PYPROJECT", manifest if manifest_selection == "default" else tmp_path / "missing"
    )
    python = "/fixture/venv/bin/python"
    inventory = tmp_path / "inventory" / "licenses.csv"
    csv_output = "Name,Version,License\nprettytable,3.16.0,BSD-3-Clause\n"
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        assert command[0] == python
        if command[1:4] == ["-m", "pip", "install"]:
            requirements = Path(command[command.index("-r") + 1]).read_text().splitlines()
            assert requirements == ["prettytable>=3", "wcwidth"]
            assert command[-2:] == ["pip-licenses==5.5.5", "setuptools>=84"]
            assert kwargs["check"]
            return SimpleNamespace(returncode=0)
        assert command[1:4] == ["-m", "piplicenses", "--with-system"]
        if "--allow-only" in command:
            assert kwargs["stdout"] == kwargs["stderr"] == subprocess.DEVNULL
            return SimpleNamespace(returncode=license_result)
        assert "--format=csv" in command and kwargs["check"]
        kwargs["stdout"].write(csv_output)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(python_licenses.subprocess, "run", run)
    if manifest_selection == "default":
        assert python_licenses.check_licenses(python, inventory) == license_result
    else:
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "check_python_licenses.py",
                "--python",
                python,
                "--inventory",
                str(inventory),
                "--pyproject",
                str(manifest),
            ],
        )
        assert python_licenses.main() == license_result
    if license_result:
        assert len(calls) == 2
        assert not inventory.exists()
        assert "::error::" in capsys.readouterr().out
    else:
        assert inventory.read_text() == csv_output


def test_python_license_install_failure_blocks_check_and_export(tmp_path, monkeypatch):
    manifest = tmp_path / "pyproject.toml"
    manifest.write_text('[project]\ndependencies = ["prettytable>=3"]\n')
    monkeypatch.setattr(python_licenses, "PYPROJECT", manifest)
    inventory = tmp_path / "inventory.csv"
    with (
        patch.object(python_licenses.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "pip")) as run,
        pytest.raises(subprocess.CalledProcessError),
    ):
        python_licenses.check_licenses("python", inventory)
    assert run.call_count == 1
    assert not inventory.exists()


def test_comparison_inventory_can_record_a_license_removed_by_the_head(tmp_path, monkeypatch):
    manifest = tmp_path / "pyproject.toml"
    manifest.write_text('[project]\ndependencies = ["old-dependency"]\n')
    output = tmp_path / "licenses.csv"
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        assert "--allow-only" not in command
        if "--format=csv" in command:
            kwargs["stdout"].write("Name,Version,License\nold-dependency,1.0,GPL-3.0-only\n")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(python_licenses.subprocess, "run", run)
    assert python_licenses.check_licenses("python", output, manifest, inventory_only=True) == 0
    assert len(calls) == 2
    assert "GPL-3.0-only" in output.read_text()
    with pytest.raises(ValueError, match="output path"):
        python_licenses.check_licenses("python", pyproject=manifest, inventory_only=True)


def test_nightly_artifact_handoff_matches_fpe_and_accuracy_consumers():
    jobs = _workflow("nightly-ci.yml")["jobs"]
    steps = jobs["build-artifacts"]["steps"]
    names = [step.get("name") for step in steps]
    assert names.index("Write checksums and provenance") < names.index("Stage to Artifactory")
    assert (
        names.index("Fetch the staged wheel back from Artifactory")
        < names.index("Upload verified nightly artifacts")
        < names.index("Smoke-test the staged wheel on every supported Python")
    )
    upload = steps[names.index("Upload verified nightly artifacts")]["with"]
    assert upload["name"] == "nightly-dist-${{ matrix.arch }}"
    assert upload["path"] == "${{ runner.temp }}/nightly-dist/*"
    assert upload["mode"] == "upload"
    artifact = upload["name"].replace("${{ matrix.arch }}", "amd64")
    assert jobs["fpe-support-matrix"]["with"]["wheel_artifact"] == artifact
    resolver = _workflow("e2e-accuracy.yml")["jobs"]["resolve"]["steps"][0]["with"]["script"]
    assert f"artifact.name === '{artifact}-manifest'" in resolver
    consumer = _workflow("e2e-accuracy-branch.yml")["jobs"]["wheel"]["steps"]
    download = next(
        step
        for step in consumer
        if step.get("uses", "").endswith("/wheel-storage") and step["with"]["mode"] == "download"
    )
    assert download["with"]["name"] == artifact


@pytest.mark.parametrize(
    "source_ref,event",
    [
        ("refs/heads/main", "schedule"),
        ("refs/heads/main", "workflow_dispatch"),
        ("refs/heads/release/0.12.0", "workflow_dispatch"),
    ],
)
@pytest.mark.parametrize("package", ["aisim", "aisimulate"])
@pytest.mark.parametrize("consumer_workflow", ["e2e-accuracy-branch.yml", "fpm-accuracy-branch.yml"])
def test_nightly_provenance_and_checksums_pass_real_accuracy_consumer(
    tmp_path, source_ref, event, package, consumer_workflow
):
    (tmp_path / "scripts").symlink_to(REPOSITORY_ROOT / "scripts", target_is_directory=True)
    directory = tmp_path / "accuracy-wheel"
    directory.mkdir()
    wheel = directory / f"{package}-0.12.0.dev20260917-cp311-abi3-manylinux_2_28_x86_64.whl"
    wheel.write_bytes(b"fixture wheel")
    (directory / f"{package}-core-0.12.0-dev.20260917.crate").write_bytes(b"fixture crate")
    manifest = tmp_path / "python/aisim/pyproject.toml"
    manifest.parent.mkdir(parents=True)
    manifest.write_text('[project]\nversion = "0.12.0.dev20260917"\n')
    python = tmp_path / "tools/venv-build/bin/python"
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    # Only distribution metadata is needed by the real provenance writer.
    metadata = tmp_path / "metadata/maturin-1.12.0.dist-info"
    metadata.mkdir(parents=True)
    (metadata / "METADATA").write_text("Metadata-Version: 2.1\nName: maturin\nVersion: 1.12.0\n")
    env = {
        **os.environ,
        "PYTHONPATH": str(metadata.parent),
        "DIST_DIR": str(directory),
        "TOOLS_DIR": str(tmp_path / "tools"),
        "RUNNER_TEMP": str(tmp_path),
        "MATRIX_ARCH": "amd64",
        "MATRIX_RUNNER": "fixture",
        "CONTAINER_IMAGE": "fixture@sha256:" + "b" * 64,
        "GH_REPOSITORY": "ai-dynamo/aisim",
        "GH_WORKFLOW_REF": "ai-dynamo/aisim/.github/workflows/nightly-ci.yml@refs/heads/main",
        "GH_REF": source_ref,
        "GH_SHA": "a" * 40,
        "GH_RUN_ID": "123",
        "GH_RUN_ATTEMPT": "1",
        "GH_EVENT_NAME": event,
        "GH_SERVER_URL": "https://github.com",
        "RUST_TOOLCHAIN": "1.98.0",
        "UV_VERSION": "0.13.10",
        "GITHUB_STEP_SUMMARY": str(tmp_path / "summary"),
        "GITHUB_OUTPUT": str(tmp_path / "output"),
        "EXPECTED_SHA": "a" * 40,
        "NIGHTLY_RUN": "123",
    }
    producer = next(
        step["run"]
        for step in _workflow("nightly-ci.yml")["jobs"]["build-artifacts"]["steps"]
        if step.get("name") == "Write checksums and provenance"
    )
    result = subprocess.run(["bash", "-e", "-c", producer], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    provenance = json.loads((directory / "provenance.json").read_text())
    assert provenance["ref"] == source_ref
    assert provenance["commit"] == env["GH_SHA"]
    assert provenance["event"] == event
    assert provenance["workflow_ref"].endswith("@refs/heads/main")
    assert provenance["version"] == "0.12.0.dev20260917"
    assert provenance["artifacts"][wheel.name] == hashlib.sha256(wheel.read_bytes()).hexdigest()
    consumer = next(
        step["run"] for step in _workflow(consumer_workflow)["jobs"]["wheel"]["steps"] if step.get("id") == "verify"
    ).replace("/opt/python/cp312-cp312/bin/python", shlex.quote(sys.executable))

    def verify():
        return subprocess.run(["bash", "-e", "-c", consumer], cwd=tmp_path, env=env, capture_output=True, text=True)

    result = verify()
    if event == "workflow_dispatch":
        # Manual staging is not a scheduled, qualified nightly producer.
        assert result.returncode != 0
        assert "nightly wheel provenance mismatch" in result.stderr
        return
    assert result.returncode == 0, result.stdout + result.stderr
    # Migration window: accept legacy aisimulate producer provenance too.
    provenance["repository"] = "ai-dynamo/aisimulate"
    (directory / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    checksums = {
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in (wheel.name, "provenance.json")
    }
    (directory / "SHA256SUMS.txt").write_text("".join(f"{digest}  {name}\n" for name, digest in checksums.items()))
    assert verify().returncode == 0, "legacy ai-dynamo/aisimulate provenance must still verify"
    provenance["repository"] = "ai-dynamo/unrelated"
    (directory / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    checksums["provenance.json"] = hashlib.sha256((directory / "provenance.json").read_bytes()).hexdigest()
    (directory / "SHA256SUMS.txt").write_text("".join(f"{digest}  {name}\n" for name, digest in checksums.items()))
    assert verify().returncode != 0
    assert "nightly wheel provenance mismatch" in verify().stderr
    provenance["repository"] = "ai-dynamo/aisim"
    (directory / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    checksums["provenance.json"] = hashlib.sha256((directory / "provenance.json").read_bytes()).hexdigest()
    (directory / "SHA256SUMS.txt").write_text("".join(f"{digest}  {name}\n" for name, digest in checksums.items()))
    wheel.write_bytes(b"tampered wheel")
    assert verify().returncode != 0
    wheel.write_bytes(b"fixture wheel")
    env["EXPECTED_SHA"] = "c" * 40
    assert verify().returncode != 0


@pytest.mark.parametrize("failure", [None, "missing-token", "http-error"])
@pytest.mark.parametrize("package", ["aisim", "aisimulate"])
def test_gitlab_security_trigger_matches_the_verified_consumer_contract(tmp_path, failure, package):
    # Interface verified against release-automation commit
    # cc63ecf80165209f630d4a5b3868983e980c6cc2 (MR !84): .gitlab-ci.yml forwarding
    # and projects/aisim.yml. This is an independent request fixture,
    # not a copy of the external pipeline implementation.
    capture = tmp_path / "request.json"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    curl = fake_bin / "curl"
    curl.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "Path(os.environ['CAPTURE_REQUEST']).write_text(json.dumps(sys.argv[1:]))\n"
        "sys.exit(int(os.environ['CURL_EXIT_CODE']))\n"
    )
    curl.chmod(0o755)
    step = next(
        step
        for step in _workflow("nightly-ci.yml")["jobs"]["trigger-gitlab-security"]["steps"]
        if step.get("name") == "Trigger internal security scan"
    )
    endpoint = "https://gitlab.invalid/api/v4/projects/123/trigger/pipeline"
    result = subprocess.run(
        ["bash", "-e", "-c", step["run"]],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "CAPTURE_REQUEST": str(capture),
            "CURL_EXIT_CODE": "22" if failure == "http-error" else "0",
            "GITLAB_TRIGGER_TOKEN": "" if failure == "missing-token" else "fixture-token",
            "GITLAB_PIPELINE_URL": endpoint,
            "WHEEL_VERSION": "0.12.0.dev202609170000001234",
            "WHEEL_PACKAGE": package,
            "TOOLING_SHA": "b" * 40,
            "GH_RUN_ID": "123456",
            "GH_SHA": "a" * 40,
            "SLACK_THREAD_TS": "1234567890.123456",
            "SLACK_CHANNEL_ID": "fixture-channel",
            "RUNNER_TEMP": str(tmp_path),
            "GITHUB_OUTPUT": str(tmp_path / "output"),
        },
    )
    assert (result.returncode == 0) == (failure is None), result.stdout + result.stderr
    if failure == "missing-token":
        assert not capture.exists()
        return
    args = json.loads(capture.read_text())
    assert args[-1] == endpoint
    assert "--fail" in args
    assert args[args.index("--output") + 1] == str(tmp_path / "gitlab-trigger.json")
    fields = dict(args[i + 1].split("=", 1) for i, arg in enumerate(args) if arg == "-F")
    assert fields == {
        "token": "fixture-token",
        "ref": "main",
        "variables[PROJECT]": package,
        "variables[PIPELINE_TYPE]": "security",
        "variables[RELEASE_TYPE]": "nightly",
        "variables[NIGHTLY_TAG]": "nightly-202609170000001234-aaaaaaa",
        "variables[WHEEL_VERSION]": "0.12.0.dev202609170000001234",
        "variables[GITHUB_RUN_ID]": "123456",
        "variables[COMMIT_SHA]": "a" * 40,
        "variables[AISIMULATE_TOOLING_SHA]": "b" * 40,
        "variables[SLACK_THREAD_TS]": "1234567890.123456",
        "variables[SLACK_CHANNEL_ID]": "fixture-channel",
        "variables[DRY_RUN]": "false",
    }


@pytest.mark.parametrize("attempt", ["1", "2"])
@pytest.mark.parametrize("gate", ["build-artifacts", "fpe-support-matrix", "license-evidence"])
@pytest.mark.parametrize("result", ["failure", "skipped", "cancelled"])
def test_manual_publish_rejects_failed_dependencies(attempt, gate, result):
    assert not _nightly_condition(
        "trigger-gitlab-security",
        **{
            "github.event_name": "workflow_dispatch",
            "github.run_attempt": attempt,
            "needs.manual-approval.result": "success",
            "needs.manual-approval.outputs.approved-attempt": attempt,
            f"needs.{gate}.result": result,
        },
    )


def test_manual_publish_rejects_wrong_ref_disabled_trigger_and_cancellation():
    approved = {
        "github.event_name": "workflow_dispatch",
        "needs.manual-approval.result": "success",
        "needs.manual-approval.outputs.approved-attempt": "1",
    }
    assert not _nightly_condition("trigger-gitlab-security", cancelled=True, **approved)
    for override in ({"github.ref": "refs/heads/release/0.12.0"}, {"vars.GITLAB_SECURITY_TRIGGER_ENABLED": "false"}):
        assert not _nightly_condition("trigger-gitlab-security", **{**approved, **override})


def _nightly_version(created, number):
    script = next(
        step["with"]["script"]
        for step in _workflow("nightly-ci.yml")["jobs"]["changes-guard"]["steps"]
        if step.get("id") == "version"
    )
    program = """
const run = JSON.parse(process.argv[1]);
const output = {};
const core = {setOutput: (key, value) => { output[key] = value; }};
const github = {rest: {actions: {getWorkflowRun: async () => ({data: run})}}};
const context = {repo: {owner: 'fixture', repo: 'fixture'}, runId: 123};
const AsyncFunction = Object.getPrototypeOf(async function(){}).constructor;
new AsyncFunction('github', 'context', 'core', process.argv[2])(github, context, core)
  .then(() => console.log(JSON.stringify(output)))
  .catch(error => { console.error(error.message); process.exitCode = 1; });
"""
    return subprocess.run(
        [shutil.which("node"), "-e", program, json.dumps({"created_at": created, "run_number": number}), script],
        text=True,
        capture_output=True,
    )


def _current_product_version():
    manifest = tomllib.loads((REPOSITORY_ROOT / "python/aisim/pyproject.toml").read_text())
    return manifest["project"]["version"]


def test_nightly_versions_are_unique_date_ordered_and_stable_across_retries():
    from packaging.version import Version

    versions = []
    for date, number in [
        ("2026-09-17T23:59:59Z", 1234),
        ("2026-09-17T23:59:59Z", 1235),
        ("2026-09-18T00:00:00Z", 1236),
    ]:
        result = _nightly_version(date, number)
        assert result.returncode == 0, result.stderr
        value = json.loads(result.stdout)
        assert value["dev-date"] == date[:10].replace("-", "")
        versions.append(value["dev-version"])
        assert json.loads(_nightly_version(date, number).stdout) == value
    assert versions == ["202609170000001234", "202609170000001235", "202609180000001236"]
    base_version = _current_product_version()
    assert Version(f"{base_version}.dev20260917") < Version(f"{base_version}.dev{versions[0]}")
    stamped_versions = [Version(f"{base_version}.dev{version}") for version in versions]
    assert stamped_versions == sorted(stamped_versions)
    for number in (0, -1, 10000000000, "invalid"):
        assert _nightly_version("2026-09-17T00:00:00Z", number).returncode != 0


@pytest.mark.parametrize("suffix", [".dev20260917", ".dev202609170000001234"])
@pytest.mark.parametrize("base_version", [None, "0.12.0"], ids=["current", "historical"])
@pytest.mark.parametrize("package", ["aisim", "aisimulate"])
@pytest.mark.parametrize("inherit_version", [True, False], ids=["workspace-version", "explicit-version"])
def test_current_release_tools_stamp_and_validate_historical_manifests(
    tmp_path, suffix, base_version, package, inherit_version
):
    current_version = _current_product_version()
    base_version = base_version or current_version
    for name in ("Cargo.toml", "crates/core/Cargo.toml", "python/aisim/pyproject.toml", "scripts/pyproject.toml"):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            (REPOSITORY_ROOT / name)
            .read_text()
            .replace(f'version = "{current_version}"', f'version = "{base_version}"')
        )
    core_manifest = tmp_path / "crates/core/Cargo.toml"
    if not inherit_version:
        core_manifest.write_text(
            core_manifest.read_text().replace("version.workspace = true", f'version = "{base_version}"')
        )
    if package == "aisimulate":
        for name in ("crates/core/Cargo.toml", "python/aisim/pyproject.toml"):
            target = tmp_path / name
            target.write_text(
                target.read_text()
                .replace('name = "aisim"', 'name = "aisimulate"')
                .replace('name = "aisim-core"', 'name = "aisimulate-core"')
                .replace('aisim = "aisim.supervision:main"\n', "")
                .replace("aisim.supervision", "aisimulate.supervision")
                .replace("aisim.legacy_cli", "aisimulate.legacy_cli")
            )
    original_core = core_manifest.read_bytes()
    for args in (
        ["init", "-q"],
        ["add", "."],
        ["-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "source"],
    ):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)
    stamp = [sys.executable, str(REPOSITORY_ROOT / "scripts/release/apply_dev_version.py"), suffix, str(tmp_path)]
    subprocess.run(stamp, check=True, capture_output=True)
    before = {str(p): p.read_bytes() for p in tmp_path.rglob("*.toml")}
    subprocess.run(stamp, check=True, capture_output=True)
    assert before == {str(p): p.read_bytes() for p in tmp_path.rglob("*.toml")}
    result = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts/release/build_release_artifacts.py"),
            "--root",
            str(tmp_path),
            "--check-only",
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"{base_version}{suffix}" in (tmp_path / "python/aisim/pyproject.toml").read_text()
    assert f"{base_version}-dev.{suffix[4:]}" in (tmp_path / "Cargo.toml").read_text()
    if inherit_version:
        assert core_manifest.read_bytes() == original_core
    else:
        assert f"{base_version}-dev.{suffix[4:]}" in core_manifest.read_text()


def test_standalone_gitlab_stamper_preserves_dependency_versions(tmp_path):
    # GitLab extracts just the stamper and runs system Python without tomllib.
    stamper = tmp_path / "stamper.py"
    shutil.copyfile(REPOSITORY_ROOT / "scripts/release/apply_dev_version.py", stamper)
    (tmp_path / "tomllib.py").write_text('raise ModuleNotFoundError("tomllib unavailable")\n')
    project = tmp_path / "python/aisim/pyproject.toml"
    project.parent.mkdir(parents=True)
    project.write_text('[project]\nversion = "0.15.0"\n')
    workspace = tmp_path / "Cargo.toml"
    dependency = '[workspace.dependencies.other]\nversion = "1.2.3"\n'
    workspace.write_text(dependency + '[workspace.package]\nversion = "0.15.0"\n')
    core = tmp_path / "crates/core/Cargo.toml"
    core.parent.mkdir(parents=True)
    core.write_text('[package]\nversion = { workspace = true }\n[dependencies.other]\nversion = "1.2.3"\n')
    original_core = core.read_bytes()

    subprocess.run([sys.executable, "-E", "-S", str(stamper), ".dev20261120", str(tmp_path)], check=True)
    assert workspace.read_text() == dependency + '[workspace.package]\nversion = "0.15.0-dev.20261120"\n'
    assert core.read_bytes() == original_core


def test_manual_fpe_uses_current_harness_and_selected_inventory_for_every_job():
    workflow = _workflow("fpe-support-matrix.yml")
    assert workflow["env"]["FPE_SOURCE_SHA"] == "${{ inputs.expected_sha }}"
    assert workflow["env"]["FPE_TOOLING_SHA"] == "${{ github.sha }}"
    for job in workflow["jobs"].values():
        checkouts = [s["with"] for s in job["steps"] if "actions/checkout@" in s.get("uses", "")]
        assert checkouts == [
            {"ref": "${{ github.sha }}", "persist-credentials": "false"},
            {"ref": "${{ inputs.expected_sha }}", "path": "release-source", "persist-credentials": "false"},
        ]
    for job, action in [
        ("prepare-wheel", "record-wheel"),
        ("discover-shards", "discover"),
        ("generate", "probe"),
        ("build-web-matrix", "package"),
    ]:
        assert f"scripts/fpe/run_release_fpe.py {action}" in _run_commands(workflow["jobs"][job])
    nightly = _workflow("nightly-ci.yml")["jobs"]
    assert nightly["fpe-support-matrix"]["with"]["source_branch"] == "${{ needs.changes-guard.outputs.target-ref }}"
    assert nightly["build-artifacts"]["env"]["DEV_DATE"] == "${{ needs.changes-guard.outputs.dev-version }}"
    assert "release-tooling/scripts/release/apply_dev_version.py" in _run_commands(nightly["build-artifacts"])
    assert "release-tooling/scripts/release/build_release_artifacts.py --root ." in _run_commands(
        nightly["build-artifacts"]
    )


def test_simulation_performance_rollout_and_revision_contract():
    workflow = _workflow("simulation-performance.yml")
    selector = workflow["jobs"]["select"]
    assert workflow["on"]["push"] == _workflow("performance.yml")["on"]["push"]
    assert "workflow_dispatch" in workflow["on"]
    assert "if" not in selector
    assert workflow["concurrency"]["group"].startswith("simulation-performance-")
    assert selector["runs-on"] == "ubuntu-24.04"
    assert "python scripts/performance/select_simulation_perf.py" in _run_commands(selector)
    comparison = workflow["jobs"]["compare"]
    assert comparison["if"] == "needs.select.outputs.run_comparison == 'true'"
    commands = _run_commands(selector)
    assert comparison["needs"] == ["select", "build"]
    build = workflow["jobs"]["build"]
    assert build["needs"] == "select"
    assert build["if"] == comparison["if"]
    assert build["strategy"]["fail-fast"] == "false"
    assert build["strategy"]["matrix"]["include"] == [
        {"side": "base", "sha": "${{ needs.select.outputs.base_sha }}"},
        {"side": "head", "sha": "${{ needs.select.outputs.head_sha }}"},
    ]
    build_commands = _run_commands(build)
    callback_check = "scripts/performance/check_dynamo_callback.py source dynamo"
    assert (
        build_commands.index("build_simulation_dynamo.py source dynamo")
        < build_commands.index('python scripts/release/package_identity.py "${RUNNER_TEMP}/wheel"')
        < build_commands.index(callback_check)
        < build_commands.index("simulation_perf_artifact.py record")
    )
    assert '"${RUNNER_TEMP}/callback-venv/bin/python"' in build_commands
    downloads = [
        step["with"]["name"]
        for step in comparison["steps"]
        if step.get("uses", "").endswith("/wheel-storage") and step["with"]["mode"] == "download"
    ]
    assert downloads == [
        "simulation-wheel-base-${{ needs.select.outputs.base_sha }}",
        "simulation-wheel-head-${{ needs.select.outputs.head_sha }}",
    ]
    assert "maturin build" not in _run_commands(comparison)
    assert "simulation_perf_artifact.py verify" in _run_commands(comparison)
    assert 'git merge-base "${PR_HEAD_SHA}" "${base_tip}"' in commands
    assert 'base_sha="${PR_HEAD_SHA}"' in commands
    assert "PROTOCOL_VERSION" in commands
    assert "**INVALID_COMPARISON**" in commands
    assert "was not benchmarked" in commands
    commands = _run_commands(comparison)
    assert "repetitions=3" in commands
    assert "${BASE_SRC}/python/aisim/tools/simulation_perf_gate/run.py" in commands
    assert "${HEAD_SRC}/python/aisim/tools/simulation_perf_gate/run.py" in commands
    assert "${BASE_SRC}/python/aisim/tools/simulation_perf_gate/worker.py" in commands
    assert "${HEAD_SRC}/python/aisim/tools/simulation_perf_gate/worker.py" in commands


@pytest.mark.parametrize(
    "path,expected",
    [
        ("crates/core/src/replay/loadgen/weka.rs", True),
        ("crates/core/src/engine/scheduler/vllm/core.rs", True),
        ("python/aisim/src/aisim/traffic.py", True),
        ("python/aisim/src/aisim/compiler.py", True),
        ("python/aisim/src/aisim_core/systems/data/b200_sxm/gemm/vllm/0.24.0/gemm_perf.parquet", True),
        ("python/aisim/tools/simulation_perf_gate/fixtures/agentx.jsonl", True),
        ("scripts/performance/build_simulation_dynamo.py", True),
        ("scripts/performance/check_dynamo_callback.py", True),
        ("scripts/performance/dynamo_callback_contract.rs", True),
        ("scripts/performance/simulation_dynamo_requirements.in", True),
        ("scripts/performance/simulation_dynamo_requirements.txt", True),
        ("python/aisim/uv.lock", True),
        ("python/aisim/tools/simulation_perf_gate/README.md", False),
        ("python/aisim/tools/forward_perf_gate/run.py", False),
        ("docs/cli/user-guide.md", False),
    ],
)
def test_simulation_performance_selects_complete_pr_files(path, expected):
    from scripts.performance import select_simulation_perf

    pull = {"head": {"sha": "a" * 40}, "base": {"sha": "b" * 40, "ref": "main"}, "draft": False, "changed_files": 1}

    def api(endpoint):
        return [[{"filename": path}]] if "/files?" in endpoint else [pull]

    selected = select_simulation_perf.select_comparison(
        "owner/repo", "push", "refs/heads/pull-request/1", "a" * 40, api=api
    )
    assert selected["run_comparison"] == str(expected).lower()


@pytest.mark.parametrize(
    "path,expected",
    [
        ("contract.py", True),
        ("worker.py", True),
        ("fixtures/agentx.jsonl", True),
        ("fixtures/agentx.json", True),
        ("fixtures/another-trace.jsonl", True),
        ("README.md", False),
        ("QUALIFICATION.md", False),
        ("fixtures/README.md", False),
        ("fixtures/LICENSE", False),
        ("fixtures/DATASET_CARD.md", False),
    ],
)
def test_simulation_perf_controller_change_detection(tmp_path, path, expected):
    steps = _workflow("simulation-performance.yml")["jobs"]["select"]["steps"]
    revisions = next(step for step in steps if step.get("id") == "revisions")["run"]
    detection = "controller_changes=" + revisions.split("controller_changes=", 1)[1].split('echo "base_sha=', 1)[0]
    gate = "python/aisim/tools/simulation_perf_gate"
    _git(tmp_path, "init", "--quiet")
    base = _commit_file(tmp_path, "base", "base\n")
    # A runtime change selects the normal comparison; only benchmark code/input
    # changes should add a second run using the head controller.
    runtime = "crates/core/src/replay.rs"
    (tmp_path / runtime).parent.mkdir(parents=True)
    _commit_file(tmp_path, runtime, "changed\n")
    changed = f"{gate}/{path}"
    (tmp_path / changed).parent.mkdir(parents=True, exist_ok=True)
    head = _commit_file(tmp_path, changed, "changed\n")
    output = tmp_path / "output"
    subprocess.run(
        ["bash", "-euc", detection],
        cwd=tmp_path,
        check=True,
        env={
            **os.environ,
            "base_sha": base,
            "PR_HEAD_SHA": head,
            "gate_path": gate,
            "GITHUB_OUTPUT": str(output),
        },
    )
    assert output.read_text() == f"validate_head_controller={str(expected).lower()}\n"


@pytest.mark.parametrize(
    "scenario", ["normal", "self", "missing_base", "missing_head", "mixed", "older_base", "malformed"]
)
def test_simulation_perf_revision_preparation(tmp_path, scenario):
    steps = _workflow("simulation-performance.yml")["jobs"]["select"]["steps"]
    script = next(step["run"] for step in steps if step.get("id") == "revisions")
    _git(tmp_path, "init", "--quiet")
    gate = tmp_path / "python/aisim/tools/simulation_perf_gate"
    (gate / "fixtures").mkdir(parents=True)
    for name in ("cases.py", "compare.py", "contract.py", "run.py", "worker.py", "fixtures/agentx.jsonl"):
        (gate / name).write_text("fixture\n")
    (gate / "__init__.py").write_text("PROTOCOL_VERSION = 3\n")
    if scenario == "missing_base":
        (gate / "worker.py").unlink()
    _git(tmp_path, "add", ".")
    base = _commit_file(tmp_path, "base", "base\n")
    _git(tmp_path, "update-ref", "refs/remotes/origin/main", base)
    if scenario == "missing_base":
        (gate / "worker.py").write_text("fixture\n")
    elif scenario == "missing_head":
        (gate / "worker.py").unlink()
    elif scenario in {"mixed", "older_base", "malformed"}:
        version = {"mixed": "2", "older_base": "5", "malformed": "None"}[scenario]
        (gate / "__init__.py").write_text("PROTOCOL_VERSION = " + version + "\n")
    _git(tmp_path, "add", ".")
    head = _commit_file(tmp_path, "head", "head\n")
    output, summary = tmp_path / "output", tmp_path / "summary"
    result = subprocess.run(
        ["bash", "-euc", script],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        env={
            **os.environ,
            "PR_HEAD_SHA": head,
            "BASE_REF": "main",
            "SELF_COMPARE": str(scenario == "self").lower(),
            "GITHUB_OUTPUT": str(output),
            "GITHUB_STEP_SUMMARY": str(summary),
        },
    )
    outputs = dict(line.split("=", 1) for line in output.read_text().splitlines())
    if scenario in {"normal", "self"}:
        assert result.returncode == 0, result.stderr
        assert outputs["run_comparison"] == "true"
        assert outputs["base_sha"] == (head if scenario == "self" else base)
        assert outputs["validate_head_controller"] == "false"
    else:
        assert outputs["run_comparison"] == "false"
        skipped = scenario in {"missing_base", "older_base"}
        assert result.returncode == (0 if skipped else 1)
        assert ("was not benchmarked" if skipped else "INVALID_COMPARISON") in summary.read_text()


@pytest.mark.parametrize("self_compare,fail_first,count", [(False, False, 1), (True, False, 3), (True, True, 3)])
def test_simulation_perf_invocations(tmp_path, self_compare, fail_first, count):
    bash = shutil.which("bash")
    if bash is None or subprocess.run([bash, "-uc", 'args=(); : "${args[@]}"'], capture_output=True).returncode:
        pytest.skip("workflow invocation test requires Bash with empty-array support under set -u")
    steps = _workflow("simulation-performance.yml")["jobs"]["compare"]["steps"]
    script = next(step["run"] for step in steps if step.get("name") == "Run paired benchmark")
    python = tmp_path / "venv/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text(
        '#!/bin/bash\nprintf "%s\\n" "$*" >> "$CALLS"\n'
        'if [[ "$FAIL_FIRST" == true && $(wc -l < "$CALLS") -eq 1 ]]; then exit 1; fi\n'
    )
    python.chmod(0o755)
    calls = tmp_path / "calls"
    result = subprocess.run(
        [bash, "-euc", script],
        text=True,
        capture_output=True,
        env={
            **os.environ,
            "BASE_VENV": str(python.parents[1]),
            "HEAD_VENV": "/head-venv",
            "BASE_SRC": "/base-src",
            "HEAD_SRC": "/head-src",
            "BASE_SHA": "base",
            "HEAD_SHA": "head",
            "RESULTS_DIR": "/results",
            "SELF_COMPARE": str(self_compare).lower(),
            "CALLS": str(calls),
            "FAIL_FIRST": str(fail_first).lower(),
        },
    )
    assert result.returncode == int(fail_first), result.stderr
    invocations = calls.read_text().splitlines()
    assert len(invocations) == count
    for attempt, invocation in enumerate(invocations, 1):
        assert invocation.startswith("/base-src/python/aisim/tools/simulation_perf_gate/run.py ")
        assert "--base-worker /base-src/python/aisim/tools/simulation_perf_gate/worker.py" in invocation
        assert "--head-worker /head-src/python/aisim/tools/simulation_perf_gate/worker.py" in invocation
        assert ("--qualification" in invocation) == self_compare
        assert (
            f"--output-dir /results/qualification-{attempt}" in invocation
            if self_compare
            else "--output-dir /results" in invocation
        )


def _write_lock(source, *versions):
    source.mkdir(parents=True, exist_ok=True)
    (source / "Cargo.lock").write_text(
        "".join(f'[[package]]\nname = "pyo3"\nversion = "{version}"\n\n' for version in versions)
        + '[[package]]\nname = "pyo3-ffi"\nversion = "0.30.1"\n'
    )


@pytest.mark.parametrize(
    ("versions", "error"), [(("0.23.5",), None), (("0.30.1",), None), ((), "found"), (("0.23.5", "0.30.1"), "found")]
)
def test_simulation_dynamo_reads_one_locked_pyo3_version(tmp_path, versions, error):
    from scripts.performance import build_simulation_dynamo as builder

    _write_lock(tmp_path, *versions)
    if error:
        with pytest.raises(ValueError, match=error):
            builder.locked_pyo3_version(tmp_path / "Cargo.lock")
    else:
        assert builder.locked_pyo3_version(tmp_path / "Cargo.lock") == versions[0]


@pytest.mark.parametrize("wrong_core", [False, True])
@pytest.mark.parametrize("package", ["aisim-core", "aisimulate-core"])
def test_simulation_dynamo_resolves_then_builds_locked(tmp_path, monkeypatch, wrong_core, package):
    from scripts.performance import build_simulation_dynamo as builder

    source, dynamo, output = (tmp_path / name for name in ("source", "dynamo", "output"))
    core = source / "crates/core/Cargo.toml"
    core.parent.mkdir(parents=True)
    core.write_text(f'[package]\nname = "{package}"\nversion = "0.13.0"\n')
    bindings = dynamo / "lib/bindings/python"
    bindings.mkdir(parents=True)
    output.mkdir()
    for manifest in (dynamo / "Cargo.toml", bindings / "Cargo.toml"):
        manifest.write_text('[dependencies]\naisim-core = "0.15.0"\n')
    lock = bindings / "Cargo.lock"
    old_lock = b'[[package]]\nname = "parquet"\nversion = "55.2.0"\n'
    resolved_lock = old_lock.replace(b"55.2.0", b"59.0.0")
    lock.write_bytes(old_lock)
    built = []

    def check_output(args, **kwargs):
        if args[:2] == ["cargo", "metadata"]:
            assert "--locked" not in args
            assert kwargs["cwd"] == bindings
            assert lock.read_bytes() == old_lock
            for manifest in (dynamo / "Cargo.toml", bindings / "Cargo.toml"):
                dependency = tomllib.loads(manifest.read_text())["dependencies"]["aisim-core"]
                assert dependency["package"] == package
                assert dependency["path"] == str(core.parent)
            lock.write_bytes(resolved_lock)
            return json.dumps(
                {"packages": [{"name": package, "manifest_path": str(core) + (".wrong" if wrong_core else "")}]}
            )
        if args[-2:] == ["rev-parse", "HEAD"]:
            return builder.DYNAMO_REVISION if args[2] == str(dynamo) else "a" * 40
        assert args == ["git", "-C", str(dynamo), "diff", "HEAD"]
        return b"dependency patch\n"

    def run(args, **kwargs):
        assert kwargs["check"]
        if args[0] == "git":
            assert args == ["git", "-C", str(dynamo), "diff", "--exit-code", "HEAD"]
            return
        assert lock.read_bytes() == resolved_lock
        if "maturin" in args:
            assert "--locked" in args
            assert kwargs["cwd"] == bindings
        else:
            assert args[:2] == ["uv", "build"]
        built.append(args)

    monkeypatch.setattr(builder.subprocess, "check_output", check_output)
    monkeypatch.setattr(builder.subprocess, "run", run)
    if wrong_core:
        with pytest.raises(ValueError, match="different AISim core"):
            builder.build(source, dynamo, output)
        assert not built
        assert not list(output.iterdir())
    else:
        builder.build(source, dynamo, output)
        assert len(built) == 2
        assert (output / "dynamo-Cargo.lock").read_bytes() == resolved_lock
        provenance = json.loads((output / "dynamo-build.json").read_text())
        assert provenance == {
            "source_sha": builder.DYNAMO_REVISION,
            "aisimulate_sha": "a" * 40,
            "features": ["ais-forward-pass"],
        }
        assert (output / "dynamo-build.patch").read_bytes() == b"dependency patch\n"


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "revision",
        "side",
        "wheel",
        "requirements",
        "extra_wheel",
        "unverified_wheel",
        "embedded_core",
        "dynamo_wheel",
        "missing_lock",
        "changed_lock",
    ],
)
def test_simulation_perf_artifact_verification(tmp_path, fault):
    from scripts.performance.simulation_perf_artifact import sha256, verify

    wheel = tmp_path / "aisim-test.whl"
    wheel.write_bytes(b"built wheel")
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("locked\n")
    additional = [
        tmp_path / name
        for name in (
            "ai_dynamo_runtime-test.whl",
            "ai_dynamo-test.whl",
            "dynamo-requirements.txt",
            "dynamo-build.json",
            "dynamo-build.patch",
            "dynamo-Cargo.lock",
        )
    ]
    for path in additional:
        path.write_text("build record\n")
    manifest = {
        "side": "base",
        "source_sha": "a" * 40,
        "files": {path.name: sha256(path) for path in (wheel, requirements, *additional)},
        "dynamo": {"aisimulate_sha": "a" * 40},
    }
    if fault == "revision":
        manifest["source_sha"] = "b" * 40
    elif fault == "side":
        manifest["side"] = "head"
    elif fault == "wheel":
        wheel.write_bytes(b"wrong wheel")
    elif fault == "requirements":
        requirements.write_text("wrong dependencies\n")
    elif fault == "extra_wheel":
        (tmp_path / "aisim-other.whl").write_bytes(b"another wheel")
    elif fault == "unverified_wheel":
        (tmp_path / "unexpected_package-1.0-py3-none-any.whl").write_bytes(b"unverified wheel")
    elif fault == "embedded_core":
        manifest["dynamo"]["aisimulate_sha"] = "b" * 40
    elif fault == "dynamo_wheel":
        additional[0].write_bytes(b"wrong Dynamo wheel")
    elif fault == "missing_lock":
        (tmp_path / "dynamo-Cargo.lock").unlink()
        del manifest["files"]["dynamo-Cargo.lock"]
    elif fault == "changed_lock":
        (tmp_path / "dynamo-Cargo.lock").write_text("different dependency graph\n")
    (tmp_path / "provenance.json").write_text(json.dumps(manifest))
    if fault:
        with pytest.raises(ValueError):
            verify(tmp_path, "base", "a" * 40)
    else:
        assert verify(tmp_path, "base", "a" * 40) == manifest


@pytest.mark.parametrize("layout", ["scripts/build_manylinux_wheel.py", "scripts/release/build_manylinux_wheel.py"])
def test_standalone_fpe_uses_current_requirements_and_revision_builder(tmp_path, layout):
    step = next(
        s
        for s in _workflow("fpe-support-matrix.yml")["jobs"]["prepare-wheel"]["steps"]
        if s.get("name") == "Build one wheel for standalone qualification"
    )
    source = tmp_path / "release-source"
    builder = source / layout
    builder.parent.mkdir(parents=True)
    builder.touch()
    if layout == "scripts/release/build_manylinux_wheel.py":
        (source / "scripts/build_manylinux_wheel.py").touch()
    requirements = tmp_path / "scripts/release/requirements.txt"
    requirements.parent.mkdir(parents=True)
    requirements.touch()
    capture = """
python() {
  if [ "$1" = "-m" ]; then
    test "$4" = "--require-hashes" || return 1
    test "$6" = "${GITHUB_WORKSPACE}/scripts/release/requirements.txt" || return 1
    test -f "$6" || return 1
  else
    test -f "$1" || return 1
    test "$2" = "--output-dir" || return 1
    test "$3" = "../fpe-wheel" || return 1
    echo "$1"
  fi
}
"""
    result = subprocess.run(
        ["bash", "-e", "-c", capture + step["run"]],
        cwd=source,
        env={**os.environ, "GITHUB_WORKSPACE": str(tmp_path)},
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == layout


@pytest.mark.parametrize("extra", ["other/pyproject.toml", "python/unknown/pyproject.toml"])
def test_release_manifest_discovery_rejects_unknown_tool_only_manifests(tmp_path, monkeypatch, extra):
    for name in ("Cargo.toml", "crates/core/Cargo.toml", "python/aisim/pyproject.toml", "scripts/pyproject.toml"):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((REPOSITORY_ROOT / name).read_bytes())
    monkeypatch.setattr(release_artifacts, "ROOT", tmp_path)
    monkeypatch.setattr(
        release_artifacts, "EXPECTED_PYTHON_PROJECTS", {tmp_path / "python/aisim/pyproject.toml": "aisim"}
    )
    monkeypatch.setattr(release_artifacts, "EXPECTED_CRATE", tmp_path / "crates/core/Cargo.toml")
    release_artifacts.check_manifests()
    target = tmp_path / extra
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("[dependency-groups]\nci = []\n")
    with pytest.raises(AssertionError, match="missing project name"):
        release_artifacts.check_manifests()


@pytest.mark.parametrize("package", ["aisim", "aisimulate"])
def test_wheel_identity_rejects_missing_and_mixed_product_builds(tmp_path, package):
    from scripts.release.package_identity import single_wheel

    (tmp_path / "ai_dynamo-1.0-py3-none-any.whl").write_bytes(b"dependency")
    with pytest.raises(ValueError, match="exactly one"):
        single_wheel(tmp_path)
    selected = tmp_path / f"{package}-0.13.0-cp311-abi3-manylinux_2_28_x86_64.whl"
    selected.write_bytes(b"selected product")
    assert single_wheel(tmp_path) == selected
    other = "aisimulate" if package == "aisim" else "aisim"
    (tmp_path / f"{other}-0.13.0-cp311-abi3-manylinux_2_28_x86_64.whl").write_bytes(b"stale build")
    with pytest.raises(ValueError, match="exactly one"):
        single_wheel(tmp_path)


@pytest.mark.parametrize("package,alias", [("aisim", True), ("aisimulate", False), ("aisimulate", True)])
@pytest.mark.parametrize(
    "failure",
    [
        None,
        "name",
        "version",
        "duplicate-name",
        "dist-info",
        "missing-metadata",
        "extra-metadata",
        "missing-entry-points",
        "wrong-entry-root",
        "missing-command",
        "alias",
        "wrong-target",
        "extra-script",
    ],
)
def test_release_output_checks_embedded_identity_and_scripts(tmp_path, monkeypatch, package, alias, failure):
    version = "0.14.0" if package == "aisim" else "0.13.0"
    monkeypatch.setattr(release_artifacts, "ROOT", tmp_path)
    monkeypatch.setattr(release_artifacts, "PYTHON_PACKAGE", package)
    monkeypatch.setattr(release_artifacts, "CRATE_PACKAGE", f"{package}-core")
    monkeypatch.setattr(release_artifacts, "REQUIRE_CLI_ALIAS", alias)
    output = tmp_path / "dist"
    output.mkdir()
    (output / f"{package}-core-{version}.crate").write_bytes(b"crate fixture")
    for name in release_artifacts.LEGAL_FILES:
        (tmp_path / name).write_text(f"fixture legal text for {name}\n")
    namespace = "aisim" if package == "aisim" else "aisimulate"
    monkeypatch.setattr(release_artifacts, "PYTHON_NAMESPACE", namespace)
    scripts = {
        "aiconfigurator": f"{namespace}.legacy_cli.entrypoint:main",
        "aisimulate": f"{namespace}.supervision:main",
    }
    if alias:
        scripts["aisim"] = f"{namespace}.supervision:main"
    if failure == "missing-command":
        del scripts["aisimulate"]
    elif failure == "alias":
        if alias:
            del scripts["aisim"]
        else:
            scripts["aisim"] = f"{namespace}.supervision:main"
    elif failure == "wrong-target":
        scripts["aisimulate"] = "wrong.module:main"
    elif failure == "extra-script":
        scripts["unexpected"] = f"{namespace}.supervision:main"
    embedded_name = "other-product" if failure == "name" else package
    embedded_version = "0.0.1" if failure == "version" else version
    metadata = f"Metadata-Version: 2.1\nName: {embedded_name}\nVersion: {embedded_version}\n"
    if failure == "duplicate-name":
        metadata += f"Name: {package}\n"
    dist_info = "other.dist-info" if failure == "dist-info" else f"{package}-{version}.dist-info"
    wheel = output / f"{package}-{version}-cp311-abi3-manylinux_2_28_x86_64.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        if failure != "missing-metadata":
            archive.writestr(f"{dist_info}/METADATA", metadata)
        if failure == "extra-metadata":
            archive.writestr("extra.dist-info/METADATA", metadata)
        if failure != "missing-entry-points":
            entry_root = "other.dist-info" if failure == "wrong-entry-root" else dist_info
            archive.writestr(
                f"{entry_root}/entry_points.txt",
                "[console_scripts]\n" + "".join(f"{name} = {target}\n" for name, target in scripts.items()),
            )
        for name in release_artifacts.LEGAL_FILES:
            archive.writestr(f"{dist_info}/licenses/{name}", (tmp_path / name).read_bytes())
    if failure is None:
        release_artifacts.verify_output(output, version, version)
    else:
        with pytest.raises(AssertionError):
            release_artifacts.verify_output(output, version, version)


def test_performance_jobs_fetch_each_revisions_pinned_data_before_benchmarking():
    """The gates run offline or under a worker timeout, so every installed revision fetches its own pin first."""
    prefetch = '/bin/python" scripts/performance/prefetch_op_latency.py'
    simulation = _workflow("simulation-performance.yml")["jobs"]
    for job, consumers, count in (
        (simulation["build"], "check_dynamo_callback.py source dynamo", 1),
        (simulation["compare"], "simulation_perf_gate/run.py", 1),  # one loop over both environments
        (_workflow("performance.yml")["jobs"]["compare"], "forward_perf_gate/run.py", 2),
    ):
        commands = _run_commands(job)
        assert commands.count(prefetch) == count
        assert commands.rindex(prefetch) < commands.index(consumers)


def test_standalone_composition_steps_install_the_download_transport():
    """compose_systems_root.py needs the Hugging Face transport, and these steps never install AISim itself."""
    documents = [yaml.safe_load(path.read_text()) for path in sorted(WORKFLOW_ROOT.glob("*.yml"))]
    documents += [yaml.safe_load(path.read_text()) for path in sorted(ACTION_ROOT.glob("*/action.yml"))]
    steps = [
        step for document in documents for job in (document.get("jobs") or {}).values() for step in job.get("steps", [])
    ]
    steps += [step for document in documents for step in (document.get("runs") or {}).get("steps", [])]
    composing = [step["run"] for step in steps if "compose_systems_root.py" in step.get("run", "")]
    assert composing
    for run in composing:
        before = run.split("compose_systems_root.py")[0]
        assert re.search(r"pip install[^\n]*\bhuggingface_hub\b", before), run
