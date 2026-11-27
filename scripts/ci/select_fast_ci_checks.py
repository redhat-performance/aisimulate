#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Skip script lock regeneration only when a complete diff proves it unrelated."""

from __future__ import annotations

import json
import os
import re
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path

SHA = re.compile(r"[0-9a-f]{40}")
LOCK_INPUTS = {
    "pyproject.toml",
    "uv.toml",
    "scripts/uv.toml",
    "scripts/pyproject.toml",
    "scripts/ci/select_fast_ci_checks.py",
    "tests/test_fast_ci_policy.py",
    ".github/workflows/fast-ci.yml",
}


def lock_input(path: str) -> bool:
    return path in LOCK_INPUTS or (path.startswith("scripts/") and path.endswith("/requirements.txt"))


def read_pull_request(repository: str, number: str) -> dict:
    result = subprocess.run(
        ["gh", "api", "--method", "GET", f"repos/{repository}/pulls/{number}"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return json.loads(result.stdout)


def select_locks(
    root: Path,
    env: Mapping[str, str],
    *,
    pr_reader: Callable[[str, str], dict] = read_pull_request,
) -> tuple[bool, str]:
    event, ref = env.get("GITHUB_EVENT_NAME", ""), env.get("GITHUB_REF", "")
    if event not in {"pull_request", "push"}:
        return True, "manual or unknown event: validate every script lock"
    target, base = env.get("TARGET_SHA", ""), env.get("BASE_SHA", "")
    copied_pr = re.fullmatch(r"refs/heads/pull-request/([0-9]+)", ref)
    try:
        if event == "push" and ref.startswith("refs/heads/pull-request/"):
            repository = env.get("GITHUB_REPOSITORY", "")
            if not copied_pr or not re.fullmatch(r"[\w.-]+/[\w.-]+", repository):
                return True, "cannot identify the trusted PR copy"
            pull = pr_reader(repository, copied_pr[1])
            if pull["head"]["sha"] != target:
                return True, "trusted copy no longer matches the PR head"
            base = pull["base"]["sha"]
        if not SHA.fullmatch(target) or not SHA.fullmatch(base) or base == "0" * 40:
            return True, "comparison commit is missing or invalid"
        # PRs and trusted copies compare the complete PR, including earlier
        # commits. Lifecycle pushes compare both sides of the whole push.
        separator = "..." if event == "pull_request" or copied_pr else ".."
        result = subprocess.run(
            ["git", "diff", "--name-only", "--no-renames", "-z", f"{base}{separator}{target}", "--"],
            cwd=root,
            check=True,
            capture_output=True,
            timeout=30,
        )
        paths = result.stdout.decode("utf-8").split("\0")
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError):
        return True, "comparison unavailable: validate every script lock"
    if any(lock_input(path) for path in paths if path):
        return True, "script manifest, lock, or validation machinery changed"
    return False, "complete change contains no script lock inputs"


def main() -> None:
    selected, reason = select_locks(Path(__file__).resolve().parents[2], os.environ)
    print(f"Script dependency lock regeneration: {str(selected).lower()} ({reason})")
    if output := os.environ.get("GITHUB_OUTPUT"):
        with Path(output).open("a", encoding="utf-8") as handle:
            handle.write(f"dependency_locks={str(selected).lower()}\n")


if __name__ == "__main__":
    main()
