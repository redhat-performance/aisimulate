#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

# These files take precedence over the managed root policy on GitHub.
if [[ -f .github/CODEOWNERS || -L .github/CODEOWNERS || -f docs/CODEOWNERS || -L docs/CODEOWNERS ]]; then
    echo "::error::The generated root CODEOWNERS must be the repository's only CODEOWNERS file."
    exit 1
fi

python -m pytest -c /dev/null .github/codeowners/test_*.py -q \
    -p no:cacheprovider --override-ini="addopts=" --override-ini="filterwarnings="
python .github/codeowners/build_codeowners.py \
    --areas .github/codeowners/areas.yaml --repo . --strict
python .github/codeowners/emit_codeowners.py \
    --areas .github/codeowners/areas.yaml --out CODEOWNERS \
    --external .github/codeowners/external_contributors.yaml \
    --contributors-out CONTRIBUTORS.md
if [[ -n "$(git status --porcelain --untracked-files=all -- CODEOWNERS CONTRIBUTORS.md)" ]]; then
    git diff -- CODEOWNERS CONTRIBUTORS.md || true
    git status --short --untracked-files=all -- CODEOWNERS CONTRIBUTORS.md
    echo "::error::Generated CODEOWNERS artifacts are out of date."
    exit 1
fi
