# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from pathlib import Path
import subprocess
from typing import Tuple

import warnings


def reinforcer_git_info() -> Tuple[str, str]:
    """Returns a tuple of (git_sha, git_branch) for the current commit and branch.

    If no git repo is found, returns ("", "").
    """
    root_path = Path(__file__).resolve().parent
    try:
        # Get commit SHA
        commit_output = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            cwd=root_path,
            check=True,
            universal_newlines=True,
        )
        commit = commit_output.stdout.strip()

        # Get branch name
        branch_output = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            cwd=root_path,
            check=True,
            universal_newlines=True,
        )
        branch = branch_output.stdout.strip()

        return commit, branch
    except (subprocess.CalledProcessError, OSError):
        warnings.warn("No git repo found! Returning empty strings.")
        return "", ""
