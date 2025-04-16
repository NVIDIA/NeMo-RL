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
import subprocess
import shlex
import logging
from functools import lru_cache

dir_path = os.path.dirname(os.path.abspath(__file__))
git_root = os.path.abspath(os.path.join(dir_path, "../.."))
DEFAULT_VENV_DIR = os.path.join(git_root, "venvs")

logger = logging.getLogger(__name__)


@lru_cache(maxsize=None)
def create_local_venv(py_executable: str, venv_name: str) -> str:
    """Create a virtual environment using uv and execute a command within it.

    The output can be used as a py_executable for a Ray worker assuming the worker
    nodes also have access to the same file system as the head node.

    This function is cached to avoid multiple calls to uv to create the same venv,
    which avoids duplicate logging.

    Args:
        py_executable (str): Command to run with the virtual environment (e.g., "uv.sh run --locked")
        venv_name (str): Name of the virtual environment (e.g., "foobar.Worker")

    Returns:
        str: Path to the python executable in the created virtual environment
    """
    # This directory is where virtual environments will be installed
    # It is local to the driver process but should be visible to all worker nodes
    # If this directory is not accessible from worker nodes (e.g., on a distributed
    # cluster with non-shared filesystems), you may encounter errors when workers
    # try to access the virtual environments
    #
    # You can override this location by setting the REINFORCER_VENV_DIR environment variable

    REINFORCER_VENV_DIR = os.environ.get("REINFORCER_VENV_DIR", DEFAULT_VENV_DIR)
    logger.info(f"REINFORCER_VENV_DIR is set to {REINFORCER_VENV_DIR}.")

    # Create the venv directory if it doesn't exist
    os.makedirs(REINFORCER_VENV_DIR, exist_ok=True)

    # Full path to the virtual environment
    venv_path = os.path.join(REINFORCER_VENV_DIR, venv_name)

    # Create the virtual environment
    uv_venv_cmd = ["uv", "venv", "--allow-existing", venv_path]
    subprocess.run(uv_venv_cmd, check=True)

    # Execute the command with the virtual environment
    env = os.environ.copy()
    # NOTE: UV_PROJECT_ENVIRONMENT is appropriate here only b/c there should only be
    #  one call to this in the driver. It is not safe to use this in a multi-process
    #  context.
    #  https://docs.astral.sh/uv/concepts/projects/config/#project-environment-path
    env["UV_PROJECT_ENVIRONMENT"] = venv_path

    # Split the py_executable into command and arguments
    exec_cmd = shlex.split(py_executable)
    # Command doesn't matter, since `uv` syncs the environment no matter the command.
    exec_cmd.extend(["echo", f"Finished creating venv {venv_path}"])

    subprocess.run(exec_cmd, env=env, check=True)

    # Return the path to the python executable in the virtual environment
    python_path = os.path.join(venv_path, "bin", "python")
    return python_path
