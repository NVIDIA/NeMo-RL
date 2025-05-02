import os

import setuptools

# --- Configuration Start ---
final_packages = []
final_package_dir = {}

# --- nemo package conditional section ---
nemo_package_source_dir = "NeMo/nemo"
nemo_package_name = "nemo"

if os.path.exists(nemo_package_source_dir):
    final_packages.append(nemo_package_name)
    final_package_dir[nemo_package_name] = nemo_package_source_dir
# --- End of nemo package conditional section ---

setuptools.setup(
    name="nemo-tron",  # Must match [project].name in pyproject.toml
    version="0.0.0",  # Must match [project].version in pyproject.toml
    description="Standalone packaging for the NeMo Tron sub-module.",  # Can be sourced from pyproject.toml too
    author="NVIDIA",
    author_email="nemo-toolkit@nvidia.com",
    packages=final_packages,
    package_dir=final_package_dir,
    py_modules=["is_nemo_installed"],
    install_requires=[
        "lightning",
        "wget",
        "onnx",
        "fiddle",
        "cloudpickle",
        "braceexpand",
        "webdataset",
        "h5py",
        "ijson",
        "matplotlib",
        "scikit-learn",
        "nemo-run",
        "hatchling",
    ],
)
