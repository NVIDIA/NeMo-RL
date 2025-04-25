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
import torch
from huggingface_hub import snapshot_download


def prefetch_hf_model(
    repo_id: str,
    *,
    revision: str | None = None,
    cache_dir: str | os.PathLike | None = None,
) -> Path:
    """Make sure *all* files for `repo_id` are present locally.

    1. Every rank first tries a *cheap* cache-only lookup.
    2. If the snapshot is missing       → rank 0 downloads it (resuming if partial).
    3. Other ranks block on a barrier   → then proceed to load.

    Returns the path to the snapshot root.
    """
    # Check if the model is already fully cached
    model_is_cached = False
    try:
        path = snapshot_download(
            repo_id,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=True,  # do not hit the network here
        )
        # Check that weight files exist, not just tokenizer/config
        weight_extensions = [".bin", ".pt", ".pth", ".safetensors"]
        weight_files = [
            f
            for f in Path(path).glob("**/*")
            if f.is_file() and f.suffix in weight_extensions
        ]

        if weight_files:
            print("Model weights already cached locally. Using that")
            model_is_cached = True
            return path
        else:
            print("Found cached files but no model weights, proceeding with download")
    except EnvironmentError:
        print("Model not cached, downloading on head")

    # Only download if model isn't fully cached
    torch.distributed.barrier()
    if not model_is_cached:
        # Only main process is allowed to fetch
        if torch.distributed.get_rank() == 0:
            print(f"[rank 0] Downloading {repo_id} from the Hub...")
            snapshot_download(
                repo_id,
                revision=revision,
                cache_dir=cache_dir,
                resume_download=True,  # continue a partial DL if one exists
            )

        # All ranks sync; after this everyone has the files on disk
        torch.distributed.barrier()

    # Make the call again, now cache-only; guaranteed to succeed
    return snapshot_download(
        repo_id,
        revision=revision,
        cache_dir=cache_dir,
        local_files_only=True,
    )
