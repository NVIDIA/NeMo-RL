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
from datasets import load_dataset

from nemo_reinforcer.data.hf_datasets.interfaces import HfDataset


## assumptions about DPO dataset:
## json files should have the following keys:
## "prompt"
## "chosen_response"
## "rejected_response"
class DPODataset(HfDataset):
    def __init__(self, train_data_path: str, val_data_path: str):
        ## TODO: assuming for now that data has been split into train and val
        ## as an offline preprocessing step

        ## TODO: update the keys to match with what's expected from apply_chat_template
        ## we need to do this outisde of the data class because we want to keep
        ## chosen and rejected responses for a given prompt together when shuffling
        self.formatted_ds = {
            "train": load_dataset("json", data_files=train_data_path, split="train"),
            "validation": load_dataset("json", data_files=val_data_path, split="train"),
        }
        super().__init__(
            dataset_name="dpo",
            ## passthrough template
            custom_template="{% for message in messages %}{{ message['content'] }}{% endfor %}",
        )
