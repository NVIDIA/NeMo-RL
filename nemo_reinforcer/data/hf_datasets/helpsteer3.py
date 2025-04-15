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


def format_helpsteer3(data):
    response_1 = data["response1"]
    response_2 = data["response2"]
    overall_preference = data["overall_preference"]

    return {
        "prompt": data["context"],
        "chosen_response": response_1 if overall_preference <= 0 else response_2,
        "rejected_response": response_2 if overall_preference < 0 else response_1,
    }


class HelpSteer3Dataset(HfDataset):
    """HelpSteer3 preference dataset for DPO training."""

    def __init__(self):
        ds = load_dataset("nvidia/HelpSteer3", "preference")
        self.formatted_ds = ds.map(format_helpsteer3)

        super().__init__(
            dataset_name="HelpSteer3",
            custom_template=None,  ## use tokenizer's template
        )
