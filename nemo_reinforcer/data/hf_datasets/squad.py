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

from typing import Optional
from datasets import load_dataset
from nemo_reinforcer.data.hf_datasets.interfaces import HfDataset


def format_squad(data):
    return {
        "messages": [
            {
                "role": "system",
                "content": data["context"],
            },
            {
                "role": "user",
                "content": data["question"],
            },
            {
                "role": "assistant",
                "content": data["answers"]["text"][0],
            },
        ]
    }


class SquadDataset(HfDataset):
    def __init__(self):
        original_ds = load_dataset("rajpurkar/squad")
        self.formatted_ds = original_ds.map(format_squad)

        custom_template = "{% for message in messages %}{%- if message['role'] == 'system'  %}{{'Context: ' + message['content'].strip()}}{%- elif message['role'] == 'user'  %}{{' Question: ' + message['content'].strip() + ' Answer:'}}{%- elif message['role'] == 'assistant'  %}{{' ' + message['content'].strip()}}{%- endif %}{% endfor %}"

        super().__init__(
            dataset_name="squad",
            custom_template=custom_template,
        )
