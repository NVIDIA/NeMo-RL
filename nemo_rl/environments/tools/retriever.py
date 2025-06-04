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
import re
from typing import Any, Dict, List, TypedDict

import ray
import torch
from datasets import load_dataset

from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn
from nemo_rl.tools.tools import BM25Retriever


class RAGEnvConfig(TypedDict):
    dataset_name: str  # Name of the dataset to load
    dataset_split: str  # Split of the dataset to use
    text_column: str  # Column name containing the text to retrieve
    num_results: int  # Number of documents to retrieve
    k1: float  # BM25 parameter
    b: float  # BM25 parameter
    device: str  # Device to compute BM25


@ray.remote
class RAGEnvironment(EnvironmentInterface):
    """RAG environment that uses BM25 for document retrieval."""

    def __init__(self, cfg: RAGEnvConfig):
        self.cfg = cfg

        # Load dataset
        dataset = load_dataset(cfg["dataset_name"], split=cfg["dataset_split"])
        documents = [sample[cfg["text_column"]] for sample in dataset]

        # Initialize BM25 retriever
        self.retriever = BM25Retriever(
            documents=documents,
            num_result=cfg["num_results"],
            k1=cfg["k1"],
            b=cfg["b"],
            device=cfg["device"],
        )

    def format_result(self, retrieved_docs: List[str]) -> str:
        result = "<result>\n"
        for i, doc in enumerate(retrieved_docs):
            result += f"<{i + 1}>\n{doc}\n</{i + 1}>\n"
        result += "</result>\n"
        return result

    def step(
        self,
        message_log_batch: List[LLMMessageLogType],
        metadata_batch: List[Dict[str, Any]],
    ) -> EnvironmentReturn:
        """Process a batch of retrieval steps."""
        # Extract queries from the last message in each log
        messages = [ml[-1]["content"] for ml in message_log_batch]

        # Retrieve documents for each query
        results = []
        for message in messages:
            match = re.search(r"<retrieve>(.*)</retrieve>", message, re.DOTALL)
            if not match:
                results.append(
                    {"role": "environment", "content": "No retrieval query found!"}
                )
                continue
            query = match.group(1)
            retrieved_docs = self.retriever(query)
            result = self.format_result(retrieved_docs)
            results.append({"role": "environment", "content": result})

        batch_size = len(message_log_batch)
        rewards_tensor = torch.zeros(batch_size, dtype=torch.float32)
        terminated_tensor = torch.ones(batch_size, dtype=torch.bool)
        next_stop_strings = [["</retrieve>"]] * batch_size

        return EnvironmentReturn(
            observations=results,
            metadata=metadata_batch,
            next_stop_strings=next_stop_strings,
            rewards=rewards_tensor,
            terminateds=terminated_tensor,
        )

    def shutdown(self):
        """Clean up resources."""
        pass

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict
    ) -> tuple[BatchedDataDict, dict]:
        """Compute metrics for the batch."""
        # No specific metrics for RAG
        return batch, {}
