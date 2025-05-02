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
from typing import Dict, List, Optional, Tuple, TypedDict

import ray
import torch

from KernelBench.src.eval import KernelExecResult

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES
from nemo_rl.environments import cuda_kernel_eval
from nemo_rl.environments.interfaces import (
    EnvironmentInterface,
    EnvironmentReturn,
)
from nemo_rl.environments.metrics import (
    calculate_pass_rate_per_prompt,
)
from nemo_rl.environments.utils import chunk_list_to_workers


class CudaEnvConfig(TypedDict):
    num_workers: int
    stop_strings: Optional[List[str]] = None  # Default stop strings for this env


@ray.remote
class HFVerifyWorker:
    DEFAULT_PY_EXECUTABLE = PY_EXECUTABLES.SYSTEM

    def verify(
        self, prompts: List[str], completions: List[str]
    ) -> List[Tuple[float, KernelExecResult]]:
        """Evaluate the custom CUDA kernel completions against the reference code.

        Args:
            prompts: List[str]. Contains pytorch reference code.
            completions: List[str]. Contains custom CUDA kernel code.

        Returns:
            List[Tuple[float, KernelExecResult]]. The rewards and the KernelExecResult (containing metadata) for each custom code.
        """
        results = []
        for prompt, completion in zip(prompts, completions):
            try:
                reward, eval_metadata = cuda_kernel_eval.get_reward(prompt, completion)
                results.append((reward, eval_metadata))
            except Exception:
                results.append(
                    (
                        0,
                        KernelExecResult(
                            compiled=False,
                            executed=False,
                            correctness=False,
                            metadata={"other_error": "HFVerifyWorker error"},
                        ),
                    )
                )
        return results


class CudaEnvironmentMetadata(TypedDict):
    ground_truth: str


@ray.remote
class CudaEnvironment(EnvironmentInterface):
    DEFAULT_PY_EXECUTABLE = PY_EXECUTABLES.SYSTEM

    def __init__(self, cfg: CudaEnvConfig):
        self.cfg = cfg
        self.num_workers = cfg["num_workers"]
        self.workers = [
            HFVerifyWorker.options(
                runtime_env={"py_executable": HFVerifyWorker.DEFAULT_PY_EXECUTABLE}
            ).remote()
            for _ in range(self.num_workers)
        ]

    def shutdown(self):
        # shutdown all workers
        for worker in self.workers:
            ray.kill(worker)

    def step(
        self,
        message_log_batch: List[List[Dict[str, str]]],
        metadata: List[CudaEnvironmentMetadata],
    ) -> EnvironmentReturn:
        """Runs a step in the cuda environment.

        Args:
            message_log: List[List[Dict[str, str]]]. A batch of OpenAI-API-like message logs that represent interactions with the LLM.
            metadata: List[CudaEnvironmentMetadata]. Empty for CUDA environment, because no ground truth.

        Returns:
            EnvironmentReturn: A tuple containing:
                - List[Dict[str, str]]: Observations/completions batch
                - List[Dict]: Updated metadata
                - List[str]: Next stop strings for the next turn
                - Tensor: Rewards tensor
                - Tensor: Done flags tensor
        """
        # Extract the user prompts and assistant's completions from the message history
        user_prompt_batch = []
        assistant_completion_batch = []
        for conversation in message_log_batch:
            # Extract user prompt (assuming the first user message is the main prompt)
            # Each message list should have at least one user prompt
            user_prompts = [
                interaction["content"]
                for interaction in conversation
                if interaction["role"] == "user"
            ]
            user_prompt_batch.append(
                user_prompts[0] if user_prompts else ""
            )  # TODO: Handle multiple user turns if necessary

            # Extract assistant completion (assuming the last assistant message is the main completion)
            # Each message list should have at least one assistant completion
            assistant_completions = [
                interaction["content"]
                for interaction in conversation
                if interaction["role"] == "assistant"
            ]
            assistant_completion_batch.append(
                assistant_completions[0] if assistant_completions else ""
            )

        # Chunk data for workers
        chunked_user_prompt_batch = chunk_list_to_workers(
            user_prompt_batch, self.num_workers
        )
        chunked_assistant_completion_batch = chunk_list_to_workers(
            assistant_completion_batch, self.num_workers
        )

        # # Process each chunk in parallel
        futures = [
            self.workers[i].verify.remote(prompt_chunk, completion_chunk)
            for i, (prompt_chunk, completion_chunk) in enumerate(
                zip(chunked_user_prompt_batch, chunked_assistant_completion_batch)
            )
        ]

        results = ray.get(futures)  # list of list of tuples (float, KernelExecResult)

        # flatten the results
        results = [item for sublist in results for item in sublist]  # list of tuples
        observations = [
            {
                "role": "environment",
                "content": f"Reward: {reward}\n"
                f"Compiled: {eval_metadata['compiled']}\n"
                f"Executed: {eval_metadata['executed']}\n"
                f"Correctness: {eval_metadata['correctness']}\n",
            }
            for reward, eval_metadata in results
        ]

        # create a tensor of rewards and done flags
        rewards = torch.tensor([reward for reward, _ in results]).cpu()
        done = torch.ones_like(rewards).cpu()

        next_stop_strings = [None] * len(message_log_batch)

        return EnvironmentReturn(
            observations=observations,
            metadata=metadata,
            next_stop_strings=next_stop_strings,
            rewards=rewards,
            terminateds=done,
        )

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict
    ) -> Tuple[BatchedDataDict, dict]:
        """Computes metrics for this environment given a global rollout batch.

        Every rank will run this function, so you're free to use distributed
        calculations if you'd prefer for heavy metrics.
        """
        batch["rewards"] = (
            batch["rewards"] * batch["is_end"]
        )  # set a reward of 0 for any incorrectly ended sequences
        if (batch["rewards"] == 1).float().sum() > 0:
            correct_solution_generation_lengths = (
                (batch["generation_lengths"] - batch["prompt_lengths"])[
                    batch["rewards"] == 1
                ]
                .float()
                .mean()
                .item()
            )
        else:
            correct_solution_generation_lengths = 0

        metrics = {
            # "table": table, TODO @sahilj WIP
            "accuracy": batch["rewards"].mean().item(),
            "pass@samples_per_prompt": calculate_pass_rate_per_prompt(
                batch["text"], batch["rewards"]
            ),
            "fraction_of_samples_properly_ended": batch["is_end"].float().mean().item(),
            "num_problems_in_batch": batch["is_end"].shape[0],
            "generation_lengths": batch["generation_lengths"].float().mean().item(),
            "prompt_lengths": batch["prompt_lengths"].float().mean().item(),
            "correct_solution_generation_lengths": correct_solution_generation_lengths,
        }

        return batch, metrics
