from typing import Dict, List, Optional, Tuple

import ray
import torch
from ray.actor import ActorHandle

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn


@ray.remote
class LLMJudgeEnvironment(EnvironmentInterface):
    """A mock environment that assigns a random reward of 0 or 1 until we have a judge model."""

    DEFAULT_PY_EXECUTABLE = PY_EXECUTABLES.SYSTEM

    def __init__(
        self, cfg: Optional[Dict] = None, judge_llm_handle: Optional[ActorHandle] = None
    ):
        self.cfg = cfg or {}
        self.judge_llm_handle = judge_llm_handle
        # Allow custom judge prompt via configuration; fallback to default.
        self.prompt_template = self.cfg.get(
            "prompt_template",
            (
                "Problem:\n{prompt}\n\n"
                "Ground Truth Answer:\n{ground_truth}\n\n"
                "Model Response:\n{response}\n\n"
                "Instructions:\nYou are a fair and consistent judge. "
                "Above is a problem, a ground truth answer, and a model's response. "
                "Evaluate if the model's response matches the ground truth answer.\n"
                'Answer "YES" or "NO", then provide a brief explanation.'
            ),
        )

    def step(
        self,
        message_log_batch: List[List[Dict[str, str]]],
        metadata: List[Optional[dict]],
        *args,
        **kwargs,
    ) -> EnvironmentReturn:
        batch_size = len(message_log_batch)

        # Default observations
        observations = [
            {"role": "environment", "content": "Environment: processed"}
            for _ in range(batch_size)
        ]

        # If a judge LLM handle is provided, run a lightweight inference call
        assert self.judge_llm_handle is not None, "Judge LLM handle must be provided"

        judge_prompts = []
        for convo, meta in zip(message_log_batch, metadata):
            problem_prompt = meta.get("prompt", "") if meta else ""
            ground_truth = meta.get("ground_truth", "") if meta else ""

            assistant_msgs = [m["content"] for m in convo if m["role"] == "assistant"]
            model_response = "\n".join(assistant_msgs)

            judge_prompt = self.prompt_template.format(
                prompt=problem_prompt,
                ground_truth=ground_truth,
                response=model_response,
            )
            judge_prompts.append(judge_prompt)

        data = BatchedDataDict({"prompts": judge_prompts})
        out = ray.get(self.judge_llm_handle.generate_text.remote(data, greedy=True))

        # Parse outputs: if starts with YES it's correct
        rewards_list = []
        for txt in out["texts"]:
            lower = txt.strip().lower()
            passed = lower.startswith("yes")
            rewards_list.append(1 if passed else 0)

        rewards = torch.tensor(rewards_list, dtype=torch.int)

        # One-step episodic task – terminate immediately
        terminateds = torch.ones_like(rewards)

        # No special stop-strings or metadata updates
        next_stop_strings = [None] * batch_size
        metadata = [None] * batch_size

        return EnvironmentReturn(
            observations=observations,
            metadata=metadata,
            next_stop_strings=next_stop_strings,
            rewards=rewards,
            terminateds=terminateds,
        )

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict
    ) -> Tuple[BatchedDataDict, dict]:
        """Provide minimal metrics required by the trainer."""
        metrics = {"accuracy": batch["rewards"].float().mean().item()}
        return batch, metrics
