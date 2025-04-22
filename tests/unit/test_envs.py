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

import ray
import torch
from typing import Dict, List, Tuple, Optional, TypedDict, Literal, Any

from nemo_reinforcer.distributed.batched_data_dict import BatchedDataDict
from nemo_reinforcer.data.interfaces import LLMMessageLogType
from nemo_reinforcer.environments.interfaces import (
    EnvironmentInterface,
    EnvironmentReturn,
)
from nemo_reinforcer.distributed.virtual_cluster import PY_EXECUTABLES
from .environments.sliding_puzzle_game import SlidingPuzzleGame


class MultiStepCalcMetadata(TypedDict):
    problem: str
    expected_final_answer: float
    max_steps: int
    current_step: int


class SlidingPuzzleMetadata(TypedDict):
    game_state: Dict[str, Any]  # Stores the dict returned by SlidingPuzzleGame methods
    num_moves: int
    max_moves: int


class _MultiStepCalculatorLogic:
    def __init__(self):
        pass

    def _parse_tool_call(self, text: str) -> Optional[Tuple[float, float, str]]:
        """Parses '[opA, opB, operation]<call: calculator>'."""
        # Use a more distinct tool call suffix
        tool_call_suffix = "<call: calculator>"
        if not text.strip().endswith(tool_call_suffix):
            return None

        content = text.strip()[: -len(tool_call_suffix)].strip()
        if not (content.startswith("[") and content.endswith("]")):
            return None
        parts = content[1:-1].split(",")
        if len(parts) != 3:
            return None
        try:
            op_a = float(parts[0].strip())
            op_b = float(parts[1].strip())
            operation = parts[2].strip().lower()
            return op_a, op_b, operation
        except ValueError:
            return None

    def _calculate(self, op_a: float, op_b: float, operation: str) -> Optional[float]:
        """Performs the calculation."""
        # (Reusing the calculation logic)
        if operation == "sum":
            return op_a + op_b
        elif operation == "diff":
            return op_a - op_b
        elif operation == "prod":
            return op_a * op_b
        elif operation == "div":
            if abs(op_b) < 1e-6:
                return None  # Division by zero error
            return op_a / op_b
        else:
            return None  # Unknown operation

    def _is_final_answer(self, text: str) -> Optional[float]:
        """Checks if the text is just a final numerical answer."""
        try:
            # Allow potential formatting like <final_answer>16.0</final_answer>
            # or just the number itself.
            processed_text = text.strip()
            if processed_text.startswith("<final_answer>") and processed_text.endswith(
                "</final_answer>"
            ):
                processed_text = processed_text[
                    len("<final_answer>") : -len("</final_answer>")
                ]

            return float(processed_text)
        except ValueError:
            return None

    def process_turn(
        self,
        message_log: LLMMessageLogType,
        metadata: MultiStepCalcMetadata,
    ) -> Tuple[
        Dict[str, str],
        float,
        bool,
        Optional[List[str]],
        Optional[MultiStepCalcMetadata],
    ]:
        """Processes a single turn for the multi-step calculator task."""
        last_assistant_msg = ""
        if message_log and message_log[-1]["role"] == "assistant":
            last_assistant_msg = message_log[-1]["content"].strip()

        current_step = metadata["current_step"]
        max_steps = metadata["max_steps"]
        expected_final_answer = metadata["expected_final_answer"]

        turn_reward = 0.0
        is_terminated = False
        next_stop_strings = [
            "<call: calculator>"
        ]  # Let model generate tool call or final answer freely
        next_metadata = metadata.copy()
        next_observation_content = ""

        # Check if max steps reached
        if current_step >= max_steps:
            is_terminated = True
            next_observation_content = "<error>Maximum steps reached."
            next_metadata = None
            return (
                {"role": "environment", "content": next_observation_content},
                0.0,
                is_terminated,
                None,
                next_metadata,
            )

        # Check for final answer first
        final_answer = self._is_final_answer(last_assistant_msg)
        if final_answer is not None:
            is_terminated = True
            next_metadata = None  # End of episode
            if abs(final_answer - expected_final_answer) < 1e-6:
                turn_reward = 1.0  # Correct final answer
                next_observation_content = (
                    f"Correct! The final answer is {final_answer:.2f}."
                )
            else:
                turn_reward = 0.0  # Incorrect final answer
                next_observation_content = f"Incorrect final answer. Expected {expected_final_answer:.2f}, got {final_answer:.2f}."
        else:
            # Check for tool call
            parsed_call = self._parse_tool_call(last_assistant_msg)
            if parsed_call:
                req_op_a, req_op_b, req_op = parsed_call
                result = self._calculate(req_op_a, req_op_b, req_op)
                if result is not None:
                    # Tool call success, provide result
                    next_observation_content = f"<result>{result:.5f}</result>"
                    next_metadata["current_step"] += 1
                    is_terminated = False
                else:  # Calculation failed
                    is_terminated = True
                    next_observation_content = "<error>Calculation failed."
                    next_metadata = None
            else:  # No final answer and no valid tool call
                is_terminated = True
                next_observation_content = (
                    "<error>Invalid response. Expected tool call or final answer."
                )
                next_metadata = None

        next_observation = {"role": "environment", "content": next_observation_content}
        return (
            next_observation,
            turn_reward,
            is_terminated,
            next_stop_strings,
            next_metadata,
        )


class _SlidingPuzzleLogic:
    def __init__(self):
        pass  # No initialization needed as game methods are static

    def _parse_action(self, text: str) -> Optional[str]:
        """Parses the action from '<action></action>'"""
        prefix = "<action>"
        suffix = "</action>"
        # Find the prefix, case-insensitive, and potentially after some thought process
        text_lower = text.lower()
        prefix_lower = prefix.lower()
        suffix_lower = suffix.lower()

        start_idx = text_lower.rfind(prefix_lower)  # Find the last occurrence

        if start_idx != -1:
            # Find the end tag after the start tag
            end_idx = text_lower.find(suffix_lower, start_idx + len(prefix_lower))
            if end_idx != -1:
                # Extract content between tags
                action_content = text[start_idx + len(prefix) : end_idx].strip()
                return action_content
        return None

    def process_turn(
        self,
        message_log: LLMMessageLogType,
        metadata: SlidingPuzzleMetadata,
    ) -> Tuple[
        Dict[str, str],
        float,
        bool,
        Optional[List[str]],
        Optional[SlidingPuzzleMetadata],
    ]:
        """Processes a single turn for the sliding puzzle task."""
        game_state = metadata["game_state"]
        current_moves = metadata["num_moves"]
        max_moves = metadata["max_moves"]

        turn_reward = 0.0
        is_terminated = False
        next_stop_strings = ["</action>"]
        next_metadata = metadata.copy()
        next_observation_content = ""

        # Check if max moves reached
        if current_moves >= max_moves:
            is_terminated = True
            next_observation_content = (
                f"<error>Maximum moves ({max_moves}) reached.</error>"
            )
            next_metadata = None
            return (
                {"role": "environment", "content": next_observation_content},
                0.0,
                is_terminated,
                None,
                next_metadata,
            )

        # Get last assistant message and parse action
        last_assistant_msg_content = ""
        if message_log and message_log[-1]["role"] == "assistant":
            last_assistant_msg_content = message_log[-1]["content"].strip()

        parsed_action = self._parse_action(last_assistant_msg_content)

        if parsed_action is None:
            # Handle cases where parsing failed or it wasn't assistant's turn properly
            # is_terminated = True  # Penalize for bad format
            rendered_board = SlidingPuzzleGame.render(game_state)
            next_observation_content = f"<environment>\n{rendered_board}\n\nInvalid response format no move made. Try <action></action> like this: <action>your_action</action></environment>"
            next_metadata = None
        elif parsed_action == "view":
            rendered_board = SlidingPuzzleGame.render(game_state)
            next_observation_content = f"<environment>\n{rendered_board}\n\nViewing the board. No move made.</environment>"
        else:
            # Execute the game step
            step_response, reward, game_over, next_game_state = SlidingPuzzleGame.step(
                parsed_action, game_state
            )

            turn_reward = reward
            is_terminated = game_over
            next_metadata["game_state"] = next_game_state
            next_metadata["num_moves"] = current_moves + 1

            # Combine rendered board and step response for the next observation
            rendered_board = SlidingPuzzleGame.render(next_game_state)
            # next_observation_content = f"<environment>\n{rendered_board}\n\n{step_response}</environment>"
            next_observation_content = f"<environment>\n{step_response}\n</environment>"
            # next_observation_content = f"\n{step_response}"

            if is_terminated:
                next_metadata = None  # Clear metadata on termination
                # next_stop_strings remains None

        return (
            {"role": "environment", "content": next_observation_content + "\n"},
            turn_reward,
            is_terminated,
            next_stop_strings,
            next_metadata,
        )


@ray.remote
class MultiStepCalculatorEnv(EnvironmentInterface):
    DEFAULT_PY_EXECUTABLE = PY_EXECUTABLES.SYSTEM
    """Multi-step calculator environment (Ray Actor)."""

    def __init__(self, cfg: Optional[Dict] = None):
        self.logic = _MultiStepCalculatorLogic()

    def step(
        self,
        message_log_batch: List[LLMMessageLogType],
        metadata_batch: List[MultiStepCalcMetadata],
    ) -> EnvironmentReturn:
        """Processes a batch of interactions using the calculator logic."""
        futures = [
            self.logic.process_turn(log, meta)
            for log, meta in zip(message_log_batch, metadata_batch)
        ]
        results = futures

        # Unpack results and format according to EnvironmentReturn tuple
        observations = []
        rewards = []
        terminateds = []
        all_stop_strings = []  # List of Lists or Nones
        all_next_metadata = []

        for obs, rew, term, stops, meta in results:
            observations.append(obs)  # obs is already Dict[str, str]
            rewards.append(rew)
            terminateds.append(term)
            all_stop_strings.append(stops)
            all_next_metadata.append(meta)

        # Convert to tensors where needed
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        # Done flag combines termination and truncation (truncation not used here)
        done_tensor = torch.tensor(terminateds, dtype=torch.bool)

        # Return tuple matching EnvironmentReturn NamedTuple
        return EnvironmentReturn(
            observations=observations,
            metadata=all_next_metadata,
            next_stop_strings=all_stop_strings,
            rewards=rewards_tensor,
            terminateds=done_tensor,
        )

    def shutdown(self):
        pass

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict
    ) -> Tuple[BatchedDataDict, dict]:
        # Example: could calculate success rate based on final reward
        final_rewards = batch.get(
            "total_reward", torch.tensor([0.0] * len(batch["idx"]))
        )
        success_rate = final_rewards.mean().item() if len(final_rewards) > 0 else 0.0
        return batch, {"success_rate": success_rate}


@ray.remote
class SlidingPuzzleEnv(EnvironmentInterface):
    DEFAULT_PY_EXECUTABLE = PY_EXECUTABLES.SYSTEM
    """Sliding Puzzle environment (Ray Actor)."""

    def __init__(self, cfg: Optional[Dict] = None):
        # cfg could contain game generation config like {'size': 3, 'shuffle_moves': 50}
        self.game_config = cfg.get("game_config", {}) if cfg else {}
        self.logic = _SlidingPuzzleLogic()

    def step(
        self,
        message_log_batch: List[LLMMessageLogType],
        metadata_batch: List[SlidingPuzzleMetadata],
    ) -> EnvironmentReturn:
        """Processes a batch of sliding puzzle interactions."""
        # Since logic is synchronous, process sequentially (can parallelize if logic becomes heavy)
        results = [
            self.logic.process_turn(log, meta)
            for log, meta in zip(message_log_batch, metadata_batch)
        ]

        # Unpack results and format according to EnvironmentReturn NamedTuple
        observations = []
        rewards = []
        terminateds = []
        all_stop_strings = []
        all_next_metadata = []

        for obs, rew, term, stops, meta in results:
            observations.append(obs)
            rewards.append(rew)
            terminateds.append(term)
            all_stop_strings.append(stops)
            all_next_metadata.append(meta)

        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        terminated_tensor = torch.tensor(terminateds, dtype=torch.bool)

        return EnvironmentReturn(
            observations=observations,
            metadata=all_next_metadata,
            next_stop_strings=all_stop_strings,
            rewards=rewards_tensor,
            terminateds=terminated_tensor,
        )

    def shutdown(self):
        pass

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict
    ) -> Tuple[BatchedDataDict, dict]:
        # Calculate success rate based on final reward == 1.0
        final_rewards = batch.get(
            "total_reward", torch.tensor([0.0] * len(batch["idx"]))
        )
        success_rate = (
            (final_rewards == 1.0).float().mean().item()
            if len(final_rewards) > 0
            else 0.0
        )
        # Could also calculate average number of moves for successful episodes, etc.
        return batch, {"sliding_puzzle_success_rate": success_rate}
