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

from typing import Dict, Any, Tuple, List, Optional, Callable


class GameInterface:
    @staticmethod
    def generate(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a new game state based on configuration.

        Args:
            config: Game configuration dictionary

        Returns:
            A dictionary containing the complete game state
        """
        raise NotImplementedError("Each game must implement generate()")

    @staticmethod
    def init(game_state: Dict[str, Any]) -> str:
        """
        Initialize a game and return welcome messages.

        Args:
            game_state: The game state dictionary

        Returns:
            String containing welcome message and instructions
        """
        raise NotImplementedError("Each game must implement init()")

    @staticmethod
    def step(
        action: str, game_state: Dict[str, Any]
    ) -> Tuple[str, float, bool, Dict[str, Any]]:
        """
        Process a game action and update the state.

        Args:
            action: String representing the player's action
            game_state: Current game state dictionary

        Returns:
            Tuple containing:
            - Response message
            - Reward for this action
            - Boolean indicating if game is terminated
            - Updated game state
        """
        raise NotImplementedError("Each game must implement step()")

    @staticmethod
    def render(game_state: Dict[str, Any]) -> str:
        """
        Render the current game state as a string.

        Args:
            game_state: The game state dictionary

        Returns:
            String representation of the game state
        """
        raise NotImplementedError("Each game must implement render()")
