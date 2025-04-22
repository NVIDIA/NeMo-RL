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

import random
import copy
from typing import List, Tuple, Dict, Any, Optional
from .game_interface import GameInterface


class SlidingPuzzleGame(GameInterface):
    @staticmethod
    def generate(config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a new Sliding Puzzle."""
        size = config.get("size", 4)  # Default to 4x4 (15-puzzle)
        shuffle_moves = config.get(
            "shuffle_moves", 100
        )  # Number of random moves for shuffling

        # Create the solved state
        grid = [[(r * size + c + 1) for c in range(size)] for r in range(size)]
        # Set the bottom-right corner to 0 (empty space)
        grid[size - 1][size - 1] = 0

        # Save the solution
        solution = [row[:] for row in grid]

        # Find the empty space
        empty_pos = (size - 1, size - 1)

        # Shuffle the grid with valid moves
        for _ in range(shuffle_moves):
            # Get possible moves
            moves = []
            r, c = empty_pos
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # Right, Down, Left, Up
                nr, nc = r + dr, c + dc
                if 0 <= nr < size and 0 <= nc < size:
                    moves.append((nr, nc))

            # Choose a random move
            if moves:
                new_r, new_c = random.choice(moves)
                # Swap the empty space with the chosen tile
                grid[r][c], grid[new_r][new_c] = grid[new_r][new_c], grid[r][c]
                empty_pos = (new_r, new_c)

        # Create and return the game state
        return {
            "size": size,
            "grid": grid,
            "solution": solution,
            "empty_pos": empty_pos,
            "commands": {
                "slide r c": "Slide tile at row r, column c into the empty space",
                "up": "Slide tile below empty space up",
                "down": "Slide tile above empty space down",
                "left": "Slide tile to the right of empty space left",
                "right": "Slide tile to the left of empty space right",
            },
        }

    @staticmethod
    def init(game_state: Dict[str, Any]) -> str:
        """Initialize Sliding Puzzle game and return welcome message."""
        size = game_state["size"]

        return (
            f"\n===== SLIDING PUZZLE =====\n"
            f"Arrange the {size}x{size} grid by sliding tiles into the empty space.\n"
            f"- The goal is to arrange numbers from 1 to {size * size - 1} in order\n"
            f"- Use 'slide r c' to slide a specific tile\n"
            f"- Or use 'up', 'down', 'left', 'right' to slide in that direction"
        )

    @staticmethod
    def step(
        action: str, game_state: Dict[str, Any]
    ) -> Tuple[str, float, bool, Dict[str, Any]]:
        """Process an action in the Sliding Puzzle game."""
        size = game_state["size"]
        grid = game_state["grid"]
        empty_r, empty_c = game_state["empty_pos"]

        # Default return values
        response = "Unknown command. Type 'help' to see available commands."
        reward = -0.05  # Small penalty for invalid actions
        is_terminated = False

        # Deep copy game state to avoid modifying the original
        new_state = copy.deepcopy(game_state)

        move_made = False

        if action.startswith("slide "):
            try:
                _, r, c = action.split()
                r, c = int(r) - 1, int(c) - 1

                # Validate input
                if not (0 <= r < size and 0 <= c < size):
                    return (
                        f"Invalid position. Row/column must be between 1 and {size}.",
                        reward,
                        is_terminated,
                        new_state,
                    )

                # Check if tile is adjacent to empty space
                if abs(r - empty_r) + abs(c - empty_c) != 1:
                    return (
                        "Tile must be adjacent to the empty space.",
                        reward,
                        is_terminated,
                        new_state,
                    )

                # Slide the tile
                new_state["grid"][empty_r][empty_c] = grid[r][c]
                new_state["grid"][r][c] = 0
                new_state["empty_pos"] = (r, c)

                move_made = True
                response = f"Slid tile {grid[r][c]} into the empty space."

            except ValueError:
                return (
                    "Invalid input format. Use: slide row col",
                    reward,
                    is_terminated,
                    new_state,
                )

        elif action in ["up", "down", "left", "right"]:
            # Convert direction to row/col offset
            if action == "up":
                r, c = empty_r + 1, empty_c  # Tile below moves up
                dir_text = "up"
            elif action == "down":
                r, c = empty_r - 1, empty_c  # Tile above moves down
                dir_text = "down"
            elif action == "left":
                r, c = empty_r, empty_c + 1  # Tile to right moves left
                dir_text = "left"
            elif action == "right":
                r, c = empty_r, empty_c - 1  # Tile to left moves right
                dir_text = "right"

            # Check if the move is valid
            if 0 <= r < size and 0 <= c < size:
                # Slide the tile
                new_state["grid"][empty_r][empty_c] = grid[r][c]
                new_state["grid"][r][c] = 0
                new_state["empty_pos"] = (r, c)

                move_made = True
                response = f"Slid tile {grid[r][c]} {dir_text}."
            else:
                return f"Cannot slide {dir_text}.", reward, is_terminated, new_state

        if move_made:
            reward = 0

            # Check if puzzle is solved
            if new_state["grid"] == new_state["solution"]:
                response = "Congratulations! You've solved the puzzle!"
                reward = 1.0  # Win reward
                is_terminated = True

        return response, reward, is_terminated, new_state

    @staticmethod
    def render(game_state: Dict[str, Any]) -> str:
        """Render the current Sliding Puzzle game state."""
        grid = game_state["grid"]
        size = game_state["size"]

        output = ["\n"]

        # Create a visual representation of the grid
        max_digits = len(str(size * size - 1))

        # Top border
        output.append("  " + "+" + "-" * (max_digits + 2) * size + "+")

        # Rows
        for i, row in enumerate(grid):
            row_str = f"{i + 1} |"
            for val in row:
                if val == 0:
                    # Empty space
                    row_str += " " * (max_digits + 2)
                else:
                    # Tile with number
                    row_str += f" {val:>{max_digits}} "
            row_str += "|"
            output.append(row_str)

        # Bottom border
        output.append("  " + "+" + "-" * (max_digits + 2) * size + "+")

        # Column labels
        col_labels = "    "
        for i in range(size):
            col_labels += f"{i + 1:^{max_digits + 2}}"
        output.append(col_labels)

        return "\n".join(output)


def is_solvable(grid: List[List[int]], size: int) -> bool:
    """Check if a sliding puzzle is solvable."""
    # Flatten the grid
    flat = [num for row in grid for num in row if num != 0]

    # Count inversions
    inversions = 0
    for i in range(len(flat)):
        for j in range(i + 1, len(flat)):
            if flat[i] > flat[j]:
                inversions += 1

    # Find row of the empty tile (0) from the bottom
    empty_row = 0
    for i in range(size - 1, -1, -1):
        for j in range(size):
            if grid[i][j] == 0:
                empty_row = size - i
                break

    # For odd-sized grids, the puzzle is solvable if the number of inversions is even
    if size % 2 == 1:
        return inversions % 2 == 0
    # For even-sized grids, the puzzle is solvable if:
    # (inversions odd and empty on even row from bottom) or (inversions even and empty on odd row from bottom)
    else:
        return (inversions % 2 == 1 and empty_row % 2 == 0) or (
            inversions % 2 == 0 and empty_row % 2 == 1
        )


def play_sliding_puzzle(config=None):
    """Wrapper function for backward compatibility."""
    from play_game import play_game

    play_game(SlidingPuzzleGame, config)
