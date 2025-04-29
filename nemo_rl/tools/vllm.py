import re
from copy import deepcopy
from pprint import pformat
from typing import Dict, List, TypedDict

import torch
from nemo_reinforcer.tools.interfaces import ToolInterface
from nemo_reinforcer.tools.tools import StatefulCodeExecutor
from transformers import PreTrainedTokenizerBase


class ProcessorState(TypedDict):
    # Variables to pass through for each sample. Necessary for batch decoding.
    is_expr: bool
    expr_ids: List[int]
    result_ids: List[int]
    tool_map: Dict[str, ToolInterface]
    executor: StatefulCodeExecutor


class CodeLogitsProcessor:
    """Process code execution on-the-fly in vLLM generation.

    We use a stateful code executor, so every code snippet have access to functions and variables
    defined in all previous snippets.

    When a model generates a code snippet like

    ``
    <code>
    x = 3
    x ** 2
    </code>
    ``

    this processor will automatically execute the code, and append the result (if any) to the generation

    ``
    <result>
    9
    </result>
    ``

    Args:
        tokenizer: tokenizer from the pretrained model
        tool_map: tools that the model can use
        tag: xml tag to detect code snippet
        result_tag: xml tag to output the result
    """

    LOGIT_INFINITY = 1000

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        tool_map: Dict[str, ToolInterface] = {},
        tag: str = "<code>",
        result_tag: str = "<result>",
    ):
        self.tokenizer = tokenizer
        self.tool_map = tool_map
        self.start_tag = tag
        self.end_tag = tag.replace("<", "</")
        self.result_start = result_tag
        self.result_end = result_tag.replace("<", "</")
        self.start_tag_len = len(
            tokenizer.encode(self.start_tag, add_special_tokens=False)
        )
        self.end_tag_len = len(tokenizer.encode(self.end_tag, add_special_tokens=False))

        self.states: Dict[str, ProcessorState] = {}

    def __call__(self, input_ids: List[int], scores: torch.Tensor) -> torch.Tensor:
        seq_id = hash(tuple(input_ids))
        if len(input_ids) > 0:
            last_seq_id = hash(tuple(input_ids[:-1]))
            state = self.states[last_seq_id].copy()
        else:
            state = {
                "is_expr": False,
                "result_ids": [],
                "executor": StatefulCodeExecutor(self.tool_map),
            }
        self.states[seq_id] = state

        if state["result_ids"]:
            # ignore generation and return unconsumed result tokens
            id, *state["result_ids"] = state["result_ids"]
            scores[id] = self.LOGIT_INFINITY
            return scores

        if not state["is_expr"]:
            # start tag contains at most len(start_tag) tokens
            generation = self.tokenizer.decode(input_ids[-self.start_tag_len :])

            if self.start_tag in generation:
                state["is_expr"] = True
                state["expr_ids"] = list(input_ids[-self.start_tag_len :])
        else:
            state["expr_ids"] = state["expr_ids"] + [input_ids[-1]]
            # end tag contains at most len(end_tag) tokens
            generation = self.tokenizer.decode(input_ids[-self.end_tag_len :])
            if self.end_tag in generation:
                state["is_expr"] = False
                expr = self.tokenizer.decode(state["expr_ids"])
                expr = re.search(
                    rf"(?<={self.start_tag}).*(?={self.end_tag})", expr, re.DOTALL
                ).group(0)

                # avoid messing up executor states of other samples
                state["executor"] = deepcopy(state["executor"])
                result = state["executor"](expr)

                if result is None:
                    # silent mode
                    return scores
                result = pformat(result)
                if "\n" in expr or "\n" in result:
                    # multi-line format
                    result = f"\n\n{self.result_start}\n{result}\n{self.result_end}"
                else:
                    # inline format
                    result = f"{self.result_start}{result}{self.result_end}"

                lookahead = generation[
                    generation.find(self.end_tag) + len(self.end_tag) :
                ]
                if result.startswith(lookahead):
                    # trim template string from result if the model has already generated it
                    result = result[len(lookahead) :]

                id, *state["result_ids"] = self.tokenizer.encode(
                    result, add_special_tokens=False
                )
                scores[id] = self.LOGIT_INFINITY

        return scores
