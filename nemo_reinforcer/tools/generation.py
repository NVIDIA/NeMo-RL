from typing import Dict, List, Any
import re
import warnings
from pprint import pformat

import ray
import torch
from torch.nn.utils.rnn import pad_sequence

from transformers import AutoTokenizer

from nemo_reinforcer.distributed.batched_data_dict import BatchedDataDict
from nemo_reinforcer.models.generation.interfaces import (
    GenerationInterface,
    GenerationDatumSpec,
    GenerationOutputSpec,
)
from nemo_reinforcer.tools.interfaces import ToolInterface
from nemo_reinforcer.tools.tools import StatefulCodeExecutor


LOGIT_INFINITY = 1000


def unpad(batched_tensor, lengths):
    lists = []
    for tensor, length in zip(batched_tensor, lengths):
        lists.append(tensor[:length].tolist())
    return lists


def generate_with_code_and_tools(
    policy: GenerationInterface,
    input_batch: BatchedDataDict[GenerationDatumSpec],
    tokenizer: AutoTokenizer,
    execute_code: bool = True,
    tool_map: Dict[str, ToolInterface] = {},
    tag: str = "<code>",
    result_tag: str = "<result>",
    *args,
    **kwargs,
) -> BatchedDataDict[GenerationOutputSpec]:
    """Invoke policy.generate() with code execution and tool use."""
    if tool_map and not execute_code:
        warnings.warn(
            "Tool use requires code execution, but code execution is disabled. All the tools will be ignored."
        )

    batch = input_batch.copy()
    start_tag = tag
    end_tag = tag.replace("<", "</")
    result_start = result_tag
    result_end = result_tag.replace("<", "</")

    batch_size = len(batch["input_ids"])
    stop_strings = batch["stop_strings"] if "stop_strings" in batch else []
    stop_strings = [stop_strings + [end_tag]] * batch_size
    batch["stop_strings"] = stop_strings
    old_logprobs = None
    # input_ids: (batch_size, max_length)
    # input_lengths: (batch_size,)
    # stop_strings: (batch_size, *)

    active_batch = batch
    active_indices = torch.arange(batch_size)
    executors = [StatefulCodeExecutor.remote(tool_map) for _ in range(batch_size)]
    completed_output_ids = [None] * batch_size
    completed_logprobs = [None] * batch_size

    i = 0
    while len(active_indices) > 0:
        print(f"==================== iteration {i} ====================")
        print(f"batch: {batch}")
        print(f"active indices: {active_indices}")
        i += 1

        generation_outputs = policy.generate(active_batch, *args, **kwargs)
        # generation_lengths: (batch_size)
        # logprobs: (batch_size, max_length) unbounded logits, default to 0
        # output_ids: (batch_size, max_length)
        # unpadded_sequence_lengths: (batch_size)
        # max length = 131072
        output_ids = generation_outputs["output_ids"]
        # only contains logprobs for newly generated tokens
        logprobs = generation_outputs["logprobs"]
        input_lengths = active_batch["input_lengths"]
        total_lengths = generation_outputs["unpadded_sequence_lengths"]
        if old_logprobs is not None:
            # recover logprobs for tokens generated in previous iterations
            for i, input_length in enumerate(input_lengths):
                logprobs[i, :input_length] = old_logprobs[i, :input_length]

        generated_ids = []
        for output_id, input_length, total_length in zip(
            output_ids, input_lengths, total_lengths
        ):
            generated_ids.append(output_id[input_length:total_length])
        print(f"generated_ids = {generated_ids}")

        # decode the tokens of incompleted samples
        generated_texts = tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )

        is_code = []
        exprs = []
        lookaheads = []
        # parse newly generated texts
        for i, (generated_text, active_index, total_length) in enumerate(
            zip(generated_texts, active_indices, total_lengths)
        ):
            print(f"generated text = {generated_text}")
            match = re.search(
                rf"{start_tag}(.*){end_tag}(.*)", generated_text, re.DOTALL
            )
            if match:
                # stop is caused by code execution
                # expr takes everything between <code> and </code>, including new lines
                # lookahead takes everything after </code>
                is_code.append(i)
                expr, lookahead = match.groups()
                exprs.append(expr)
                lookaheads.append(lookahead)
            else:
                # stop is not caused by code execution
                # e.g. eos token, max length or other stop strings
                completed_output_ids[active_index] = output_ids[i, :total_length]
                completed_logprobs[active_index] = logprobs[i, :total_length]
        if len(is_code) == 0:
            break

        futures = []
        for i, expr, lookahead in zip(is_code, exprs, lookaheads):
            active_index = active_indices[i]
            future = executors[active_index].__call__.remote(expr)
            futures.append(future)
        results = ray.get(futures)
        print(f"got results:\n{results}")

        new_results = []
        for result in results:
            if result is None:
                result = ""
                new_results.append(result)
                continue
            result = pformat(result)
            if "\n" in expr or "\n" in result:
                # multi-line format
                result = f"\n\n{result_start}\n{result}\n{result_end}"
            else:
                # inline format
                result = f"{result_start}{result}{result_end}"
            if lookahead:
                if result.startswith(lookahead):
                    # The generation may look like "</code>\n" if ">\n" is a single token.
                    # We trim \n from the result if the model has already generated it.
                    result = result[len(lookahead) :]
                else:
                    warnings.warn(
                        f"Expect the generation to stop at {repr(end_tag)}, but got {repr(end_tag + lookahead)}. "
                        "This is because some characters are merged into a single token by the tokenizer. "
                        "These extra characters will be kept in the generation."
                    )
            new_results.append(result)

        encodings = tokenizer(
            new_results,
            add_special_tokens=False,
            padding=True,
            padding_side="right",
            return_tensors="pt",
        )
        result_ids = encodings["input_ids"]
        result_lengths = encodings["attention_mask"].sum(dim=1).to(torch.int32)
        print(f"result_ids = {result_ids}")
        print(f"result_lengths = {result_lengths}")

        is_code = torch.tensor(is_code)
        print(f"is_code = {is_code}")
        # reduce active batch
        print(f"active batch before reduction = {active_batch}")
        active_batch = active_batch.select_indices(is_code)
        print(f"active batch after reduction = {active_batch}")
        active_indices = active_indices[is_code]
        output_ids = output_ids[is_code]
        logprobs = logprobs[is_code]
        total_lengths = total_lengths[is_code]
        # max length before appending results
        old_max_length = total_lengths.max()
        # max length after appending results
        new_max_length = (total_lengths + result_lengths).max()
        new_output_ids = torch.full(
            (len(active_indices), new_max_length),
            tokenizer.pad_token_id,
            dtype=output_ids.dtype,
        )
        new_logprobs = torch.full(
            (len(active_indices), new_max_length), 0, dtype=logprobs.dtype
        )
        new_output_ids[:, :old_max_length] = output_ids[:, :old_max_length]
        new_logprobs[:, :old_max_length] = logprobs[:, :old_max_length]

        # append results to generation
        for i, (old_length, result_length) in enumerate(
            zip(total_lengths, result_lengths)
        ):
            new_length = old_length + result_length
            new_output_ids[i, old_length:new_length] = result_ids[i, :result_length]
            new_logprobs[i, old_length:new_length] = LOGIT_INFINITY

        active_batch["input_ids"] = new_output_ids
        active_batch["input_lengths"] = total_lengths + result_lengths
        print(f"new_output_ids = {new_output_ids}")
        print(f"input_lengths = {total_lengths + result_lengths}")
        old_logprobs = new_logprobs

    output_ids = pad_sequence(
        completed_output_ids,
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
        padding_side="right",
    )
    logprobs = pad_sequence(
        completed_logprobs, batch_first=True, padding_value=0.0, padding_side="right"
    )
    total_lengths = torch.tensor([len(output_id) for output_id in completed_output_ids])
    generation_lengths = total_lengths - input_batch["input_lengths"]

    return {
        "output_ids": output_ids,
        "logprobs": logprobs,
        "generation_lengths": generation_lengths,
        "unpadded_sequence_lengths": total_lengths,
    }
