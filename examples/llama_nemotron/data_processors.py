from typing import Any, Dict, List

import torch
from transformers import AutoTokenizer

from nemo_rl.data.interfaces import DatumSpec, LLMMessageLogType, TaskDataSpec


def _append_message(tokenizer: AutoTokenizer, msg_log: LLMMessageLogType, role: str, text: str, gen_prompt: bool):
    """Render a single chat message and append token_ids in-place."""
    message = {"role": role, "content": text}
    rendered = tokenizer.apply_chat_template(
        [message], tokenize=False, add_generation_prompt=gen_prompt, add_special_tokens=False
    )
    message["token_ids"] = tokenizer(rendered, return_tensors="pt")["input_ids"][0]
    message["content"] = rendered
    msg_log.append(message)


def _base_processor(
    prompt_text: str,
    extra_info: Dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer: AutoTokenizer,
    max_seq_length: int,
    idx: int,
    task_name: str,
    row_system_prompt: str | None = None,
) -> DatumSpec:
    """Common logic shared by all processors."""
    message_log: LLMMessageLogType = []

    # optional system prompt – row-level overrides spec-level
    sys_prompt = row_system_prompt if row_system_prompt is not None else task_data_spec.system_prompt
    if sys_prompt:
        _append_message(tokenizer, message_log, "system", sys_prompt, gen_prompt=False)

    # user message
    user_prompt = task_data_spec.prompt.format(prompt_text) if task_data_spec.prompt else prompt_text
    _append_message(tokenizer, message_log, "user", user_prompt, gen_prompt=True)

    length = sum(len(m["token_ids"]) for m in message_log)
    loss_multiplier = 1.0
    if max_seq_length and length > max_seq_length:
        # truncate aggressively and mask out of loss
        for m in message_log:
            m["token_ids"] = m["token_ids"][: min(4, max_seq_length // len(message_log))]
        loss_multiplier = 0.0

    return {
        "message_log": message_log,
        "length": length,
        "extra_env_info": extra_info,
        "loss_multiplier": loss_multiplier,
        "idx": idx,
        "task_name": task_name,
    }


# -----------------------------------------------------------------------------
# Processors per task
# -----------------------------------------------------------------------------


def llm_judge_scp_116k_processor(
    datum_dict: Dict[str, Any], task_data_spec: TaskDataSpec, tokenizer, max_seq_length: int, idx: int
) -> DatumSpec:
    prompt_text = datum_dict.get("prompt", "")
    args_dict = datum_dict.get("args", {})
    extra_info = {
        "ground_truth": args_dict.get("expected_answer"),
        "prompt": args_dict.get("prompt"),
        "extract_box": args_dict.get("extract_box", False),
    }
    return _base_processor(
        prompt_text,
        extra_info,
        task_data_spec,
        tokenizer,
        max_seq_length,
        idx,
        datum_dict.get("task_name", "llm_judge_scp116k"),
        datum_dict.get("system_prompt"),
    )


def code_processor(
    datum_dict: Dict[str, Any], task_data_spec: TaskDataSpec, tokenizer, max_seq_length: int, idx: int
) -> DatumSpec:
    prompt_text = datum_dict["prompt"]

    args_dict = datum_dict.get("args", {})
    extra_info = {
        "unittests": args_dict.get("unittests"),
        "test_type": args_dict.get("test_type"),
        "fn_name": args_dict.get("fn_name"),
    }
    return _base_processor(
        prompt_text,
        extra_info,
        task_data_spec,
        tokenizer,
        max_seq_length,
        idx,
        datum_dict.get("task_name", "code"),
        datum_dict.get("system_prompt"),
    )


def genrm_processor(
    datum_dict: Dict[str, Any], task_data_spec: TaskDataSpec, tokenizer, max_seq_length: int, idx: int
) -> DatumSpec:
    prompt_text = datum_dict["prompt"]
    args_dict = datum_dict.get("args")
    extra_info = {
        "num_responses": args_dict.get("num_responses"),
        "helpfulness_1": args_dict.get("helpfulness_1"),
        "helpfulness_2": args_dict.get("helpfulness_2", None),
        "preference_ranking": args_dict.get("preference_ranking", None),
    }
    return _base_processor(
        prompt_text,
        extra_info,
        task_data_spec,
        tokenizer,
        max_seq_length,
        idx,
        datum_dict.get("task_name", "genrm"),
        datum_dict.get("system_prompt"),
    )


def math_processor(
    datum_dict: Dict[str, Any], task_data_spec: TaskDataSpec, tokenizer, max_seq_length: int, idx: int
) -> DatumSpec:
    problem = datum_dict["problem"]
    extra_info = {"ground_truth": datum_dict["expected_answer"]}
    return _base_processor(
        problem,
        extra_info,
        task_data_spec,
        tokenizer,
        max_seq_length,
        idx,
        datum_dict.get("task_name", "math"),
        datum_dict.get("system_prompt"),
    )
