# SFT on OpenMathInstruct-2

This guide explains how to use NeMo RL to run SFT on the [nvidia/OpenMathInstruct-2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2) math instruction tuning dataset. We then evaluate the trained model on the [MATH-500 benchmark](https://huggingface.co/datasets/HuggingFaceH4/MATH-500).


## Training the Model
To train the model using NeMo RL, use the `examples/configs/recipes/tutorials/sft/sft_openmathinstruct2.yaml` config file. This file closely matches the experiment settings in the [original OpenMathInstruct-2 paper](https://arxiv.org/abs/2410.01560).

```
uv run examples/run_sft.py --config=examples/configs/recipes/tutorials/sft/sft_openmathinstruct2.yaml
```

## Evaluating the Model
To evaluate the model, we first need to convert our PyTorch distributed checkpoints to HuggingFace format:

```
uv run examples/convert_dcp_to_hf.py \
    --config=/path/to/checkpoint/config.yaml \
    --dcp-ckpt-path=/path/to/checkpoint/policy/weights \
    --hf-ckpt-path=/path/to/checkpoint/hf
```

Then, to evaluate on the MATH-500 dataset:

```
uv run examples/run_eval.py \
    --config=examples/configs/recipes/tutorials/sft/eval_math500.yaml \
    generation.model_name=/path/to/hf/checkpoint
```

Running the above, we see the following results at 16,000 steps:

```
============================================================
model_name='hf' dataset_name='MATH-500'
max_new_tokens=2048 temperature=0.0 top_p=1.0 top_k=-1

metric='pass@1' num_tests_per_prompt=1

score=0.6000 (300.0/500)
============================================================
```

Evaluating the NeMo-Aligner checkpoint using the same method, we see:

```
============================================================
model_name='hf' dataset_name='MATH-500'
max_new_tokens=2048 temperature=0.0 top_p=1.0 top_k=-1

metric='pass@1' num_tests_per_prompt=1

score=0.5940 (297.0/500)
============================================================
```

For comparison, evaluating on Llama3.1-8B-Instruct results in:
```
============================================================
model_name='Llama-3.1-8B-Instruct' dataset_name='MATH-500'
max_new_tokens=2048 temperature=0.0 top_p=1.0 top_k=-1

metric='pass@1' num_tests_per_prompt=1

score=0.4480 (224.0/500)
============================================================
```

## WIP Notes
Slight difference between NeMo-Skills method and NeMo RL method:

For the NeMo-Skills method, the header of the assistant response is part of the prompt and not predicted in SFT training:

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nSolve the following math problem. Make sure to put the answer (and only answer) inside \\boxed{}.\n\nA circle is inscribed in a regular hexagon. What is the ratio of the area of the circle to the area of the hexagon?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n
```

In NeMo RL SFT, the entire assistant response including the headers is part of the prediction (i.e., it is not masked out):
```
Prompt: '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nSolve the following math problem. Make sure to put the answer (and only answer) inside \\\\boxed{}.\n\nFactor the following expression: $\\frac{1}{x^2} + \\frac{2}{x} + 1$.<|eot_id|>'

Completion: "<|start_header_id|>assistant<|end_header_id|>\n\nLet's analyze the given expression:\n\\[ \\frac{1}{x^2} + \\frac{2}{x} + 1 \\]\n\nWe can notice that this expression resembles the perfect square of a binomial:\n\\[ \\left( \\frac{1}{x} + 1 \\right)^2 \\]\n\nExpanding the square:\n\\[ \\left( \\frac{1}{x} + 1 \\right)^2 = \\frac{1}{x^2} + 2\\left( \\frac{1}{x} \\right) + 1^2 = \\frac{1}{x^2} + \\frac{2}{x} + 1 \\]\n\nThis matches the original expression.\n\nThus, the factored form of the expression is:\n\\[ \\frac{1}{x^2} + \\frac{2}{x} + 1 = \\boxed{\\left( \\frac{1}{x} + 1 \\right)^2} \\]<|eot_id|>"
```

We take the loss over all the "Completion" tokens.


Issue saving last checkpoint