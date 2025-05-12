# SFT on OpenMathInstruct-2

This guide explains how to use NeMo RL to run SFT on the [nvidia/OpenMathInstruct-2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2) math instruction tuning dataset. We then show how to use NeMo RL's evaluation scripts to evaluate the trained model on the [MATH-500 benchmark](https://huggingface.co/datasets/HuggingFaceH4/MATH-500).


## Training the Model
To train the model using NeMo RL, use the `examples/configs/recipes/tutorials/sft/sft_openmathinstruct2.yaml` config file. This file closely matches the experiment settings in the [original OpenMathInstruct-2 paper](https://arxiv.org/abs/2410.01560).

```
uv run examples/run_sft.py --config=examples/configs/sft_openmathinstruct2.yaml
```

### Dataset Splits

The OpenMathInstruct-2 has several versions of different sizes. Configure the version of the dataset via the `data.split` config:

* `train`: full 14 M problem–solution pairs
* `train_1M`, `train_2M`, `train_5M`: fair-downsampled subsets of 1M, 2M, or 5M examples

By default, the config uses the 1M subset (`data.split=train_1M`).

## Evaluating the Model
Throughout training, the checkpoints of the model will be saved to the `results/sft_openmathinstruct2` folder (specified by `checkpointing.checkpoint_dir`). To evaluate the model, we first need to convert the PyTorch distributed checkpoint to HuggingFace format:

```
uv run examples/convert_dcp_to_hf.py \
    --config=results/sft_openmathinstruct2/step_1855/config.yaml \
    --dcp-ckpt-path=results/sft_openmathinstruct2/step_1855/policy/weights \
    --hf-ckpt-path=results/sft_openmathinstruct2/step_1855/hf
```

Replace `results/sft_openmathinstruct2/step_1855` with the path to the checkpoint you are evaluating. The resulting HuggingFace checkpoint will be saved to `--hf-ckpt-path`.

To evaluate on the MATH-500 dataset:

```
uv run examples/run_eval.py \
    --config=examples/configs/eval_math500.yaml \
    generation.model_name=results/sft_openmathinstruct2/step_1855/hf
```

where `generation.model_name` is the path to the HuggingFace checkpoint. Running the above after training the Llama-3.1-8B model for 1 epoch on the train_1M version of the OpenMathInstruct-2 dataset, we get the following result:

```
============================================================
model_name='hf' dataset_name='MATH-500'
max_new_tokens=2048 temperature=0.0 top_p=1.0 top_k=-1

metric='pass@1' num_tests_per_prompt=1

score=0.6000 (300.0/500)
============================================================
```

As a reference, using NeMo-Aligner and NeMo-Skills to train and evaluate the model on the same dataset (as is done in the [original OpenMathInstruct-2 paper](https://arxiv.org/abs/2410.01560)) results in a score of 0.5020

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