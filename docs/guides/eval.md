# Evaluation

This document explains how to use an evaluation script for assessing model capabilities.

## Prepare For Evaluation

### Convert DCP to HF (Optional)
If you have trained a model and saved the checkpoint in the Pytorch DCP format, you first need to convert it to the Hugging Face format before running evaluation.

Use the `examples/convert_dcp_to_hf.py` script. You'll need the path to the training configuration file (`config.yaml`), the DCP checkpoint directory, and specify an output path for the HF format model.

```sh
# Example for a GRPO checkpoint at step 170
uv run python examples/convert_dcp_to_hf.py \
    --config results/grpo/step_170/config.yaml \
    --dcp-ckpt-path results/grpo/step_170/policy/weights/ \
    --hf-ckpt-path results/grpo/hf
```
*Note: Adjust the paths according to your training output directory structure.*

Once the conversion is complete, you can override the `generation.model_name` to point to the directory containing the converted HF model in [this section](#run-evaluation-script).

### Prepare Configuration
**Override with custom settings**

To run the evaluation, you can use the [default configuration file](../../examples/configs/eval.yaml) or specify a custom one or override some settings via command line.

The default configuration file will use greedy sampling strategy to evaluate Qwen2.5-Math-1.5B-Instruct on AIME-2024.

**Prompt Template Configuration**

Always remember to use the same prompt and chat_template that were used during training.

For open-source models, we recommend setting `tokenizer.chat_template=default`, `data.prompt_file=null` and `data.system_prompt_file=null` to allow them to use their native chat templates.

## Run Evaluation Script

We will use the `run_eval.py` script to run evaluation using a model directly from Hugging Face Hub or a local path already in HF format.

Note that the eval script only supports for the HF format model. If you haven't converted your DCP format model, you should back to [Convert DCP to HF](#convert-dcp-to-hf-optional) and follow the guide to convert your model.

```sh
# Run evaluation script with default config (examples/configs/eval.yaml)
uv run python examples/run_eval.py

# Run evaluation script with converted model
uv run python examples/run_eval.py generation.model_name=$PWD/results/grpo/hf

# Run evaluation script with custom config file
uv run python examples/run_eval.py --config path/to/custom_config.yaml

# Override specific config values via command line
# Example: Evaluation of DeepScaleR-1.5B-Preview on MATH-500 using 8 GPUs
#          Pass@1 accuracy averaged over 16 samples for each problem
uv run python examples/run_eval.py \
    generation.model_name=agentica-org/DeepScaleR-1.5B-Preview \
    generation.temperature=0.6 \
    generation.top_p=0.95 \
    generation.vllm_cfg.max_model_len=32768 \
    data.dataset_name=HuggingFaceH4/MATH-500 \
    data.dataset_key=test \
    eval.num_tests_per_prompt=16 \
    cluster.gpus_per_node=8
```

## Example Evaluation Output

When you complete the evaluation, you will receive a summary similar to the following.

```
============================================================
model_name='Qwen2.5-Math-1.5B-Instruct' dataset_name='aime_2024'
max_new_tokens=2048 temperature=0.0 top_p=1.0 top_k=-1

metric='pass@1' num_tests_per_prompt=1

score=0.1000 (3.0/30)
============================================================
```
