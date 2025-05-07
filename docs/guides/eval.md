# Evaluation

This document explains how to use an evaluation script for assessing model capabilities.

## Start Evaluation

To run the evaluation, you can use the [default configuration file](../../examples/configs/eval.yaml) or specify a custom one or overwrite some settings via command line.

The default configuration file will use greedy sampling strategy to evaluate Qwen2.5-Math-1.5B-Instruct on AIME-2024.

### Start Script

**Evaluate Standard HF Format Models:**

To run evaluation using a model directly from Hugging Face Hub or a local path already in HF format, use the `run_eval.py` script.

```sh
# To run the evaluation with default config (examples/configs/eval.yaml)
uv run python examples/run_eval.py

# To run the evaluation with your own config file
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

**Evaluate Models Trained with DCP Checkpoints (DPO/GRPO/SFT):**

If you have trained a model using DPO, GRPO or SFT and saved the checkpoint in the Pytorch DCP format, you first need to convert it to the Hugging Face format before running evaluation.

1.  **Convert DCP to HF:**
    Use the `examples/convert_dcp_to_hf.py` script. You'll need the path to the training configuration file (`config.yaml`), the DCP checkpoint directory, and specify an output path for the HF format model.

    ```sh
    # Example for a GRPO checkpoint at step 170
    uv run python examples/convert_dcp_to_hf.py \
        --config results/grpo/step_170/config.yaml \
        --dcp-ckpt-path results/grpo/step_170/policy/weights/ \
        --hf-ckpt-path results/grpo/hf
    ```
    *Note: Adjust the paths according to your training output directory structure.*

2.  **Run Evaluation on Converted Model:**
    Once the conversion is complete, run the evaluation script, overriding the `generation.model_name` to point to the directory containing the converted HF model.

    ```sh
    # Example using the converted HF model from the previous step
    uv run python examples/run_eval.py generation.model_name=$PWD/results/grpo/hf
    ```

### Example Output

```
============================================================
model_name='Qwen2.5-Math-1.5B-Instruct' dataset_name='aime_2024'
max_new_tokens=2048 temperature=0.0 top_p=1.0 top_k=-1

metric='pass@1' num_tests_per_prompt=1

score=0.1000 (3.0/30)
============================================================
```

## Example Configuration File

You can find an example evaluation configuration file [here](../../examples/configs/eval.yaml).

### Prompt Template Configuration

Always remember to use the same prompt and chat_template that were used during training.

For open-source models, we recommend setting `tokenizer.chat_template=default`, `data.prompt_file=null` and `data.system_prompt_file=null` to allow them to use their native chat templates.
