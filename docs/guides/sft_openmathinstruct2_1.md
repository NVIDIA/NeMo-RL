# SFT on OpenMathInstruct-2

This guide explains how to use NeMo RL to run SFT on the [nvidia/OpenMathInstruct-2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2) math instruction tuning dataset. 

## NeMo-Skills
We closely follow the [official instructions](https://nvidia.github.io/NeMo-Skills/openmathinstruct2/) of the [original OpenMathInstruct-2 paper](https://arxiv.org/abs/2410.01560). First, install NeMo-Skills:

```
pip install git+https://github.com/NVIDIA/NeMo-Skills.git 
```

Next, run `ns setup` to configure your local/slurm environment.

## Preparing the Data
### Download the Data
Download the data from [nvidia/OpenMathInstruct-2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2) and convert the data to jsonl format:


```python
import json

from datasets import load_dataset
from tqdm import tqdm

dataset = load_dataset('nvidia/OpenMathInstruct-2', split='train')

print("Converting dataset to jsonl format")
output_file = "openmathinstruct2.jsonl"
with open(output_file, 'w', encoding='utf-8') as f:
    for item in tqdm(dataset):
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"Conversion complete. Output saved as {output_file}")
```

### Convert to SFT Format
We then convert the data to SFT format by applying the Llama3 Instruct chat template and adding a generic math prompt:

```python
ns run_cmd --cluster=local \
python -m nemo_skills.training.prepare_data \
    ++prompt_template=llama3-instruct \
    ++prompt_config=generic/math \
    ++preprocessed_dataset_files=/workspace/openmathinstruct2.jsonl \
    ++output_key=generated_solution \
    ++output_path=/workspace/openmathinstruct2-sft.jsonl \
    ++hf_model_name="meta-llama/Meta-Llama-3.1-8B" \
    ++filters.drop_multi_boxed=false \
    ++filters.trim_prefix=false \
    ++filters.trim_solutions=false \
    ++filters.drop_incorrect_arithmetic=false \
    ++filters.split_arithmetic=false \
    ++filters.remove_contaminated=false
```

## Training the Model
To train the model using NeMo RL, use the `examples/configs/sft_openmath_instruct.yaml` config file. This file matches the experiment settings in the [original OpenMathInstruct-2 paper](https://arxiv.org/abs/2410.01560):

```
uv run examples/run_sft.py --config=examples/configs/sft_openmathinstruct.yaml
```


## Evaluating the Model
Throughout training, the checkpoints of the model will be saved to the `results/sft_openmath` folder. To evaluate the model, we first need to convert our PyTorch distributed checkpoints to HuggingFace format:

```
uv run examples/convert_dcp_to_hf.py \
    --config=/path/to/checkpoint/config.yaml \
    --dcp-ckpt-path=/path/to/checkpoint/policy/weights \
    --hf-ckpt-path=/path/to/checkpoint/hf
```

We can then use NeMo-Skills to evaluate our model on the MATH dataset:

```
ns eval \
    --cluster=slurm \
    --server_type=vllm \
    --model=/path/to/checkpoint/hf \
    --server_gpus=8 \
    --benchmarks=math:0 \
    --output_dir=/path/to/output \
    ++prompt_template=llama3-instruct \
```

To get the final scores, run:
```
ns summarize_results /path/to/output/eval-results --cluster local
```

## Results
Using the above steps to train the model to 16,000 steps, we see the following results:

```
--------------------------- math ---------------------------
evaluation_mode | num_entries | symbolic_correct | no_answer
greedy          | 5000        | 59.08%           | 5.44%
```

Using the official instructions to train a model using NeMo-Aligner, we a comparable accuracy:
```
--------------------------- math ---------------------------
evaluation_mode | num_entries | symbolic_correct | no_answer
greedy          | 5000        | 60.44%           | 4.04%
```



---
## WIP Notes

Alternatively, we can use NeMo-RL to evaluate on the MATH-500 dataset:

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