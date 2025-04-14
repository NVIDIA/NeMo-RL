# Direct Preference Optimization in Reinforcer

## Launch a DPO Run

The script [examples/run_dpo.py](../../examples/run_dpo.py) can be used to launch a DPO experiment. This script can either be launched locally or via Slurm. For details on how to set up Ray and launch a job using Slurm, refer to the [cluster documentation](../cluster.md).

Be sure to launch the job using `uv`. The command to launch a DPO job is as follows:
```bash
uv run examples/run_dpo.py --config <PATH TO YAML CONFIG> <OVERRIDES>
```
If not specified, `config` will default to [examples/configs/dpo.yaml](../../examples/configs/dpo.yaml).

## Configuration

Reinforcer allows users to configure DPO experiments using `yaml` config files. An example DPO configuration file can be found [here](../../examples/configs/dpo.yaml).

To override a value in the config, either update the value in the `yaml` file directly, or pass the override via the command line. For example:

```bash
uv run examples/run_dpo.py \
    cluster.gpus_per_node=8 \
    dpo.sft_loss_weight=0.1 \
    dpo.preference_average_log_probs=True \
    logger.wandb.name="dpo-dev-8-gpu"
```

**Reminder**: Don't forget to set your HF_HOME and WANDB_API_KEY (if needed). You'll need to do a `huggingface-cli login` as well for Llama models.

## Datasets

Each class representing a Reinforcer DPO dataset is expected to have the following attributes:
1. `formatted_ds`: The dictionary of formatted datasets. This dictionary should contain `train` and `validation` splits, and each split should conform to the format described below.
2. `task_spec`: The `TaskDataSpec` for this dataset. This should specify the name you choose for this dataset.

DPO datasets are expected to follow a specific format with three key fields:
- `prompt`: The input prompt/context
- `chosen_response`: The preferred/winning response
- `rejected_response`: The non-preferred/losing response

[data/hf_datasets/helpsteer3.py](../../nemo_reinforcer/data/hf_datasets/helpsteer3.py) provides an example of how to format data for DPO:

```python
def format_helpsteer3(data):
    response_1 = data["response1"]
    response_2 = data["response2"]
    overall_preference = data["overall_preference"]

    return {
        "prompt": data["context"],
        "chosen_response": response_1 if overall_preference < 0 else response_2,
        "rejected_response": response_2 if overall_preference < 0 else response_1,
    }
```

We also provide a [DPODataset](../../nemo_reinforcer/data/hf_datasets/dpo.py) class that is compatible with jsonl-formatted preference datsets. This class assumes train and validation datasets have been split and processed into the expected format offline. The jsonl files should consist of examples with `prompt`, `chosen_response`, and `rejected_response` keys.

## Adding Custom DPO Datasets

Adding a new DPO dataset is straightforward. Your custom dataset class should:
1. Implement the required format conversion in the constructor
2. Set up the appropriate `task_spec`

Here's a minimal example which simply re-keys an existing jsonl dataset:

```python
from datasets import load_dataset
from nemo_reinforcer.data.hf_datasets.interfaces import HfDataset
from nemo_reinforcer.data.interfaces import TaskDataSpec

class CustomDPODataset:
    def preprocess_dataset(
        self,
        data,
        prompt_key: str = "context",
        chosen_key: str = "chosen",
        rejected_key: str = "rejected"
    ):

        return {
            "prompt": data[prompt_key],
            "chosen_response": data[chosen_key],
            "rejected_response": data[rejected_key],
        }

    
    def __init__(
        self,
        train_data_path: str,
        val_data_path: str,
        prompt_key: str,
        chosen_key: str,
        rejected_key: str,
    ):

        # Load and format your dataset
        fn_kwargs={
                "prompt_key": prompt_key, 
                "chosen_key": chosen_key, 
                "rejected_key": rejected_key
            }
        formatted_ds = {
            "train": load_dataset("json", data_files=train_data_path, split="train").map(
                self.preprocess_dataset, 
                fn_kwargs=fn_kwargs,
            ),
            "validation": load_dataset("json", data_files=val_data_path, split="train").map(
                self.preprocess_dataset, 
                fn_kwargs=fn_kwargs,
            ),
        }
        
        # Initialize task spec with dataset name
        self.task_spec = TaskDataSpec(
            dataset_name="custom_dpo",
        )
```

## DPO-Specific Parameters

The DPO implementation in Reinforcer supports several key parameters that can be adjusted:

- `dpo.reference_policy_kl_penalty`: Controls the strength of the KL penalty term
- `dpo.preference_loss_weight`: Weight for the preference loss
- `dpo.sft_loss_weight`: Weight for the auxiliary SFT loss
- `dpo.preference_average_log_probs`: Whether to average log probabilities over tokens in the preference loss term
- `dpo.sft_average_log_probs`: Whether to average log probabilities over tokens in the SFT loss term

These parameters can be adjusted in the config file or via command-line overrides to optimize training for your specific use case.
