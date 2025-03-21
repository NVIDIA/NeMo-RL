# Supervised Fine-tuning in Reinforcer

## Launch an SFT Run

The script [examples/run_sft.py](../../examples/run_sft.py) can be used to launch an experiment. This script can either be launched locally or via Slurm. For details on how to set up Ray and launch a job using Slurm, refer to the [cluster documentation](../cluster.md).

Be sure to launch the job using `uv`. The command to launch an SFT job is as follows:
```bash
uv run examples/run_sft.py --config <PATH TO YAML CONFIG> <OVERRIDES>
```
If not specified, `config` will default to [examples/configs/sft.yaml](../../examples/configs/sft.yaml).

## Configuration

Reinforcer allows users to configure experiments using `yaml` config files. An example SFT configuration file can be found [here](../../examples/configs/sft.yaml).

To override a value in the config, either update the value in the `yaml` file directly, or pass the override via the command line. For example:

```bash
uv run examples/run_sft.py \
    cluster.gpus_per_node=1 \
    logger.wandb.name="sft-dev-1-gpu"
```
**Reminder**: Don't forget to set your HF_HOME and WANDB_API_KEY (if needed). You'll need to do a `huggingface-cli login` as well for Llama models.

## Datasets

SFT datasets in Reinforcer are encapsulated using classes. Each SFT data class is expected to have the following attributes:
  1. `formatted_ds`: The dictionary of formatted datasets. This dictionary should contain `train` and `validation` splits, and each split should conform to the format described below.
  2. `task_spec`: The `TaskDataSpec` for this dataset. This should specify the name you choose for this dataset as well as the `custom_template` for this dataset. More on custom templates below.

SFT datasets are expected to follow the HuggingFace chat format. Refer to the [chat dataset document](../design_docs/chat_datasets.md) for details. If your data is not in the correct format, simply write a preprocessing script to convert the data into this format. [data/hf_datasets/squad.py](../../nemo_reinforcer/data/hf_datasets/squad.py) has an example:

```python
def format_squad(data):
    return {
        "messages": [
            {
                "role": "system",
                "content": data["context"],
            },
            {
                "role": "user",
                "content": data["question"],
            },
            {
                "role": "assistant",
                "content": data["answers"]["text"][0],
            },
        ]
    }
```

Reinforcer SFT uses HuggingFace chat templates to format the individual examples. If you would like to use a custom template, create a string template in [jinja format](https://huggingface.co/docs/transformers/v4.34.0/en/chat_templating#how-do-i-create-a-chat-template) and pass it to the dataset's `TaskDataSpec`. For example,

```python
custom_template = (
    "{% for message in messages %}{%- if message['role'] == 'system'  %}{{'Context: ' + message['content'].strip()}}{%- elif message['role'] == 'user'  %}{{' Question: ' + message['content'].strip() + ' Answer: '}}{%- elif message['role'] == 'assistant'  %}{{message['content'].strip()}}{%- endif %}{% endfor %}"
)
task_spec = TaskDataSpec(
    task_name="squad",
    custom_template=custom_template,
)
```

By default, NeMo-Reinforcer has support for `Squad` and `OpenAssistant` datasets. Both of these datasets are downloaded from HuggingFace and preprocessed on-the-fly, so there's no need to provide a path to any datasets on disk.

Adding a new dataset is a straightforward process.
As long as your custom dataset has the `formatted_ds` and `task_spec` attributes described above, it can serve as a drop-in replacement for Squad and OpenAssistant.