# An in-depth walkthrough of GRPO in Reinforcer

## Quickstart: Launch a GRPO Run

If you want to get running quickly, the script [examples/run_grpo_math.py](../../examples/run_grpo_math.py) has an example implementation of using GRPO to train a model on math problems. This script can either be launched locally or via Slurm. For details on how to set up Ray and launch a job using Slurm, refer to the [cluster documentation](../cluster.md).

We recommend launching the job using `uv`:
```bash
uv run examples/run_grpo_math.py --config <PATH TO YAML CONFIG> {overrides}
```
If not specified, `config` will default to [examples/configs/grpo.yaml](../../examples/configs/grpo.yaml)
**Reminder**: Don't forget to set your HUGGINGFACE_TOKEN, HF_HOME, and WANDB_API_KEY (if needed)

## Now, for the details:

In this guide, we'll walk through we handle
* Data
* Model training
* Fast generation
* Overall Resource Flow

### Data
We support training with multiple RL "Environments" at the same time.

An [Environment](../../nemo_reinforcer/environments/interfaces.py) is an object that accepts a state/action history and returns an update state and rewards for the step. They run as Ray Remote Actors. Example [MathEnvironment](../../nemo_reinforcer/environments/math_environment.py).

To support this, we need to know:
* What environments you have
* Which data should go to which environments
* How to prepare the data from your dataset into a form we can use

#### Common Data Format
We define a [DatumSpec](../../nemo_reinforcer/data/interfaces.py) that holds all relevant information for each training example:
```python
class DatumSpec(TypedDict):
    message_log: LLMMessageLogType
    length: int  # total (concatenated) length of the message tensors
    extra_env_info: Dict[str, Any] # anything your environment requires goes here, for example the 'answer' of a math problem
    loss_multiplier: float  # multiplier for the loss for this datum. 0 to mask out (say the sample is invalid)
    idx: int
    task_name: Optional[str] = "default"
    __extra__: Any  # This allows additional fields of any type
```

#### Data Processors
We name all distinct "environments your model wants to optimize against" "tasks". So you might define a "math" task or a "code" task. 
For each task, you should provide a data processor that reads from your dataset and returns a [DatumSpec](../../nemo_reinforcer/data/interfaces.py)

```python
def my_data_processor(
    datum_dict: Dict[str, Any], # loaded directly from your dataset (i.e. single line of jsonl data)
    task_data_spec: TaskDataSpec,
    tokenizer,
    max_seq_length: int,
    idx: int,
) -> DatumSpec:
```
We have an example of this as `math_data_processor` in [run_grpo_math.py](../../examples/run_grpo_math.py)

#### Putting it all together:
GRPO expects datasets to have the following form:
```json
{"task_name": "math", <actual data>}
```
Then, you can set data up as such:
```python
base_dataset = load_dataset("json", data_files=data_config["dataset_name"])["train"]
tokenizer = AutoTokenizer.from_pretrained(policy_config["model_name"])

task_data_processors = defaultdict(lambda: (math_task_spec, math_data_processor))
task_data_processors["math"] = (math_task_spec, math_data_processor)

math_env = MathEnvironment.remote(env_configs["math"]) # ray remote actor

dataset = AllTaskProcessedDataset(
    base_dataset,
    tokenizer,
    math_task_spec,
    task_data_processors,
    max_seq_length=data_config["max_input_seq_length"],
)
```
Notice that you provide a mapping of tasks to their processors so the dataset knows what to use when processing samples.


### Policy Model
We define a [PolicyInterface]() that contains everything you need to train a Policy model.

This Policy object holds a [RayWorkerGroup](../../nemo_reinforcer/distributed/worker_groups.py) of SPMD (1 proc/gpu) processes that run HF/MCore, all coordinated by this object so it appears to you like 1 GPU!

### Fast Generation
We support vLLM through the [VllmGeneration](../../nemo_reinforcer/models/generation/vllm.py) class right now.

The function [grpo_train](../../nemo_reinforcer/algorithms/grpo.py) contains the core GRPO training loop.