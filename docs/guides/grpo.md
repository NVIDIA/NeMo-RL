# An in-depth walkthrough of GRPO in Reinforcer

## Quickstart: Launch a GRPO Run

If you want to get running quickly, the script [examples/run_grpo_math.py](../../examples/run_grpo_math.py) has an example implementation of using GRPO to train a model on math problems. This script can either be launched locally or via Slurm. For details on how to set up Ray and launch a job using Slurm, refer to the [cluster documentation](../cluster.md).

We recommend launching the job using `uv`:
```bash
uv run examples/run_grpo_math.py --config <PATH TO YAML CONFIG> {overrides}
```
If not specified, `config` will default to [examples/configs/grpo.yaml](../../examples/configs/grpo.yaml)

**Reminder**: Don't forget to set your HF_HOME and WANDB_API_KEY (if needed). You'll need to do a `huggingface-cli login` as well for Llama models.

## Now, for the details:

In this guide, we'll walk through how we handle
* Data
* Model training
* Fast generation
* Overall Resource Flow
* Loss

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

### Loss
We use the [ClippedPGLossFn](../../nemo_reinforcer/algorithms/loss_functions.py) to calculate the loss for GRPO. Formally,

$$
L(\theta) = E_{x \sim \pi_{\theta_{\text{old}}}} \Big[ \min \Big(\frac{\pi_\theta(x)}{\pi_{\theta_{\text{old}}}(x)}A_t, \text{clip} \big( \frac{\pi_\theta(x)}{\pi_{\theta_{\text{old}}}(x)}, 1 - \varepsilon, 1 + \varepsilon \big) A_t \Big) \Big] - \beta D_{\text{KL}} (\pi_\theta \| \pi_\text{ref}) \tag{1}
$$

where:

- $\pi_\theta$ is the policy model we are currently optimizing
- $\pi_{\theta_{\text{old}}}$ is the previous policy model
- $A_t$ is the advantage estimate
- $\varepsilon$ is a clipping hyperparameter
- $\beta$ is the KL penalty coefficient
- $\pi_{\text{ref}}$ is the reference policy

#### On-Policy KL Approximation

In practice, we calculate the KL divergence using the estimator from Schulman 2020 (http://joschu.net/blog/kl-approx.html), which is unbiased and guaranteed to be positive.

$$
D_{\text{KL}} (\pi_\theta \| \pi_\text{ref}) \approx E_{x \sim \pi_{\theta}} \Big[ \frac{\pi_\text{ref}(x)}{\pi_\theta(x)} - \log \frac{\pi_\text{ref}(x)}{\pi_\theta(x)} - 1 \Big]
$$

Note that the loss function above (Equation 1) samples from $\pi_{\theta_{\text{old}}}$ instead of $\pi_\theta$, meaning that the kl approximation is off-policy if we use samples from $\pi_{\theta_{\text{old}}}$. This is the default formulation used in the [original GRPO paper](https://arxiv.org/abs/2402.03300). In order to use an _on-policy_ KL approximation while sampling from $\pi_{\theta_{\text{old}}}$, we can incorporate importance weights:

$$
\begin{align*}
D_{\text{KL}} (\pi_\theta \| \pi_\text{ref}) &\approx E_{x \sim \pi_{\theta}} \Big[ \frac{\pi_\text{ref}(x)}{\pi_\theta(x)} - \log \frac{\pi_\text{ref}(x)}{\pi_\theta(x)} - 1 \Big] \\
&= \frac{1}{N}\sum_x \pi_{\theta}(x) \Big[ \frac{\pi_\text{ref}(x)}{\pi_\theta(x)} - \log \frac{\pi_\text{ref}(x)}{\pi_\theta(x)} - 1 \Big] \\
&= \frac{1}{N}\sum_x \pi_{\theta_{\text{old}}}(x) \frac{\pi_{\theta}(x)}{\pi_{\theta_{\text{old}}}(x)} \Big[ \frac{\pi_\text{ref}(x)}{\pi_\theta(x)} - \log \frac{\pi_\text{ref}(x)}{\pi_\theta(x)} - 1 \Big] \\
&= E_{x \sim \pi_{\theta_\text{old}}} \frac{\pi_{\theta}(x)}{\pi_{\theta_{\text{old}}}(x)} \Big[ \frac{\pi_\text{ref}(x)}{\pi_\theta(x)} - \log \frac{\pi_\text{ref}(x)}{\pi_\theta(x)} - 1 \Big] \\
\end{align*}
$$

where $N$ is the total number of samples. To enable the on-policy kl approximation, set the config `use_on_policy_kl_approximation=True` in the `ClippedPGLossConfig`.


#### Importance Sampling Correction
The policy we use to draw samples, $\pi_{\theta_{\text{old}}}$, is used in both the inference framework and the training framework. To account for this distinction, we refer to the inference framework policy as $\pi_{\text{inference}}$ and the training framework policy as $\pi_{\text{training}}$. As noted in [Adding New Models](../adding_new_models.md#understanding-discrepancies-between-backends), it is possible that the token probabilities from $\pi_{\text{training}}$ and $\pi_{\text{inference}}$ to have discrepancies, leading to off-policy samples. We can correct for this by introducing importance weights between $\pi_{\text{training}}$ and $\pi_{\text{inference}}$ to the first term of the loss function. Let $f_\theta(x) = \min \Big(\frac{\pi_\theta(x)}{\pi_{\theta_{\text{old}}}(x)}A_t, \text{clip} \big( \frac{\pi_\theta(x)}{\pi_{\theta_{\text{old}}}(x)}, 1 - \varepsilon, 1 + \varepsilon \big) A_t \Big)$ represent the first term of Equation 1. Then,

$$
\begin{align*}
E_{x \sim \pi_\text{training}}(x) f_\theta(x) &= \frac{1}{N}\sum_x \pi_\text{training}(x) f_\theta(x) \\
&= \frac{1}{N}\sum_x \pi_\text{inference}(x) \frac{\pi_\text{training}(x)}{\pi_\text{inference}(x)} f_\theta(x) \\
&= E_{x \sim \pi_\text{inference}} \frac{\pi_\text{training}(x)}{\pi_\text{inference}(x)} f_\theta(x)
\end{align*}
$$

By multiplying the first term of Equation 1 by the importance weights $\frac{\pi_\text{training}(x)}{\pi_\text{inference}(x)}$, we can correct for the distribution mismatch between $\pi_{\text{training}}$ and $\pi_{\text{inference}}$ while still sampling from $\pi_{\text{inference}}$.

To enable the importance sampling correction, set the config `use_importance_sampling_correction=True` in the `ClippedPGLossConfig`.
