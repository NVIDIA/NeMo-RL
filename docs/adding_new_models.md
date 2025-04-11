# Adding New Models

- [Adding New Models](#adding-new-models)
  - [Importance of Log Probability Consistency in Training and Inference](#importance-of-log-probability-consistency-in-training-and-inference)
  - [Understanding Discrepancies Between Backends](#understanding-discrepancies-between-backends)
  - [1. Hugging Face–Based Models](#1-hugging-facebased-models)
    - [Validation Workflow](#validation-workflow)
  - [2. Megatron Models](#2-megatron-models)
    - [Additional Validation](#additional-validation)
  - [3. Expected Error Threshold](#3-expected-error-threshold)
  - [4. Stress Testing Your Model in Reinforcer](#4-stress-testing-your-model-in-reinforcer)

This guide outlines how to integrate and validate a new model within **NeMo-Reinforcer**. Each new model must pass a standard set of compatibility tests before being considered ready to be used in RL pipelines.

:::{tip}
Jump to [Stress Testing Your Model in Reinforcer](#4-stress-testing-your-model-in-reinforcer) if 
you are looking for the script to test models.
:::

## Importance of Log Probability Consistency in Training and Inference

In on-policy RL, we sample tokens (actions) from the latest version of the policy, meaning the sampling distribution of token probabilities produced by the inference framework must closely match those from the training framework. If the inference framework produces significantly different probabilities, we effectively sample from a different distribution, leading to errors in the loss estimation.

As an example, we would see errors in naive KL estimation:

$$\text{KL} = E_{x \sim \pi}[\pi(x) - \pi_{\text{ref}}(x)]$$  

When summed/integrated, replacing the $x \sim \pi$ with $x \sim \pi_{\text{wrong}}$ leads to an error of:

$$\sum_{x} \left( \pi(x) - \pi_{\text{ref}}(x) \right) \left( \pi_{\text{wrong}}(x) - \pi(x) \right)$$  

So, to verify correctness, we calculate

$$
\frac{1}{n}\sum_{i=1}^{n\text{(tokens)}}\exp\left(\left\|\text{logprobs-train-fwk}_i - \text{logprobs-sampling-fwk}_i\right\|\right)
$$

where samples are drawn as $x \sim \pi_{\text{sampling-framework}}$

as a measure of multiplicative probability error for sampled tokens. Note that this is not exhaustive (the sampling framework could lack distribution support and we wouldn't catch it here, as $x \sim \pi_{\text{sampling-framework}}$). To get a much stricter guarantee on correctness, you should run this metric twice and average the results, where in the second run, you sample $x \sim \pi_{\text{training-framework}}$. In practice, we use just the former in our tests and find it sufficient.

## Understanding Discrepancies Between Backends

When validating models across different backends, you may encounter discrepancies in log probabilities. These differences can stem from various sources with effects ranging from negligible to significant:

- **Numerical precision differences**: Training and inference backends may differ in precision formats (FP32, FP16, BF16, FP8).
  - Training may use mixed precision while the inference backend may not
  - High-precision training with FP8 inference may not be numerically stable for certain models
  - Differences can occur at the layer level, with some layers in FP32 while others use lower precision

- **Implementation variations**: Subtle differences in how layer implementations like softmax, layer normalization, or attention mechanisms are implemented.
  - Attention/Norm layers (which could be fused) in TransformerEngine may not be bit-wise identical to implementations in inference backends
  - Inference backends may re-implement kernels (e.g., for SSM layers) leading to differences
  - Softmax in training frameworks may be calculated differently than in inference backends for numerical stability

- **KV/Prefill cache handling**: Differences in how key-value/prefill caches are managed during autoregressive generation.
  - In some cases, disabling the inference backend cache can resolve discrepancies

- **Parallelism effects**: Parallelisms like Tensor parallelism may introduce small variations

- **Inherent non-determinism**: Some neural network operations are inherently non-deterministic (e.g., `torch.cumsum`)

- **Prefill/Decoding kernel mismatch**: Different kernels for prefill and decoding phases may produce different log probabilities.
  - Training frameworks typically use prefill kernels, while inference backends may use both prefill kernels and specialized decoding kernels

- **Imperfect Refit**: Weight conversion from the training framework to the inference backend may be incomplete or data formats may be incorrect
  - If weights are reshaped or reordered incorrectly, generations tend to be very wrong
  - In some cases, if some weights in the inference backend are not refit after each training step, the error between training and inference log probabilities can diverge as training progresses

- **Batch size**: In some cases, `batch_size>1` may produce larger errors than `batch_size=1`

When investigating discrepancies beyond the acceptable threshold, focus on these areas and determine whether the differences appear systematically or only in specific contexts.


---

## 1. Hugging Face–Based Models

### Validation Workflow

When validating Hugging Face-based models, perform the following checks:

- **Compare log probabilities**  
  Ensure the generation log probabilities from inference backends like **vLLM** match those computed by HuggingFace. This comparison helps diagnose potential mismatches.

- **Test parallelism**  
  Verify consistency with other parallelism settings. 

- **Variance**  
  Repeat tests multiple times (e.g., 10 runs) to confirm that behavior is deterministic or within acceptable variance.

- **Check sequence lengths**  
  Perform inference on sequence lengths of 100, 1,000, and 10,000 tokens.  
  Ensure the model behaves consistently at each length.

- **Use real and dummy data**  
  - **Real data:** Tokenize and generate from actual text samples.  
  - **Dummy data:** Simple numeric sequences to test basic generation.

- **Vary sampling parameters**  
  Test both greedy and sampling generation modes.  
  Adjust temperature and top-p to confirm output consistency across backends.

- **Test different batch sizes**  
  Try with batch sizes of 1, 8, and 32 to ensure consistent behavior across different batch configurations.

---

## 2. Megatron Models

### Additional Validation

- **Compare Megatron outputs**  
  Ensure the Megatron forward pass aligns with HuggingFace and the generation log probabilities from inference backends like **vLLM**.

- **Parallel settings**  
  Match the same parallelism configurations used for the HuggingFace-based tests.  
  Confirm outputs remain consistent across repeated runs.

---

## 3. Expected Error Threshold

When comparing log probabilities between training and inference backends, we use an error threshold of `1.05` to determine acceptable variance (for equal precision). An error of `1.0` indicates a perfect match, and values exceeding `1.05` require further investigation.

When validating your model, you should analyze the results across different configurations. Your analysis should include:

| Sequence Length | Data Type  | Generation Method | Batch Size | HF vs VLLM | Megatron vs VLLM |
|-----------------|------------|-------------------|------------|------------|------------------|
| 100             | Real       | Greedy            | 1          | 1.02       | 1.01             |
| 100             | Real       | Sampling          | 8          | 1.03       | 1.02             |
| 100             | Synthetic  | Greedy            | 1          | 1.01       | 1.02             |
| 1,000           | Real       | Greedy            | 32         | 1.04       | 1.03             |
| ...             | ...        | ...               | ...        | ...        | ...              |

---

By following these validation steps and ensuring your model's outputs remain consistent across backends, you can confirm that your new model meets **NeMo-Reinforcer**'s requirements.

## 4. Stress Testing Your Model in Reinforcer

We provide a script to help stress test your model by running generation and computing the multiplicative probability error between 
the generation log probabilities and those produced by the training framework. You can use it like so:

```bash
# Download and test a model (hf backend vs vllm backend)
uv run examples/stress_test_model.py generation.model_name=meta-llama/Llama-3.2-1B-Instruct

# Test a model and also adjust max_seq_len (relevant for models like qwen with max_position_embeddings=4096)
uv run examples/stress_test_model.py generation.model_name=Qwen/Qwen2.5-Math-1.5B-Instruct generation.vllm_cfg.max_model_len=4096

# Test a huggingface model trained by Reinforcer or one on local disk
uv run examples/stress_test_model.py generation.model_name=/path/to/huggingface/checkpoint

# Test the huggingface backend for generation (Note: this is much slower, so consider making your configuration and experiment smaller)
uv run examples/stress_test_model.py generation.model_name=meta-llama/Llama-3.2-1B-Instruct generation.backend=hf

# Providing --test_refit will also test the refit code path
uv run examples/stress_test_model.py --test_refit generation.model_name=meta-llama/Llama-3.2-1B-Instruct
```

After the script completes, it will display the following, in this order:

1. All the examples it processed 
2. A summary table of results
3. Location of the log and JSON files to review the results (with the model_name and datetime as part of the filename)

The summary will look something like this and will depend on your configuration:
```SUMMARY RESULTS TABLE
                                                                                                                      
  ISL (avg)   OSL (avg)   Max New Tokens   Data Type   Generation Method   Batch Size   Refit        HF vs vllm(gen)  
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
     233.00      512.00              512   numbers     greedy                       8   no_refit              1.0001  
      88.94      466.50              512   random      greedy                       8   no_refit              1.0033  
      10.88      399.00              512   literal     greedy                       8   no_refit              1.0053  
     233.00      512.00              512   numbers     default                      8   no_refit              1.0001  
      88.94      466.50              512   random      default                      8   no_refit              1.0033  
      10.88      399.00              512   literal     default                      8   no_refit              1.0053  
     233.00     7960.00             8192   numbers     greedy                       8   no_refit              1.0000  
      88.94     6627.81             8192   random      greedy                       8   no_refit              1.0024  
      10.88     6154.38             8192   literal     greedy                       8   no_refit              1.0032  
     233.00     7960.00             8192   numbers     default                      8   no_refit              1.0000  
      88.94     6627.81             8192   random      default                      8   no_refit              1.0024  
      10.88     6154.38             8192   literal     default                      8   no_refit              1.0032  
```

To configure the default data used for this test, see the examples in [stress.yaml](../examples/configs/stress.yaml) under the `data.*` key.

