# FSDP2 Parallel Plan

This guide outlines the parallelization strategy for FSDP2 training in NeMo-RL.

## Fallback Priority

Three parallelization approaches are supported, with the following fallback priority.

**Custom Parallel Plan**

User-defined custom parallel plans take precedence when available.

For implementation details and usage guidelines, please refer to [Custom Parallel Plan Example](#custom-parallel-plan-example).

**Optimized Parallel Plan**

Optimized parallel plans are available for specific model architectures and may offer superior performance compared to the Hugging Face tensor parallel implementation.

This approach is used when no custom parallel plan is specified and the model class supports optimized parallelization.

**Hugging Face Tensor Parallel Plan**

Hugging Face provides tensor parallelism for most models through `._tp_plan`.

It serves as the default when neither custom nor optimized parallel plans are available.

## Custom Parallel Plan Example

Custom parallel plan should be defined in a file, exemplified by `examples/custom_parallel.py`.

To implement the custom parallel plan, configure `policy.dtensor_cfg.custom_parallel_plan=examples.custom_parallel.custom_parallel_plan`.

```python
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel
from torch.distributed.tensor.placement_types import Replicate, Shard


custom_parallel_plan = {
    "model.embed_tokens": RowwiseParallel(input_layouts=Replicate()),
    "model.layers.*.self_attn.q_proj": ColwiseParallel(),
    "model.layers.*.self_attn.k_proj": ColwiseParallel(),
    "model.layers.*.self_attn.v_proj": ColwiseParallel(),
    "model.layers.*.self_attn.o_proj": RowwiseParallel(),
    "model.layers.*.mlp.up_proj": ColwiseParallel(),
    "model.layers.*.mlp.gate_proj": ColwiseParallel(),
    "model.layers.*.mlp.down_proj": RowwiseParallel(),
    "lm_head": ColwiseParallel(output_layouts=Shard(-1), use_local_output=False),
}
```
