# Checkpointing with HuggingFace Models

## Checkpoint Format
Reinforcer provides two checkpoint formats for HuggingFace models: Torch distributed and HuggingFace format. Torch distributed is used by default for efficiency, and HuggingFace format is provided for compatibility with HuggingFace's `AutoModel.from_pretrained` API. Note that HuggingFace format checkpoints save only the model weights, ignoring the optimizer states. It is recommended to use Torch distributed format to save intermediate checkpoints and to save a HuggingFace checkpoint only at the end of training. 

There are two ways to get a Reinforcer checkpoint in HuggingFace format.

1. (Recommended) Save the HuggingFace checkpoint directly by passing `save_hf=True` to `HFPolicy`'s `save_checkpoint`:
    
    ```python
    policy.save_checkpoint(
        weights_path=<WHERE_TO_SAVE_MODEL_WEIGHTS>,
        optimizer_path=<WHERE_TO_SAVE_OPTIM_STATE>,
        save_torch_dist=True,
        save_hf=True,
    )
    ```
2. Convert a Torch distributed checkpoint checkpoint to HuggingFace format after training. We provide a conversion script for this purpose.

    ```python
    uv run examples/convert_dcp_to_hf.py --config=<YAML CONFIG USED DURING TRAINING> <ANY CONFIG OVERRIDES USED DURING TRAINING> --dcp-ckpt-path=<PATH TO DIST CHECKPOINT TO CONVERT> --hf-ckpt-path=<WHERE TO SAVE HF CHECKPOINT>
    ```