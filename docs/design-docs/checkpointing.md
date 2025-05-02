# Checkpointing with HuggingFace Models 

## Checkpoint Format 
NeMo-RL provides two checkpoint formats for HuggingFace models: Torch distributed and HuggingFace format. Torch distributed is used by default for efficiency, and HuggingFace format is provided for compatibility with HuggingFace's `AutoModel.from_pretrained` API. Note that HuggingFace format checkpoints save only the model weights, ignoring the optimizer states. It is recommended to use Torch distributed format to save intermediate checkpoints and to save a HuggingFace checkpoint only at the end of training. 

A checkpoint converter is provided to convert a Torch distributed checkpoint checkpoint to HuggingFace format after training: 

    ```python
    uv run examples/convert_dcp_to_hf.py --config=<YAML CONFIG USED DURING TRAINING> <ANY CONFIG OVERRIDES USED DURING TRAINING> --dcp-ckpt-path=<PATH TO DIST CHECKPOINT TO CONVERT> --hf-ckpt-path=<WHERE TO SAVE HF CHECKPOINT>
    ```

