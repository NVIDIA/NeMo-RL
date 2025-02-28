from typing import Any, Dict, Tuple
import torch

from nemo_reinforcer.distributed.batched_data_dict import BatchedDataDict


def simple_loss(
    next_token_logits: torch.Tensor, data: BatchedDataDict
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    # Just return mean of logprobs as the loss for testing
    loss = next_token_logits.mean()
    metrics = {"test_metric": loss.item() * 0.5}
    return loss, metrics


# Create a simple masked NLL loss function
def nll_loss(
    next_token_logits: torch.Tensor, data: BatchedDataDict
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    # logits shape: [batch_size, seq_len, vocab_size]
    # Get the next token logits for each position
    next_tokens = data.get("input_ids")[:, 1:].cuda()  # Skip first token
    next_token_logprobs = torch.nn.functional.log_softmax(next_token_logits, dim=-1)
    logprobs = next_token_logprobs[:, :-1]  # Remove last position's logits

    # Gather the logprobs for the actual next tokens
    token_logprobs = logprobs.gather(dim=-1, index=next_tokens.unsqueeze(-1)).squeeze(
        -1
    )

    # Only compute loss on generated tokens (not input tokens)
    # by applying the token_loss_mask (shifted by 1 since we're predicting next tokens)
    token_loss_mask = data.get("token_loss_mask")[:, 1:].cuda()
    loss = -torch.sum(token_logprobs * token_loss_mask)

    return loss, {"loss": loss.item()}
