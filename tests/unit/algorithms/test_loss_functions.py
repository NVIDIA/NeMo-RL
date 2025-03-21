# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pytest
import torch
from nemo_reinforcer.algorithms.loss_functions import NLLLoss


def test_nll_loss():
    if not torch.cuda.is_available():
        pytest.skip("No GPU available")

    loss_fn = NLLLoss()

    vocab_size = 8
    data = {
        "input_ids": torch.arange(vocab_size / 2)
        .unsqueeze(0)
        .to(torch.int64)
        .to("cuda"),
        "token_mask": torch.tensor([[0, 0, 1, 1]]).to("cuda"),
        "sample_mask": torch.tensor([[1]]).to("cuda"),
    }

    ### assume we predict the correct token with high probability
    next_token_logits = (
        torch.tensor(
            [
                [0, 999.0, 0, 0, 0, 0, 0, 0],
                [0, 0, 999.0, 0, 0, 0, 0, 0],
                [0, 0, 0, 999.0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0.0, 0, 0, 0],  ## unused because we don't have a label
            ]
        )
        .unsqueeze(0)
        .to("cuda")
    )
    loss, metrics_dict = loss_fn(next_token_logits, data)
    torch.testing.assert_allclose(loss.cpu(), torch.tensor(0.0))
    # Check the metrics dictionary contains the expected values
    assert metrics_dict["num_unmasked_tokens"] == 2
    assert metrics_dict["total_tokens"] == 3

    ## now assume we predict the incorrect token with high probability
    next_token_logits = (
        torch.tensor(
            [
                [999.0, 0, 0, 0, 0, 0, 0, 0],
                [0, 999.0, 0, 0, 0, 0, 0, 0],
                [0, 0, 999.0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )
        .unsqueeze(0)
        .to("cuda")
    )
    loss, metrics_dict = loss_fn(next_token_logits, data)
    ## loss per token is 999, and we have two unmasked tokens
    ## with the updated loss function, we now average the loss over unmasked tokens
    torch.testing.assert_allclose(loss.cpu(), torch.tensor(999.0))
    # Check the metrics dictionary contains the expected values
    assert metrics_dict["num_unmasked_tokens"] == 2
    assert metrics_dict["total_tokens"] == 3
