import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np

# load model and the modified model
from model import Model
from model import get_inputs
from model import get_init_inputs
from model_new import ModelNew

torch.cuda.synchronize()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def check_correctness():
    # run the model and check correctness
    with torch.no_grad():

        # generate inputs and init_inputs, and instantiate models
        set_seed(42)
        inputs = get_inputs()
        set_seed(42)
        init_inputs = get_init_inputs()

        # move to GPU
        inputs = [x.cuda() if isinstance(x, torch.Tensor) else x for x in inputs]
        init_inputs = [
            x.cuda() if isinstance(x, torch.Tensor) else x for x in init_inputs
        ]

        set_seed(42)
        model = Model(*init_inputs).cuda()
        set_seed(42)
        model_new = ModelNew(*init_inputs).cuda()

        # forward pass
        output = model(*inputs)
        output_new = model_new(*inputs)

        # move to CPU
        torch.cuda.synchronize()
        output = output.cpu()
        output_new = output_new.cpu()

        # check correctness
        assert output.shape == output_new.shape
        assert torch.allclose(output, output_new, atol=1e-02)

    return "PASS"


def run(random_seed=42):

    # run both models and check correctness
    check_correctness()

    return "PASS"


if __name__ == "__main__":
    print(run())
