import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a GEMM, Group Normalization, Minimum operation, and Bias addition.
    """
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.gemm(x)
        x = self.group_norm(x)
        x = torch.min(x, dim=1, keepdim=True)[0] 
        x = x + self.bias
        return x

batch_size = 128
in_features = 512
out_features = 256
num_groups = 8
bias_shape = (1, out_features, 1, 1)

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, num_groups, bias_shape]