import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a 3D convolution, applies HardSwish, ReLU, Softmax, and then calculates the mean.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super(Model, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, bias=bias)

    def forward(self, x):
        x = self.conv(x)
        x = torch.nn.functional.hardswish(x)
        x = torch.relu(x)
        x = torch.softmax(x, dim=1)
        x = torch.mean(x, dim=[2, 3, 4])
        return x

batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]