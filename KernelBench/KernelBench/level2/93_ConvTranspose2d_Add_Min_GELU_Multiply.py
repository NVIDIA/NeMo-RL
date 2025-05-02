import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a transposed convolution, adds a value, takes the minimum, applies GELU, and multiplies by a value.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.add_value = add_value
        self.multiply_value = multiply_value

    def forward(self, x):
        x = self.conv_transpose(x)
        x = x + self.add_value
        x = torch.min(x, torch.tensor(0.0))
        x = torch.nn.functional.gelu(x)
        x = x * self.multiply_value
        return x

batch_size = 128
in_channels = 32
out_channels = 16
height, width = 32, 32
kernel_size = 4
stride = 2
add_value = 0.5
multiply_value = 2.0

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, add_value, multiply_value]