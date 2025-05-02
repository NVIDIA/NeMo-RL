# tests/test_kernel_eval.py

import pytest
import torch
import time
import os
import re
import signal
from unittest.mock import patch  # For mocking lock errors

# Adjust the import path if your tests directory is structured differently
from KernelBench.src import eval
from KernelBench.src.eval import KernelExecResult, EvaluationTimeoutError

# --- Helper Dummy Model Sources ---

# A valid, simple original model
original_model_src_valid = """
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.tensor(1.0)) # Ensure movement to device

    def forward(self, x):
        return x * 2.0 + self.param

def get_init_inputs():
    return []

def get_inputs():
    # Return tensor on CPU, eval function moves it
    return [torch.randn(5, 10, dtype=torch.float32)]
"""

# A valid custom model matching the original
custom_model_src_valid_correct = """
import torch
import torch.nn as nn

class ModelNew(nn.Module): # Note: Must be ModelNew
    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        # Same logic as original
        return x * 2.0 + self.param

# Must be defined even if identical
def get_init_inputs():
    return []

def get_inputs():
    return [torch.randn(5, 10, dtype=torch.float32)]
"""

# Custom model with incorrect output (Keep as is - natural way)
custom_model_src_incorrect_output = """
import torch
import torch.nn as nn

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        # Different logic
        return x * 3.0 + self.param # Incorrect logic

def get_init_inputs():
    return []

def get_inputs():
    return [torch.randn(5, 10, dtype=torch.float32)]
"""

# Custom model that causes a runtime error during forward (e.g., index out of bounds error)
custom_model_src_runtime_error = """
import torch
import torch.nn as nn

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        # Force a definite runtime error by accessing an invalid index
        # This will always cause a runtime error, not just an output mismatch
        invalid_index = x.shape[1] + 100  # Way beyond the tensor dimensions
        bad_access = x[:, invalid_index]  # This will cause IndexError: index out of range
        return bad_access + self.param

def get_init_inputs():
    return []

def get_inputs():
    return [torch.randn(5, 10, dtype=torch.float32)]
"""

custom_model_src_perf_runtime_error = """
import torch
import torch.nn as nn

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.tensor(1.0))
        # Counter to track forward passes
        self.forward_count = 0
        # A tensor that might become zero
        self.divisor = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        self.forward_count += 1
        print(f"Custom model forward count: {self.forward_count}")

        # Only cause an error after, say, the 2nd forward pass
        # This should pass 1 correctness trial but fail during performance trials
        if self.forward_count > 1:
             print("Triggering runtime error in performance phase...")
             # Force a divide-by-zero error on the GPU
             self.divisor.data.fill_(0.0)
             bad_calc = x / self.divisor # RuntimeError: division by zero
             return bad_calc + self.param
        else:
             # Correct behavior for the first pass(es)
             return x * 2.0 + self.param

def get_init_inputs():
    return []

def get_inputs():
    return [torch.randn(5, 10, dtype=torch.float32)]
"""

# Custom model that sleeps (for timeout tests - Keep as is)
custom_model_src_sleep = """
import torch
import torch.nn as nn
import time

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        print(f"Custom model forward: Sleeping for 3 seconds...")
        time.sleep(3) # Sleep longer than typical test timeout
        print("Custom model forward: Woke up!")
        return x * 2.0 + self.param # Correct logic

def get_init_inputs():
    return []

def get_inputs():
    return [torch.randn(5, 10, dtype=torch.float32)]
"""

# Original model with compile error (SyntaxError - Keep as is)
original_model_src_compile_error = """
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x * 2.0 + self.param

def get_init_inputs():
    return []

def get_inputs():
    return [torch.randn(5, 10, dtype=torch.float32)]

invalid syntax here
"""

# Custom model with compile error (SyntaxError - Keep as is)
custom_model_src_compile_error = """
import torch
import torch.nn as nn

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x * 2.0 + self.param

def get_init_inputs():
    return []

def get_inputs():
    return [torch.randn(5, 10, dtype=torch.float32)]

invalid syntax here for ModelNew
"""


custom_model_src_compile_sleep = """
import torch
import torch.nn as nn
import time

print("Custom model source: Starting execution...")
# Add sleep at the top level to simulate long load/compile time
time.sleep(3)
print("Custom model source: Finished sleeping.")

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        # Logic doesn't matter much as it won't be reached if timeout is short
        return x * 2.0 + self.param

# Must be defined even if identical
def get_init_inputs():
    return []

def get_inputs():
    return [torch.randn(5, 10, dtype=torch.float32)]
"""


# Original model with instantiation error (e.g., divide by zero in __init__)
original_model_src_instantiate_error = """
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        a = 1
        b = 0
        c = a / b # ZeroDivisionError during instantiation
        self.param = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x * 2.0 + self.param

def get_init_inputs():
    return []

def get_inputs():
    return [torch.randn(5, 10, dtype=torch.float32)]
"""

# Custom model with instantiation error (e.g., invalid op in __init__)
custom_model_src_instantiate_error = """
import torch
import torch.nn as nn

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        # Example: Try to use an attribute that doesn't exist yet
        self.another_param = self.non_existent_param + 1 # NameError during instantiation
        self.param = nn.Parameter(torch.tensor(1.0))


    def forward(self, x):
        return x * 2.0 + self.param

def get_init_inputs():
    return []

def get_inputs():
    return [torch.randn(5, 10, dtype=torch.float32)]
"""


# --- Test Fixtures ---
# (Keep fixtures as they were)
@pytest.fixture(scope="function")
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device not available")
    return torch.device("cuda:0")


@pytest.fixture(scope="function")
def build_dir(tmp_path_factory):
    # Create a unique build directory for each test function run
    return tmp_path_factory.mktemp("eval_build_")


# --- Test Cases ---

gelu_original_src = """
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return F.gelu(x, approximate='tanh')


def get_inputs():
    # randomly generate input tensors based on the model architecture
    x = torch.randn(1024, 1024).cuda()
    return [x]


def get_init_inputs():
    # randomly generate tensors required for initialization based on the model architecture
    return []
"""

gelu_custom_src_correct = """
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

source = '''
#include <torch/extension.h>
#include <stdio.h>
#include <c10/cuda/CUDAException.h>

// Macro to check if a tensor is a CUDA tensor
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")

// Macro to check if a tensor is contiguous in memory
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

// Macro to check both CUDA and contiguity requirements
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Utility function for ceiling division
inline unsigned int cdiv(unsigned int a, unsigned int b) { return (a + b - 1) / b;}

__global__ void my_gelu_kernel(float* out, float* inp, int n) {
    // Calculate global thread index
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Return if thread index is out of bounds
    if (i >= n) return;

    // Load input value
    float x = inp[i];

    // Compute GELU (Gaussian Error Linear Unit) activation
    // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    out[i] = 0.5f * x * (1.0f + tanhf(sqrtf(2.0f/3.141592653589793f) * (x + 0.044715f * (x*x*x))));
}

torch::Tensor my_gelu_out(torch::Tensor output, const torch::Tensor& inp) {
    CHECK_INPUT(inp);  // Validate input tensor
    int n = inp.numel();  // Get total number of elements in input tensor

    // Ensure output tensor has same properties as input tensor
    TORCH_CHECK((output.sizes() == inp.sizes()) || (output.device() == inp.device())
                || (output.scalar_type() == inp.scalar_type()));

    int threads = 256;  // Set number of threads per block

    // Launch CUDA kernel
    my_gelu_kernel<<<cdiv(n, threads), threads>>>(
        output.data_ptr<float>(), inp.data_ptr<float>(), n);

    C10_CUDA_KERNEL_LAUNCH_CHECK();  // Check for CUDA errors
    return output;
}

torch::Tensor my_gelu(const torch::Tensor& inp) {
    CHECK_INPUT(inp);  // Validate input tensor
    auto output = torch::empty_like(inp);  // Create output tensor with same properties as input
    my_gelu_out(output, inp);  // Compute GELU activation
    return output;
}
'''

# Define C++ source code as a string
cpp_src = '''
torch::Tensor my_gelu(const torch::Tensor& inp);
torch::Tensor my_gelu_out(torch::Tensor output, const torch::Tensor& inp);
'''

# Load and compile the CUDA extension
fused_gelu = torch.utils.cpp_extension.load_inline(
    name="fused_gelu",  # Name of the extension
    cpp_sources=cpp_src,  # C++ source code
    cuda_sources=source,  # CUDA source code
    functions=['my_gelu', 'my_gelu_out'],  # Functions to expose
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fused_gelu = fused_gelu

    def forward(self, x):
        return self.fused_gelu.my_gelu(x)
"""


# def test_eval_gelu_correct(device, build_dir):
#     """Tests the provided GELU example for correctness."""
#     result = eval.eval_kernel_against_ref(
#         original_model_src=gelu_original_src,
#         custom_model_src=gelu_custom_src_correct,
#         seed_num=101,
#         num_correct_trials=1,
#         verbose=True,
#         measure_performance=False,
#         build_dir=build_dir,
#         device=device,
#         timeout=60,  # Allow ample time for compilation
#     )
#     print(result)
#     assert isinstance(result, KernelExecResult)
#     assert result.compiled is True
#     assert result.executed is True
#     assert result.correctness is True
#     # Check cheat detection did not trigger (implicitly passed if correctness is True)
#     assert result.metadata.get("cheating") is False


def test_eval_gelu_cheat_detected(device, build_dir):
    """
    Tests if cheat detection works when the custom kernel is bypassed
    in the forward pass.
    """
    # Modify the forward method to compute GELU directly and ignore the kernel
    gelu_custom_src_cheating = re.sub(
        r"return self\.fused_gelu\.my_gelu\(x\)",
        r"correct_result = F.gelu(x, approximate='tanh')\n        # self.fused_gelu.my_gelu(x) # Call commented out or irrelevant\n        return correct_result",
        gelu_custom_src_correct,
        count=1,  # Only replace the first occurrence in forward
    )
    # Verify replacement occurred
    assert "correct_result = F.gelu(x, approximate='tanh')" in gelu_custom_src_cheating
    assert "return correct_result" in gelu_custom_src_cheating
    assert (
        "return self.fused_gelu.my_gelu(x)" not in gelu_custom_src_cheating
    )  # Make sure original return is gone

    print("--- Cheating Source ---")
    print(gelu_custom_src_cheating)
    print("----------------------")

    result = eval.eval_kernel_against_ref(
        original_model_src=gelu_original_src,
        custom_model_src=gelu_custom_src_cheating,
        seed_num=102,
        num_correct_trials=1,
        verbose=True,
        measure_performance=False,
        build_dir=build_dir,
        device=device,
        timeout=60,
    )
    print(result)
    assert isinstance(result, KernelExecResult)
    assert result.compiled is True  # It still compiles
    assert result.executed is True  # It runs
    assert (
        result.correctness is False
    )  # Should fail correctness because cheating was detected
    assert result.metadata.get("cheating") is True  # Explicitly check the cheating flag


def test_eval_success(device, build_dir):
    """Tests a successful evaluation with correct matching code."""
    result = eval.eval_kernel_against_ref(
        original_model_src=original_model_src_valid,
        custom_model_src=custom_model_src_valid_correct,
        seed_num=42,
        num_correct_trials=2,
        num_perf_trials=0,
        verbose=True,
        measure_performance=False,
        build_dir=build_dir,
        device=device,
        timeout=30,  # Ample time
    )
    print(result)
    assert isinstance(result, KernelExecResult)
    assert result.compiled is True
    assert result.executed is True
    assert result.correctness is True
    assert "error" not in result.metadata  # Check general error keys aren't present
    assert "timeout" not in result.metadata


def test_eval_success_with_perf(device, build_dir):
    """Tests a successful evaluation measuring performance."""
    result = eval.eval_kernel_against_ref(
        original_model_src=original_model_src_valid,
        custom_model_src=custom_model_src_valid_correct,
        seed_num=43,
        num_correct_trials=1,
        num_perf_trials=5,  # Reduced for faster testing
        verbose=True,
        measure_performance=True,
        build_dir=build_dir,
        device=device,
        timeout=45,  # Ample time for perf
    )
    print(result)
    assert isinstance(result, KernelExecResult)
    assert result.compiled is True
    assert result.executed is True
    assert result.correctness is True
    assert result.runtime > 0
    assert "mean" in result.runtime_stats
    assert "performance_runtime_error" not in result.metadata
    assert "timeout" not in result.metadata


def test_eval_original_compile_error(device, build_dir):
    """Tests failure when the original model source has a compile error."""
    result = eval.eval_kernel_against_ref(
        original_model_src=original_model_src_compile_error,
        custom_model_src=custom_model_src_valid_correct,  # Doesn't matter
        seed_num=44,
        device=device,
        build_dir=build_dir,
        verbose=True,
    )
    print(result)
    assert isinstance(result, KernelExecResult)
    assert result.compiled is False
    assert result.executed is False
    assert result.correctness is False
    assert "ref_model_compilation_error" in result.metadata
    assert isinstance(result.metadata["ref_model_compilation_error"], str)


def test_eval_custom_compile_error(device, build_dir):
    """Tests failure when the custom model source has a compile error."""
    result = eval.eval_kernel_against_ref(
        original_model_src=original_model_src_valid,
        custom_model_src=custom_model_src_compile_error,
        seed_num=46,
        device=device,
        build_dir=build_dir,
        verbose=True,
    )
    print(result)
    assert isinstance(result, KernelExecResult)
    assert result.compiled is False
    assert result.executed is False
    assert result.correctness is False
    assert "custom_model_compilation_error" in result.metadata
    # The error stored might be the actual Exception object or its string representation
    assert result.metadata["custom_model_compilation_error"] is not None


def test_eval_original_instantiate_error(device, build_dir):
    """Tests failure when the original model fails during instantiation."""
    result = eval.eval_kernel_against_ref(
        original_model_src=original_model_src_instantiate_error,
        custom_model_src=custom_model_src_valid_correct,  # Doesn't matter
        seed_num=45,
        device=device,
        build_dir=build_dir,
        verbose=True,
    )
    print(result)
    assert isinstance(result, KernelExecResult)
    assert result.compiled is True
    assert result.executed is False
    assert result.correctness is False
    assert "instantiation_error" in result.metadata
    assert isinstance(result.metadata["instantiation_error"], str)


def test_eval_custom_instantiate_error(device, build_dir):
    """Tests failure when the custom model compiles but fails instantiation."""
    result = eval.eval_kernel_against_ref(
        original_model_src=original_model_src_valid,
        custom_model_src=custom_model_src_instantiate_error,
        seed_num=47,
        device=device,
        build_dir=build_dir,
        verbose=True,
    )
    print(result)
    assert isinstance(result, KernelExecResult)
    assert result.compiled is True
    assert result.executed is False
    assert result.correctness is False
    assert "instantiation_error" in result.metadata
    assert result.metadata["instantiation_error"] is not None


def test_eval_correctness_mismatch(device, build_dir):
    """Tests failure due to mismatch in output between original and custom."""
    result = eval.eval_kernel_against_ref(
        original_model_src=original_model_src_valid,
        custom_model_src=custom_model_src_incorrect_output,
        seed_num=48,
        device=device,
        build_dir=build_dir,
        verbose=True,
        num_correct_trials=1,
    )
    print(result)
    assert isinstance(result, KernelExecResult)
    assert result.compiled is True
    assert result.executed is True
    assert result.correctness is False
    assert result.metadata.get("correctness_issue") == "Output mismatch"
    assert "max_difference" in result.metadata


def test_eval_correctness_runtime_error(device, build_dir):
    """Tests failure when custom model raises runtime error during correctness check."""
    result = eval.eval_kernel_against_ref(
        original_model_src=original_model_src_valid,
        custom_model_src=custom_model_src_runtime_error,
        seed_num=49,
        device=device,
        build_dir=build_dir,
        verbose=True,
        num_correct_trials=1,
    )
    print(result)
    assert isinstance(result, KernelExecResult)
    assert result.compiled is True
    assert result.executed is False
    assert result.correctness is False
    assert "correctness_runtime_error" in result.metadata
    assert result.metadata["correctness_runtime_error"] is not None


def test_eval_correctness_timeout(device, build_dir):
    """Tests timeout during the correctness checking phase."""
    timeout_seconds = 2
    original_handler = signal.getsignal(signal.SIGALRM)
    result = eval.eval_kernel_against_ref(
        original_model_src=original_model_src_valid,
        custom_model_src=custom_model_src_sleep,  # Sleeps for 3s
        seed_num=50,
        device=device,
        build_dir=build_dir,
        verbose=True,
        num_correct_trials=1,
        measure_performance=False,
        timeout=timeout_seconds,
    )
    print(result)
    restored_handler = signal.getsignal(signal.SIGALRM)

    assert isinstance(result, KernelExecResult)
    assert result.compiled is True  # Compilation succeeded
    assert result.executed is False
    assert result.correctness is False
    assert "correctness_timeout_error" in result.metadata
    assert result.metadata.get("timeout") == timeout_seconds
    # Check signal handler restoration (important for timeout tests)
    assert restored_handler == original_handler


def test_eval_performance_timeout(device, build_dir):
    """Tests timeout during the performance measurement phase."""
    timeout_seconds = (
        4  # Enough for 1 correctness run (sleeps 3s) but not multiple perf runs
    )
    original_handler = signal.getsignal(signal.SIGALRM)
    result = eval.eval_kernel_against_ref(
        original_model_src=original_model_src_valid,
        custom_model_src=custom_model_src_sleep,  # Sleeps for 3s per run
        seed_num=52,
        device=device,
        build_dir=build_dir,
        verbose=True,
        num_correct_trials=1,
        measure_performance=True,
        num_perf_trials=5,
        timeout=timeout_seconds,
    )
    print(result)
    restored_handler = signal.getsignal(signal.SIGALRM)

    assert isinstance(result, KernelExecResult)
    assert result.compiled is True
    assert result.executed is False
    assert result.correctness is False  # Timeout occurred, failing overall run
    assert (
        "performance_timeout_error" in result.metadata
        or "correctness_timeout_error" in result.metadata
    )  # Might timeout during either phase depending on timing
    assert result.metadata.get("timeout") == timeout_seconds
    assert restored_handler == original_handler


# Mocking example for lock file error
@patch("KernelBench.src.eval.load_custom_model")
def test_eval_compilation_lock_error(mock_load_custom, device, build_dir):
    """Tests the specific handling of lock file errors during compilation."""
    # Configure the mock to raise an exception containing 'lock'
    mock_load_custom.side_effect = Exception(
        "Failed to acquire lock file /path/to/lock"
    )

    result = eval.eval_kernel_against_ref(
        original_model_src=original_model_src_valid,
        custom_model_src="dummy_source_lock_test",  # Content doesn't matter due to mock
        seed_num=53,
        device=device,
        build_dir=build_dir,
        verbose=True,
        timeout=10,
    )
    print(result)
    # According to the code, it should return None on lock errors currently
    assert result is None
    mock_load_custom.assert_called_once()  # Ensure our mock was actually called


def test_eval_compilation_timeout(device, build_dir):
    """Tests timeout occurring specifically during the custom model compilation/loading phase."""
    # Timeout should be > time for original model load, but < sleep time in custom source
    timeout_seconds = 2
    original_handler = signal.getsignal(signal.SIGALRM)
    result = eval.eval_kernel_against_ref(
        original_model_src=original_model_src_valid,
        custom_model_src=custom_model_src_compile_sleep,  # Use the new model with sleep
        seed_num=54,
        device=device,
        build_dir=build_dir,
        verbose=True,
        timeout=timeout_seconds,
    )
    print(result)
    restored_handler = signal.getsignal(signal.SIGALRM)

    print(f"Test Result Metadata: {result.metadata}")  # Debug print

    assert isinstance(result, KernelExecResult)
    # Compilation failed specifically due to timeout during the custom model load
    assert result.compiled is False, "Compilation should fail due to timeout"
    assert result.executed is False
    assert result.correctness is False
    # Check that the specific timeout key for custom model compilation is present
    assert "custom_model_compilation_timeout_error" in result.metadata, (
        "Metadata should contain custom_model_compilation_timeout_error key"
    )
    # Ensure other timeout keys (like correctness/performance) are NOT present
    assert "correctness_timeout_error" not in result.metadata
    assert "performance_timeout_error" not in result.metadata
    # Verify the timeout value and signal handler restoration
    assert result.metadata.get("timeout") == timeout_seconds
    assert restored_handler == original_handler, "Original SIGALRM handler not restored"
