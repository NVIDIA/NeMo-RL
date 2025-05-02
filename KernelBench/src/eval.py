"""
Helpers for Evaluations
"""

import signal
import time
import types
from typing import Any
import requests
import torch
import torch.nn as nn
import os
import subprocess
from pydantic import BaseModel
import numpy as np
import random
import json
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
import sys

from . import utils

REPO_TOP_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)
KERNEL_BENCH_PATH = os.path.join(REPO_TOP_PATH, "KernelBench")


# Define a custom exception for timeouts
class EvaluationTimeoutError(TimeoutError):
    pass


# Define the handler for the alarm signal
def _handle_timeout(signum, frame):
    print(f"[Eval Timeout] Signal handler called for signal {signum}")
    raise EvaluationTimeoutError("Kernel evaluation timed out")


def fetch_kernel_from_database(
    run_name: str, problem_id: int, sample_id: int, server_url: str
):
    """
    Intenral to us with our django database
    Return a dict with kernel hash, kernel code, problem_id
    """
    response = requests.get(
        f"{server_url}/get_kernel_by_run_problem_sample/{run_name}/{problem_id}/{sample_id}",
        json={"run_name": run_name, "problem_id": problem_id, "sample_id": sample_id},
    )
    assert response.status_code == 200
    response_json = response.json()
    assert str(response_json["problem_id"]) == str(problem_id)
    return response_json


def fetch_ref_arch_from_problem_id(problem_id, problems, with_name=False) -> str:
    """
    Fetches the reference architecture in string for a given problem_id
    """
    if isinstance(problem_id, str):
        problem_id = int(problem_id)

    problem_path = problems[problem_id]

    # problem_path = os.path.join(REPO_ROOT_PATH, problem)
    if not os.path.exists(problem_path):
        raise FileNotFoundError(f"Problem file at {problem_path} does not exist.")

    ref_arch = utils.read_file(problem_path)
    if not with_name:
        return ref_arch
    else:
        return (problem_path, ref_arch)


def fetch_ref_arch_from_level_problem_id(level, problem_id, with_name=False):
    PROBLEM_DIR = os.path.join(KERNEL_BENCH_PATH, "level" + str(level))
    dataset = utils.construct_problem_dataset_from_problem_dir(PROBLEM_DIR)
    return fetch_ref_arch_from_problem_id(problem_id, dataset, with_name)


def set_seed(seed: int):
    torch.manual_seed(seed)
    # NOTE: this only sets on current cuda device
    torch.cuda.manual_seed(seed)


class KernelExecResult(BaseModel):
    """
    Single Kernel Execution
    """

    compiled: bool = False
    executed: bool = False
    correctness: bool = False
    metadata: dict = {}
    runtime: float = -1.0  # in us, only recorded if we decide to measure performance
    runtime_stats: dict = {}  # only recorded if we decide to measure performance

    ref_exec_eager_time: float = -1.0
    ref_exec_compile_time: float = -1.0


def load_original_model_and_inputs(
    model_original_src: str, context: dict
) -> tuple[nn.Module, callable, callable]:
    """
    Load class from original NN.module pytorch code
    this is pytorch reference and we feed that to model to see if there will be any improvement
    """

    try:
        compile(model_original_src, "<string>", "exec")
    except SyntaxError as e:
        print(f"Syntax Error in original code {str(e)}")
        return None

    try:
        exec(model_original_src, context)  # expose to current namespace
    except Exception as e:
        print(f"Error in executing original code {str(e)}")
        return None

    # these should be defined in the original model code and present in the context
    get_init_inputs_fn = context.get("get_init_inputs")
    get_inputs_fn = context.get("get_inputs")
    Model = context.get("Model")
    return (Model, get_init_inputs_fn, get_inputs_fn)


def load_custom_model(
    model_custom_src: str, context: dict, build_directory: str = None
) -> nn.Module:
    """
    Load class from custom NN.module pytorch code
    this is the code output by LLM with calls to custom cuda kernels
    """
    if build_directory:
        context["BUILD_DIRECTORY"] = build_directory
        # Add import at the start of the source code
        model_custom_src = (
            f"import os\nos.environ['TORCH_EXTENSIONS_DIR'] = '{build_directory}'\n"
        ) + model_custom_src

    try:
        compile(model_custom_src, "<string>", "exec")
        exec(model_custom_src, context)
        # DANGER: need to delete refernece from global namespace
    except SyntaxError as e:
        print(f"Syntax Error in custom generated code or Compilation Error {str(e)}")
        return None

    ModelNew = context.get("ModelNew")
    return ModelNew


def _cleanup_cuda_extensions():
    """Helper function to cleanup compiled CUDA extensions"""
    # SIMON NOTE: is this necessary?
    import shutil

    torch_extensions_path = os.path.join(
        os.path.expanduser("~"), ".cache", "torch_extensions"
    )
    if os.path.exists(torch_extensions_path):
        shutil.rmtree(torch_extensions_path)


def graceful_eval_cleanup(curr_context: dict, device: torch.device):
    """
    Clean up env, gpu cache, and compiled CUDA extensions after evaluation
    """  # delete ran-specific function definitions before next eval run
    del curr_context
    # Clear CUDA cache and reset GPU state
    with torch.cuda.device(device):
        torch.cuda.empty_cache()

        # does this help?
        torch.cuda.reset_peak_memory_stats(device=device)

        torch.cuda.synchronize(
            device=device
        )  # Wait for all CUDA operations to complete

    # _cleanup_cuda_extensions() # SIMON NOTE: is this necessary?


def build_compile_cache_legacy(
    custom_model_src: str,
    verbose: bool = False,
    build_dir: os.PathLike = None,
) -> tuple[bool, str, str]:
    """
    Try to build the compiled cuda code for sample and store in the cache directory
    Should be able to run on CPUs to do this massively in parallel

    Don't limit ninja to set default number of workers, let it use all the cpu cores possible

    NOTE: currently stdout_buffer does not capture all the compiler warning and failure messages
    Returns:
        tuple[bool, str]: whether compilation is successful, stdout content as string
    """
    context = {}
    stdout_buffer = StringIO()

    if verbose:
        print("[Compilation] Pre-compile custom cuda binaries")

    try:
        os.environ["TORCH_USE_CUDA_DSA"] = "1"  # compile with device side assertion
        # sys.stdout.flush()

        # Capture stdout during compilation
        with redirect_stdout(stdout_buffer), redirect_stderr(stdout_buffer):
            load_custom_model(custom_model_src, context, build_dir)
            # sys.stdout.flush()

        if verbose:
            print(f"[Compilation] Compilation Successful, saved cache at: {build_dir}")
    except Exception as e:
        print(
            f"[Compilation] Failed to compile custom CUDA kernel. Unable to cache, \nError: {str(e)}"
        )
        return False, stdout_buffer.getvalue(), str(e)

    return True, stdout_buffer.getvalue(), None


def build_compile_cache(
    custom_model_src: str,
    verbose: bool = False,
    build_dir: os.PathLike = None,
) -> tuple[bool, str, str]:
    """
    Try to build the compiled cuda code for sample and store in the cache directory
    Should be able to run on CPUs to do this massively in parallel

    Don't limit ninja to set default number of workers, let it use all the cpu cores possible
    # try do this with a subprocess
    NOTE: currently stdout_buffer does not capture all the compiler warning and failure messages
    Returns:
        tuple[bool, str]: whether compilation is successful, stdout content as string
    """
    context = {}
    stdout_buffer = StringIO()

    if verbose:
        print("[Compilation] Pre-compile custom cuda binaries")

    try:
        os.environ["TORCH_USE_CUDA_DSA"] = "1"  # compile with device side assertion
        # sys.stdout.flush()

        # Capture stdout during compilation
        with redirect_stdout(stdout_buffer), redirect_stderr(stdout_buffer):
            load_custom_model(custom_model_src, context, build_dir)
            # sys.stdout.flush()

        if verbose:
            print(f"[Compilation] Compilation Successful, saved cache at: {build_dir}")
    except Exception as e:
        print(
            f"[Compilation] Failed to compile custom CUDA kernel. Unable to cache, \nError: {str(e)}"
        )
        return False, stdout_buffer.getvalue(), str(e)

    return True, stdout_buffer.getvalue(), None


def build_compile_cache_with_capturing(
    custom_model_src: str, verbose: bool = False, build_dir: os.PathLike = None
) -> tuple[int, str, str]:
    """
    Write a temporary python file to compile the custom model on CPU
    Captures the return code, stdout, and stderr
    This works for capturing, build_compile_cache does not
    """
    if build_dir:
        # Add import at the start of the source code
        custom_model_src = (
            f"import os\nos.environ['TORCH_EXTENSIONS_DIR'] = '{build_dir}'\n"
        ) + custom_model_src

    kernel_hash = hash(custom_model_src)
    # tmp is a temp python file we write to for compilation
    tmp = os.path.join(build_dir, f"tmp_{kernel_hash}.py")
    os.makedirs(os.path.dirname(tmp), exist_ok=True)

    with open(tmp, "w", encoding="utf-8") as f:
        f.write(custom_model_src)

    # Execute the temporary Python file and capture output
    process = subprocess.Popen(
        ["python", tmp], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate()
    returncode = process.returncode

    # Clean up temporary file
    os.remove(tmp)

    if verbose:
        print("[CPU Precompile] return code: ", returncode)
        print("[CPU Precompile] stdout: \n", stdout.decode("utf-8"))
        print("[CPU Precompile] stderr: \n", stderr.decode("utf-8"))

    return returncode, stdout.decode("utf-8"), stderr.decode("utf-8")


def eval_kernel_against_ref(
    original_model_src: str,
    custom_model_src: str,
    seed_num: int = 42,
    num_correct_trials: int = 1,
    num_perf_trials: int = 10,
    verbose: bool = False,
    measure_performance: bool = False,
    build_dir: os.PathLike = None,
    device: torch.device = torch.cuda.current_device()
    if torch.cuda.is_available()
    else None,  # have to run on GPU
    timeout: int = 300,
) -> KernelExecResult:
    """
    Evaluate the custom kernel against the original model.
    Returns a KernelExecResult object with the following fields:
    - compiled: whether the custom kernel was compiled successfully
    - executed: whether the custom kernel was executed successfully
    - correctness: whether the custom kernel is correct
    - metadata: a dictionary containing the metadata for the evaluation
    - runtime: the runtime of the custom kernel, -1.0 if not measured because kernel was incorrect

    num_correct_trials: number of trials to initialize different random inputs; correctness pass only if all trials pass
    num_perf_trials: run the evalutation many times to take the average
    device: GPU (cuda) device to run the evalutation on
    """
    # TODO: check device is busy
    assert torch.cuda.is_available(), "CUDA is not available, cannot run Eval"
    torch.set_printoptions(
        precision=4,  # Decimal places
        threshold=10,  # Total number of elements before truncating
        edgeitems=3,  # Number of elements at beginning and end of dimensions
        linewidth=80,  # Maximum width before wrapping
    )

    # set CUDA device
    torch.cuda.set_device(device)

    context = {}
    metadata = {}  # for storing result metadata
    metadata["hardware"] = torch.cuda.get_device_name(device=device)
    metadata["device"] = str(device)  # for debugging

    original_sigalrm_handler = None

    try:
        # --- Timeout Setup ---
        if timeout is not None and timeout > 0:
            original_sigalrm_handler = signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(timeout)
            if verbose:
                print(f"[Eval] Timeout set to {timeout} seconds.")

        # --- Load Original Model ---
        if verbose:
            print(f"[Eval] Start Evaluation! on device: {device}")
            print("[Eval] Loading Original Model")
        try:
            Model, get_init_inputs, get_inputs = load_original_model_and_inputs(
                original_model_src, context
            )
        except EvaluationTimeoutError as e:
            if verbose:
                print(
                    f"[Eval] Failed to load original model: Timeout during model loading. \nError: {str(e)}"
                )
            metadata["ref_model_compilation_timeout_error"] = str(e)
            metadata["timeout"] = timeout
            # No need to call cleanup/alarm(0) here, finally block handles it
            return KernelExecResult(
                compiled=False, executed=False, correctness=False, metadata=metadata
            )
        except Exception as e:
            if verbose:
                print(
                    f"[Eval] Failed to load original model: Error during model loading. \nError: {str(e)}"
                )
            metadata["ref_model_compilation_error"] = str(e)
            return KernelExecResult(
                compiled=False, executed=False, correctness=False, metadata=metadata
            )

        # If components are None here, compilation failed without raising an expected exception
        if not all([Model, get_init_inputs, get_inputs]):
            metadata["ref_model_compilation_error"] = (
                "[Eval] Failed to load original model components"
            )
            return KernelExecResult(
                compiled=False, executed=False, correctness=False, metadata=metadata
            )

        # ---  Load Custom Model ---
        ModelNew = None  # Define ModelNew here to check if compilation succeeded
        if verbose:
            print("[Eval] Loading New Model with Custom CUDA Kernel")
        try:
            os.environ["TORCH_USE_CUDA_DSA"] = "1"  # compile with device side assertion
            # add hash for later to distinguish between multi-turn kernels
            ModelNew = load_custom_model(custom_model_src, context, build_dir)
            torch.cuda.synchronize(device=device)  # not sure if this is too much
            if verbose:
                print("[Eval] Custom Model Loaded.")
        except EvaluationTimeoutError as e:
            if verbose:
                print(
                    f"[Eval] Failed to compile custom CUDA kernel: Timeout during custom model compilation. \nError: {str(e)}"
                )
            metadata["custom_model_compilation_timeout_error"] = str(e)
            metadata["timeout"] = timeout
            return KernelExecResult(
                compiled=False, executed=False, correctness=False, metadata=metadata
            )
        except Exception as e:
            if verbose:
                print(
                    f"[Eval] Failed to compile custom CUDA kernel: Record as compilation failure. \nError: {str(e)}"
                )
            # TODO: add metadata for compilation error (how to we get the compilation error message?)

            if "lock" in str(e) or "No such file or directory" in str(e):
                # this is a lock file error, likely due to concurrent compilation
                # this does not necessarily mean the compilation failed, but we should retry
                if verbose:
                    print(
                        f"[Eval] Lock file error during compilation, Please retry. Error: {str(e)}"
                    )
                return None
            else:
                metadata["custom_model_compilation_error"] = str(e)
                return KernelExecResult(
                    compiled=False, executed=False, correctness=False, metadata=metadata
                )  # skip further steps

        # If ModelNew is None here, compilation failed without raising an expected exception
        if ModelNew is None:
            metadata["custom_model_compilation_error"] = (
                "[Eval] Custom model compilation failed unexpectedly without specific error."
            )
            return KernelExecResult(
                compiled=False, executed=False, correctness=False, metadata=metadata
            )

        # at this point we passed compilation
        # --- Instantiate, Check Correctness, Measure Performance ---
        try:
            set_seed(seed_num)  # set seed for reproducible input
            init_inputs = get_init_inputs()
            init_inputs = [
                x.cuda(device=device) if isinstance(x, torch.Tensor) else x
                for x in init_inputs
            ]

            # Instantiate custom model
            with torch.no_grad():
                set_seed(seed_num)  # set seed for reproducible weights
                original_model = Model(*init_inputs)
                assert hasattr(original_model, "forward")
                custom_model = ModelNew(*init_inputs)
                assert hasattr(custom_model, "forward")
                torch.cuda.synchronize(device=device)
            if verbose:
                print("[Eval] Original and Custom Model with Custom CUDA Kernel Loaded")
        except Exception as e:
            if verbose:
                print(
                    f"[Eval] Failed to load custom CUDA kernel; Compiled but not able to run. \nError: {str(e)}"
                )
            # TODO: add metadata for runtime error e.g. error in launching kernel, illegal memory access, ...
            metadata["instantiation_error"] = str(e)
            return KernelExecResult(
                compiled=True, executed=False, correctness=False, metadata=metadata
            )  # skip further steps

        kernel_exec_result = None

        # Check Correctness
        if verbose:
            print("[Eval] Checking Correctness")
        try:
            kernel_exec_result = run_and_check_correctness(
                original_model,
                custom_model,
                get_inputs,
                metadata=metadata,
                num_correct_trials=num_correct_trials,
                verbose=verbose,
                seed=seed_num,
                device=device,
            )
        except EvaluationTimeoutError as e:
            if verbose:
                print(f"[Eval] Timeout during correctness execution: {str(e)}")
            metadata["correctness_timeout_error"] = str(e)
            metadata["timeout"] = timeout
            return KernelExecResult(
                compiled=True, executed=False, correctness=False, metadata=metadata
            )
        except Exception as e:
            # TODO: add metadata for runtime error e.g. error in launching kernel, illegal memory access, ...
            if verbose:
                print(f"[Eval] Error in Measuring Correctness: {str(e)}")
            metadata["correctness_runtime_error"] = str(e)
            return KernelExecResult(
                compiled=True, executed=False, correctness=False, metadata=metadata
            )

        # Measure Performance [Optional] | conditioned on compilation + correctness + no exception so far
        if measure_performance:
            try:
                if kernel_exec_result and kernel_exec_result.correctness:
                    if verbose:
                        print("[Eval] Measuring Performance as Sample is Correct")

                    torch.cuda.synchronize(device=device)
                    set_seed(seed_num)
                    inputs = get_inputs()
                    inputs = [
                        x.cuda(device=device) if isinstance(x, torch.Tensor) else x
                        for x in inputs
                    ]
                    model_new = custom_model.cuda(device=device)
                    torch.cuda.synchronize(device=device)

                    elapsed_times = time_execution_with_cuda_event(
                        model_new,
                        *inputs,
                        num_trials=num_perf_trials,
                        verbose=verbose,
                        device=device,
                    )
                    runtime_stats = get_timing_stats(elapsed_times, device=device)

                    if verbose:
                        print(f"[Eval] Performance Stats: {runtime_stats}")
                    kernel_exec_result.runtime = runtime_stats["mean"]
                    kernel_exec_result.runtime_stats = runtime_stats
            except EvaluationTimeoutError as e:  # Not sure when we would pass correctness but fail performance check
                if verbose:
                    print(f"[Eval] Timeout during performance execution: {str(e)}")
                metadata["performance_timeout_error"] = str(e)
                metadata["timeout"] = timeout
                return KernelExecResult(
                    compiled=True, executed=False, correctness=False, metadata=metadata
                )
            except Exception as e:  # Not sure when we would pass correctness but fail performance check
                if verbose:
                    print(f"[Eval] Error in Measuring Performance: {str(e)}")
                metadata["performance_runtime_error"] = str(e)
                return KernelExecResult(
                    compiled=True, executed=False, correctness=False, metadata=metadata
                )

        # --- Cancel Timeout if execution finished normally ---
        if timeout is not None and timeout > 0:
            signal.alarm(0)
            if verbose:
                print("[Eval] Evaluation finished within timeout.")

    except EvaluationTimeoutError as e:
        if verbose:
            print(f"[Eval] Caught EvaluationTimeoutError: {str(e)}")
        # Should a specific result indicating timeout
        metadata = {"timeout_error": str(e), "timeout": timeout}
        return KernelExecResult(
            compiled=None, metadata=metadata
        )  # compiled=None indicates indeterminate state due to timeout

    except Exception as e:
        # Catch any other unexpected errors during the process
        if verbose:
            print(f"[Eval] Caught unexpected error during evaluation: {str(e)}")
        metadata = {"unexpected_error": str(e)}
        # Return a generic failure result
        return KernelExecResult(compiled=None, metadata=metadata)

    finally:
        # --- Restore Original Signal Handler ---
        # This block executes whether an exception occurred or not.
        # It's crucial to restore the original handler if we changed it.
        if timeout is not None and timeout > 0:
            signal.alarm(0)

        if original_sigalrm_handler is not None:
            signal.signal(signal.SIGALRM, original_sigalrm_handler)
            if verbose:
                print("[Eval] Restored original SIGALRM handler.")

        # Clean up resources
        graceful_eval_cleanup(context, device)

    # Return the result obtained before any timeout or error if successful
    return (
        kernel_exec_result
        if kernel_exec_result is not None
        else KernelExecResult(
            compiled=None,
            metadata={"error": "Evaluation ended unexpectedly without result"},
        )
    )


def register_and_format_exception(
    exception_type: str,
    exception_msg: Exception | str,
    metadata: dict,
    verbose: bool = False,
    truncate=False,
    max_length=200,
):
    """
    max_length characters

    NOTE: I can't get torch truncate to work during exception handling so I have this for now
    """
    # Truncate exception message if too long
    exception_str = str(exception_msg)
    if truncate and len(exception_str) > max_length:
        exception_str = exception_str[: max_length - 3] + "..."

    if verbose:
        print(f"[Exception {exception_type}] {exception_str} ")
    metadata[exception_type] = exception_str

    return metadata


def time_execution_with_cuda_event(
    kernel_fn: callable,
    *args,
    num_warmup: int = 3,
    num_trials: int = 10,
    verbose: bool = True,
    device: torch.device = None,
) -> list[float]:
    """
    Time a CUDA kernel function over multiple trials using torch.cuda.Event

    Args:
        kernel_fn: Function to time
        *args: Arguments to pass to kernel_fn
        num_trials: Number of timing trials to run
        verbose: Whether to print per-trial timing info
        device: CUDA device to use, if None, use current device

    Returns:
        List of elapsed times in milliseconds
    """
    if device is None:
        if verbose:
            print(f"Using current device: {torch.cuda.current_device()}")
        device = torch.cuda.current_device()

    # Warm ups
    for _ in range(num_warmup):
        kernel_fn(*args)
        torch.cuda.synchronize(device=device)

    print(
        f"[Profiling] Using device: {device} {torch.cuda.get_device_name(device)}, warm up {num_warmup}, trials {num_trials}"
    )
    elapsed_times = []

    # Actual trials
    for trial in range(num_trials):
        # create event marker default is not interprocess
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        kernel_fn(*args)
        end_event.record()

        # Synchronize to ensure the events have completed
        torch.cuda.synchronize(device=device)

        # Calculate the elapsed time in milliseconds
        elapsed_time_ms = start_event.elapsed_time(end_event)
        if verbose:
            print(f"Trial {trial + 1}: {elapsed_time_ms:.3g} ms")
        elapsed_times.append(elapsed_time_ms)

    return elapsed_times


def is_cheating(
    model_new: nn.Module,
    inputs: list[torch.Tensor | Any],
    original_output: torch.Tensor,
    device: torch.device,
    verbose=False,
) -> bool:
    """
    Attempts to detect if the custom kernel call is bypassed.

    Monkey-patches the JIT-compiled kernel function(s) with a mock
    that returns zeros, re-runs the forward pass, and compares the output
    to the original. If they match, it suggests the kernel wasn't used.

    Args:
        model_new: The custom model instance (ModelNew).
        inputs: The inputs used for the forward pass.
        original_output: The output from the original forward pass.
        device: The torch device being used.
        verbose: Flag for verbose logging.

    Returns:
        True if cheating is suspected, False otherwise.
    """
    jit_module_attr_name = None
    original_jit_module = None
    original_kernel_functions = {}
    mock_functions = {}
    cheating_suspected = False

    try:
        # Find the JIT compiled module attribute in model_new
        # Heuristic: Find the first attribute that is a ModuleType and has a __file__
        for name, attr in vars(model_new).items():
            # Check if it's a module and likely a compiled extension
            if isinstance(attr, types.ModuleType) and hasattr(attr, "__file__"):
                # Optional further check: is attr.__file__ inside a torch_extensions dir?
                original_jit_module = attr
                jit_module_attr_name = name
                if verbose:
                    print(
                        f"[Cheat Detection] Found JIT module attribute: '{name}' of type {type(attr)}"
                    )
                break
        # Fallback check using named_modules for modules added via nn.Module methods
        if not original_jit_module:
            for name, module in model_new.named_modules():
                if "." not in name and name != "":  # Check direct children
                    if isinstance(module, types.ModuleType) and hasattr(
                        module, "__file__"
                    ):
                        original_jit_module = module
                        jit_module_attr_name = name
                        if verbose:
                            print(
                                f"[Cheat Detection] Found JIT module attribute (via named_modules): '{name}' of type {type(module)}"
                            )
                        break

        if not original_jit_module:
            if verbose:
                print(
                    "[Cheat Detection] Could not find JIT compiled module attribute in ModelNew. Skipping check."
                )
            return False  # Cannot perform check

        # Find potential kernel function(s) within the JIT module
        for func_name in dir(original_jit_module):
            # Avoid private/magic methods and ensure it's directly part of the module instance
            if not func_name.startswith("_") and hasattr(
                original_jit_module, func_name
            ):
                original_func = getattr(original_jit_module, func_name)
                # Check if it's a function/method bound to the instance
                if callable(original_func) and hasattr(original_func, "__call__"):
                    if verbose:
                        print(
                            f"[Cheat Detection] Found callable JIT function: {func_name}"
                        )
                    original_kernel_functions[func_name] = original_func

                    # 3. Create mock function(s)
                    def create_mock_func(original_out, captured_func_name):
                        # Closure to capture the necessary variables
                        def mock_func(*args, **kwargs):
                            if verbose:
                                print(
                                    f"[Cheat Detection] Mock function {captured_func_name} called"
                                )
                            # Return tensor(s) of zeros with the same metadata as original output
                            if isinstance(original_out, torch.Tensor):
                                return torch.zeros_like(original_out)
                            elif isinstance(original_out, (list, tuple)):
                                mocked_outputs = []
                                for t in original_out:
                                    if isinstance(t, torch.Tensor):
                                        mocked_outputs.append(torch.zeros_like(t))
                                    else:
                                        mocked_outputs.append(
                                            t
                                        )  # Keep non-tensors as is
                                return type(original_out)(mocked_outputs)
                            else:
                                # Cannot determine output shape for non-tensor/sequence, return None
                                print(
                                    f"[Cheat Detection] Warning: Cannot mock output for type {type(original_out)}"
                                )
                                return None

                        return mock_func

                    mock_functions[func_name] = create_mock_func(
                        original_output, func_name
                    )

        if not mock_functions:
            if verbose:
                print(
                    "[Cheat Detection] No callable functions found in JIT module. Skipping check."
                )
            return False  # Cannot perform check

        # Monkey-patch
        for func_name, mock_func in mock_functions.items():
            setattr(original_jit_module, func_name, mock_func)
            if verbose:
                print(f"[Cheat Detection] Patched {jit_module_attr_name}.{func_name}")

        # Re-run forward pass
        with torch.no_grad():
            patched_output = model_new(*inputs)
        torch.cuda.synchronize(device=device)

        # Compare outputs (using the existing compare_outputs function)
        # Tolerance needs to be strict as we expect zeros if kernel is used
        outputs_match = torch.allclose(
            original_output, patched_output, atol=1e-06, rtol=1e-07
        )
        if verbose:
            print(
                f"[Cheat Detection] Original vs Patched output match: {outputs_match}"
            )

        cheating_suspected = outputs_match

    except Exception as e:
        print(f"[Cheat Detection] Error during cheat detection: {e}")
        raise
    finally:
        # Restore original functions
        if original_jit_module:
            for func_name, original_func in original_kernel_functions.items():
                # Ensure the attribute exists before trying to set it back
                if hasattr(original_jit_module, func_name):
                    setattr(original_jit_module, func_name, original_func)
                    if verbose:
                        print(
                            f"[Cheat Detection] Restored {jit_module_attr_name}.{func_name}"
                        )

    return cheating_suspected


def run_and_check_correctness(
    original_model_instance: nn.Module,
    new_model_instance: nn.Module,
    get_inputs_fn: callable,
    metadata: dict,
    num_correct_trials: int,
    verbose=False,
    seed=42,
    device=None,
) -> KernelExecResult:
    """
    run the model and check correctness,
    assume model already loaded and compiled (loaded and compiled in the caller)
    this is all on GPU, requiring cuda device and transfer .cuda()

    num_correct_trials: run the evalutation multiple times with (ideally) different random inputs to ensure correctness
    """
    pass_count = 0

    # Generate num_correct_trials seeds deterministically from the initial seed
    torch.manual_seed(seed)
    correctness_trial_seeds = [
        torch.randint(0, 2**32 - 1, (1,)).item() for _ in range(num_correct_trials)
    ]

    with torch.no_grad():
        for trial in range(num_correct_trials):
            trial_seed = correctness_trial_seeds[trial]
            if verbose:
                print(f"[Correctness] Generating Random Input with seed {trial_seed}")

            set_seed(trial_seed)
            inputs = get_inputs_fn()
            inputs = [
                x.cuda(device=device) if isinstance(x, torch.Tensor) else x
                for x in inputs
            ]

            set_seed(trial_seed)
            model = original_model_instance.cuda(device=device)

            set_seed(trial_seed)
            model_new = new_model_instance.cuda(device=device)

            try:
                output = model(*inputs)
                torch.cuda.synchronize(device=device)
                output_new = model_new(*inputs)
                torch.cuda.synchronize(device=device)
                # ensure all GPU operations are completed before checking results

                if output.shape != output_new.shape:
                    metadata = register_and_format_exception(
                        "correctness_issue",
                        f"Output shape mismatch: Expected {output.shape}, got {output_new.shape}",
                        metadata,
                    )
                    if verbose:
                        print(
                            f"[Correctness] [FAIL] trial {trial}: Output shape mismatch: Expected {output.shape}, got {output_new.shape}"
                        )
                    return KernelExecResult(
                        compiled=True,
                        executed=True,
                        correctness=False,
                        metadata=metadata,
                    )

                try:
                    cheating_detected = is_cheating(
                        model_new, inputs, output_new, device, verbose=verbose
                    )
                    if cheating_detected:
                        print(
                            "[Correctness] [FAIL]: Potential cheat detected! Output matched even when kernel was no-op."
                        )
                        metadata["cheating"] = True
                        return KernelExecResult(
                            compiled=True,
                            executed=True,
                            correctness=False,
                            metadata=metadata,
                        )
                except Exception as e:
                    if verbose:
                        print(f"[Correctness] Error during cheat detection: {str(e)}")
                    return KernelExecResult(
                        compiled=True,
                        executed=True,
                        correctness=False,
                        metadata=metadata,
                    )  # TODO: add more info about why cheating detection failed e.g. failed monkey patch?

                # check output value difference
                if not torch.allclose(
                    output, output_new, atol=1e-02, rtol=1e-02
                ):  # fail
                    max_diff = torch.max(torch.abs(output - output_new)).item()
                    avg_diff = torch.mean(torch.abs(output - output_new)).item()
                    metadata.setdefault("max_difference", []).append(f"{max_diff:.6f}")
                    metadata.setdefault("avg_difference", []).append(f"{avg_diff:.6f}")
                    metadata["correctness_issue"] = "Output mismatch"
                    if verbose:
                        print(f"[Correctness] [FAIL] trial {trial}: Output mismatch")
                else:  # pass
                    pass_count += 1
                    if verbose:
                        print(
                            f"[Correctness] [PASS] trial {trial}: New Model matches Model"
                        )

            except EvaluationTimeoutError:  # Explicitly catch and re-raise timeout
                print(
                    "[Correctness] Timeout occurred during correctness check execution"
                )
                raise
            except Exception as e:
                print(
                    f"[Correctness] Exception happened during correctness check: {str(e)}"
                )
                raise

    if verbose:
        print(
            f"[Correctness] Pass count: {pass_count}, num_correct_trials: {num_correct_trials}"
        )

    # put all the useful info here!
    metadata["correctness_trials"] = f"({pass_count} / {num_correct_trials})"
    if (
        "cheating" not in metadata or not metadata["cheating"]
    ):  # Redundant because we usually return early if cheating is detected
        metadata["cheating"] = False

    if pass_count == num_correct_trials:
        return KernelExecResult(
            compiled=True, executed=True, correctness=True, metadata=metadata
        )
    else:
        return KernelExecResult(
            compiled=True, executed=True, correctness=False, metadata=metadata
        )


def check_metadata_serializable(metadata: dict):
    """
    Ensure metadata is JSON serializable,
    if not, convert non-serializable values to strings
    """
    try:
        json.dumps(metadata)
    except (TypeError, OverflowError) as e:
        print(f"[WARNING] Metadata is not JSON serializable, error: {str(e)}")
        # Convert non-serializable values to strings
        metadata = {
            "eval_0": {
                k: (
                    str(v)
                    if not isinstance(
                        v, (dict, list, str, int, float, bool, type(None))
                    )
                    else v
                )
                for k, v in metadata["eval_0"].items()
            }
        }
        print(
            f"[WARNING] Metadata now converted to string: {metadata} to be JSON serializable"
        )

    return metadata


def check_metadata_serializable_all_types(metadata: dict):
    """
    Ensure metadata is JSON serializable,
    if not, convert non-serializable values to strings recursively
    """

    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(v) for v in obj]
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            return str(obj)

    try:
        json.dumps(metadata)
        return metadata
    except (TypeError, OverflowError) as e:
        print(f"[WARNING] Metadata is not JSON serializable, error: {str(e)}")
        # Convert non-serializable values to strings recursively
        converted_metadata = convert_to_serializable(metadata)
        print(
            f"[WARNING] Metadata now converted to be JSON serializable: {converted_metadata}"
        )
        return converted_metadata


################################################################################
# Performance Eval
################################################################################


def fetch_baseline_time(
    level_name: str, problem_id: int, dataset: list[str], baseline_time_filepath: str
) -> dict:
    """
    Fetch the baseline time from the time
    """
    if not os.path.exists(baseline_time_filepath):
        raise FileNotFoundError(
            f"Baseline time file not found at {baseline_time_filepath}"
        )

    with open(baseline_time_filepath, "r") as f:
        baseline_json = json.load(f)

    problem_name = dataset[problem_id].split("/")[-1]
    baseline_time = baseline_json[level_name].get(problem_name, None)
    return baseline_time


def get_timing_stats(elapsed_times: list[float], device: torch.device = None) -> dict:
    """Get timing statistics from a list of elapsed times.

    Args:
        elapsed_times: List of elapsed times in milliseconds
        device: CUDA device, record device info
    Returns:
        Dict containing mean, std, min, max and num_trials
        all timing are in ms
    """

    stats = {
        "mean": float(f"{np.mean(elapsed_times):.3g}"),
        "std": float(f"{np.std(elapsed_times):.3g}"),
        "min": float(f"{np.min(elapsed_times):.3g}"),
        "max": float(f"{np.max(elapsed_times):.3g}"),
        "num_trials": len(elapsed_times),
    }

    if device:
        stats["hardware"] = torch.cuda.get_device_name(device=device)
        stats["device"] = str(device)  # for debugging

    return stats


# if __name__ == "__main__":
# fetch_kernel_from_database("kernelbench_prompt_v2_level_2", 1, 1, "http://localhost:9091")
# print(fetch_ref_arch_from_level_problem_id("2", 1, with_name=True))
# fetch_baseline_time("level1", 0, ["1_Square_matrix_multiplication_.py"], "tests/baseline_time_matx3.json")
