import os
import re
from typing import Tuple
import torch
import torch.distributed as dist
from KernelBench.src import eval as kernel_eval
from KernelBench.src import utils as kernel_utils
from KernelBench.src.eval import KernelExecResult
from KernelBench.scripts.generate_baseline_time import measure_program_time
from KernelBench.scripts.run_and_check import ScriptConfig, evaluate_single_sample_src


def get_code(s: str, language: str = "python") -> str:
    """Extract code from a string that may be wrapped in code fences.

    Args:
        s: A string that may contain code wrapped in fences like ```python ... ```

    Returns:
        The extracted code without the fences, or error if no fences are found.
    """
    pattern = f"```{language}\n(.*?)\n```"
    match = re.search(pattern, s, re.DOTALL)

    if match:
        # Return the code inside the fences
        return match.group(1)
    else:
        # Return the error if no fences with specified language are found
        raise ValueError(
            f"No code fences with language '{language}' found in the input string"
        )


def get_reward(
    prompt: str,
    completion: str,
    language: str = "python",
    weights: dict[str, float] = {
        "compiled": 0.5,
        "executed": 0.5,
        "correctness": 2,
        "performance": 4,
    },
    correctness_trials: int = 1,
    performance_trials: int = 1,
    timeout: int = 300,
    verbose: bool = True,
    measure_performance: bool = True,
) -> Tuple[float, KernelExecResult]:
    """Get reward for a given prompt and completion."""
    if not check_inline_format(completion, language):
        return 0.0, KernelExecResult(
            compiled=False,
            executed=False,
            correctness=False,
            metadata={"error": "Invalid completion format"},
        )

    current_reward = 0.0
    kernel_eval_result = None
    try:
        reference_code, solution_code = get_code(prompt), get_code(completion)
        eval_config = ScriptConfig()
        eval_config.num_correct_trials = correctness_trials
        eval_config.num_perf_trials = performance_trials
        eval_config.timeout = timeout
        eval_config.verbose = verbose
        eval_config.measure_performance = measure_performance
        kernel_eval_result = get_kernel_eval(reference_code, solution_code, eval_config)

        # Check for compilation success
        if not kernel_eval_result.compiled:
            return 0.0, kernel_eval_result
        current_reward += weights["compiled"]

        # Check for execution success
        if not kernel_eval_result.executed:
            return current_reward, kernel_eval_result
        current_reward += weights["executed"]

        # Check for correctness
        if not kernel_eval_result.correctness:
            return current_reward, kernel_eval_result
        current_reward += weights["correctness"]

        # Add performance success
        if kernel_eval_result.runtime != -1.0:
            current_reward += weights["performance"] * (
                kernel_eval_result.ref_exec_eager_time / kernel_eval_result.runtime
            )

        return current_reward, kernel_eval_result
    except Exception as e:
        if kernel_eval_result is None:
            return 0.0, KernelExecResult(
                compiled=False,
                executed=False,
                correctness=False,
                metadata={"error": str(e)},
            )

        # We already have an eval result, something else happened
        kernel_eval_result.metadata.update({"error": str(e)})
        return current_reward, kernel_eval_result


def check_inline_format(completion: str, language: str = "python") -> bool:
    """Format reward function specifically for code responses with inline CUDA.

    Check for class ModelNew(nn.Module).

    Args:
        language: Programming language for code fences.
    """
    # Check for think fences, code language fence, and ModelNew class
    # Pattern: <think>...</think> followed by anything, then ```language ... class ModelNew ... ```, then anything to the end.
    pattern = rf"^<think>\n.*?\n</think>\n.*?```{language}.*?class\s+ModelNew\s*\(\s*nn\.Module\s*\).*?```.*?$"
    match = re.search(pattern, completion, re.DOTALL | re.MULTILINE)
    return bool(match)


def get_kernel_eval(
    ref_arch_src: str,
    kernel_src: str,
    config: ScriptConfig,
) -> KernelExecResult:
    """Get KernelExecResult for a given reference and kernel source code.

    Args:
        ref_arch_src (str): code without fences of reference architecture
        kernel_src (str): code without fences of custom architecture

    Returns:
        KernelExecResult: containing metadata of the evaluation results
    """
    # Start Evaluation
    global_rank = dist.get_rank()  # TODO: figure out rank stuff for ray
    local_rank = int(os.environ.get("LOCAL_RANK"))
    device = torch.device(f"cuda:{local_rank}")
    kernel_utils.set_gpu_arch(config.gpu_arch)

    if config.verbose:
        print(f"[Rank {global_rank}] Evaluating kernel against reference code")
    # Evaluate kernel against reference code
    kernel_eval_result = evaluate_single_sample_src(
        ref_arch_src=ref_arch_src,
        kernel_src=kernel_src,
        configs=config.to_dict(),
        device=device,
    )
    kernel_exec_time = kernel_eval_result.runtime

    # Measure baseline time
    if kernel_eval_result.correctness and kernel_eval_result.runtime != -1.0:
        if config.verbose:
            print(f"[Rank {global_rank}][INFO] Measuring reference program time")
        # Default using PyTorch Eager here
        ref_time_eager_result = measure_program_time(
            ref_arch_name="Reference Program",
            ref_arch_src=ref_arch_src,
            num_trials=config.num_perf_trials,
            use_torch_compile=False,
            device=device,
        )
        ref_exec_eager_time = ref_time_eager_result.get("mean", None)
        kernel_eval_result.ref_exec_eager_time = ref_exec_eager_time

        # Measure Torch Compile time
        ref_time_compile_result = measure_program_time(
            ref_arch_name="Reference Program",
            ref_arch_src=ref_arch_src,
            num_trials=config.num_perf_trials,
            use_torch_compile=True,
            torch_compile_backend="inductor",
            torch_compile_options="default",
            device=device,
        )
        ref_exec_compile_time = ref_time_compile_result.get("mean", None)
        kernel_eval_result.ref_exec_compile_time = ref_exec_compile_time
    else:
        if config.verbose:
            print(
                f"[Rank {global_rank}][INFO] Skipping reference program time measurement as kernel did not pass correctness"
            )

    if config.verbose:
        print("=" * 40)
        print(f"[Rank {global_rank}][Eval] Kernel eval result: {kernel_eval_result}")
        print("-" * 40)

        if kernel_eval_result.correctness and kernel_eval_result.runtime != -1.0:
            print(
                f"[Rank {global_rank}][Timing] Custom Kernel exec time: {kernel_exec_time} ms\n"
                f"[Rank {global_rank}][Timing] PyTorch Reference Eager exec time: {ref_exec_eager_time} ms\n"
                f"[Rank {global_rank}][Speedup] Speedup over eager: {ref_exec_eager_time / kernel_exec_time:.2f}x\n"
                f"[Rank {global_rank}][Timing] PyTorch Reference torch.compile time: {ref_exec_compile_time} ms\n"
                f"[Rank {global_rank}][Speedup] Speedup over torch.compile: {ref_exec_compile_time / kernel_exec_time:.2f}x"
            )
        else:
            print(
                f"[Rank {global_rank}][Speedup] Speedup Not Available as Kernel did not pass correctness"
            )

        print("=" * 40)

    return kernel_eval_result
