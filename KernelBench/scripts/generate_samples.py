import pydra
from pydra import REQUIRED, Config
import os, sys
import torch
import json
from dataclasses import dataclass


from datasets import load_dataset

from src.dataset import construct_kernelbench_dataset
from src.eval import eval_kernel_against_ref
from src.prompt_constructor import prompt_generate_custom_cuda_from_prompt_template
from src.utils import extract_first_code, set_gpu_arch, read_file, create_inference_server_from_presets, maybe_multithread

"""
Batch Generate Samples for Particular Level

Assume 1 sample per problem here
"""

REPO_TOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

torch.set_printoptions(precision=4, threshold=10)

class GenerationConfig(Config):
    def __init__(self):
        
        self.dataset_src = REQUIRED # either huggingface or local

        # name of dataset name on Hugging Face
        self.dataset_name = "ScalingIntelligence/KernelBench"

        # Problem Specification
        self.level = REQUIRED
        
        # subset of problems to generate, otherwise generate on all problems in the level
        self.subset = (None, None) # (problem_id, problem_name), these are the logical index

        self.run_name = REQUIRED # name of the run

        # num of thread pool to call inference server in parallel
        self.num_workers = 1
        self.api_query_interval = 0.0

        # Inference config
        self.server_type = "deepseek"
        self.model_name = "deepseek-coder"
        self.max_tokens = 4096
        self.temperature = 0.0
        
        # Logging
        # Top Directory to Store Runs
        self.runs_dir = os.path.join(REPO_TOP_DIR, "runs")
    
        self.verbose = False
        self.store_type = "local" # TODO: add Database Integration

        # Future support
        # Migrate Monkeys code base to KernelBench
        # self.num_samples = 0 # for sampling multiple samples per problem

        self.log_prompt = False

    def greedy(self):
        # For greedy decoding, epsecially baseline eval
        self.greedy_sample = True

    def __repr__(self):
        return f"EvalConfig({self.to_dict()})"
    

@dataclass
class WorkArgs:
    problem_id: int # logically indexed
    sample_id: int

def generate_sample_single(work: WorkArgs, config: GenerationConfig, dataset, inference_server: callable, run_dir: str) -> bool:
    # 1. Fetch Problem
    if config.dataset_src == "huggingface":
        curr_problem_row = dataset.filter(lambda x: x["problem_id"] == work.problem_id, desc=None)

        ref_arch_src = curr_problem_row["code"][0]
        problem_name = curr_problem_row["name"][0]

    elif config.dataset_src == "local":
        problem_idx_in_dataset = work.problem_id - 1 # due to dataset list being 0-indexed locally
        ref_arch_path = dataset[problem_idx_in_dataset]

        problem_name = os.path.basename(ref_arch_path)
        ref_arch_src = read_file(ref_arch_path)

    # Extract problem number from problem name (e.g. "1" from "1_Square_matrix_multiplication_.py")
    problem_number = int(problem_name.split("_")[0])
    assert problem_number == work.problem_id, f"Problem number in filename ({problem_number}) does not match config problem_id ({config.problem_id})"
    
    

    # Construct Prompt   
    custom_cuda_prompt = prompt_generate_custom_cuda_from_prompt_template(ref_arch_src)
    if config.log_prompt:
        prompt_path = os.path.join(run_dir, f"level_{config.level}_problem_{work.problem_id}_sample_{work.sample_id}_prompt.txt")
        with open(prompt_path, "w") as f:
            f.write(custom_cuda_prompt)

    # Query server with constructed prompt
    custom_cuda = inference_server(custom_cuda_prompt)
    custom_cuda = extract_first_code(custom_cuda, ["python", "cpp"])
    # check LLM is able to generate custom CUDA code
    assert custom_cuda is not None, "Custom CUDA code generation failed"

    if config.verbose:
        print(f"Generated sample {work.sample_id} for problem {problem_number}: {problem_name}")

    # Store to local file
    kernel_path = os.path.join(run_dir, f"level_{config.level}_problem_{work.problem_id}_sample_{work.sample_id}_kernel.py")
    with open(kernel_path, "w") as f:
        f.write(custom_cuda)
    
    return True
    

def generate_sample_launcher(work: WorkArgs, config: GenerationConfig, dataset, inference_server: callable, run_dir: str):
    try:
        return generate_sample_single(work, config, dataset, inference_server, run_dir)
    except Exception as e:
        print(f"Error generating sample {work.problem_id} {work.sample_id}: {e}")
        return None


def check_kernel_exists(run_dir: str, level: int, problem_id: int, sample_id: int) -> bool:
    """
    Check if a kernel for a given problem and sample ID already exists in the run directory
    """
    kernel_path = os.path.join(run_dir, f"level_{level}_problem_{problem_id}_sample_{sample_id}_kernel.py")
    return os.path.exists(kernel_path)
    

@pydra.main(base=GenerationConfig)
def main(config: GenerationConfig):
    """
    Batch Generate Samples for Particular Level
    Store generated kernels in the specified run directory
    """
    print(f"Starting Batch Generation with config: {config}")

    # Dataset Configurations
    if config.dataset_src == "huggingface":
        dataset = load_dataset(config.dataset_name)
        curr_level_dataset = dataset[f"level_{config.level}"]
    elif config.dataset_src == "local":
        curr_level_dataset = construct_kernelbench_dataset(config.level)


    num_problems_in_level = len(curr_level_dataset)

    if config.subset == (None, None):
        problem_id_range = range(1, num_problems_in_level)
    else:
        assert config.subset[0] >= 1 and config.subset[1] <= num_problems_in_level, f"Subset range {config.subset} out of range for Level {config.level}"
        problem_id_range = range(config.subset[0], config.subset[1])

    print(f"Generating on 1 sample each for level {config.level} problems: {problem_id_range}")

    # set up run directory
    run_dir = os.path.join(config.runs_dir, config.run_name)
    os.makedirs(run_dir, exist_ok=True)
    pydra.save_yaml(config.to_dict(), os.path.join(run_dir, "generation_config.yaml"))


    assert config.store_type == "local", "supporting local file-system based storage for now" # database integreation coming soon, need to migrate from CUDA Monkeys code

    problems_to_run = []
    for problem_id in range(problem_id_range.start, problem_id_range.stop + 1): # end index is inclusive
        # assume sample id is 0 for now
        if not check_kernel_exists(run_dir, config.level, problem_id, sample_id=0):
            problems_to_run.append(
                WorkArgs(
                    problem_id=int(problem_id),
                    sample_id=0 # fix to 0 for now
                )
        )
    

    # Create inference function with config parameters
    # We provide some presets in utils but you can also pass in your own, see query_server for more details
    inference_server = create_inference_server_from_presets(server_type=config.server_type,
                                                        model_name=config.model_name,
                                                        temperature=config.temperature,
                                                        max_tokens=config.max_tokens,
                                                        verbose=config.verbose)

    # Launch workers
    generation_results = maybe_multithread(generate_sample_launcher, 
                      problems_to_run, 
                      config.num_workers, 
                      time_interval=config.api_query_interval, 
                      # extra args
                      config=config, 
                      dataset=curr_level_dataset, 
                      inference_server=inference_server,
                      run_dir=run_dir
                      )
    
    num_generated_samples = len(generation_results)
    total_problems = len(problems_to_run)
    num_failed_problems = total_problems - num_generated_samples
    print(f"Generated {num_generated_samples} samples for total {total_problems} problems, Please retry for the {num_failed_problems} failed problems.")


if __name__ == "__main__":
    main()

