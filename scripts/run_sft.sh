export HF_TOKEN=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
export WANDB_API_KEY=295af9bc6092e5591b44b322bafeb4ab49d3fd47
export RAY_DEDUP_LOGS=0

export CONTAINER="/home/zhaochengz/containers/ray+uv.sqsh"
export COMMAND="cd /home/zhaochengz/lustre/reinforcer; uv pip install -e .; uv run examples/run_sft.py"
# store experiment results in current working directory
# recursively expand soft links in PWDz
WD=$(readlink -f $PWD)
export MOUNTS="/home/zhaochengz/lustre/reinforcer:/home/zhaochengz/lustre/reinforcer,$WD:$WD"

sbatch --nodes=2 --account=llmservice_modelalignment_ppo --job-name=sft --partition=batch --time=4:0:0 --gres=gpu:8 \
    /home/zhaochengz/lustre/reinforcer/ray.sub