#!/bin/bash
#SBATCH --job-name=llm_reasoning_eval
#SBATCH --output=logs/reasoning_eval_%j.out
#SBATCH --error=logs/reasoning_eval_%j.err
#SBATCH --partition=<PARTITION>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --mem=480G
#SBATCH --time=7-00:00:00

# Set these for your environment
BASE=/path/to/ECA_alpha_async      # directory containing ECA_Data_New/ and the scripts
VENV=/path/to/venv                 # Python environment with torch and transformers
export BASE_DIR="$BASE"
export MODELS_DIR=/path/to/models  # directory containing the model checkpoints

cd "$BASE"
mkdir -p logs results/zero_shot_reasoning_only results/zero_shot_tools_only results/zero_shot_both
source "$VENV/bin/activate"

echo "========================================================"
echo "Job     : $SLURM_JOB_ID"
echo "GPUs    : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
echo "Start   : $(date)"
echo "========================================================"

N_SAMPLES=50

for CONDITION in reasoning_only tools_only both; do
    for MODEL in \
        "Qwen2.5-7B-Instruct" \
        "Qwen2.5-72B-Instruct" \
        "Llama-3.1-8B-Instruct" \
        "Llama-3.1-70B-Instruct" \
        "Mistral-7B-Instruct-v0.3" \
        "Mixtral-8x7B-Instruct-v0.1"
    do
        echo "========================================================"
        echo "  Running condition=$CONDITION: $MODEL  |  $(date)"
        echo "========================================================"
        python3 -u zero_shot_alpha_async_reasoning_eval.py \
            --model "$MODEL" \
            --condition "$CONDITION" \
            --n_samples $N_SAMPLES
        EXIT=$?
        if [ $EXIT -ne 0 ]; then
            echo "ERROR: $MODEL ($CONDITION) failed with exit code $EXIT -- continuing"
        else
            echo "DONE: $MODEL ($CONDITION) at $(date)"
        fi
        python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null
    done
done
echo "========================================================"
echo "All reasoning/tool-use evaluations complete: $(date)"
echo "========================================================"
