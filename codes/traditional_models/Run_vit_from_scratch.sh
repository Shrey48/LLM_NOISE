#!/bin/bash
#SBATCH --job-name=vit_from_scratch
#SBATCH --output=logs/vit_%j.out
#SBATCH --error=logs/vit_%j.err
#SBATCH --partition=<PARTITION>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=4-00:00:00

# Set these for your environment
BASE=/path/to/ECA_alpha_async      # directory containing ECA_Data_New/ and this script
VENV=/path/to/venv                 # Python environment with torch and numpy

cd "$BASE"
mkdir -p logs
source "$VENV/bin/activate"

echo "========================================================"
echo "  From-scratch ViT baseline (alphaECA)"
echo "  Job $SLURM_JOB_ID on $SLURM_NODELIST"
echo "  Started: $(date)"
echo "========================================================"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

python3 -u vit_from_scratch_alpha.py \
    --data_dir ECA_Data_New \
    --epochs 30 \
    --batch_size 32 \
    --output_dir vit_results
EXIT=$?
if [ $EXIT -ne 0 ]; then
    echo "ERROR: ViT training failed with exit code $EXIT"
    exit $EXIT
fi

echo "========================================================"
echo "  Done: $(date)"
echo "========================================================"
