#!/bin/bash
#SBATCH --job-name=4node_test
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=4node_test_%j.out
#SBATCH --error=4node_test_%j.err

# Environment setup
module load python/3.9
module load cuda/12.2
source /mnt/lustre/user/llm-env/bin/activate

# Basic config
export OMP_NUM_THREADS=16
export CUDA_VISIBLE_DEVICES=0

# Gloo backend settings
export GLOO_DEVICE_TRANSPORT=TCP
export GLOO_TIMEOUT_SECONDS=600

# Debug output
export PYTHONUNBUFFERED=1

echo "========================================"
echo "4-NODE LAYER SPLITTING TEST"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Node list: $SLURM_JOB_NODELIST"
echo "========================================"

# Display node info
echo "Node information:"
srun --ntasks=4 --ntasks-per-node=1 bash -c '
    echo "[$(hostname)] GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)"
    echo "[$(hostname)] Memory: $(free -h | grep Mem | awk "{print \$2}")"
    echo ""
'

echo "========================================"
echo "Starting 4-node test..."
echo "========================================"

# Run test
srun python model_parallel_4node.py

echo "========================================"
echo "4-node test completed at $(date)"
echo "========================================"