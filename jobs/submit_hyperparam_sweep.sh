#!/bin/bash
# SLURM Job Array for Hyperparameter Tuning
# Submits multiple training jobs with different configs in parallel

#SBATCH --partition=rtx4090
#SBATCH --array=0-11                    # Run 12 different configs (0-11)
#SBATCH --time=12:00:00                 # 12 hours per job
#SBATCH --job-name=hyperparam_sweep
#SBATCH --output=logs/sweep-%A_%a.out  # %A=array ID, %a=task ID
#SBATCH --mail-user=pardot@post.bgu.ac.il
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Activate conda environment
source /storage/modules/packages/anaconda/etc/profile.d/conda.sh
conda activate de_SGDD

# Navigate to project directory
cd /home/pardot/decoder_sgdd

# Create logs directory
mkdir -p logs

# Array task ID determines which config to use
# SLURM_ARRAY_TASK_ID is automatically set (0, 1, 2, ..., 11)
CONFIG_FILE="configs/hyperparam_sweep/config_${SLURM_ARRAY_TASK_ID}.yaml"

echo "================================"
echo "SLURM Job Array Task: $SLURM_ARRAY_TASK_ID"
echo "Config: $CONFIG_FILE"
echo "Job ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "================================"

# Check if config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Run training with the selected config
python train.py --config "$CONFIG_FILE"

# Capture exit code
EXIT_CODE=$?
echo "Training completed with exit code: $EXIT_CODE"
exit $EXIT_CODE
