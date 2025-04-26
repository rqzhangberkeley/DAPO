#!/bin/bash
#SBATCH --job-name=test-valid       # Job name
#SBATCH --output=./logs/rloo_dapo_Cl_1.5B_%j.out  # Output file (%j will be replaced by job ID)
#SBATCH --error=./logs/rloo_dapo_Cl_1.5B_%j.err   # Error file
#SBATCH --nodes=1                 # Number of nodes
#SBATCH --ntasks-per-node=1       # Number of tasks per node
#SBATCH --cpus-per-task=256         # Number of CPU cores per task
#SBATCH --gpus-per-node=4              # Number of GPUs (4 GPUs per node)
#SBATCH --mem=500G                # Memory per node
#SBATCH --time=2:00:00           # Time limit (24 hours)
#SBATCH --account=bdwy-dtai-gh    # Account name (adjust to your account)
#SBATCH --mail-user=rqzhang@berkeley.edu  # Email address to receive notifications
#SBATCH --mail-type=BEGIN,END,FAIL         # Send email at begin, end, or fail of job


# Load modules (Everytime before running any code on DeltaAI, please run this command)
# module default
module load cuda/12.6.1
module load gcc/11.4.0
conda activate verl
wandb login 363018e9dc8339fae726d3b48a839f262c457194

./recipe/dapo/test_dapo_1.5b.sh