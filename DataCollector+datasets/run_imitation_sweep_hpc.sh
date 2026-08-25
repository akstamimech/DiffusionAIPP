#!/bin/sh
#
#SBATCH --job-name="sweep_imit"
#SBATCH --partition=gpu-a100-small
#SBATCH --time=00:25:00
#SBATCH --array=0-49%2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --account=Education-AE-MSc-AE
#SBATCH --output=slurm-imitation-%A_%a.out
#SBATCH --error=slurm-imitation-%A_%a.err

module load 2025
module load cuda
module load python

source ~/.bashrc
conda activate diffusion_env

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMPI_MCA_opal_cuda_support=true

export MAP_START=41
export MAP_END=50
export REPEATS_PER_MAP=5
export TIMEALLOTED=3000
export WALLCLOCK_SECONDS=300
export UTILITY_THRESHOLD=0.3
export PLANNING_HORIZON=8
export EXECUTION_CHUNK=40
export ENFORCE_MIN_STEP_TIME=1
export SKIP_VIZ=1

EXPERIMENT_DIR="${EXPERIMENT_DIR:-$HOME/experiments}"
cd "$EXPERIMENT_DIR" || exit 1
SCRIPT_DIR="$(pwd)"

export CSV_DIR="$SCRIPT_DIR/csv"
export RESULTS_ROOT="$SCRIPT_DIR/results_hpc"
export IMITATE_CHECKPOINT="$SCRIPT_DIR/checkpoints/imitate_best.pth"

/scratch/ajain3/conda/envs/diffusion_env/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
srun /scratch/ajain3/conda/envs/diffusion_env/bin/python -u "$SCRIPT_DIR/sweep_imitation_hpc.py"
