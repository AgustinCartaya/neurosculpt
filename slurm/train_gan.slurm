#!/bin/bash
# Train a 3D GAN on a SLURM cluster. Submit from the repository root:
#   sbatch --export=ALL,MODEL=mgan slurm/train_gan.slurm      # mgan | mgan_seg | mpgan | mpgan_seg
# Fill in the placeholders (<...>) for your cluster and run `mkdir -p slurm/logs` before submitting.

#SBATCH --job-name=train_gan
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --nodes=1
#SBATCH --time=1-00:00:00            # D-HH:MM:SS (MGAN used 1 day per job, MPGAN 8 h; training resumes from the last checkpoint)
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
##SBATCH --mail-user=<EMAIL>
##SBATCH --mail-type=ALL

MODEL="${MODEL:-mgan}"
CONDA_ENV="${CONDA_ENV:-<CONDA_ENV>}"   # environment with brainpred[tf] installed (TF 2.15 + tensorflow-addons)

cd "${SLURM_SUBMIT_DIR}"
echo "Job ${SLURM_JOB_NAME} (${SLURM_JOB_ID}) on ${SLURM_JOB_NODELIST}, model: ${MODEL}"

module purge
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

python scripts/train_gan.py --model "${MODEL}"
