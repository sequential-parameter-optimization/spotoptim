#!/bin/bash
# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later
# =============================================================================
# run_spotoptim.sh — submit a spotoptim experiment on the GWDG NHR cluster.
#
# Usage:
#     sbatch scripts/slurm/run_spotoptim.sh <prefix>_exp.pkl
#
# The experiment pickle is produced locally by
#     opt = SpotOptim(..., n_jobs=16)
#     opt.save_experiment(prefix="myrun")
# and copied to the cluster with `scp`. This script runs the corresponding
# `scripts/slurm/run_spotoptim.py` runner inside the repo's `uv` environment
# and writes `<prefix>_res.pkl` next to the input.
#
# The default request matches what the spotoptim docs/slurm.qmd chapter
# describes: 16 CPUs, 16 GB RAM, 24 h walltime on the shared CPU partition.
# Override at submit time if needed:
#     sbatch --cpus-per-task=8 --mem=8G scripts/slurm/run_spotoptim.sh ...
# =============================================================================

#SBATCH --job-name=spotoptim
#SBATCH --partition=standard96s:shared
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/spotoptim_%j.out
#SBATCH --error=logs/spotoptim_%j.err
#SBATCH --constraint=inet

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: sbatch $0 <prefix>_exp.pkl" >&2
    exit 1
fi
EXP_PKL="$1"

# --- GWDG proxy (compute nodes need this for any outbound traffic) -----------
export http_proxy="http://www-cache.gwdg.de:3128"
export https_proxy="http://www-cache.gwdg.de:3128"

# --- Pin BLAS/OMP to one thread per worker -----------------------------------
# spotoptim's ProcessPoolExecutor spawns one process per n_jobs worker; without
# this each worker would in turn spin up cpu_count() BLAS threads, leading to
# 16 x 16 oversubscription on standard96s:shared.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONHASHSEED=0
export PYTHONUNBUFFERED=1

mkdir -p logs

module purge 2>/dev/null || true
module load gcc uv

REPO_DIR="${SPOTOPTIM_REPO:-$HOME/workspace/spotoptim}"
cd "$REPO_DIR"

echo "=== spotoptim job ==="
echo "Job ID    : ${SLURM_JOB_ID:-local}"
echo "Node      : $(hostname)"
echo "Partition : ${SLURM_JOB_PARTITION:-?}"
echo "CPUs      : ${SLURM_CPUS_PER_TASK:-?}"
echo "Mem       : ${SLURM_MEM_PER_NODE:-?} MB"
echo "Repo      : $REPO_DIR"
echo "Experiment: $EXP_PKL"
echo "Start     : $(date)"
echo "====================="

uv run python scripts/slurm/run_spotoptim.py "$EXP_PKL"

echo "=== Job completed at $(date) ==="
