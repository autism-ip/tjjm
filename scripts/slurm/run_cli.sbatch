#!/usr/bin/env bash
# -*- coding: utf-8 -*-
#
# [INPUT]  Slurm sbatch runtime, bash, optional conda init script, and the CLI command to execute.
# [OUTPUT] A cluster job that activates the project environment and runs the provided command.
# [POS]    scripts/slurm/ launch layer; it binds resources and environment only.
# [PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: sbatch [slurm-options] scripts/slurm/run_cli.sbatch <command> [args...]" >&2
  exit 1
fi

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/hpc/${USER:-${LOGNAME:-user}}/tjjm}"
CONDA_ENV="${CONDA_ENV:-tjjm}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"

if [[ ! -d "$PROJECT_ROOT" ]]; then
  PROJECT_ROOT="$HOME/sourcecode/tjjm"
fi

if [[ ! -d "$PROJECT_ROOT" ]]; then
  echo "PROJECT_ROOT not found: $PROJECT_ROOT" >&2
  exit 2
fi

if [[ ! -f "$CONDA_SH" ]]; then
  CONDA_SH="$HOME/anaconda3/etc/profile.d/conda.sh"
fi

if [[ ! -f "$CONDA_SH" ]]; then
  echo "Conda init script not found. Set CONDA_SH before submitting the job." >&2
  exit 3
fi

# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$CONDA_ENV"

cd "$PROJECT_ROOT"

echo "[slurm] host=$(hostname)"
echo "[slurm] job_id=${SLURM_JOB_ID:-local}"
echo "[slurm] partition=${SLURM_JOB_PARTITION:-unknown}"
echo "[slurm] pwd=$(pwd)"
echo "[slurm] command=$*"

exec "$@"
