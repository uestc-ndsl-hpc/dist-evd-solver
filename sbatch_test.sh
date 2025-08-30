#!/bin/bash

#================================================
# Slurm BATCH SCRIPT
#================================================
#SBATCH --job-name=my-gpu-job      # 作业名称，方便识别
#SBATCH --partition=gpu7           # 指定分区
#SBATCH --account=$(whoami)        # 指定账户（通常可以省略，系统会自动识别）
#SBATCH --ntasks=2                 # 请求 2 个任务
#SBATCH --cpus-per-task=32          # 每个任务请求 8 个 CPU核心
#SBATCH --mem-per-cpu=8000M        # 每个 CPU 核心请求 8000 MB 内存
#SBATCH --gpus=a800:2              # 请求 2 块 a800 GPU
#SBATCH --time=1-00:00:0           # 运行时长上限（1天）
#SBATCH --output=slurm-%j.out      # 将标准输出重定向到 slurm-<job_id>.out 文件

#================================================
# 运行环境和诊断信息（推荐）
#================================================
echo "=========================================="
echo "Job started on $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node list: $SLURM_JOB_NODELIST"
echo "GPU assigned: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

if [ -f /etc/profile.d/modules.sh ]; then
   source /etc/profile.d/modules.sh
fi

export OMPI_MCA_hwloc_base_binding_policy=none

mpirun build/dist-evd-solver-mpi --double -n=$1 -b=32 -nb=1024 -v > log/a800_2_$1.txt

echo "=========================================="
echo "Job finished on $(date)"
echo "=========================================="
