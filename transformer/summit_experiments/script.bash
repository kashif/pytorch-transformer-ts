#!/bin/bash
#BSUB -P CSC499
#BSUB -W 23:00
#BSUB -nnodes 20
#BSUB -q batch-hm
#BSUB -J mldl_test_job
#BSUB -o /gpfs/alpine/csc499/scratch/hstellar/job%J.out
#BSUB -e /gpfs/alpine/csc499/scratch/hstellar/job%J.out
#BSUB -alloc_flags gpudefault
module load cuda
module load job-step-viewer
module load ums
module load ums-gen119
module load nvidia-rapids/cucim_21.08
conda activate py37n
 
nodes=($(cat ${LSB_DJOB_HOSTFILE} | sort | uniq | grep -v login | grep -v batch)) 
head=${nodes[0]}                                                                   
export MASTER_ADDR=$head                                                           
export MASTER_PORT=29500 
# default from torch launcher 
echo $MASTER_ADDR   
echo $MASTER_PORT 
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

jsrun -n20 -bpacked:8 -g6 -a6 -c42 -r1 python scaling.py
