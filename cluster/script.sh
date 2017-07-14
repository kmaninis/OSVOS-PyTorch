#!/bin/bash
#$ -S /bin/bash
#$ -l gpu
#$ -l h_rt=00:59:00
#$ -l h_vmem=20G
#S -pe multicore 2
#$ -o /scratch_net/reinhold/Kevis/logs/
#$ -e /scratch_net/reinhold/Kevis/logs/
#$ -l hostname=biwirender0[5-9]|biwirender1[0-2]
#$ -j y

EXP_PATH=/home/kmaninis/glusterfs/pytorch_experiments/OSVOS-PyTorch
PYTHON_EXEC=/home/kmaninis/apps/anaconda2/bin/python
SEQ_NAME=$1
export SEQ_NAME=$SEQ_NAME

cd $EXP_PATH

$PYTHON_EXEC $EXP_PATH/train_online.py
