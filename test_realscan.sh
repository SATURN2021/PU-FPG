#!/usr/bin/env bash

cur_dir=$PWD
logdir=$1
datadir=$2
GPU=$3
PY_ARGS=${@:4}
echo "cur_dir: $cur_dir"
echo "logdir: $logdir"
echo "datadir: $datadir"
echo "GPU: $GPU"
echo "PY_ARGS: $PY_ARGS"
read -p "Press Enter to continue."

CUDA_VISIBLE_DEVICES=${GPU} python main.py --phase test --restore $logdir --data_dir  $datadir ${PY_ARGS}
mkdir $logdir/result-realscan
cp -r $cur_dir/evaluation_code/result/ $logdir/result-realscan
