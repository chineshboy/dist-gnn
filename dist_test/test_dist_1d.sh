#!/bin/sh

export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7;
WORLD_SIZE=4
GRAPH="Reddit"
for i in {1..$WORLD_SIZE}
do
  nohup python dist_1d.py --graphname=${GRAPH} --world_size=${WORLD_SIZE} --local_rank=${i} > node_${i}.log &
done
