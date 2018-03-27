#!/bin/bash

cores="15"
memory="4500"  # per core
scratch="5000"
gpus="2"
clock="24:00"
model="GeForceGTX1080Ti"
warn="-wt 15 -wa INT"

cmd="bsub
    -n $cores
    -W $clock $output
    $warn
    -R 'select[gpu_model0 == $model] rusage[mem=$memory,scratch=$scratch,ngpus_excl_p=$gpus]'
    $*"

echo $cmd
eval $cmd
