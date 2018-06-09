#!/bin/zsh
export LD_LIBRARY_PATH=~/openpose/3rdparty/caffe/build/lib:$LD_LIBRARY_PATH
alias openpose='~/openpose/build/examples/openpose/openpose.bin'
dir=$1
openpose --image_dir $dir --write_json $dir
