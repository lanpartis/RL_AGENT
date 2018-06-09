#!/bin/zsh
export LD_LIBRARY_PATH=~/openpose/3rdparty/caffe/build/lib:$LD_LIBRARY_PATH
alias openpose='~/openpose/build/examples/openpose/openpose.bin' # set your own openpose directory
dir='RL_DATA/EP1/STATE/'

for folder in $(ls $dir); do
    subfolder=${dir}"/$folder"
     openpose --image_dir ${subfolder} -write_json ${subfolder} --display 0 --render_pose 0
done

