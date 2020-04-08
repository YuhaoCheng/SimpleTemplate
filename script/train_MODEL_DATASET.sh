#! /bin/bash
cfg_folder='MODEL/DATASET'
project_path='default_path'
cfg_name='default.yaml'
verbose='default'

#-----------------useage-------------------------
# sh xxx.sh GPUS project_path configure_name verbose
#------------------------------------------------

if test -n "$2" # $2 is not null, it is true 
then
    project_path=$2
    echo "the project path is:" $project_path
else
    echo "Using the default project path:"$project_path
fi

cd $project_path

if test -n "$3" # $2 is not null, it is true 
then
    cfg_name=$3
    echo "the config file is:" $cfg_name
else
    echo "Using the default config:"$cfg_name
fi

if test -n "$4"
then
    verbose=$4
    echo "The verbose is:"$verbose
else
    echo "Using the default verbose:"$verbose
fi


CUDA_VISIBLE_DEVICES=$1 python main_MODEL.py --project_path $project_path --cfg_folder $cfg_folder --cfg_name $cfg_name --verbose $verbose 
