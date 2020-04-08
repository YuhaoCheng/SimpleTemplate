cd ./script
MODEL="test"
DATASET="test"
project_path=""
GPU="0"
config=""
verobose=""
while getopts "m:d:p:g:c:v" opt
do
    case $opt in 
    m) 
    MODEL=$OPTARG
    echo "model:"$MODEL
    ;;
    d)
    DATASET=$OPTARG
    echo "dataset":$DATASET
    ;;
    p)
    project_path=$OPTARG
    echo "project_path:"$project_path
    ;;
    g)
    GPU=$OPTARG
    echo "GPUS:"$GPU
    ;;
    c)
    config=$OPTARG
    echo "config:"$config
    ;;
    v)
    verobose=$OPTARG
    echo "verbose:"$verobose
    ;;
    ?)
    exit 1
    ;;
    esac
done
SCRIPT_NAME="train_"$MODEL"_"$DATASET".sh"
echo "Using the training script: $SCRIPT_NAME"
sh $SCRIPT_NAME $GPU $project_path $config $verobose
