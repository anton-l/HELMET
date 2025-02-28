#!/bin/bash
#SBATCH --job-name=helmet_eval
#SBATCH --partition hopper-prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=22
#SBATCH --mem-per-cpu=11G
#SBATCH --gpus=2
#SBATCH --output=logs/%x_%A_%a_%N.out
#SBATCH --error=logs/%x_%A_%a_%N.err
#SBATCH --time=24:00:00

set -x -e

function print_help {
    echo "Usage: $0 --model MODEL_PATH [--context_length CONTEXT_LENGTH] [--use_chat_template]"
    echo ""
    echo "Options:"
    echo "  --model PATH            Path to the model to evaluate (required)"
    echo "  --context_length LEN    Context length to use (8k, 16k, 32k, 8192, 16384, 32768)"
    echo "                          Default: 16k"
    echo "  --use_chat_template     Use chat template for the model (flag)"
    echo "  --help                  Display this help message"
    exit 1
}

source ~/.bashrc
source "/admin/home/anton/miniforge3/etc/profile.d/conda.sh"
source activate /fsx/anton/conda_envs/helmet
module load cuda/12.1

MODEL=""
CONTEXT_LENGTH="16k"
USE_CHAT_TEMPLATE="False"

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --model)
            if [[ -z "$2" || "$2" == --* ]]; then
                echo "Error: --model requires a value"
                print_help
            fi
            MODEL="$2"
            shift 2
            ;;
        --context_length)
            if [[ -z "$2" || "$2" == --* ]]; then
                echo "Error: --context_length requires a value"
                print_help
            fi
            CONTEXT_LENGTH="$2"
            shift 2
            ;;
        --use_chat_template)
            USE_CHAT_TEMPLATE="True"
            shift
            ;;
        --help)
            print_help
            ;;
        *)
            echo "Error: Unknown option: $1"
            print_help
            ;;
    esac
done

if [ -z "$MODEL" ]; then
    echo "Error: --model parameter is required"
    print_help
fi

CONTEXT_LENGTH_NUM=""
case $CONTEXT_LENGTH in
    "8k"|"8K")
        CONTEXT_LENGTH_NUM=8192
        CONFIG_DIR="configs/configs_8k"
        ;;
    "16k"|"16K")
        CONTEXT_LENGTH_NUM=16384
        CONFIG_DIR="configs/configs_16k"
        ;;
    "32k"|"32K")
        CONTEXT_LENGTH_NUM=32768
        CONFIG_DIR="configs/configs_32k"
        ;;
    8192)
        CONTEXT_LENGTH_NUM=8192
        CONFIG_DIR="configs/configs_8k"
        ;;
    16384)
        CONTEXT_LENGTH_NUM=16384
        CONFIG_DIR="configs/configs_16k"
        ;;
    32768)
        CONTEXT_LENGTH_NUM=32768
        CONFIG_DIR="configs/configs_32k"
        ;;
    *)
        echo "Error: context_length must be 8k, 16k, 32k, 8192, 16384, or 32768"
        exit 1
        ;;
esac

export CONTEXT_LENGTH_NUM=$CONTEXT_LENGTH_NUM

echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo ""
echo "Number of Nodes Allocated      = $SLURM_JOB_NUM_NODES"
echo "Number of Tasks Allocated      = $SLURM_NTASKS"
echo "Number of Cores/Task Allocated = $SLURM_CPUS_PER_TASK"
echo "Context Length                 = $CONTEXT_LENGTH ($CONTEXT_LENGTH_NUM)"
echo "Model                          = $MODEL"

IDX=$SLURM_ARRAY_TASK_ID
NGPU=2

PORT=$(shuf -i 30000-65000 -n 1)
echo "Port                          = $PORT"

export OMP_NUM_THREADS=10
export HF_DATASETS_TRUST_REMOTE_CODE=True

TAG=v1

CONFIGS=(recall.yaml icl.yaml longqa.yaml rag.yaml rerank.yaml)

SEED=42

# Set model parameters
MODEL_NAME="$MODEL"
OUTPUT_DIR="output/${MODEL_NAME##*/}"

# Set options based on use_chat_template
OPTIONS=""
if [[ "$USE_CHAT_TEMPLATE" == "False" ]]; then
    OPTIONS="$OPTIONS --use_chat_template False"
fi

echo "Evaluation output dir         = $OUTPUT_DIR"
echo "Tag                           = $TAG"
echo "Model name                    = $MODEL_NAME"
echo "Use chat template             = $USE_CHAT_TEMPLATE"
echo "Options                       = $OPTIONS"
echo "Config directory              = $CONFIG_DIR"

# Create results directory
RESULTS_DIR="./results/${MODEL_NAME##*/}"
mkdir -p "$RESULTS_DIR"

# Display configuration summary
echo "======== Configuration Summary ========"
echo "Model:                      $MODEL_NAME"
echo "Context length:             $CONTEXT_LENGTH ($CONTEXT_LENGTH_NUM)"
echo "Config directory:           $CONFIG_DIR"
echo "Results directory:          $RESULTS_DIR"
echo "Output file:                $RESULTS_DIR/${CONTEXT_LENGTH}.csv"
echo "Use chat template:          $USE_CHAT_TEMPLATE"
echo "======================================="

# Run evaluations for all configs
for CONFIG in "${CONFIGS[@]}"; do
    echo "Config file: $CONFIG_DIR/$CONFIG"

    python eval.py \
        --config "$CONFIG_DIR/$CONFIG" \
        --seed $SEED \
        --output_dir $OUTPUT_DIR \
        --tag $TAG \
        --model_name_or_path $MODEL_NAME \
        $OPTIONS
done

MODEL_NAME_SHORT="${MODEL_NAME##*/}"  # Extract just the model name without path

python scripts/collect_results.py \
    --output_file "$RESULTS_DIR/${CONTEXT_LENGTH}.csv" \
    --model_name "$MODEL_NAME_SHORT" \
    --context_length $CONTEXT_LENGTH_NUM \
    --use_chat_template $USE_CHAT_TEMPLATE

echo "Results saved to $RESULTS_DIR/${CONTEXT_LENGTH}.csv"
echo "Finished with exit code $?"