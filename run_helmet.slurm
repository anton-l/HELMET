#!/bin/bash
#SBATCH --job-name=helmet_short
#SBATCH --partition hopper-prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=11
#SBATCH --mem-per-cpu=11G
#SBATCH --gpus=1
#SBATCH --output=logs/%x_%A_%a_%N.out
#SBATCH --error=logs/%x_%A_%a_%N.err
#SBATCH --time=03:00:00
#SBATCH --array=0-4%5

set -x -e
source ~/.bashrc
source "/admin/home/anton/miniforge3/etc/profile.d/conda.sh"
source activate /fsx/anton/conda_envs/helmet

module load cuda/12.1

echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo ""
echo "Number of Nodes Allocated      = $SLURM_JOB_NUM_NODES"
echo "Number of Tasks Allocated      = $SLURM_NTASKS"
echo "Number of Cores/Task Allocated = $SLURM_CPUS_PER_TASK"
echo "Array Job ID                   = $SLURM_ARRAY_JOB_ID"
echo "Array Task ID                  = $SLURM_ARRAY_TASK_ID"
echo "Cache                          = $TRANSFORMERS_CACHE"

IDX=$SLURM_ARRAY_TASK_ID
NGPU=2

PORT=$(shuf -i 30000-65000 -n 1)
echo "Port                          = $PORT"

export OMP_NUM_THREADS=10
export HF_DATASETS_TRUST_REMOTE_CODE=True

TAG=v1

CONFIGS=(recall_short.yaml rag_short.yaml longqa_short.yaml icl_short.yaml rerank_short.yaml)

SEED=42

MODELS=(
#  "meta-llama/Llama-3.2-1B" # 0
#  "meta-llama/Llama-3.2-1B-Instruct" # 1
#  "Qwen/Qwen2.5-1.5B" # 2
#  "Qwen/Qwen2.5-1.5B-Instruct" # 3
#  "HuggingFaceTB/SmolLM2-1.7B-final" # 4
#  "HuggingFaceTB/SmolLM2-1.7B-Intermediate-SFT-v2" # 5
#  "ibm-granite/granite-3.0-1b-a400m-base" # 6
#  "ibm-granite/granite-3.0-1b-a400m-instruct" # 7
#
#  "HuggingFaceTB/smollmv2-1.7B-32k-final" # 8
#  "HuggingFaceTB/smollmv2-1.7B-16k-final" # 9
#  "HuggingFaceTB/smollmv2-lc-base-16k-lr1e-5" # 10
#  "HuggingFaceTB/smollmv2-lc-base-16k-lr2e-5" # 11
#  "HuggingFaceTB/smollmv2-lc-base-8k-lr2e-5-800steps" # 12
#  "HuggingFaceTB/smollmv2-lc-base-8k-lr2e-5" # 13
#  "HuggingFaceTB/smollmv2-lc-co1-base-8k-16k-lr1e-5-400steps" # 14
#
#  "HuggingFaceTB/smollmv2-lc-co1-base-16k-32k-lr1e-5-400steps" # 15
#  "HuggingFaceTB/smollmv2-lc-co1-base-8k-32k-lr1e-5-400steps" # 16
#  "HuggingFaceTB/smollmv2-lc-m2-base-16k-lr1e-5-200steps" # 17
#  "HuggingFaceTB/smollmv2-lc-m2-base-16k-lr1e-5-400steps" # 18
#  "HuggingFaceTB/smollmv2-lc-m2-base-16k-lr1e-5-800steps" # 19
#  "HuggingFaceTB/smollmv2-lc-m2-base-8k-lr1e-5-800step" # 20
#  "HuggingFaceTB/smollmv2-lc-m2-base-8k-lr1e-5-400steps" # 21
#  "HuggingFaceTB/smollmv2-lc-m2-base-8k-lr1e-5-200steps" # 22
#
#  "HuggingFaceTB/smollmv2-temp-1.7B-16k" # 23
#  "HuggingFaceTB/smollm2-lc-merge-no-base" # 24
#  "HuggingFaceTB/smollmv2-lc-m4-base-16k-lr1e-5-800steps" # 25
#
#  "HuggingFaceTB/smollm2-1.7B-16k-m4-magpieu-ifeval" # 26
#  "HuggingFaceTB/smollm2-1.7B-16k-m4-magpieu-ifeval-longalign" # 27
#  "HuggingFaceTB/smollm2-1.7B-16k-m4-mix5-ep1" # 28
#  "HuggingFaceTB/smollm2-1.7B-16k-m4-mix5-longalign-ep1" # 29
#
#  "HuggingFaceTB/decay-75B_8k_lc" # 30
#  "HuggingFaceTB/decay-200B_8k_lc-5100000" # 31
#  "HuggingFaceTB/decay-200B_8k_lc-5112000" # 32
#  "HuggingFaceTB/decay-200B_8k_lc-5120000" # 33
#  "HuggingFaceTB/smollm2-1.7B-decay-75B-8k-magpieu-ifeval-longalign" # 34
#  "HuggingFaceTB/smollm2-1.7B-decay-75B-8k-mix5-longalign-ep1" # 35
#
#  "HuggingFaceTB/smollm2-8k-dpo-mix5-longalign-ultraf-ep1" # 36
#  "HuggingFaceTB/smollm2-8k-dpo-mix5-longalign-ultraf-ep2" # 37
#  "HuggingFaceTB/smollm2-18k-dpo-magpieu-ifeval-longalign-ultraf-ep3" # 38
#
#  "HuggingFaceTB/fix-rope-50B-8k-5216000" # 39
#  "HuggingFaceTB/fix-rope-50B-8k-5218000" # 40
#  "HuggingFaceTB/fix-rope-50B-8k-5220000" # 41
#  "HuggingFaceTB/smollm2-1.7B-decay-75B-8k-mix5-longalign-rewritev2" # 42
#  "HuggingFaceTB/ft-16k-1e5" # 43
#  "HuggingFaceTB/50B-16k-320k-rope" # 44
#  "HuggingFaceTB/big-50B-8k-130k-rope" # 45
#  "HuggingFaceTB/135M-lc-6B" # 46
#  "HuggingFaceTB/360M-50B-8k-2592k" # 47
#  "HuggingFaceTB/50B-decay-last-try" # 48
#  "HuggingFaceTB/360M-lc-100k-rope-12B" # 49
#  "HuggingFaceTB/135M-lc-12B" # 50
#  "HuggingFaceTB/135M-lc-100k-rope-12B" # 51
#  "HuggingFaceTB/smollm2-1.7B-8k-mix7-half-ep2-v2-dpo-ultraf-ep3" # 52
#  "HuggingFaceTB/smollm2-1.7B-8k-mix7-ep2-v2-dpo-ultraf-ep3" # 53
#  "HuggingFaceTB/SmolLM2-1.7B"

#  "HuggingFaceTB/SmolLM2-1.7B-Instruct-rope300k-reup"
#  "HuggingFaceTB/SmolLM2-Instruct-16k-SeaLong-ST5k-rope300k"
#  "HuggingFaceTB/SmolLM2-Instruct-16k-SeaLong-LongAlign-ST5k-rope300k"
#  "HuggingFaceTB/SmolLM2-16k-SmolTalk1M-AceMath500k-CodeIO-ep2-8k"
#  "HuggingFaceTB/SmolLM2-16k-SmolTalk1M-AceMath500k-CodeIO-ep1-8k"
#  "HuggingFaceTB/SmolLM2-16k-SmolTalk1M-AceMath500k-ep1-8k"
#  "HuggingFaceTB/SmolLM2-16k-SmolTalk1M-AceMath100k-CodeIO-ep1-rope300k"
#  "HuggingFaceTB/SmolLM2-16k-SmolTalk1M-AceMath100k-CodeIO-ep1-rope100k"
#  "HuggingFaceTB/SmolLM2-Instruct-16k-SeaLong-LongAlign-ST10k-v2-rope300k"
#  "HuggingFaceTB/SmolLM2-Instruct-16k-SeaLong-LongAlign-ST10k-v2-rope300k-1e-4-ep2"
#  "HuggingFaceTB/SmolLM2-Ins-16k-SeaLong-LongAlign-ST5k-v2-rope300k-1e-4"
#  "HuggingFaceTB/SmolLM2-Instruct-16k-SeaLong-LongAlign-ST10k-v2-rope500k-1e-4"

#  "HuggingFaceTB/SmolLM2-Instruct-16k-SeaLong-LongAlign-ST10k-v2-rope500k-5e-5-ep2"
#  "HuggingFaceTB/SmolLM2-Instruct-16k-SeaLong-LongAlign-ST10k-v2-rope300k-1e4-2"
#  "HuggingFaceTB/SmolLM2-Instruct-16k-SeaLong-LongAlign-ST10k-v2-rope300k-5e-5"
#  "/fsx/loubna/data/SmolLM2-1.7B-Instruct-rope500k"
  "/fsx/loubna/data/SmolLM2-1.7B-Instruct-rope600k-32k"
  "/fsx/loubna/data/SmolLM2-1.7B-Instruct-rope500k-32k"
  "/fsx/loubna/data/SmolLM2-1.7B-Instruct-rope700k-32k"
  "/fsx/loubna/data/SmolLM2-1.7B-Instruct-rope800k-32k"
)

MODEL_NAME="${MODELS[$IDX]}"
OUTPUT_DIR="output/${MODEL_NAME##*/}"

shopt -s nocasematch
chat_models=".*(instruct|sft|it$|16k-m4|longalign|dpo).*"
echo $MODEL_NAME
if ! [[ $MODEL_NAME =~ $chat_models ]]; then
    OPTIONS="$OPTIONS --use_chat_template False"
fi

echo "Evaluation output dir         = $OUTPUT_DIR"
echo "Tag                           = $TAG"
echo "Model name                    = $MODEL_NAME"
echo "Options                       = $OPTIONS"

for CONFIG in "${CONFIGS[@]}"; do
    echo "Config file: $CONFIG"

    python eval.py \
        --config configs/smollm_16k/$CONFIG \
        --seed $SEED \
        --output_dir $OUTPUT_DIR \
        --tag $TAG \
        --model_name_or_path $MODEL_NAME \
        $OPTIONS
done

echo "finished with $?"



