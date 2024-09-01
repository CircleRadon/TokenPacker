#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT="0831-tokenpacker-retrain-llava1-1-3-1"
SPLIT="llava_vqav2_mscoco_test-dev2015"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} /workspace/conda_env/llava-raw/bin/python -m llava.eval.model_vqa_loader \
        --model-path /workspace/checkpoints/0831-tokenpacker-retrain-llava1-1-3-stage2 \
        --question-file ./playground/data/eval/vqav2/$SPLIT.jsonl \
        --image-folder /hy/zitong/code/gpt4roi/data/VQAv2/test2015/ \
        --answers-file ./playground/data/eval/vqav2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=./playground/data/eval/vqav2/answers/$SPLIT/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/vqav2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

/workspace/conda_env/llava-raw/bin/python scripts/convert_vqav2_for_submission.py --split $SPLIT --ckpt $CKPT

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 /workspace/conda_env/torch_1.9/bin/python /hy/zitong/code/gpu/A100/gpu_bs96.py