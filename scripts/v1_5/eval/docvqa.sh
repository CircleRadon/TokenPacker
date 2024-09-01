#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT="0516-ours-13b-mini-gemini-data"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} /workspace/conda_env/llava_raw/bin/python -m llava.eval.eval_docvqa \
        --model-path /hy/zitong/code/LLaVA-1.1.3/llava-1.5-13b \
        --question-file /hy/zitong/data/docvqa/test_v1.0.json \
        --image-folder /hy/zitong/data/docvqa/images \
        --answers-file ./playground/data/eval/docvqa/answers/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=./playground/data/eval/docvqa/answers/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/docvqa/answers/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

/workspace/conda_env/llava_raw/bin/python scripts/convert_docvqa_for_eval.py --src $output_file --dst ./playground/data/eval/docvqa/answers/$CKPT/submit.json


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 /workspace/conda_env/torch_1.9/bin/python /hy/zitong/code/gpu/A100/gpu_bs96.py
