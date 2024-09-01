#!/bin/bash

/workspace/conda_env/llava-raw/bin/python -m llava.eval.model_vqa \
    --model-path /workspace/checkpoints/0831-tokenpacker-retrain-llava1-1-3-stage2 \
    --question-file ./playground/data/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder ./playground/data/eval/mm-vet/images \
    --answers-file ./playground/data/eval/mm-vet/answers/0831-tokenpacker-retrain-llava1-1-3.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p ./playground/data/eval/mm-vet/results

/workspace/conda_env/llava-raw/bin/python scripts/convert_mmvet_for_eval.py \
    --src ./playground/data/eval/mm-vet/answers/0831-tokenpacker-retrain-llava1-1-3.jsonl \
    --dst ./playground/data/eval/mm-vet/results/0831-tokenpacker-retrain-llava1-1-3.json

# /workspace/conda_env/torch_1.9/bin/python /hy/zitong/code/gpu/A100/gpu_bs96.py