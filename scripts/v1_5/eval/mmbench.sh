#!/bin/bash

SPLIT="mmbench_dev_20230712"

/workspace/conda_env/llava-raw/bin/python -m llava.eval.model_vqa_mmbench \
    --model-path /workspace/checkpoints/0831-tokenpacker-retrain-llava1-1-3-stage2 \
    --question-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --answers-file ./playground/data/eval/mmbench/answers/$SPLIT/0831-tokenpacker-retrain-llava1-1-3.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT

/workspace/conda_env/llava-raw/bin/python scripts/convert_mmbench_for_submission.py \
    --annotation-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --result-dir ./playground/data/eval/mmbench/answers/$SPLIT \
    --upload-dir ./playground/data/eval/mmbench/answers_upload/$SPLIT \
    --experiment 0831-tokenpacker-retrain-llava1-1-3

# /workspace/conda_env/torch_1.9/bin/python /hy/zitong/code/gpu/A100/gpu_bs96.py
