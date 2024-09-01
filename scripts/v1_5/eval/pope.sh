#!/bin/bash

NAME="0831-tokenpacker-retrain-llava1-1-3"

/workspace/conda_env/llava-raw/bin/python -m llava.eval.model_vqa_loader_pope \
    --model-path /workspace/checkpoints/0831-tokenpacker-retrain-llava1-1-3-stage2 \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder /hy/zitong/code/gpt4roi/data/coco_imgs \
    --answers-file ./playground/data/eval/pope/answers/$NAME.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

/workspace/conda_env/llava-raw/bin/python llava/eval/eval_pope.py \
    --annotation-dir ./playground/data/eval/pope/coco \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file ./playground/data/eval/pope/answers/$NAME.jsonl


# /workspace/conda_env/torch_1.9/bin/python /hy/zitong/code/gpu/A100/gpu_bs96.py