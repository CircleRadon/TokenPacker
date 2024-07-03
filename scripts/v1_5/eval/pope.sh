#!/bin/bash


NAME="llava-tokenpacker-7b"

python -m llava.eval.model_vqa_loader_pope \
    --model-path llava-tokenpacker-7b \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder /path/tp/coco_imgs \
    --answers-file ./playground/data/eval/pope/answers/$NAME.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_pope.py \
    --annotation-dir ./playground/data/eval/pope/coco \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file ./playground/data/eval/pope/answers/$NAME.jsonl
