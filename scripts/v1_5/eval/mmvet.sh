#!/bin/bash

python -m llava.eval.model_vqa \
    --model-path llava-tokenpacker-7b \
    --question-file /path/to/llava-mm-vet.jsonl \
    --image-folder /path/to/mm-vet/images \
    --answers-file ./playground/data/eval/mm-vet/answers/llava-tokenpacker-7b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p ./playground/data/eval/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src ./playground/data/eval/mm-vet/answers/llava-tokenpacker-7b.jsonl \
    --dst ./playground/data/eval/mm-vet/results/llava-tokenpacker-7b.json
