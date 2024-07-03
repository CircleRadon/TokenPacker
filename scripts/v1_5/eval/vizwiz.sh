#!/bin/bash

python -m llava.eval.model_vqa_loader \
    --model-path llava-tokenpacker-7b\
    --question-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --image-folder /path/to/vizwiz/test \
    --answers-file ./playground/data/eval/vizwiz/answers/llava-tokenpacker-7b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python scripts/convert_vizwiz_for_submission.py \
    --annotation-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --result-file ./playground/data/eval/vizwiz/answers/llava-tokenpacker-7b.jsonl \
    --result-upload-file ./playground/data/eval/vizwiz/answers_upload/llava-tokenpacker-7b.json
