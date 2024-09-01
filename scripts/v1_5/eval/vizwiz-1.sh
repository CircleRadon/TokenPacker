#!/bin/bash

/workspace/conda_env/llava-raw/bin/python -m llava.eval.model_vqa_loader \
    --model-path /workspace/checkpoints/0831-tokenpacker-retrain-llava1-1-3-stage2 \
    --question-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --image-folder ./playground/data/eval/vizwiz/test \
    --answers-file ./playground/data/eval/vizwiz/answers/0831-tokenpacker-retrain-llava1-1-3-1.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

/workspace/conda_env/llava-raw/bin/python scripts/convert_vizwiz_for_submission.py \
    --annotation-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --result-file ./playground/data/eval/vizwiz/answers/0831-tokenpacker-retrain-llava1-1-3-1.jsonl \
    --result-upload-file ./playground/data/eval/vizwiz/answers_upload/0831-tokenpacker-retrain-llava1-1-3-1.json

# /workspace/conda_env/torch_1.9/bin/python /hy/zitong/code/gpu/A100/gpu_bs96.py 