#!/bin/bash

/workspace/conda_env/llava-raw/bin/python -m llava.eval.model_vqa_loader \
    --model-path /workspace/checkpoints/0831-tokenpacker-retrain-llava1-1-3-stage2/ \
    --question-file ./playground/data/eval/MME/llava_mme.jsonl \
    --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file ./playground/data/eval/MME/answers/0831-tokenpacker-retrain-llava1-1-3.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

cd ./playground/data/eval/MME

/workspace/conda_env/llava-raw/bin/python convert_answer_to_mme.py --experiment 0831-tokenpacker-retrain-llava1-1-3

cd eval_tool

/workspace/conda_env/llava-raw/bin/python calculation.py --results_dir answers/0831-tokenpacker-retrain-llava1-1-3


# /workspace/conda_env/torch_1.9/bin/python /hy/zitong/code/gpu/A100/gpu_bs96.py

