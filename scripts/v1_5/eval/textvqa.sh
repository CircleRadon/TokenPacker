#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python -m llava.eval.model_vqa_loader \
    --model-path llava-tokenpacker-7b \
    --question-file /path/tp/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder /path/tp/textvqa/train_images \
    --answers-file ./playground/data/eval/textvqa/answers/llava-tokenpacker-7b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

CUDA_VISIBLE_DEVICES=0 python -m llava.eval.eval_textvqa \
    --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./playground/data/eval/textvqa/answers/llava-tokenpacker-7b.jsonl
