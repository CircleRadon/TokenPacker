
python -m llava.eval.eval_ocr_bench \
    --model_path llava-tokenpacker-7b  \
    --image_folder /path/to/OCR-Bench/OCRBench_Images \
    --output_folder ./playground/data/eval/ocr_bench \
    --OCRBench_file /path/to/OCRBench.json \
    --save_name llava-tokenpacker-7b \
    --temperature 0 \
    --conv_mode vicuna_v1

