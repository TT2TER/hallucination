#!/bin/bash

python -m llava.eval.llava_model_vqa_loader \
    --model-path ../llava-v1.5-7b \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder ./playground/data/eval/pope/val2014 \
    --answers-file ./playground/data/eval/pope/answers/memvr.jsonl \
    --temperature 0 \
    --cuda-device 'cuda:0' \
    --apply-memvr 'memvr' \
    --retracing-ratio 0.12 \
    --entropy-threshold 0.75 \
    --max-new-tokens 1 \
    --starting-layer 5 \
    --ending-layer 16 \

python llava/eval/eval_pope.py \
    --annotation-dir ./playground/data/eval/pope/coco \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file ./playground/data/eval/pope/answers/memvr.jsonl