#!/bin/bash

python -m llava.eval.llava_model_vqa_loader_our \
    --model-path ../llava-v1.5-7b \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder ./playground/data/eval/pope/val2014 \
    --answers-file ./playground/data/eval/pope/answers/our_q25.jsonl \
    --temperature 0 \
    --cuda-device 'cuda:0' \
    --apply-memvr 'memvr' \
    --retracing-ratio 0.12 \
    --entropy-threshold 0.75 \
    --max-new-tokens 1 \
    --starting-layer 5 \
    --ending-layer 16 \
    --look-rate 25 \

python llava/eval/eval_pope.py \
    --annotation-dir ./playground/data/eval/pope/coco \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file ./playground/data/eval/pope/answers/our_q25.jsonl

python -m llava.eval.llava_model_vqa_loader_our \
    --model-path ../llava-v1.5-7b \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder ./playground/data/eval/pope/val2014 \
    --answers-file ./playground/data/eval/pope/answers/our_q50.jsonl \
    --temperature 0 \
    --cuda-device 'cuda:0' \
    --apply-memvr 'memvr' \
    --retracing-ratio 0.12 \
    --entropy-threshold 0.75 \
    --max-new-tokens 1 \
    --starting-layer 5 \
    --ending-layer 16 \
    --look-rate 50 \

python llava/eval/eval_pope.py \
    --annotation-dir ./playground/data/eval/pope/coco \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file ./playground/data/eval/pope/answers/our_q50.jsonl

    python -m llava.eval.llava_model_vqa_loader_our \
    --model-path ../llava-v1.5-7b \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder ./playground/data/eval/pope/val2014 \
    --answers-file ./playground/data/eval/pope/answers/our_q75.jsonl \
    --temperature 0 \
    --cuda-device 'cuda:0' \
    --apply-memvr 'memvr' \
    --retracing-ratio 0.12 \
    --entropy-threshold 0.75 \
    --max-new-tokens 1 \
    --starting-layer 5 \
    --ending-layer 16 \
    --look-rate 75 \

python llava/eval/eval_pope.py \
    --annotation-dir ./playground/data/eval/pope/coco \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file ./playground/data/eval/pope/answers/our_q75.jsonl

python -m llava.eval.llava_model_vqa_loader_our \
    --model-path ../llava-v1.5-7b \
    --question-file /home/ma/Documents/hallucination/LLaVA/playground/data/eval/cocoqa/cocoqa_valid_llava_1w.jsonl \
    --image-folder /home/ma/Documents/hallucination/LLaVA/playground/data/eval/cocoqa/train2017 \
    --answers-file /home/ma/Documents/hallucination/LLaVA/playground/data/eval/cocoqa/answers/our_q25.jsonl \
    --temperature 0 \
    --cuda-device 'cuda:0' \
    --apply-memvr 'memvr' \
    --retracing-ratio 0.12 \
    --entropy-threshold 0.75 \
    --max-new-tokens 1024 \
    --starting-layer 5 \
    --ending-layer 16 \
    --look-rate 25 \

python -m llava.eval.llava_model_vqa_loader_our \
    --model-path ../llava-v1.5-7b \
    --question-file /home/ma/Documents/hallucination/LLaVA/playground/data/eval/cocoqa/cocoqa_valid_llava_1w.jsonl \
    --image-folder /home/ma/Documents/hallucination/LLaVA/playground/data/eval/cocoqa/train2017 \
    --answers-file /home/ma/Documents/hallucination/LLaVA/playground/data/eval/cocoqa/answers/our_q50.jsonl \
    --temperature 0 \
    --cuda-device 'cuda:0' \
    --apply-memvr 'memvr' \
    --retracing-ratio 0.12 \
    --entropy-threshold 0.75 \
    --max-new-tokens 1024 \
    --starting-layer 5 \
    --ending-layer 16 \
    --look-rate 50 \

python -m llava.eval.llava_model_vqa_loader_our \
    --model-path ../llava-v1.5-7b \
    --question-file /home/ma/Documents/hallucination/LLaVA/playground/data/eval/cocoqa/cocoqa_valid_llava_1w.jsonl \
    --image-folder /home/ma/Documents/hallucination/LLaVA/playground/data/eval/cocoqa/train2017 \
    --answers-file /home/ma/Documents/hallucination/LLaVA/playground/data/eval/cocoqa/answers/our_q75.jsonl \
    --temperature 0 \
    --cuda-device 'cuda:0' \
    --apply-memvr 'memvr' \
    --retracing-ratio 0.12 \
    --entropy-threshold 0.75 \
    --max-new-tokens 1024 \
    --starting-layer 5 \
    --ending-layer 16 \
    --look-rate 75 \
