import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math

import transformers
# MemVR
from memvr import apply_memvr_llama, LlamaMLP, activate_memvr, set_visual_mask

from PIL import Image, ImageDraw
from run import vicrop_qa, vicrop_qa_hf

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor, image.size

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids, image_tensors, image_sizes = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader


def eval_model(args):
    import time
    from PIL import Image, ImageDraw

    device = args.cuda_device

    # MemVR 这句话一定要有
    transformers.models.llama.modeling_llama.LlamaMLP = LlamaMLP

    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    # 加载模型（使用attn_implementation='eager'）
    tokenizer, model, image_processor, max_length = load_pretrained_model(
        model_path=model_path,
        model_base=args.model_base,
        model_name=model_name,
        attn_implementation="eager",
        device_map=device,
    )

    # 读取问题
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    # 初始化 MemVR
    if args.apply_memvr == 'memvr':
        apply_memvr_llama(
            self=model,
            starting_layer=args.starting_layer,
            ending_layer=args.ending_layer,
            entropy_threshold=args.entropy_threshold,
            retracing_ratio=args.retracing_ratio
        )

    for line in tqdm(questions, total=len(questions)):
        image_path = os.path.join(args.image_folder, line["image"])
        question = line["text"]
        question_id = line["question_id"]

        # ViCrop QA，获得mask与bbox
        activate_memvr(model, False) 
        ori_answer, crop_answer, bbox, mask = vicrop_qa(
            model_name='llava',
            method_name='grad_att',
            image_path=image_path,
            question=question,
            model=model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            short_question=args.short_question if hasattr(args, "short_question") else question
        )
        # print(mask)
        image = Image.open(image_path).convert("RGB")
        # 图像处理
        image_tensor = process_images([image], image_processor, model.config)
        image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]
        image_sizes = [image.size]
        # print(image_tensor)

        
        qs = DEFAULT_IMAGE_TOKEN + '\n' + question

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        # print(prompt)

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
        # print(input_ids)
        # 设置视觉token mask
        activate_memvr(model, True)
        visual_token_mask = torch.from_numpy(mask).to(device)
        set_visual_mask(model, visual_token_mask)

        # 推理
        with torch.inference_mode():
            start_time = time.time()
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True                              
            )
            end_time = time.time()

        # 解码输出
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        print("\n Outputs: ", outputs)

        # print(f"\nQuestion: {question}")
        # print(f"Original Answer: {ori_answer}")
        # print(f"ViCrop Answer:   {crop_answer}")
        # print(f"Model Output:    {text_outputs}")
        # print(f"Time taken: {end_time - start_time:.2f} seconds")

        # 保存答案
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": question_id,
                                   "prompt": question,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")

    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="./playground/data/eval/pope/val2014")
    parser.add_argument("--question-file", type=str, default="./playground/data/eval/pope/llava_pope_test.jsonl")
    parser.add_argument("--answers-file", type=str, default="./playground/data/eval/pope/answers/llava-v1.5-13b.jsonl")
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=1)

    # MemVR
    parser.add_argument("--cuda-device", type=str, default="cuda:0")
    parser.add_argument("--vision-retracing", type=str, default="default")
    parser.add_argument("--retracing-ratio", type=float, default=0.0)
    parser.add_argument("--entropy-threshold", type=float, default=0.75)
    parser.add_argument("--starting-layer", type=int, default=5)
    parser.add_argument("--ending-layer", type=int, default=16)
    parser.add_argument("--apply-memvr", type=str, default='default')
    args = parser.parse_args()

    eval_model(args)
