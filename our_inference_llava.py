from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import transformers
from llava.model.llava_arch import LlavaMetaForCausalLM

from PIL import Image
import requests
import copy
import torch
import time

from memvr import apply_memvr_llama, LlamaMLP, activate_memvr, set_visual_mask


# w2l(where to look)
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image, ImageDraw
from run import vicrop_qa, vicrop_qa_hf
#w2l


model_path = "../llava-v1.5-7b"
device = "cuda:0"

#w2l
# llava_methods.py utils.py run.py三个抄过来的文件
hf_model_name = "llava-hf/llava-1.5-7b-hf"
revision = "a272c74"  # Use the specific revision for old transformers versions
model_name = 'llava'
method_name = 'grad_att'   #pure_grad有问题，我也没仔细看论文原理，就先这样了
image_path = 'images/demo1.png'
question = 'what is the date of the photo?'
short_question = 'what is the date of the photo?'

# MemVR 这句话一定要有
transformers.models.llama.modeling_llama.LlamaMLP = LlamaMLP

tokenizer, model, image_processor, max_length = load_pretrained_model(
    model_path = model_path,
    model_base = None, 
    model_name = get_model_name_from_path(model_path), 
    attn_implementation="eager", # 计算梯度时需要attention矩阵。
    device_map = device,
    ) # Add any other thing you want to pass in llava_model_args

ori_answer, crop_answer, bbox, mask = vicrop_qa(model_name, method_name, image_path, question, model, tokenizer, image_processor, short_question)

# model = LlavaForConditionalGeneration.from_pretrained(hf_model_name,
#                                                      revision=revision, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, attn_implementation="eager").to('cuda')
# processor = AutoProcessor.from_pretrained(hf_model_name, revision=revision)
# ori_answer, crop_answer, bbox = vicrop_qa_hf(model_name, method_name, image_path, question, model, processor, short_question)

print(f'Model\'s original answer:  {ori_answer}')
print(f'Answer with Vicrop:       {crop_answer}')

image = Image.open(image_path).convert("RGB")
image_draw = ImageDraw.Draw(image)
image_draw.rectangle(bbox, outline='red', width=4)
image.save('images/demo1_bbox.png')


model.eval()
model.tie_weights()

apply_memvr_llama(
    self=model,
    starting_layer = 5,    
    ending_layer = 16,
    entropy_threshold = 0.75,
    retracing_ratio = 0.1
)
activate_memvr(model, True)

image_name = 'images/demo1.png'
image = Image.open(image_name).convert("RGB")
image_tensor = process_images([image], image_processor, model.config)
image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

# 原本memvr的prompt
# conv_template = "vicuna_v1" # Make sure you use correct chat template for different models
# prompt = DEFAULT_IMAGE_TOKEN + f"\n{question}"
# conv_mode = "vicuna_v1"
# conv = conv_templates[conv_mode].copy()
# inp =  prompt
# conv.append_message(conv.roles[0], inp)
# conv.append_message(conv.roles[1], None)
# prompt = conv.get_prompt()
# print(prompt)

# 采用w2l的prompt
prompt = f"<image>\nUSER: {question} Answer the question using a single word or phrase.\nASSISTANT:"

    
input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
image_sizes = [image.size]


visual_token_mask = torch.from_numpy(mask).to(device)

# 加mask
set_visual_mask(model, visual_token_mask)
#不加mask
with torch.inference_mode():
    start_time = time.time()
    cont = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=image_sizes,
        do_sample=False,
        temperature=1,
        max_new_tokens=64,
    )
    end_time = time.time()
text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
print(text_outputs)
print(f"Time taken: {end_time - start_time} seconds")


with torch.inference_mode():
    start_time = time.time()
    cont = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=image_sizes,
        do_sample=False,
        temperature=1,
        max_new_tokens=64,
    )
    end_time = time.time()
text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
print(text_outputs)
print(f"Time taken: {end_time - start_time} seconds")