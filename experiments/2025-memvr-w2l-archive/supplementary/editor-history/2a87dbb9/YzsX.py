import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

# 参数设置
# source_dir = "/cpfs/user/UPSTREAM_USER/workspace/LLaVA/checkpoints/pretrainMixdata_sft_vicuna"

# source_dir = "/cpfs/user/UPSTREAM_USER/workspace/LLaVA/checkpoints/pretrain_SFT"
source_dir = "/root/autodl-tmp/llava-v1.5-7b"
output_dir = "/root/autodl-tmp/llava-v1.5-7b-hf"
script_path = "/root/transformers-4.40.0/src/transformers/models/llava/convert_llava_weights_to_hf.py"
text_model = "/cpfs/user/UPSTREAM_USER/workspace/LLaVA/vicuna-7b-v1.5"
vision_model = "/cpfs/user/UPSTREAM_USER/workspace/LLaVA/clip-vit-large-patch14-336"
max_workers = 16 # 可根据CPU核心/内存适当增减

# 获取待处理模型目录
model_dirs = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

def convert_model(model_dir):
    model_path = os.path.join(source_dir, model_dir)
    output_path = os.path.join(output_dir, model_dir)

    if os.path.exists(os.path.join(output_path, "config.json")):
        return f"[SKIP] {model_dir} 已转换"

    os.makedirs(output_path, exist_ok=True)
    
    command = [
        "python", script_path,
        "--local_state_dir", model_path,
        "--output_path", output_path,
        "--text_model_id", text_model,
        "--vision_model_id", vision_model,
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if result.returncode == 0:
        return f"[DONE] {model_dir} 转换成功"
    else:
        return f"[ERROR] {model_dir} 转换失败\n{result.stderr.decode()}"

# 使用线程池并行转换
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = [executor.submit(convert_model, mdir) for mdir in model_dirs]
    for future in as_completed(futures):
        print(future.result())

print("✅ 所有任务完成")