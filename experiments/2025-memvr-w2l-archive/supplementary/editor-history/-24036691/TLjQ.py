import argparse
import glob
import os
import torch
from safetensors import safe_open

from transformers import (
    AddedToken,
    AutoConfig,
    AutoImageProcessor,
    AutoTokenizer,
    LlavaConfig,
    LlavaForConditionalGeneration,
    LlavaProcessor,
    SiglipVisionConfig,
)

KEYS_TO_MODIFY_MAPPING = {
    "model.vision_tower.": "",
    "model.mm_projector": "multi_modal_projector",
    "model": "model.model",
    "vision_model.model": "vision_model",
    "lm_head": "language_model.lm_head",
    "model.model": "language_model.model",
    "multi_modal_projector.0": "multi_modal_projector.linear_1",
    "multi_modal_projector.2": "multi_modal_projector.linear_2",
}

def load_original_state_dict_from_local(local_dir):
    original_state_dict = {}
    index_path = os.path.join(local_dir, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"No model index file found at {index_path}")

    with open(index_path, "r") as f:
        index = sorted(set(json_file for _, json_file in eval(f.read())["weight_map"].items()))

    for safetensor_name in index:
        path = os.path.join(local_dir, safetensor_name)
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                original_state_dict[key] = f.get_tensor(key)

    # patch missing keys
    if "lm_head.weight" not in original_state_dict:
        original_state_dict["lm_head.weight"] = original_state_dict["model.embed_tokens.weight"].clone()

    if "model.image_newline" in original_state_dict:
        del original_state_dict["model.image_newline"]

    return original_state_dict


def convert_state_dict_to_hf(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.endswith(".inv_freq"):
            continue
        for key_to_modify, new_key in KEYS_TO_MODIFY_MAPPING.items():
            if key_to_modify in key:
                key = key.replace(key_to_modify, new_key)
        new_state_dict[key] = value
    return new_state_dict


def convert_llava_llama_to_hf(text_model_id, vision_model_id, output_path, local_state_dir):
    torch.set_default_dtype(torch.float16)

    text_config = AutoConfig.from_pretrained(text_model_id)
    tokenizer = AutoTokenizer.from_pretrained(text_model_id)
    tokenizer.add_tokens(AddedToken("<image>", special=True, normalized=False), special_tokens=True)
    if "Qwen" not in text_model_id:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    image_processor = AutoImageProcessor.from_pretrained(vision_model_id)
    processor = LlavaProcessor(tokenizer=tokenizer, image_processor=image_processor)

    if "siglip" in vision_model_id:
        vision_config = SiglipVisionConfig(
            hidden_size=1152,
            image_size=384,
            intermediate_size=4304,
            num_attention_heads=16,
            num_hidden_layers=26,
            patch_size=14,
            vision_use_head=False,
        ).to_dict()
    else:
        vision_config = None
        # vision_config = CLIPVisionConfig.from_pretrained("/cpfs/user/UPSTREAM_USER/workspace/LLaVA/clip-vit-large-patch14-336").to_dict()

    config = LlavaConfig(
        text_config=text_config,
        vision_config=vision_config,
    )

    if "Qwen" in text_model_id:
        config.image_token_index = 151646
        if "siglip" in vision_model_id:
            config.vision_feature_select_strategy = "full"
            config.vision_feature_layer = -1
    else:
        config.pad_token_id = 32001
        config.image_token_index = 32000

    with torch.device("meta"):
        model = LlavaForConditionalGeneration(config)

    print(f"Loading local state dict from: {local_state_dir}")
    state_dict = load_original_state_dict_from_local(local_state_dir)
    state_dict = convert_state_dict_to_hf(state_dict)

    model.load_state_dict(state_dict, strict=True, assign=True)

    # resize for <image> and <pad> tokens
    pre_expansion_embeddings = model.language_model.model.embed_tokens.weight.data
    mu = torch.mean(pre_expansion_embeddings, dim=0).float()
    n = pre_expansion_embeddings.size()[0]
    sigma = ((pre_expansion_embeddings - mu).T @ (pre_expansion_embeddings - mu)) / n
    dist = torch.distributions.multivariate_normal.MultivariateNormal(mu, covariance_matrix=1e-5 * sigma)

    pad_shape = 64
    vocab_size = config.text_config.vocab_size
    model.resize_token_embeddings(config.text_config.vocab_size + 2, pad_to_multiple_of=pad_shape)
    model.language_model.model.embed_tokens.weight.data[vocab_size:] = torch.stack(
        [dist.sample() for _ in range(model.language_model.model.embed_tokens.weight.data[vocab_size:].shape[0])]
    )
    model.language_model.lm_head.weight.data[vocab_size:] = torch.stack(
        [dist.sample() for _ in range(model.language_model.lm_head.weight.data[vocab_size:].shape[0])]
    )

    model.save_pretrained(output_path)
    processor.save_pretrained(output_path)
    print(f"✅ Conversion completed! Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_model_id", required=True)
    parser.add_argument("--vision_model_id", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--local_state_dir", required=True)
    args = parser.parse_args()

    convert_llava_llama_to_hf(
        args.text_model_id,
        args.vision_model_id,
        args.output_path,
        args.local_state_dir,
    )


if __name__ == "__main__":
    main()

'''
conda activate mllms_know

python convert_llava_weights_to_hf_local.py \
  --text_model_id /cpfs/user/UPSTREAM_USER/workspace/LLaVA/vicuna-7b-v1.5 \
  --vision_model_id /cpfs/user/UPSTREAM_USER/workspace/LLaVA/clip-vit-large-patch14-336 \
  --output_path /cpfs/user/UPSTREAM_USER/workspace/mllms_know/converted/llava-v1.5-7b \
  --local_state_dir /cpfs/user/UPSTREAM_USER/workspace/LLaVA/llava-v1.5-7b
  

python convert_llava_weights_to_hf_local.py \
  --text_model_id /cpfs/user/UPSTREAM_USER/workspace/LLaVA/vicuna-7b-v1.5 \
  --vision_model_id /cpfs/user/UPSTREAM_USER/workspace/LLaVA/clip-vit-large-patch14-336 \
  --output_path /cpfs/user/UPSTREAM_USER/workspace/mllms_know/converted/llava-v1.5-7b-lr10k_td10k_norepeat_10th \
  --local_state_dir /cpfs/user/UPSTREAM_USER/workspace/LLaVA/checkpoints/pretrainMixdata_sft_vicuna/llava-v1.5-7b-lr10k_td10k_norepeat_10th

'''