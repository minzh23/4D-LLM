import os
import json
import re
import torch
import cv2
import numpy as np
from tqdm import tqdm
import sys
import logging
import gc
from src.qwen2_5_vl_custom import Qwen2_5_VLForConditionalGeneration
from src.qwen2_5_vl_custom import Qwen2_5_VLProcessor
from src.training.data import image2tensor, get_video_info, smart_scale
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer
from qwen_vl_utils import process_vision_info
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Minimal memory VSI evaluation')
parser.add_argument('--model_path', type=str, default="/cephfs/ylshare/zihan/4D-LLM/output/sft_stage2/checkpoint-10349")
parser.add_argument('--output_dir', type=str, default="./src/eval_results/")
parser.add_argument('--dataset_name', type=str, default="vsibench")
args = parser.parse_args()

# 极限内存优化设置
BSZ = 1  # 固定为1
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:256'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# 创建输出目录
os.makedirs(args.output_dir, exist_ok=True)

print("Loading model with maximum memory optimization...")

# 模型加载配置 - 极限优化
model_init_kwargs = {
    "torch_dtype": torch.float16,  # 使用float16而不是bfloat16
    "low_cpu_mem_usage": True,
    "device_map": "auto",  # 自动设备映射
    "load_in_8bit": True,  # 8位量化
    "max_memory": {0: "70GB"},  # 限制GPU 0最大使用70GB
}

try:
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, 
        **model_init_kwargs
    )
    print("Model loaded successfully with 8-bit quantization")
except Exception as e:
    print(f"8-bit loading failed: {e}")
    print("Trying with 4-bit quantization...")
    # 尝试4位量化
    model_init_kwargs.update({
        "load_in_4bit": True,
        "load_in_8bit": False,
    })
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, 
        **model_init_kwargs
    )

model.eval()
model.config.use_cache = False

# 处理器加载
processor = Qwen2_5_VLProcessor.from_pretrained(
    args.model_path,
    padding_side="left"
)

# 极限优化的生成配置
generation_config = {
    "temperature": 0.1,
    "top_p": 0.001,
    "max_new_tokens": 256,  # 进一步减少到256
    "do_sample": False,  # 关闭采样以节省内存
    "pad_token_id": processor.tokenizer.pad_token_id,
    "eos_token_id": processor.tokenizer.eos_token_id,
    "use_cache": False,
    "num_beams": 1,  # 使用贪婪搜索
}

# 数据加载
PROMPT_PATH = f"./VSI-Bench/eval_{args.dataset_name}.json"
OUTPUT_PATH = os.path.join(args.output_dir, f"eval_{args.dataset_name}_minimal_output.json")

with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Processing {len(data)} samples with minimal memory usage...")

# 简化的问题模板
QUESTION_TEMPLATE = "{Question}\nPlease provide your answer between <answer> and </answer> tags."

TYPE_TEMPLATE = {
    "multiple choice": " Provide only the option letter (e.g., A, B, C, D).",
    "numerical": " Provide the numerical value.",
    "OCR": " Transcribe the text.",
    "free-form": " Provide your answer.",
    "regression": " Provide the numerical value."
}

def process_single_sample(sample_idx, sample):
    """逐个处理样本以最小化内存使用"""
    try:
        # 准备消息
        if sample["problem_type"] == 'multiple choice':
            question = sample['problem'] + "Options:\n"
            for op in sample["options"]:
                question += op + "\n"
        else:
            question = sample['problem']

        msg = [{
            "role": "user", 
            "content": [
                {
                    "type": sample['data_type'],
                    sample['data_type']: os.getcwd() + "/VSI-Bench" + sample['path'][1:]
                },
                {
                    "type": "text",
                    "text": QUESTION_TEMPLATE.format(Question=question) + TYPE_TEMPLATE[sample['problem_type']]
                }
            ]
        }]

        # 处理输入
        prompt = processor.apply_chat_template([msg], tokenize=False, add_generation_prompt=True)[0]
        image_inputs, video_inputs, _ = process_vision_info([msg])
        
        inputs = processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        # 移动到GPU
        inputs = {k: v.to('cuda') if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        # 生成
        with torch.no_grad():
            torch.cuda.empty_cache()  # 清理缓存
            generated_ids = model.generate(**inputs, **generation_config)
            
        # 解码
        output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # 提取答案
        template_end = output_text.find("assistant\n")
        if template_end != -1:
            output_text = output_text[template_end + len("assistant\n"):]
        
        # 清理内存
        del inputs, generated_ids
        torch.cuda.empty_cache()
        gc.collect()
        
        return output_text.strip()
        
    except Exception as e:
        print(f"Error processing sample {sample_idx}: {e}")
        torch.cuda.empty_cache()
        gc.collect()
        return ""

# 逐个处理样本
final_output = []
for i, sample in enumerate(tqdm(data, desc="Processing samples")):
    output = process_single_sample(i, sample)
    
    # 提取答案
    answer_pattern = r'<answer>\s*(.*?)\s*</answer>'
    match = re.search(answer_pattern, output, re.DOTALL)
    final_answer = match.group(1).strip() if match else output
    
    sample["output"] = output
    sample["prediction"] = final_answer
    final_output.append(sample)
    
    # 每10个样本保存一次
    if (i + 1) % 10 == 0:
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump({"results": final_output}, f, indent=2, ensure_ascii=False)
        print(f"Saved progress: {i+1}/{len(data)} samples")

# 最终保存
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump({"results": final_output}, f, indent=2, ensure_ascii=False)

print(f"Evaluation completed! Results saved to {OUTPUT_PATH}")
print(f"Processed {len(final_output)} samples total") 