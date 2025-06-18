import os
import json
import re
import torch
from tqdm import tqdm
from src.qwen2_5_vl_custom import Qwen2_5_VLForConditionalGeneration
from src.qwen2_5_vl_custom import Qwen2_5_VLProcessor
from qwen_vl_utils import process_vision_info
import argparse

# 强制使用CPU
os.environ['CUDA_VISIBLE_DEVICES'] = ''

parser = argparse.ArgumentParser(description='CPU-only VSI evaluation')
parser.add_argument('--model_path', type=str, default="/cephfs/ylshare/zihan/4D-LLM/output/sft_stage2/checkpoint-10349")
parser.add_argument('--output_dir', type=str, default="./src/eval_results/")
parser.add_argument('--dataset_name', type=str, default="vsibench")
args = parser.parse_args()

print("Loading model on CPU (slow but no memory limit)...")

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    args.model_path,
    torch_dtype=torch.float32,
    device_map="cpu",
    low_cpu_mem_usage=True
)

processor = Qwen2_5_VLProcessor.from_pretrained(args.model_path)

generation_config = {
    "max_new_tokens": 128,
    "do_sample": False,
    "use_cache": False,
    "num_beams": 1,
}

# 数据处理
PROMPT_PATH = f"./VSI-Bench/eval_{args.dataset_name}.json"
OUTPUT_PATH = os.path.join(args.output_dir, f"eval_{args.dataset_name}_cpu_output.json")

with open(PROMPT_PATH, "r") as f:
    data = json.load(f)

print(f"Processing {len(data)} samples on CPU...")

final_output = []
for i, sample in enumerate(tqdm(data, desc="CPU processing")):
    try:
        # 准备输入
        if sample["problem_type"] == 'multiple choice':
            question = sample['problem'] + "Options:\n" + "\n".join(sample["options"])
        else:
            question = sample['problem']
            
        msg = [{
            "role": "user",
            "content": [
                {"type": sample['data_type'], sample['data_type']: os.getcwd() + "/VSI-Bench" + sample['path'][1:]},
                {"type": "text", "text": f"{question}\nAnswer:"}
            ]
        }]
        
        # 处理输入
        prompt = processor.apply_chat_template([msg], tokenize=False, add_generation_prompt=True)[0]
        image_inputs, video_inputs, _ = process_vision_info([msg])
        
        inputs = processor(text=[prompt], images=image_inputs, videos=video_inputs, return_tensors="pt")
        
        # 生成 (CPU)
        with torch.no_grad():
            generated_ids = model.generate(**inputs, **generation_config)
        
        output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # 提取答案
        if "assistant\n" in output:
            output = output.split("assistant\n")[-1]
            
        sample["output"] = output.strip()
        sample["prediction"] = output.strip()
        final_output.append(sample)
        
        # 每5个样本保存一次
        if (i + 1) % 5 == 0:
            with open(OUTPUT_PATH, "w") as f:
                json.dump({"results": final_output}, f, indent=2)
            
    except Exception as e:
        print(f"Error processing sample {i}: {e}")
        continue

print(f"CPU evaluation completed! Results: {OUTPUT_PATH}") 