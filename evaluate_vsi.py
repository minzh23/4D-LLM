import os
import json
import re
import torch
import cv2
import numpy as np
from tqdm import tqdm
# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from src.qwen2_5_vl_custom import Qwen2_5_VLForConditionalGeneration
from src.qwen2_5_vl_custom import Qwen2_5_VLProcessor
from src.training.data import image2tensor, get_video_info, smart_scale
# from rouge_score import rouge_scorer
from PIL import Image

from transformers import AutoProcessor, AutoTokenizer
# from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
import argparse


BSZ = 2

MODEL_PATH = "/cephfs/ylshare/zihan/4D-LLM/output/sft_stage2/checkpoint-10349"
file_name = "./VSI-Bench/output.json"
os.environ['PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT'] = '1.0'

compute_dtype = torch.bfloat16
model_init_kwargs = {
    "torch_dtype": compute_dtype,
    "attn_implementation": "flash_attention_2",
    "device_map": "auto",
}

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_PATH, **model_init_kwargs).to("cuda")
model.eval()
model.config.use_cache = True

processor = Qwen2_5_VLProcessor.from_pretrained(
    MODEL_PATH,
    padding_side="left"  # 评估时使用left padding
)
model.config.tokenizer_padding_side = processor.tokenizer.padding_side

generation_config = {
    "temperature": 0.1,
    "top_p": 0.001,
    "max_new_tokens": 1024,
    "do_sample": True,
    "pad_token_id": processor.tokenizer.pad_token_id,
    "eos_token_id": processor.tokenizer.eos_token_id,
    "use_cache": True,  # 评估时启用缓存
}

# llm = LLM(
#     model=MODEL_PATH,
#     tensor_parallel_size=torch.cuda.device_count(),
#     model_loader="qwen2_5_vl_custom.wrapper:get_model_architecture",
#     max_model_len = 8192 * 2,
#     gpu_memory_utilization=0.8,
#     limit_mm_per_prompt={"image": 1, "video": 1},
#     device="cuda", 
# )


# sampling_params = SamplingParams(
#     temperature=0.1,
#     top_p=0.001,
#     max_tokens=1024,
#     stop_token_ids=[],
# )


tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.padding_side = "left"
processor.tokenizer = tokenizer

for dataset_name in ['vsibench']:

    OUTPUT_PATH = f"./src/eval_results/eval_{dataset_name}_{file_name}_greedy_output.json"
    PROMPT_PATH = f"./VSI-Bench/eval_{dataset_name}.json"
    
    if PROMPT_PATH.endswith('.jsonl'):
        with open(PROMPT_PATH, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
    elif PROMPT_PATH.endswith('.json'):
        with open(PROMPT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        raise ValueError("Input file must be .json or .jsonl")

    QUESTION_TEMPLATE = (
        "{Question}\n"
        "Please think about this question as if you were a human pondering deeply. "
        "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions "
        "It's encouraged to include self-reflection or verification in the reasoning process. "
        "Provide your detailed reasoning between the <think> and </think> tags, and then give your final answer between the <answer> and </answer> tags."
    )

    TYPE_TEMPLATE = {
        "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
        "numerical": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
        "OCR": " Please transcribe text from the image/video clearly and provide your text answer within the <answer> </answer> tags.",
        "free-form": " Please provide your text answer within the <answer> </answer> tags.",
        "regression": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags."
    }


    messages = []
    for x in data:
        if x["problem_type"] == 'multiple choice':
            question = x['problem'] + "Options:\n"
            for op in x["options"]:
                question += op + "\n"
        else:
            question = x['problem']

        msg = [{
            "role": "user",
            "content": [
                {
                    "type": x['data_type'],
                    x['data_type']: os.getcwd() + "/VSI-Bench" + x['path'][1:]
                },
                {
                    "type": "text",
                    "text": QUESTION_TEMPLATE.format(Question=question) + TYPE_TEMPLATE[x['problem_type']]
                }
            ]
        }]
        messages.append(msg)
        

    final_output = []
    start_idx = 0
    if os.path.exists(OUTPUT_PATH):
        try:
            with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
                existing = json.load(f)
                final_output = existing.get("results", [])
                start_idx = len(final_output)
                print(f"Resuming from sample index {start_idx}")
        except Exception as e:
            print(f"Error reading existing output file: {e}")


    def extract_think(output_str):
        pattern = r'<think>\s*(.*?)\s*</think>'
        match = re.search(pattern, output_str, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    def extract_answer(text):
        pattern = r'<answer>\s*(.*?)\s*</answer>'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    def normalize_number(num_str):
        try:
            num_str = num_str.replace(',', '')
            return float(num_str)
        except Exception as e:
            return None
        
    def mean_relative_accuracy(pred, target, start=0.5, end=0.95, interval=0.05):

        if not torch.is_tensor(pred):
            pred = torch.tensor(pred, dtype=torch.float32)
        if not torch.is_tensor(target):
            target = torch.tensor(target, dtype=torch.float32)
        
        epsilon = 1e-8
        rel_error = torch.abs(pred - target) / (torch.abs(target) + epsilon)
        
        thresholds = torch.arange(start, end + interval/2, interval, dtype=torch.float32)
        
        conditions = rel_error < (1 - thresholds)  
        mra = conditions.float().mean()  
        return mra.item()


    def reward_fn(sample, model_output, question_type):
        try:
            output_ans = extract_answer(model_output)
            if output_ans == '':
                output_ans = model_output
            gt_ans = extract_answer(sample.get("solution", ""))
            if question_type == "multiple choice":
                return 1.0 if output_ans.strip() == gt_ans.strip() else 0.0
            elif question_type == "numerical":
                gt_has_decimal = ("." in gt_ans) or ("," in gt_ans)
                out_has_decimal = ("." in output_ans) or ("," in output_ans)
                if gt_has_decimal != out_has_decimal:
                    return 0.0
                gt_number = normalize_number(gt_ans)
                out_number = normalize_number(output_ans)
                if gt_number is None or out_number is None:
                    return 0.0
                return 1.0 if round(gt_number, 2) == round(out_number, 2) else 0.0
            elif question_type == "regression":
                gt_number = normalize_number(gt_ans)
                out_number = normalize_number(output_ans)
                if gt_number is None or out_number is None:
                    return 0.0
                mra = mean_relative_accuracy(out_number, gt_number)
                return mra
            else:
                return 0.0
        except Exception as e:
            return 0.0

    mean_acc = []
    mean_mra = []
    for i in tqdm(range(start_idx, len(messages), BSZ), desc="Processing batches"):
        batch_messages = messages[i:i+BSZ]

        # 文本 + 图像处理
        prompts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
        image_inputs, video_inputs, video_kwargs = process_vision_info(batch_messages, return_video_kwargs=True)
        inputs = processor(
            text=prompts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        # Process videos if any
        if any(msg[0]['content'][0]['type'] == 'video' for msg in batch_messages):
            all_video_tensors = []
            video_mask = []
            
            for b_idx, msg in enumerate(batch_messages):
                if msg[0]['content'][0]['type'] == 'video':
                    video_file = msg[0]['content'][0]['video']
                    video_min_pixel, video_max_pixel, fps = (128 * 28 * 28), (128 * 28 * 28), 1.0
                    
                    # Process individual video
                    video_input, vid_kwargs = get_video_info(video_file, video_min_pixel, video_max_pixel, None, None, fps)
                    all_frame_tensors = []
                    
                    for raw_image in video_input:
                        np_img = (raw_image.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
                        np_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
                        image_tensor, _ = image2tensor(np_img, None, video_input.shape[2] * 14, video_input.shape[3] * 14)
                        image_tensor = image_tensor.to(dtype=torch.bfloat16)
                        image_tensor_flat = image_tensor.permute(1, 0, 2, 3).contiguous().view(image_tensor.shape[1], -1)
                        all_frame_tensors.append(image_tensor_flat)
                    
                    video_tensor = torch.cat(all_frame_tensors, dim=1)
                    all_video_tensors.append(video_tensor)
                    video_mask.append(b_idx)
            
            # Add video tensors to inputs if any videos were processed
            if all_video_tensors:
                D = all_video_tensors[0].shape[1]
                # Stack video tensors along batch dimension
                inputs["depth_values"] = torch.stack(all_video_tensors, dim=0).view(-1, D).to(model.device)
        
        inputs = inputs.to(model.device)
        
        with torch.no_grad():
            generated_ids = model.generate(**inputs, **generation_config)
            
        # except Exception as e:
        #     print('error:', data[i]['path'])
        #     print('Exception:', e)
        #     batch_output_text = ['<answer>error</answer>'] * BSZ
        # inputs = processor(
        #     text=[processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages],
        #     images=[msg[0]['content'][0].get('image') for msg in batch_messages if msg[0]['content'][0]['type'] == 'image'],
        #     videos=[msg[0]['content'][0].get('video') for msg in batch_messages if msg[0]['content'][0]['type'] == 'video'],
        #     return_tensors="pt"
        # )
        batch_output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)

        for j, (sample, model_output) in enumerate(zip(data[i:i+BSZ], batch_output_text), start=i):
            think_chain = extract_think(model_output)
            final_ans = extract_answer(model_output)
            if final_ans == "":
                final_ans = model_output
            sample["output"] = model_output
            sample["prediction"] = final_ans
            q_type = sample.get("problem_type", "")
            sample["reward"] = reward_fn(sample, model_output, q_type)
            sample['correct'] = True if sample["reward"]==1.0 else False
            if sample['problem_type'] != 'regression':
                mean_acc.append(sample["reward"])
            else:
                mean_mra.append(sample["reward"])
            if think_chain:
                sample["process"] = f"<think>{think_chain}</think>"
            final_output.append(sample)
        

        try:
            with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
                json.dump({"results": final_output}, f, indent=2, ensure_ascii=False)
            print(f"Processed batch {(i - start_idx)//BSZ + 1}, saved {len(final_output)} samples.")
        except Exception as e:
            print(f"Error writing to output file: {e}")

    final_acc={'mean_acc': 0.0, 'mean_mra': 0.0}
    final_acc['mean_acc'] = torch.tensor(mean_acc).mean().item()
    if mean_mra != []:
        final_acc['mean_mra'] = torch.tensor(mean_mra).mean().item()
    
    try:
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump({"results": final_output, "final_acc": [final_acc]}, f, indent=2, ensure_ascii=False)
        print(f"Final accuracy saved to {OUTPUT_PATH}")
    except Exception as e:
        print(f"Error writing final accuracy to output file: {e}")
    
    print(f"Results saved to {OUTPUT_PATH}")