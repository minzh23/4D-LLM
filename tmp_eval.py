import os
import json
import re
import torch
import cv2
import numpy as np
from tqdm import tqdm
from accelerate import Accelerator
import sys
import logging
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam
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


# Parse command line arguments
parser = argparse.ArgumentParser(description='Evaluate VSI Benchmark with Qwen2.5-VL model')
parser.add_argument('--batch_size', type=int, default=2, help='Batch size for evaluation')
parser.add_argument('--model_path', type=str, default="/cephfs/ylshare/zihan/4D-LLM/output/sft_stage2/checkpoint-10349", 
                   help='Path to the model checkpoint')
parser.add_argument('--output_dir', type=str, default="./src/eval_results/", 
                   help='Directory to save evaluation results')
parser.add_argument('--dataset_name', type=str, default="vsibench", 
                   help='Dataset name for evaluation')
# DeepSpeed related arguments
parser.add_argument('--deepspeed_config', type=str, default=None,
                   help='Path to DeepSpeed config file')
parser.add_argument('--use_deepspeed', action='store_true',
                   help='Whether to use DeepSpeed for inference')
parser.add_argument('--cpu_offload', action='store_true',
                   help='Enable CPU offloading for DeepSpeed')
args = parser.parse_args()

BSZ = args.batch_size

MODEL_PATH = args.model_path
file_name = "./VSI-Bench/output.json"
os.environ['PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT'] = '1.0'

# Create output directory if it doesn't exist
os.makedirs(args.output_dir, exist_ok=True)

# DeepSpeed configuration
deepspeed_config = None
if args.use_deepspeed:
    if args.deepspeed_config and os.path.exists(args.deepspeed_config):
        with open(args.deepspeed_config, 'r') as f:
            deepspeed_config = json.load(f)
    else:
        # Default DeepSpeed configuration for inference
        deepspeed_config = {
            "bf16": {
                "enabled": True
            },
            "zero_optimization": {
                "stage": 3,
                "offload_param": {
                    "device": "cpu" if args.cpu_offload else "none",
                    "pin_memory": True
                },
                "offload_optimizer": {
                    "device": "cpu" if args.cpu_offload else "none",
                    "pin_memory": True
                },
                "overlap_comm": True,
                "contiguous_gradients": True,
                "sub_group_size": 1e9,
                "reduce_bucket_size": "auto",
                "stage3_prefetch_bucket_size": "auto",
                "stage3_param_persistence_threshold": "auto",
                "stage3_max_live_parameters": 1e9,
                "stage3_max_reuse_distance": 1e9,
                "gather_16bit_weights_on_model_save": True
            },
            "gradient_accumulation_steps": 1,
            "gradient_clipping": 1.0,
            "steps_per_print": 2000,
            "train_batch_size": BSZ,
            "train_micro_batch_size_per_gpu": 1,
            "wall_clock_breakdown": False
        }

# Initialize accelerator with DeepSpeed
if args.use_deepspeed:
    accelerator = Accelerator(
        deepspeed_plugin=deepspeed_config,
        mixed_precision='bf16'
    )
else:
    accelerator = Accelerator()

is_main_process = accelerator.is_main_process
if not is_main_process:
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')
    logging.getLogger().setLevel(logging.ERROR)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
compute_dtype = torch.bfloat16
model_init_kwargs = {
    "torch_dtype": compute_dtype,
    "attn_implementation": "flash_attention_2",
}

# Load model without device_map to let accelerate handle device placement
if args.use_deepspeed:
    # For DeepSpeed, we need to load the model without moving it to CUDA first
    model_init_kwargs["device_map"] = None
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_PATH, **model_init_kwargs)
else:
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_PATH, **model_init_kwargs)

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
    "use_cache": True, 
}

# Prepare model with accelerator (this will handle DeepSpeed if enabled)
if args.use_deepspeed:
    # For DeepSpeed, we might need to prepare the model differently
    model = accelerator.prepare(model)
    print(f"Model prepared with DeepSpeed. Memory usage optimization enabled.")
else:
    model = accelerator.prepare(model)



tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.padding_side = "left"
processor.tokenizer = tokenizer

for dataset_name in [args.dataset_name]:

    OUTPUT_PATH = os.path.join(args.output_dir, f"eval_{dataset_name}_{os.path.basename(file_name)}_greedy_output.json")
    PROMPT_PATH = f"./VSI-Bench/eval_{dataset_name}.json"
    
    if PROMPT_PATH.endswith('.jsonl'):
        with open(PROMPT_PATH, "r", encoding="utf-8") as f:
            for line in f:
                data = []
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
    
    with accelerator.split_between_processes(messages) as messages_split:
        for i in tqdm(range(0, len(messages_split), BSZ), desc="Processing batches", disable=not is_main_process):
            batch_messages = messages_split[i:i+BSZ]

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
                            
                            # 立即清理中间变量
                            del image_tensor, image_tensor_flat
                        
                        video_tensor = torch.cat(all_frame_tensors, dim=1)
                        all_video_tensors.append(video_tensor)
                        video_mask.append(b_idx)
                        
                        # 清理视频处理的中间变量
                        del all_frame_tensors, video_input
                        if args.use_deepspeed:
                            torch.cuda.empty_cache()
                
                # Add video tensors to inputs if any videos were processed
                if all_video_tensors:
                    D = all_video_tensors[0].shape[1]
                    # Stack video tensors along batch dimension
                    inputs["depth_values"] = torch.stack(all_video_tensors, dim=0).view(-1, D)
            
            # Move inputs to the appropriate device using accelerator
            inputs = {k: v.to(accelerator.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            # For DeepSpeed, ensure no gradients are computed
            if args.use_deepspeed:
                with torch.no_grad():
                    # Disable gradient computation completely for inference
                    torch.set_grad_enabled(False)
                    with accelerator.autocast():
                        generated_ids = accelerator.unwrap_model(model).generate(**inputs, **generation_config)
                    torch.set_grad_enabled(True)  # Re-enable for safety
            else:
                with torch.no_grad():
                    # Use accelerator.unwrap_model for generation
                    with accelerator.autocast():
                        generated_ids = accelerator.unwrap_model(model).generate(**inputs, **generation_config)
            
            batch_output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)

            for k in range(len(batch_output_text)):
                template_end = batch_output_text[k].find("assistant\n<think>")
                if template_end != -1:
                    batch_output_text[k] = batch_output_text[k][template_end + len("assistant\n"):]
                
                if not batch_output_text[k].strip():
                    continue
                
                batch_output_text[k] = batch_output_text[k].strip()

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
            
            # Only write output from main process
            if is_main_process:
                try:
                    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
                        json.dump({"results": final_output}, f, indent=2, ensure_ascii=False)
                    print(f"Processed batch {(i - start_idx)//BSZ + 1}, saved {len(final_output)} samples.")
                except Exception as e:
                    print(f"Error writing to output file: {e}")

            # 在每个批次处理完成后添加显存清理
            # del prompts, image_inputs, video_inputs, inputs
            # del batch_messages, batch_output_text
            
            # Enhanced memory cleanup for DeepSpeed
            if args.use_deepspeed:
                # More aggressive cleanup for DeepSpeed
                del prompts, image_inputs, video_inputs, inputs
                del batch_messages, batch_output_text, generated_ids
                if 'all_video_tensors' in locals():
                    del all_video_tensors
                
                # Force garbage collection
                import gc
                gc.collect()
                
                # DeepSpeed specific memory cleanup
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                if hasattr(torch.cuda, 'synchronize'):
                    torch.cuda.synchronize()
                    
                # Reset peak memory stats for monitoring
                if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                    torch.cuda.reset_peak_memory_stats()
            else:
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, 'synchronize'):
                    torch.cuda.synchronize()

                # 显式删除不需要的变量
                del generated_ids, inputs
                if 'all_video_tensors' in locals():
                    del all_video_tensors

    # Gather metrics from all processes
    if accelerator.num_processes > 1:
        mean_acc = accelerator.gather(torch.tensor(mean_acc, device=accelerator.device)).cpu().tolist()
        if mean_mra:
            mean_mra = accelerator.gather(torch.tensor(mean_mra, device=accelerator.device)).cpu().tolist()

    final_acc={'mean_acc': 0.0, 'mean_mra': 0.0}
    if mean_acc:
        final_acc['mean_acc'] = sum(mean_acc) / len(mean_acc)
    if mean_mra:
        final_acc['mean_mra'] = sum(mean_mra) / len(mean_mra)
    
    # Only write final results from main process
    if is_main_process:
        try:
            with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
                json.dump({"results": final_output, "final_acc": [final_acc]}, f, indent=2, ensure_ascii=False)
            print(f"Final accuracy saved to {OUTPUT_PATH}")
        except Exception as e:
            print(f"Error writing final accuracy to output file: {e}")
        
        print(f"Results saved to {OUTPUT_PATH}")