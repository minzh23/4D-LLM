import json
import re
import gc
import cv2
import torch
from PIL import Image
from tqdm import tqdm  # 添加进度条
from src.qwen2_5_vl_custom import Qwen2_5_VLForConditionalGeneration
# from transformers import Qwen2_5_VLForConditionalGeneration
from src.training.data import image2tensor
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info  # 需要确保 qwen_vl_utils 可用

# import debugpy
# debugpy.listen(("127.0.0.1", 5678))
# print("Waiting for debugger to attach...")
# debugpy.wait_for_client()
# print("Debugger attached, starting execution...")

# **1. 加载 Qwen2.5-VL 模型**
checkpoint_path = "output/depth_finetune_all/checkpoint-612"  # 你的 checkpoint 目录
original_model_path = "Qwen/Qwen2.5-eVL-3B-Instruct"  # 原始模型路径

# 加载模型
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    checkpoint_path, torch_dtype="auto", device_map="auto", attn_implementation="flash_attention_2",
)
model.eval()  # 设为推理模式

# 加载 processor
processor = AutoProcessor.from_pretrained(checkpoint_path)

# **2. 读取测试集**
test_file = "OmniSpatial/test_data.json"  # 测试数据 JSON 文件
with open(test_file, "r", encoding="utf-8") as f:
    test_data = json.load(f)

# **3. 处理测试集**
correct_count = 0  # 统计正确个数
total_count = len(test_data)  # 用于存储所有图片的 tensor

with tqdm(total=total_count, desc="Processing", unit="sample") as pbar:
    for sample in test_data:
        image_path = "OmniSpatial//" + sample["image"]  # 图片路径
        all_image_tensors = []
        conversation = sample["conversations"]
        
        # **构造输入对话**
        input_text = conversation[0]["value"]
        input_text += "\nAnswer in the format: The correct answer is A/B/C/D."
        true_answer = conversation[1]["value"].strip()  # 获取真实答案
        
        # **加载图片**
        image = Image.open(image_path).convert("RGB")
        image_tensor, _ = image2tensor(cv2.imread(image_path))
        image_tensor = image_tensor.to(dtype=torch.float16, device=model.device)
        all_image_tensors.append(image_tensor)

        # **转换为 Qwen2.5-VL 的对话格式**
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image, 
                     "resized_height": image_tensor.shape[2], "resized_width": image_tensor.shape[3]
                    },
                    {"type": "text", "text": input_text},

                ],
            }
        ]

        # **4. 预处理文本 + 图片**
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs["depth_values"] = torch.cat(all_image_tensors, dim=0).to(torch.bfloat16)
        inputs = inputs.to(model.device)

        # **5. 让模型生成回答**
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=100)

        # **6. 解析模型输出**
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        # **7. 用正则表达式提取答案**
        match = re.search(r"The correct answer is ([A-D])", response)
        predicted_answer = match.group(1) if match else "None"

        # **8. 统计正确率**
        if predicted_answer in true_answer:
            correct_count += 1

        # **9. 释放 GPU 变量，防止显存泄漏**
        del inputs, generated_ids
        torch.cuda.empty_cache()
        gc.collect()

        # **10. 更新进度条**
        accuracy = correct_count / (pbar.n + 1) * 100
        pbar.set_postfix(Correct=correct_count, Accuracy=f"{accuracy:.2f}%")
        pbar.update(1)

# **11. 计算准确率**
accuracy = correct_count / total_count * 100
print(f"Total: {total_count}, Correct: {correct_count}, Accuracy: {accuracy:.2f}%")
