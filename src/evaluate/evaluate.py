import json
import re
import gc
import cv2
import torch
from PIL import Image
from tqdm import tqdm
from src.qwen2_5_vl_custom import Qwen2_5_VLForConditionalGeneration
from src.training.data import image2tensor
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
import wandb

# import debugpy
# debugpy.listen(("127.0.0.1", 5678))
# print("Waiting for debugger to attach...")
# debugpy.wait_for_client()
# print("Debugger attached, starting execution...")

wandb.init(
    project="4d-llm-eval",  # 项目名称，替换为你自己的项目名
    name="2_stage_training",  # 本次 run 的名字
)

# 1. 加载模型
checkpoint_path = "output/stage2_frozen_none/checkpoint-1224"
original_model_path = "Qwen/Qwen2.5-eVL-3B-Instruct"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    checkpoint_path,
    torch_dtype="auto",
    device_map="auto",
    attn_implementation="flash_attention_2",
)
model.eval()

processor = AutoProcessor.from_pretrained(checkpoint_path)

# 2. 加载测试集
test_file = "OmniSpatial/test_data.json"
with open(test_file, "r", encoding="utf-8") as f:
    test_data = json.load(f)

# 3. 初始化统计
correct_count = 0
total_count = len(test_data)
interval = 10
accuracy_log = []

# 4. 主循环
with tqdm(total=total_count, desc="Processing", unit="sample") as pbar:
    for sample in test_data:
        image_path = "OmniSpatial//" + sample["image"]
        all_image_tensors = []
        conversation = sample["conversations"]

        input_text = conversation[0]["value"]
        input_text += "Answer in the format: The correct answer is A/B/C/D."
        true_answer = conversation[1]["value"].strip()

        # 加载图像
        image = Image.open(image_path).convert("RGB")
        image_tensor, _ = image2tensor(cv2.imread(image_path))
        image_tensor = image_tensor.to(dtype=torch.float16, device=model.device)
        all_image_tensors.append(image_tensor)

        # 构造消息格式
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                        "resized_height": image_tensor.shape[2],
                        "resized_width": image_tensor.shape[3],
                    },
                    {"type": "text", "text": input_text},
                ],
            }
        ]

        # 文本 + 图像处理
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

        # 推理
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=100)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        match = re.search(r"The correct answer is ([A-D])", response)
        predicted_answer = match.group(1) if match else "None"

        # 判断是否正确
        if predicted_answer in true_answer:
            correct_count += 1

        # 清理显存
        del inputs, generated_ids
        torch.cuda.empty_cache()
        gc.collect()

        # 更新进度条
        accuracy = correct_count / (pbar.n + 2) * 100
        pbar.set_postfix(Correct=correct_count, Accuracy=f"{accuracy:.2f}%")
        pbar.update(1)

        # 每 interval 步记录准确率
        if (pbar.n + 1) % interval == 0 or (pbar.n + 1) == total_count:
            print(f"Step {pbar.n + 1}: Correct: {correct_count}, Accuracy: {accuracy:.2f}%")
            accuracy_log.append({
                "step": pbar.n + 1,
                "correct": correct_count,
                "accuracy": round(accuracy, 2)
            })
            wandb.log({
                "step": pbar.n + 1,
                "correct": correct_count,
                "accuracy": accuracy
            })

# 5. 最终结果
accuracy = correct_count / total_count * 100
print(f"\nTotal: {total_count}, Correct: {correct_count}, Accuracy: {accuracy:.2f}%")

wandb.finish()