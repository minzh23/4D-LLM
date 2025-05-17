import json
import re
import gc
import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from src.qwen2_5_vl_custom import Qwen2_5_VLForConditionalGeneration
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
import wandb
from torchvision import transforms
import decord
from decord import VideoReader, cpu
from src.training.data import image2tensor


# import debugpy
# debugpy.listen(("127.0.0.1", 5678))
# print("Waiting for debugger to attach...")
# debugpy.wait_for_client()
# print("Debugger attached, starting execution...")

wandb.init(
    project="4d-llm-eval",  # 项目名称，替换为你自己的项目名
    name="video-r1_2-stage-training_all_omni",  # 本次 run 的名字
)

def load_video_frames(video_path, num_frames=16, size=(224, 224)):
    import cv2
    from PIL import Image

    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
    frames = vr.get_batch(frame_indices).asnumpy()  # (T, H, W, C), dtype=uint8

    # 转为 PIL 图像并 resize
    pil_frames = [
        Image.fromarray(frame).resize(size, Image.BICUBIC) for frame in frames
    ]
    return pil_frames, frame_indices, vr

# 1. 加载模型
import os
from transformers import AutoProcessor

# 强制使用绝对路径
checkpoint_path = os.path.abspath("output/video-r1_stage2/checkpoint-6363")

# 加载模型
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    checkpoint_path,
    torch_dtype="auto",
    device_map="auto",
    attn_implementation="flash_attention_2",
)
model.eval()

# 加载 processor（tokenizer + vision processor）
processor = AutoProcessor.from_pretrained(checkpoint_path)

# 2. 加载测试集
test_file = "/cephfs/shared/yicheng/4D-LLM/VSI-Bench/vsi-bench.json"
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
        video_path = "/cephfs/shared/yicheng/4D-LLM/VSI-Bench/" + sample["video"]
        all_video_tensors = []
        conversation = sample["conversations"]

        input_text = conversation[0]["value"]
        input_text += "If you are given options to choose, output your choice as \\boxed{...}. Otherwise output your numeric answer as \\boxed{...}."
        true_answer = conversation[1]["value"].strip()

        # 加载视频帧并准备深度值
        frames, frame_indices, vr = load_video_frames(video_path, num_frames=16, size=(224, 224))
        
        # 提取每一帧的深度信息
        all_depth_tensors = []
        for idx in frame_indices:
            # 获取原始帧（BGR格式）
            frame = vr[idx].asnumpy()
            # 转换为BGR格式（如果需要）
            if frame.shape[2] == 3 and frame.dtype == np.uint8:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame
            
            # 提取深度值
            image_tensor, _ = image2tensor(frame_bgr)
            image_tensor = image_tensor.to(dtype=torch.float16, device=model.device)
            # 按照data.py中相同的方式处理图像张量
            image_tensor_flat = image_tensor.permute(1, 0, 2, 3).contiguous().view(image_tensor.shape[1], -1)
            all_depth_tensors.append(image_tensor_flat)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": frames,
                        "resized_height": 224,
                        "resized_width": 224,
                    },
                    {"type": "text", "text": input_text},
                ],
            }
        ]

        # 构造输入
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        # 添加深度值到输入
        # 按照data.py中的处理方式，应该是按dim=1连接
        depth_values = torch.cat(all_depth_tensors, dim=1)
        inputs["depth_values"] = depth_values.to(torch.bfloat16)
        inputs = inputs.to(model.device)

        # 推理
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=4096)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        match = re.search(r"<answer>(.*?)</answer>", response)
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