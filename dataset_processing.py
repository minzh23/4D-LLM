from datasets import Dataset
import json
import string

# 加载 .arrow 文件为 Dataset 对象
ds = Dataset.from_file("./VSI-Bench/test/data-00000-of-00001.arrow")  # 替换成你的文件名

output = []

for item in ds:
    # 构造 video 字段
    video = f"{item['dataset']}/{item['scene_name']}.mp4"

    # 构造 human value
    human_value = f"<video>\n{item['question']}"
    
    # 如果存在 options 且不为 None 或空
    options = item.get("options")
    if options:
        human_value += "\nChoose from these options to answer:\n"
        for idx, option in enumerate(options):
            label = string.ascii_uppercase[idx]  # A, B, C, ...
            human_value += f"{label}. {option}\n"
        human_value = human_value.strip()  # 去除多余换行

    # 构造 conversations
    gt = item["ground_truth"]
    conversations = [
        {
            "from": "human",
            "value": human_value
        },
        {
            "from": "gpt",
            "value": f"My answer is {gt}."
        }
    ]

    ans = item["ground_truth"]

    # 构造最终 JSON 条目
    output.append({
        "id": item["id"],
        "video": video,
        "conversations": conversations, 
        "problem_type": "multiple choice" if item["options"] is not None else "numerical", 
        "path": "./" + video, 
        "data_type": "video",
        "problem": item["question"],  
        "options": item["options"], 
        "solution": f"<answer>{ans}</answer>"
    })

# 写入 JSON 文件
with open("output.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)