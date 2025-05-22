import json

input_json = '/cephfs/shared/yicheng/4D-LLM/Video-R1/Video-R1-COT-165k.json'
output_json = '/cephfs/shared/yicheng/4D-LLM/Video-R1/Video-R1-COT-Converted-Fixed-cleaned.json'

target_video = 'ytb_XepZDqfvFeU.mp4'  # 要删除的视频文件名

# 加载原始 JSON 数据
with open(input_json, 'r', encoding='utf-8') as f:
    data = json.load(f)

original_count = len(data)

# 过滤掉 path 中包含目标视频名的条目
filtered_data = [
    item for item in data
    if target_video not in item.get("path", "")
]

filtered_count = len(filtered_data)
deleted_count = original_count - filtered_count

# 输出信息
print(f"原始条目数: {original_count}")
print(f"删除后剩余: {filtered_count}")
print(f"✅ 删除的条目数: {deleted_count}")

# 保存为新 JSON 文件
with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(filtered_data, f, indent=2, ensure_ascii=False)

print(f"✅ 清理后的文件已保存到: {output_json}")

no_path_items = [item for item in data if "path" not in item]
print(f"⚠️ 共有 {len(no_path_items)} 条目缺失 'path' 字段")

# print(json.dumps(data[0], indent=2, ensure_ascii=False))