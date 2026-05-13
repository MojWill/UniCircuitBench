import json
from collections import defaultdict


# ===================== path =====================
filtered_dataset_path = "fliter.json"   # 新筛后的题集
model_result_path = "InternVL3-8B-result.json"            # 模型完整结果


# ===================== load =====================
with open(filtered_dataset_path, "r", encoding="utf-8") as f:
    filtered_data = json.load(f)

with open(model_result_path, "r", encoding="utf-8") as f:
    model_data = json.load(f)


# ===================== 获取保留 uid =====================
keep_uids = set(item["uid"] for item in filtered_data)

print(f"Filtered benchmark size: {len(keep_uids)}")


# ===================== stats =====================
level_total = defaultdict(int)
level_correct = defaultdict(int)

total = 0
correct = 0
matched = 0


for item in model_data:
    uid = item["uid"]

    # 只统计筛后保留题
    if uid not in keep_uids:
        continue

    matched += 1

    level = item["level"]
    gt = item["gt"]
    pred = item["pred"]

    level_total[level] += 1
    total += 1

    if gt == pred:
        level_correct[level] += 1
        correct += 1


# ===================== print result =====================
print(f"Matched samples in model result: {matched}\n")

print("========== Per-level Accuracy ==========")

for level in sorted(level_total.keys()):
    c = level_correct[level]
    t = level_total[level]
    acc = c / t if t > 0 else 0

    print(f"{level}: {c}/{t} = {acc:.4f}")


overall_acc = correct / total if total > 0 else 0

print("\n========== Overall Accuracy ==========")
print(f"Overall: {correct}/{total} = {overall_acc:.4f}")


# ===================== markdown table =====================
print("\nMarkdown Table:")
print("| Level | Correct | Total | Accuracy |")
print("|---|---:|---:|---:|")

for level in sorted(level_total.keys()):
    c = level_correct[level]
    t = level_total[level]
    acc = c / t if t > 0 else 0
    print(f"| {level} | {c} | {t} | {acc:.4f} |")

print(f"| Overall | {correct} | {total} | {overall_acc:.4f} |")