import os
import re
import json
import glob
import argparse
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor


# ================== args ==================
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="/DATA/DATA4/wxy/pretrained_model/Qwen3-VL-8B-Instruct")
parser.add_argument("--data_root", type=str, default="./understanding_question")
parser.add_argument("--save_path", type=str, default="qwen3-vl-8B_result.json")
parser.add_argument("--gpu", type=str, default="0")
parser.add_argument("--num_samples", type=int, default=None)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ================== prompt ==================
def build_mcq_prompt(question, options):
    option_text = "\n".join([f"{k}. {v}" for k, v in options.items()])
    return (
        f"{question}\n\n"
        f"{option_text}\n\n"
        "Answer with only one option letter: A, B, C, or D."
    )


# ================== model ==================
def load_model():
    print("Loading Qwen3-VL...")

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        local_files_only=True
    ).eval()

    processor = AutoProcessor.from_pretrained(
        args.model_path,
        local_files_only=True
    )

    return model, processor


# ================== dataset ==================
def get_level_from_path(file_path):
    file_path = file_path.lower()
    for lv in ["level1", "level2", "level3", "level4", "level5"]:
        if lv in file_path:
            return lv
    return "unknown"


def load_dataset(data_root, num_samples=None):
    files = glob.glob(os.path.join(data_root, "**/*.json"), recursive=True)

    dataset = []
    for file in files:
        try:
            with open(file, "r", encoding="utf-8") as f:
                sample = json.load(f)

            dataset.append({
                "uid": sample.get("uid", ""),
                "image_path": sample.get("image_path", None),
                "question": sample["question"],
                "options": sample["options"],
                "gt": sample["gt"],
                "level": get_level_from_path(file)
            })

        except Exception as e:
            print(f"Skip {file}: {e}")

    if num_samples:
        dataset = dataset[:num_samples]

    print(f"Loaded {len(dataset)} samples")
    return dataset


# ================== answer parse ==================
def extract_answer(text):
    text = text.strip().upper()
    matches = re.findall(r"\b([ABCD])\b", text)
    return matches[-1] if matches else "INVALID"


# ================== inference ==================
def infer_one(model, processor, sample):

    image_path = sample.get("image_path", None)

    has_image = (
        image_path is not None and
        image_path != "" and
        os.path.exists(image_path)
    )

    prompt = build_mcq_prompt(sample["question"], sample["options"])

    # ✅ 关键修复：必须用 PIL.Image，而不是 path string
    if has_image:
        image = Image.open(image_path).convert("RGB")
        content = [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt}
        ]
    else:
        content = [
            {"type": "text", "text": prompt}
        ]

    messages = [
        {
            "role": "user",
            "content": content
        }
    ]

    # ================== preprocess ==================
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )

    inputs = {k: v.to(model.device) if torch.is_tensor(v) else v for k, v in inputs.items()}

    # ================== generate ==================
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False
        )

    # ================== decode（正确方式） ==================
    generated_text = processor.batch_decode(outputs[:, inputs["input_ids"].shape[1]:],
                                            skip_special_tokens=True)[0]

    pred = extract_answer(generated_text)
    return pred, generated_text


# ================== batch ==================
def run_batch(model, processor, dataset):

    results = []
    correct = 0
    level_stats = {}

    for sample in tqdm(dataset):

        pred, raw = infer_one(model, processor, sample)

        sample["pred"] = pred
        sample["raw_response"] = raw

        lv = sample["level"]
        level_stats.setdefault(lv, {"correct": 0, "total": 0})

        level_stats[lv]["total"] += 1

        if pred == sample["gt"]:
            correct += 1
            level_stats[lv]["correct"] += 1

        results.append(sample)

        print("=" * 60)
        print("Level:", lv)
        print("GT:", sample["gt"])
        print("Pred:", pred)
        print("Raw:", raw)

    # ================== overall ==================
    acc = correct / len(dataset)
    print("\n" + "=" * 60)
    print(f"Overall Accuracy: {acc:.4f}")

    # ================== per level ==================
    print("\nPer-level Accuracy:")
    for lv in sorted(level_stats.keys()):
        c = level_stats[lv]["correct"]
        t = level_stats[lv]["total"]
        print(f"{lv}: {c}/{t} = {c/t:.4f}")

    # ================== save ==================
    with open(args.save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to {args.save_path}")


# ================== main ==================
if __name__ == "__main__":
    model, processor = load_model()
    dataset = load_dataset(args.data_root, args.num_samples)
    run_batch(model, processor, dataset)