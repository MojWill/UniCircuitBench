import os
import re
import json
import glob
import argparse
import torch
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm


# ================== 参数 ==================
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="/DATA/DATA4/wxy/pretrained_model/InternVL3-8B")
parser.add_argument("--data_root", type=str, default="./understanding_question")
parser.add_argument("--save_path", type=str, default="InternVL3-8B-result.json")
parser.add_argument("--gpu", type=str, default="0")
parser.add_argument("--num_samples", type=int, default=None)

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


# ================== 配置 ==================
DEVICE = "cuda"
DTYPE = torch.float32
MAX_NUM = 12

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ================== 图像处理 ==================
def build_transform(input_size=448):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def dynamic_preprocess(image, image_size=448, max_num=12):
    width, height = image.size
    aspect_ratio = width / height

    target_ratios = [
        (i, j)
        for n in range(1, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num
    ]

    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    best_ratio = min(target_ratios, key=lambda r: abs(aspect_ratio - r[0] / r[1]))

    target_w = image_size * best_ratio[0]
    target_h = image_size * best_ratio[1]

    resized = image.resize((target_w, target_h))
    patches = []

    for i in range(best_ratio[0] * best_ratio[1]):
        box = (
            (i % best_ratio[0]) * image_size,
            (i // best_ratio[0]) * image_size,
            ((i % best_ratio[0]) + 1) * image_size,
            ((i // best_ratio[0]) + 1) * image_size,
        )
        patches.append(resized.crop(box))

    return patches


def load_image(path, input_size=448, max_num=12):
    if path is None or path == "":
        return None

    if not os.path.exists(path):
        return None

    image = Image.open(path).convert("RGB")
    transform = build_transform(input_size)
    patches = dynamic_preprocess(image, input_size, max_num)
    pixel_values = torch.stack([transform(p) for p in patches])
    return pixel_values


# ================== prompt ==================
def build_mcq_prompt(question, options, use_image=True):
    option_text = "\n".join([f"{k}. {v}" for k, v in options.items()])

    if use_image:
        prompt = (
            "<image>\n"
            f"{question}\n\n"
            f"{option_text}\n\n"
            "Answer with only one option letter: A, B, C, or D."
        )
    else:
        prompt = (
            f"{question}\n\n"
            f"{option_text}\n\n"
            "Answer with only one option letter: A, B, C, or D."
        )

    return prompt


# ================== 模型 ==================
def load_model():
    print("Loading model...")

    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        use_flash_attn=False,
        trust_remote_code=True
    ).to(DEVICE).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        use_fast=False
    )

    return model, tokenizer


# ================== 数据 ==================
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

            item = {
                "uid": sample.get("uid", ""),
                "image_path": sample.get("image_path", None),
                "question": sample["question"],
                "options": sample["options"],
                "gt": sample["gt"],
                "level": get_level_from_path(file)
            }

            dataset.append(item)

        except Exception as e:
            print(f"Skip {file}: {e}")

    if num_samples is not None:
        dataset = dataset[:num_samples]

    print(f"Loaded {len(dataset)} samples")
    return dataset


# ================== 推理 ==================
def extract_answer(response):
    response = response.strip().upper()

    matches = re.findall(r"\b([ABCD])\b", response)
    if len(matches) > 0:
        return matches[-1]

    return "INVALID"


def infer_one(model, tokenizer, sample):

    image_path = sample.get("image_path", None)

    has_image = (
        image_path is not None and
        image_path != "" and
        os.path.exists(image_path)
    )

    if has_image:
        pixel_values = load_image(image_path, max_num=MAX_NUM)
        pixel_values = pixel_values.to(DTYPE).to(DEVICE)
        prompt = build_mcq_prompt(sample["question"], sample["options"], True)
    else:
        pixel_values = None
        prompt = build_mcq_prompt(sample["question"], sample["options"], False)

    generation_config = dict(
        max_new_tokens=10,
        do_sample=False
    )

    response = model.chat(
        tokenizer,
        pixel_values,
        prompt,
        generation_config
    )

    pred = extract_answer(response)
    return pred, response


# ================== batch ==================
def run_batch(model, tokenizer, dataset):
    results = []
    correct = 0
    level_stats = {}

    for sample in tqdm(dataset):
        pred, raw_response = infer_one(model, tokenizer, sample)

        sample["pred"] = pred
        sample["raw_response"] = raw_response

        level = sample["level"]
        if level not in level_stats:
            level_stats[level] = {"correct": 0, "total": 0}

        level_stats[level]["total"] += 1

        if pred == sample["gt"]:
            correct += 1
            level_stats[level]["correct"] += 1

        results.append(sample)

        print("=" * 60)
        print("Level:", level)
        print("GT:", sample["gt"])
        print("Pred:", pred)
        print("Raw:", raw_response)

    # overall
    acc = correct / len(dataset)
    print("\n" + "=" * 60)
    print(f"Overall Accuracy: {acc:.4f}")

    # per level
    print("\nPer-level Accuracy:")
    for lv in sorted(level_stats.keys()):
        c = level_stats[lv]["correct"]
        t = level_stats[lv]["total"]
        print(f"{lv}: {c}/{t} = {c/t:.4f}")

    with open(args.save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to {args.save_path}")


# ================== main ==================
if __name__ == "__main__":
    model, tokenizer = load_model()
    dataset = load_dataset(args.data_root, args.num_samples)
    run_batch(model, tokenizer, dataset)