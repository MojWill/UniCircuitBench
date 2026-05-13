import os
import json
import argparse
import base64
import hashlib
import re
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI

# ==================== API 配置 ====================

#bailian-api
# DEFAULT_API_KEY = "sk-33cd5d3b36c0412e9790576196b53d72"
# DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


#ai-api
DEFAULT_API_KEY = "sk-SjxtV5BdKUqtWV1fCvR2mt5M8WwR9k5ntIemcPltGGjosi8d"
DEFAULT_BASE_URL = "http://35.220.164.252:3888/v1/"


# ==================== utils ====================
def encode_image(image_path):
    """Encode image to base64"""
    try:
        if image_path and os.path.exists(image_path):
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        print("encode_image error:", e)
    return None


def normalize_answer(pred):
    """
    Normalize model output:
    'The answer is A.' -> A
    """
    pred = str(pred).strip()

    match = re.search(r"\b(A|B|C|D|True|False)\b", pred, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    return pred.strip()


def fix_image_path(image_path):
    if not image_path:
        return None

    image_path = str(image_path).strip()
    return os.path.normpath(image_path)


def build_unique_id(q):
    base = f"{q['source_file']}|{q['q_id']}|{q['question']}"
    return hashlib.md5(base.encode("utf-8")).hexdigest()


def split_task(items, tot, idx):
    k, m = divmod(len(items), tot)
    return [
        items[i * k + min(i, m):(i + 1) * k + min(i + 1, m)]
        for i in range(tot)
    ][idx]


# ==================== load json ====================
def load_questions(folder):
    questions = []

    json_files = list(Path(folder).rglob("*.json"))
    print(f"📂 Found {len(json_files)} json files")

    for f in json_files:
        try:
            with open(f, "r", encoding="utf-8") as fp:
                q = json.load(fp)

            q["image_path"] = fix_image_path(q.get("image_path", ""))

            # save level info
            q["level"] = Path(f).parent.name

            questions.append(q)

        except Exception as e:
            print(f"skip {f}: {e}")

    return questions


# ==================== prompt ====================
def get_input_prompt(q):
    opt = q["options"]

    return (
        "You are given a multiple-choice circuit question.\n\n"
        f"Question:\n{q['question']}\n\n"
        "Options:\n"
        f"A. {opt.get('A', '')}\n"
        f"B. {opt.get('B', '')}\n"
        f"C. {opt.get('C', '')}\n"
        f"D. {opt.get('D', '')}\n\n"
        "Answer with ONLY one letter: A, B, C, or D."
    )


# ==================== LLM ====================
def llm_generator(messages, api_key, model, base_url):
    client = OpenAI(
        base_url=base_url,
        api_key=api_key
    )

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
        max_tokens=1024
    )

    return response.choices[0].message.content


# ==================== main ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        default="./understanding_question"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o"
    )
    parser.add_argument("--tot", type=int, default=1)
    parser.add_argument("--id", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    api_key = DEFAULT_API_KEY
    base_url = DEFAULT_BASE_URL

    # ==================== load ====================
    all_questions = load_questions(args.data_root)
    print(f"📊 Total questions: {len(all_questions)}")

    if args.limit is not None:
        all_questions = all_questions[:args.limit]

    task = split_task(all_questions, args.tot, args.id)
    print(f"🚀 Processing {len(task)} questions")

    # ==================== stats ====================
    total = 0
    correct = 0

    level_total = {}
    level_correct = {}

    # root result folder
    result_root = Path(args.data_root) / f"{args.model}_results"
    result_root.mkdir(parents=True, exist_ok=True)

    # ==================== inference ====================
    for q in tqdm(task):
        uid = build_unique_id(q)

        level = q.get("level", "unknown")

        save_folder = result_root / level
        save_folder.mkdir(parents=True, exist_ok=True)

        save_path = save_folder / f"{uid}.json"

        if save_path.exists():
            continue

        prompt = get_input_prompt(q)

        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
        }]

        img_path = q.get("image_path", None)

        if args.debug:
            print("\n====================")
            print("IMAGE:", img_path)
            print("EXISTS:", os.path.exists(img_path) if img_path else False)

        # add image
        if img_path and os.path.exists(img_path):
            base64_img = encode_image(img_path)

            if base64_img:
                messages[0]["content"].append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_img}"
                    }
                })

        try:
            response = llm_generator(
                messages,
                api_key,
                args.model,
                base_url
            )

            pred = normalize_answer(response)
            gt = normalize_answer(q["gt"])
            is_correct = pred == gt

            total += 1
            level_total[level] = level_total.get(level, 0) + 1

            if is_correct:
                correct += 1
                level_correct[level] = level_correct.get(level, 0) + 1

            result = q.copy()
            result["pred_raw"] = response
            result["pred"] = pred
            result["correct"] = is_correct

            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print("❌ Error:", e)

    # ==================== summary ====================
    acc = correct / total if total > 0 else 0

    summary = {
        "model": args.model,
        "total": total,
        "correct": correct,
        "accuracy": acc,
        "per_level": {}
    }

    for level in sorted(level_total.keys()):
        level_acc = (
            level_correct.get(level, 0) / level_total[level]
            if level_total[level] > 0 else 0
        )

        summary["per_level"][level] = {
            "total": level_total[level],
            "correct": level_correct.get(level, 0),
            "accuracy": level_acc
        }

    summary_path = result_root / f"summary_part{args.id}.json"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ==================== print ====================
    print("=" * 60)
    print("Model:", args.model)
    print("Total:", total)
    print("Correct:", correct)
    print("Accuracy:", round(acc, 4))
    print()

    for level in sorted(summary["per_level"].keys()):
        level_info = summary["per_level"][level]
        print(
            f"{level}: "
            f"{level_info['correct']}/{level_info['total']} "
            f"= {level_info['accuracy']:.4f}"
        )

    print("=" * 60)
    print(f"✅ Saved to {result_root}")