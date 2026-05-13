import os
import json
from pathlib import Path
from tqdm import tqdm
import requests
import dashscope
from dashscope import MultiModalConversation

# ===================== 配置 =====================
API_KEY = "sk-33cd5d3b36c0412e9790576196b53d72"   # 或者用 os.getenv("DASHSCOPE_API_KEY")

# 北京区域
dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/api/v1"

MODEL_NAME = "wan2.7-image"

INPUT_JSONL = "generation.jsonl"
OUTPUT_DIR = "./generation_answer"

LIMIT = None   # 调试时改成1/2；全部跑改成 None
# ===============================================


def save_image(image_url, save_path):
    """
    下载图片并保存
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(image_url, timeout=60)
    response.raise_for_status()

    with open(save_path, "wb") as f:
        f.write(response.content)


def generate_image(prompt):
    """
    调用 DashScope 生成图片
    返回图片URL
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"text": prompt}
            ]
        }
    ]

    response = MultiModalConversation.call(
        api_key=API_KEY,
        model=MODEL_NAME,
        messages=messages,
        result_format="message",
        stream=False,
        watermark=False,
        prompt_extend=True,
        size="1024*1024"
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"status={response.status_code}, "
            f"code={response.code}, "
            f"message={response.message}"
        )

    # 如需debug可打开
    # print(json.dumps(response, ensure_ascii=False, indent=2))

    content = response["output"]["choices"][0]["message"]["content"][0]

    # DashScope通常返回图片URL
    if "image" in content:
        return content["image"]

    elif "image_url" in content:
        return content["image_url"]

    else:
        raise ValueError(f"Unknown response format: {content}")


def process_jsonl():
    with open(INPUT_JSONL, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if LIMIT is not None:
        lines = lines[:LIMIT]

    print(f"Total samples to process: {len(lines)}")

    for idx, line in enumerate(tqdm(lines), start=1):
        try:
            sample = json.loads(line.strip())

            prompt = sample["recommended_prompt"]
            source_name = sample["source_name"]

            # xxx.json -> xxx.png
            image_name = Path(source_name).stem + ".png"
            save_path = Path(OUTPUT_DIR) / image_name

            # 已存在则跳过
            if save_path.exists():
                print(f"[{idx}] Skip existing: {save_path}")
                continue

            image_url = generate_image(prompt)
            save_image(image_url, save_path)

            print(f"[{idx}] Saved: {save_path}")

        except Exception as e:
            print(f"[{idx}] Failed: {e}")


if __name__ == "__main__":
    process_jsonl()