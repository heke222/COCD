# -*- coding: utf-8 -*-
"""
LLM-assisted named entity recognition (NER) for Chinese olfactory language.

This script uses the Gemini API to identify and classify olfactory-related
entities in Chinese sentences and exports the results to Excel files.

IMPORTANT:
- Set the environment variable GEMINI_API_KEY before running.
- All paths are relative for reproducibility.
"""

import os
import requests
import json
import time
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm


# ============================================================
# Gemini API configuration
# ============================================================

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise EnvironmentError(
        "GEMINI_API_KEY is not set. "
        "Please set it as an environment variable before running."
    )

API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"gemini-2.0-flash:generateContent?key={API_KEY}"
)

HEADERS = {"Content-Type": "application/json"}


# ============================================================
# Rate limit configuration
# ============================================================

RPM_LIMIT = 15     # requests per minute
RPD_LIMIT = 1500   # requests per day

request_count = 0
last_minute_reset = datetime.now()
last_day_reset = datetime.now()


# ============================================================
# Full annotation prompt (fixed for reproducibility)
# ============================================================

FULL_PROMPT = (
    "你是一名研究中文嗅觉语言的语言学专家，负责识别句子中所有与“嗅觉”相关的词语，"
    "并为每个实体标注最合适的语义类型。\n\n"

    "【任务说明】\n"
    "请基于句子语境，全面识别所有与嗅觉相关的词语或短语，"
    "并严格依据下述实体类型进行标注。"
    "不符合任何类型的词语请勿输出。\n"
    "以下步骤仅用于内部分析，不应在输出中呈现。\n\n"

    "【实体类型】\n"
    "1. 嗅觉关键词：\n"
    "   - 嗅觉动作：描述闻、嗅、吸入、散发、飘出等与嗅觉相关的动词\n"
    "   - 嗅觉名词：在语境中直接指代气味或香气的名词，如气味、香气、酒香等\n"
    "   - 嗅觉形容词：描述气味属性或品质的形容词，如腥臭、刺鼻、清新等\n"
    "2. 嗅觉描述符：\n"
    "   - 气味来源：产生气味的具体物质、植物、食物或物体，如玫瑰、咖啡、垃圾等\n"
    "   - 气味特征：描述气味性质或类型的词语，如烟熏、焦糖味、木质调等\n"
    "   - 嗅觉感受：气味引发的主观心理或生理感受，如温暖、沉醉、心旷神怡等\n\n"

    "【输出格式】\n"
    "每个实体单独一行，格式为：\n"
    "实体 - [类型]\n"
    "不同句子的输出之间请用一个空行分隔。\n"
    "请勿输出任何解释性文字。\n\n"

    "【示例】\n"
    "句子：雨后，泥土散发出清新的气息，令人心旷神怡。\n"
    "输出：\n"
    "泥土 - [气味来源]\n"
    "散发 - [嗅觉动作]\n"
    "清新 - [气味特征]\n"
    "气息 - [嗅觉名词]\n"
    "心旷神怡 - [嗅觉感受]\n\n"

    "现在请处理以下句子列表：\n"
)


# ============================================================
# LLM NER request
# ============================================================

def perform_ner_on_text_batch(sentences):
    global request_count, last_minute_reset, last_day_reset

    now = datetime.now()

    # Daily reset
    if (now - last_day_reset).days >= 1:
        request_count = 0
        last_day_reset = now

    if request_count >= RPD_LIMIT:
        raise RuntimeError("Daily API request limit reached.")

    # Minute reset
    if now - last_minute_reset >= timedelta(minutes=1):
        request_count = 0
        last_minute_reset = now

    if request_count >= RPM_LIMIT:
        time.sleep(60 - (now - last_minute_reset).seconds)
        request_count = 0
        last_minute_reset = datetime.now()

    prompt = FULL_PROMPT
    for i, s in enumerate(sentences):
        prompt += f"{i+1}. {s.strip()}\n"

    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }

    response = requests.post(API_URL, headers=HEADERS, json=payload)
    request_count += 1

    if response.status_code == 200:
        try:
            return response.json()["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            raise RuntimeError("Unexpected API response structure.")
    else:
        raise RuntimeError(
            f"API request failed: {response.status_code} {response.text}"
        )


def perform_ner_with_retry(sentences, max_retries=3, delay=5):
    for _ in range(max_retries):
        try:
            return perform_ner_on_text_batch(sentences)
        except Exception:
            time.sleep(delay)
    raise RuntimeError("NER request failed after multiple retries.")


# ============================================================
# Parse LLM output
# ============================================================

def parse_ner_output(ner_text, sentences, seed_terms):
    records = []
    sentence_index = -1

    for line in ner_text.splitlines():
        line = line.strip()
        if not line:
            continue

        if line[0].isdigit() and ". " in line:
            try:
                sentence_index = int(line.split(". ", 1)[0]) - 1
            except ValueError:
                sentence_index = -1
            continue

        if " - [" in line and sentence_index >= 0:
            entity, type_part = line.rsplit(" - [", 1)
            entity_type = type_part.rstrip("]")

            records.append({
                "seed_term": seed_terms[sentence_index],
                "sentence": sentences[sentence_index],
                "entity": entity.strip(),
                "entity_type": entity_type.strip()
            })

    return records


# ============================================================
# Main pipeline
# ============================================================

def main():
    input_dir = "./data/input_files"
    output_dir = "./data/output_files"
    failed_dir = "./data/failed_files"

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(failed_dir, exist_ok=True)

    excel_files = [f for f in os.listdir(input_dir) if f.endswith(".xlsx")]
    BATCH_SIZE = 80

    for filename in excel_files:
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, f"ner_{filename}")
        failed_path = os.path.join(failed_dir, f"failed_{filename}")

        df = pd.read_excel(input_path, usecols=[0, 1], skiprows=1, header=None)
        df.columns = ["seed_term", "sentence"]
        df = df.dropna(subset=["sentence"])

        results, failed = [], []
        sentences_batch, seed_batch = [], []

        with tqdm(total=len(df), desc=f"Processing {filename}") as pbar:
            for _, row in df.iterrows():
                sentences_batch.append(str(row["sentence"]).strip())
                seed_batch.append(str(row["seed_term"]).strip())

                if len(sentences_batch) == BATCH_SIZE:
                    try:
                        ner_text = perform_ner_with_retry(sentences_batch)
                        results.extend(
                            parse_ner_output(ner_text, sentences_batch, seed_batch)
                        )
                    except Exception:
                        for s, st in zip(sentences_batch, seed_batch):
                            failed.append({"seed_term": st, "sentence": s})

                    sentences_batch, seed_batch = [], []
                    pbar.update(BATCH_SIZE)

        if results:
            pd.DataFrame(results).to_excel(output_path, index=False)
        if failed:
            pd.DataFrame(failed).to_excel(failed_path, index=False)


if __name__ == "__main__":
    main()
