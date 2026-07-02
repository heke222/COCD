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
# Reference Excel file path
# ============================================================

REFERENCE_EXCEL_PATH = "./data/Olfactory_terms.xlsx"


# ============================================================
# Prompt template (with reference terms placeholder)
# ============================================================

PROMPT_TEMPLATE = (
    "你是一名研究中文嗅觉语言的语言学专家，负责识别句子中所有与\"嗅觉\"相关的词语，并为每个实体标注合适的类型。\n"
    "请严格按照以下步骤执行任务，并展示每一步的工作：\n"
    "步骤1：仔细阅读并理解句子。分析句子的整体含义和上下文。\n"
    "步骤2：根据下面的实体类型定义，找出句中所有可能匹配的词语或短语。\n"
    "步骤3：对每个找出的词语，逐一判断其最符合哪一种实体类型。如果不符合任何类型，则排除。\n"
    "步骤4：检查是否有遗漏的实体，确保全面识别。\n"
    "步骤5：按照指定格式整理并输出最终结果。\n"
    "\n"
    "【实体类型定义】\n"
    "一、 嗅觉关键词\n"
    "嗅觉关键词是指一组预先选定的、仅与一般性气味感知相关、而不指向任何具体气味或风味来源的词汇。其主要功能是在自然语言文本中作为“指示词”，用于识别和提取与嗅觉相关的语境（即句子或上下文窗口），从而将嗅觉相关语境与嗅觉无关语境区分开来。具体分为以下三个子类：\n"
    "1、嗅觉动作：\n"
    "定义：描述嗅闻行为或气味散发行为的术语。\n"
    "示例：“闻”、“嗅”、“散发”等。\n"
    "2、气味指称：\n"
    "定义：在特定语境下直接指代气味本身的术语。\n"
    "示例：“香味”、“气味”、“臭味”等。\n"
    "3、气味属性：\n"
    "定义：直接刻画气味本身所具备的特质、品性、感官特征的术语。\n"
    "示例：“臊腥”、“馊臭”、“芳馨”等。\n"
    "\n"
    "二、嗅觉描述符\n"
    "嗅觉描述符是指用于描述、表达或评价气味品质的词汇或短语。具体分为以下三个子类：\n"
    "1、基于来源：\n"
    "定义：通过参照构成气味来源的现实世界实体来描述某气味品质的术语。\n"
    "示例：“大蒜味”、“迷迭香”、“甜橙香”等。\n"
    "2、抽象属性：\n"
    "定义：表达气味本身的内在品质，或指称其来源的某种具体物理状态、制备状态的术语。\n"
    "示例：“花香”、“果木调”、“清醇”等。\n"
    "3、主观评价：\n"
    "定义：从主观评价角度描述气味体验的词语，涵盖情绪效价、熟悉度、强度或对感知者的主观影响。\n"
    "示例：“令人作呕”、“厌恶”、“愉悦”等。\n"
    "\n"
    "【输出格式要求】\n"
    "实体 - [类型]\n"
    "每个实体单独一行\n"
    "不同句子的输出之间用空行分隔。\n"
    "\n"
    "【示例】\n"
    "句子：雨后，泥土散发出清新的气息，令人心旷神怡。\n"
    "步骤1：理解句子描述了雨后环境中泥土气味及其给人的感受。\n"
    "步骤2：找出可能实体：泥土、清新、气息、散发、心旷神怡。\n"
    "步骤3：类型判断：\n"
    "  - 泥土：产生气味的物质 → [嗅觉描述符/基于来源]\n"
    "  - 清新：描述气味性质 → [嗅觉描述符/抽象属性]\n"
    "  - 气息：指代气味本身 → [嗅觉关键词/气味指称]\n"
    "  - 散发：气味扩散动作 → [嗅觉关键词/嗅觉动作]\n"
    "  - 心旷神怡：主观感受 → [嗅觉描述符/主观评价]\n"
    "步骤4：检查确认没有遗漏。\n"
    "步骤5：整理输出：\n"
    "泥土 - [嗅觉描述符/基于来源]\n"
    "清新 - [嗅觉描述符/抽象属性]\n"
    "气息 - [嗅觉关键词/气味指称]\n"
    "散发 - [嗅觉关键词/嗅觉动作]\n"
    "心旷神怡 - [嗅觉描述符/主观评价]\n"
    "\n"
    "句子：一股腥臭从垃圾桶里飘散出来。\n"
    "步骤1：理解句子描述了垃圾桶散发的气味。\n"
    "步骤2：找出可能实体：腥臭、垃圾、飘散。\n"
    "步骤3：类型判断：\n"
    "  - 腥臭：描述气味性质 → [嗅觉关键词/气味属性]\n"
    "  - 垃圾：产生气味的物体 → [嗅觉描述符/基于来源]\n"
    "  - 飘散：气味扩散动作 → [嗅觉关键词/嗅觉动作]\n"
    "步骤4：检查确认没有遗漏。\n"
    "步骤5：整理输出：\n"
    "腥臭 - [嗅觉关键词/气味属性]\n"
    "垃圾 - [嗅觉描述符/基于来源]\n"
    "飘散 - [嗅觉关键词/嗅觉动作]\n"
    "\n"
    "句子：她深深地嗅了一下那浓烈的茉莉花香，感到神清气爽。\n"
    "步骤1：理解句子描述了人物主动闻花香的行为。\n"
    "步骤2：找出可能实体：嗅、浓烈、茉莉花、香、神清气爽。\n"
    "步骤3：类型判断：\n"
    "  - 嗅：主动闻的动作 → [嗅觉关键词/嗅觉动作]\n"
    "  - 浓烈：描述气味性质 → [嗅觉描述符/抽象属性]\n"
    "  - 茉莉花：产生气味的植物 → [嗅觉描述符/基于来源]\n"
    "  - 香：描述气味性质 → [嗅觉描述符/抽象属性]\n"
    "  - 神清气爽：主观感受 → [嗅觉描述符/主观评价]\n"
    "步骤4：检查确认没有遗漏。\n"
    "步骤5：整理输出：\n"
    "嗅 - [嗅觉关键词/嗅觉动作]\n"
    "浓烈 - [嗅觉描述符/抽象属性]\n"
    "茉莉花 - [嗅觉描述符/基于来源]\n"
    "香 - [嗅觉描述符/抽象属性]\n"
    "神清气爽 - [嗅觉描述符/主观评价]\n"
    "\n"
    "注意：我上传了一个Excel文件 (Olfactory_terms.xlsx)，其中包含了中文“关键词”和“嗅觉描述符”的参考词汇列表。\n"
    "你可以参考以下词汇来辅助判断，但最终标注须严格基于上述实体类型定义。\n"
    "{reference_terms}\n"
    "\n"
    "现在，请处理以下句子列表：\n"
)


# ============================================================
# Helper function to load reference terms from Excel
# ============================================================

def load_reference_terms(excel_path):
    """
    Load reference terms from an Excel file.
    Expected format:
        - Column 0: term (string)
        - Column 1: category (string), e.g., "嗅觉动作", "气味指称", etc.
    Returns a formatted string for inclusion in the prompt.
    If file not found or error, returns an empty string with a warning.
    """
    if not os.path.exists(excel_path):
        print(f"Warning: Reference Excel file not found at {excel_path}. Proceeding without reference terms.")
        return "（未提供参考词汇列表）"

    try:
        df = pd.read_excel(excel_path, header=None)  # assume no header
        if df.empty:
            return "（参考词汇列表为空）"

        # Assume first column is term, second column is category
        # If only one column, treat all as generic terms
        if df.shape[1] >= 2:
            # Group by category
            terms_by_cat = {}
            for _, row in df.iterrows():
                term = str(row[0]).strip()
                cat = str(row[1]).strip()
                if term and cat:
                    terms_by_cat.setdefault(cat, []).append(term)
            # Build formatted string
            lines = []
            for cat, terms in terms_by_cat.items():
                lines.append(f"{cat}：{', '.join(terms)}")
            if lines:
                return "参考词汇列表：\n" + "\n".join(lines)
            else:
                # If no valid pairs, just list all terms
                terms = [str(row[0]).strip() for _, row in df.iterrows() if str(row[0]).strip()]
                if terms:
                    return "参考词汇列表：\n" + ", ".join(terms)
                else:
                    return "（参考词汇列表为空）"
        else:
            # Only one column: treat as generic list
            terms = [str(row[0]).strip() for _, row in df.iterrows() if str(row[0]).strip()]
            if terms:
                return "参考词汇列表：\n" + ", ".join(terms)
            else:
                return "（参考词汇列表为空）"
    except Exception as e:
        print(f"Warning: Failed to read reference Excel file: {e}. Proceeding without reference terms.")
        return "（参考词汇列表读取失败）"


# ============================================================
# LLM NER request (now accepts a prompt parameter)
# ============================================================

def perform_ner_on_text_batch(sentences, prompt):
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

    # Build the full prompt by appending numbered sentences
    full_prompt = prompt
    for i, s in enumerate(sentences):
        full_prompt += f"{i+1}. {s.strip()}\n"

    payload = {
        "contents": [{"parts": [{"text": full_prompt}]}]
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


def perform_ner_with_retry(sentences, prompt, max_retries=3, delay=5):
    for _ in range(max_retries):
        try:
            return perform_ner_on_text_batch(sentences, prompt)
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

    # Load reference terms from Excel and build prompt
    reference_terms = load_reference_terms(REFERENCE_EXCEL_PATH)
    prompt = PROMPT_TEMPLATE.format(reference_terms=reference_terms)

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
                        ner_text = perform_ner_with_retry(sentences_batch, prompt)
                        results.extend(
                            parse_ner_output(ner_text, sentences_batch, seed_batch)
                        )
                    except Exception:
                        for s, st in zip(sentences_batch, seed_batch):
                            failed.append({"seed_term": st, "sentence": s})

                    sentences_batch, seed_batch = [], []
                    pbar.update(BATCH_SIZE)

        # Process remaining sentences (if any)
        if sentences_batch:
            try:
                ner_text = perform_ner_with_retry(sentences_batch, prompt)
                results.extend(
                    parse_ner_output(ner_text, sentences_batch, seed_batch)
                )
            except Exception:
                for s, st in zip(sentences_batch, seed_batch):
                    failed.append({"seed_term": st, "sentence": s})
            pbar.update(len(sentences_batch))

        if results:
            pd.DataFrame(results).to_excel(output_path, index=False)
        if failed:
            pd.DataFrame(failed).to_excel(failed_path, index=False)


if __name__ == "__main__":
    main()
