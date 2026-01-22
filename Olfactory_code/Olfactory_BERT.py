# -*- coding: utf-8 -*-
"""
Masked Language Model (MLM) pretraining for a Chinese Olfactory Corpus.

This script loads sentence-level text data from a single Excel file,
tokenizes them using a pretrained Chinese BERT tokenizer,
and performs MLM pretraining using HuggingFace Transformers.
"""

import os
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    BertTokenizer,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)

# ============================================================
# Configuration (modify before running)
# ============================================================

# Path to the Excel file containing the olfactory corpus
CORPUS_EXCEL_PATH = "./data/Olfactory_Corpus.xlsx"

# Column name that contains sentences
SENTENCE_COLUMN = "sentence"

# Pretrained Chinese BERT model (HuggingFace model name or local path)
PRETRAINED_MODEL = "bert-base-chinese"

# Output directory for the trained MLM model
OUTPUT_DIR = "./outputs/bert_odor_mlm"

# Tokenization and MLM parameters
MAX_SEQ_LENGTH = 128
MLM_PROBABILITY = 0.15


# ============================================================
# Step 1: Load sentences from Excel
# ============================================================

if not os.path.isfile(CORPUS_EXCEL_PATH):
    raise FileNotFoundError(
        f"Corpus file not found: {CORPUS_EXCEL_PATH}"
    )

print(f"Loading corpus from: {CORPUS_EXCEL_PATH}")

df = pd.read_excel(CORPUS_EXCEL_PATH)

if SENTENCE_COLUMN not in df.columns:
    raise ValueError(
        f"Column '{SENTENCE_COLUMN}' not found. "
        f"Available columns: {df.columns.tolist()}"
    )

sentences = (
    df[SENTENCE_COLUMN]
    .dropna()
    .astype(str)
    .tolist()
)

print(f"Total sentences loaded: {len(sentences)}")

if len(sentences) == 0:
    raise ValueError("The sentence column is empty.")


# ============================================================
# Step 2: Build HuggingFace Dataset
# ============================================================

dataset = Dataset.from_dict({"text": sentences})

tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL)

def tokenize_function(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_SEQ_LENGTH
    )

dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)


# ============================================================
# Step 3: Data collator for MLM
# ============================================================

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=MLM_PROBABILITY
)


# ============================================================
# Step 4: Training configuration
# ============================================================

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=6,
    per_device_train_batch_size=16,
    logging_steps=100,
    save_steps=2000,
    save_total_limit=2,
    report_to="none"
)

model = BertForMaskedLM.from_pretrained(PRETRAINED_MODEL)

device = "GPU" if torch.cuda.is_available() else "CPU"
print(f"Training device: {device}")


# ============================================================
# Step 5: Train MLM model
# ============================================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator
)

trainer.train()

# Save final model and tokenizer
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"Training completed. Model saved to: {OUTPUT_DIR}")
