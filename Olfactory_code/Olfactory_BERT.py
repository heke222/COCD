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
MAX_SEQ_LENGTH = 128          # Table: 128
MLM_PROBABILITY = 0.15        # Table: 0.15

# Training hyperparameters (from table)
LEARNING_RATE = 2e-5
TOTAL_BATCH_SIZE = 1024       # Effective batch size after gradient accumulation
PER_DEVICE_BATCH_SIZE = 64    # Adjust based on GPU memory
GRADIENT_ACCUMULATION_STEPS = TOTAL_BATCH_SIZE // PER_DEVICE_BATCH_SIZE  # =16
NUM_EPOCHS = 16


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
# Step 2: Build HuggingFace Dataset and split 9:1
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

# Split into train (90%) and validation (10%)
split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]
print(f"Train size: {len(train_dataset)}, Validation size: {len(eval_dataset)}")


# ============================================================
# Step 3: Data collator for MLM
# ============================================================

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=MLM_PROBABILITY
)


# ============================================================
# Step 4: Training configuration with hyperparameters
# ============================================================

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,

    # Hyperparameters from table
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    num_train_epochs=NUM_EPOCHS,
    optim="adamw_torch",          # AdamW optimizer

    # Evaluation settings (evaluate on validation loss)
    evaluation_strategy="steps",
    eval_steps=500,               # Evaluate every 500 steps
    save_steps=500,               # Save checkpoint every 500 steps
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    # Logging and saving
    logging_steps=100,
    save_total_limit=2,
    report_to="none",             # Disable external logging (optional)
    seed=42,
)

# Detect device
device = "GPU" if torch.cuda.is_available() else "CPU"
print(f"Training device: {device}")
print(f"Effective batch size: {PER_DEVICE_BATCH_SIZE} * {GRADIENT_ACCUMULATION_STEPS} = {TOTAL_BATCH_SIZE}")

model = BertForMaskedLM.from_pretrained(PRETRAINED_MODEL)


# ============================================================
# Step 5: Train MLM model with evaluation
# ============================================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,      # Provide validation set
    data_collator=data_collator,
)

trainer.train()

# Save final model and tokenizer
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"Training completed. Model saved to: {OUTPUT_DIR}")
