# %%
from accelerate import Accelerator

accelerator = Accelerator()

import os
import json
import torch
import pandas as pd

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from datasets import load_dataset, Dataset
from trl import DPOTrainer, DPOConfig
from typing import Dict, List, Union

# %%
# Load envs
with open("env.json") as env_file:
    for k, v in json.load(env_file).items():
        os.environ[k] = v

# %%
with open("dpo_pairs_bacchus_v2.jsonl") as json_file:
    dataset = list(json_file)

# %%
dataset = list(map(json.loads, dataset))
dataset = Dataset.from_pandas(pd.DataFrame(dataset))

# %%
# Set model
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# %%
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# %%
# Setting pad_token
tokenizer.eos_token = "<|end_of_text|>"
tokenizer.pad_token = tokenizer.eos_token

# Create DPOConfig
dpo_config = DPOConfig(
    output_dir="./results_standard_dpo",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    save_steps=1000,
    logging_steps=2,
    save_total_limit=2,
    learning_rate=2e-4,
    fp16=False,
    bf16=True,
    max_grad_norm=1.0,
    warmup_ratio=0.1,
    group_by_length=False,
    lr_scheduler_type="cosine",
    beta=0.1,
    max_prompt_length=512,
    max_length=4096,
    remove_unused_columns=False,
    deepspeed="/workspace/bacchus-dpo/ds-config.json"
)

# Model to fine-tune
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    # load_in_4bit=True,
)

# Create the standard DPO Trainer
trainer = DPOTrainer(
    model=model,
    # ref_model
    args=dpo_config,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

# %%
# Prepare for distributed training
model, trainer = accelerator.prepare(model, trainer)

# %%
# Train the model
trainer.train()

# %%
# Save the final model
trainer.save_model("./standard_dpo_llama3.1-8b")

# %%



