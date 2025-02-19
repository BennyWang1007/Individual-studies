import torch

from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

import numpy as np

from training_utils import *

model_name = "Qwen/Qwen2.5-0.5B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(torch.device("cuda"))

# tokenizer = AutoTokenizer.from_pretrained(f"./fine_tuned_{model_name}")
# model = AutoModelForCausalLM.from_pretrained(f"./fine_tuned_{model_name}").to(torch.device("cuda"))

data = load_data("stored_data/news_with_rationales.jsonl")
data = [d for d in data if len(d["input_text"]) <= MAX_LEN - 9]  # Filter out long data
data = data[10:11]

def preprocess_testing(data):
    return {"input_text": [d["input_text"] for d in data]}

# Tokenize the data
def tokenize_function(examples):
    tokenized = tokenizer(
        examples["input_text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN  # Adjust based on your use case
    )
    tokenized["labels"] = tokenized["input_ids"]
    return tokenized

processed_data = preprocess_testing(data)
tokenized_dataset = Dataset.from_dict(processed_data).map(tokenize_function, batched=True)
for i in range(min(len(tokenized_dataset), 3)):
    print(tokenized_dataset[i])
    for k, v in tokenized_dataset[i].items():
        print("key:", k, type(k), type(v))

# use model to generate text
def generate_text(examples):
    input_texts = examples["input_text"]
    input_ids = torch.tensor(examples["input_ids"]).to(torch.device("cuda"))
    attention_mask = torch.tensor(examples["attention_mask"]).to(torch.device("cuda"))
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=MAX_LEN,
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    output_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return {"input_text": input_texts, "output_text": output_texts}

generated_dataset = tokenized_dataset.map(generate_text, batched=True)

for i in range(min(len(generated_dataset), 3)):
    print(generated_dataset[i]['input_text'], "\n")
    print(generated_dataset[i]['output_text'], "\n")


