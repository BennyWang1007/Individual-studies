import torch

from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

from news_with_rationale import NewsWithRationale
from xai import XAI

from training_utils import *

# 1. Load Pre-trained Model and Tokenizer
# model_name = "gemma-2-2b"  # Replace with the correct Hugging Face model name if needed
model_name = "Qwen/Qwen2.5-0.5B"
# model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

MAX_LEN = 1024

# 2. Prepare Training Data
data = load_data("stored_data/news_with_rationales.jsonl")
data = [d for d in data if len(d["input_text"]) + len(d["output_text"]) <= MAX_LEN - 9]  # Filter out long data
data = data[:112]  # For demonstration purposes only

# Combine input and output texts for training
def preprocess_training(data):
    return {"text": [d["input_text"] + d["output_text"] for d in data]}

processed_data = preprocess_training(data)
# del data

# print max length in the dataset
print("max length:", max([len(d) for d in processed_data["text"]]))

# # Tokenize the data
def tokenize_function(examples):
    tokenized = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN  # Adjust based on your use case
    )
    tokenized["labels"] = tokenized["input_ids"]
    return tokenized

# Convert to Hugging Face Dataset
tokenized_dataset = Dataset.from_dict({"text": processed_data["text"]}).map(tokenize_function, batched=True)
# del processed_data

# Split the dataset
split_dataset = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

# print(len(train_dataset))
# print(len(eval_dataset))
# for d in train_dataset:
#     print(tokenizer.decode(d["input_ids"][-1]))
# print()
# for d in eval_dataset:
#     # print(f'{len(d["input_ids"])} {len(d["labels"])}')
#     print(tokenizer.decode(d["input_ids"][-1]))
# print()
# print(train_dataset)
# print(eval_dataset)
# for i in range(3):
#     print(train_dataset[i])
#     print()

#     print(tokenizer.decode(train_dataset[i]["input_ids"]))
#     print(tokenizer.decode(train_dataset[i]["labels"]))
#     print()

# exit()

# 3. Define Training Arguments
training_args = TrainingArguments(
    output_dir="./results",            # Directory to save checkpoints
    num_train_epochs=3,                # Total number of training epochs
    per_device_train_batch_size=2,     # Adjust batch size for your resources
    per_device_eval_batch_size=2,      # Batch size during evaluation
    learning_rate=5e-5,                # Learning rate
    weight_decay=0.01,                 # Weight decay
    save_steps=500,                    # Save checkpoint every X steps
    logging_dir="./logs",              # Logging directory
    logging_steps=10,                  # Log every X steps
    eval_strategy="epoch",       # Evaluate at the end of each epoch
    save_total_limit=2,                # Limit the total number of checkpoints
    fp16=True,                         # Use mixed precision if available
    fp16_opt_level="O1",               # Mixed precision optimization level
    # use_cpu=True,                      # Use CPU since limited GPU resources
)

# 4. Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# 5. Train the Model
trainer.train()

# 6. Save the Fine-Tuned Model
model.save_pretrained(f"./fine_tuned_{model_name}")
tokenizer.save_pretrained(f"./fine_tuned_{model_name}")

# 7. Load the Fine-Tuned Model
model = AutoModelForCausalLM.from_pretrained(f"./fine_tuned_{model_name}")
