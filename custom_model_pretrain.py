import json
import os

import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
)

from curriculum_training.constants import MAX_TRAINING_INPUT_LENGTH

# from custome_model import Qwen2ForCausalLM
from transformers import Qwen2ForCausalLM


USE_GPU = True and torch.cuda.is_available()
USE_LORA = False
MAX_INPUT_LENGTH = MAX_TRAINING_INPUT_LENGTH


if USE_GPU:
    print("Using GPU for training.")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use only first GPU
    device = torch.device("cuda:0")
else:
    print("Using CPU for training.")
    device = torch.device("cpu")

MODEL_NAME = "CustomQwen2Model"
PRETRAIN_DATASET = "stored_data/processed_data.jsonl"
BATCH_SIZE = 10

model = Qwen2ForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",
)
tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.padding_side = "left"


def load_pretrain_data(dataset_path: str, max_samples=None) -> list[dict]:
    """
    Load pretraining data from a JSONL file.
    :param dataset_path: Path to the JSONL file.
    :param max_samples: Maximum number of samples to load (for testing).
    :return: List of dictionaries containing the data.
    """
    data = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            try:
                dat = json.loads(line)
                data.append(dat)
            except json.JSONDecodeError:
                print(f"Error decoding JSON: {line}")
                continue
            if max_samples and len(data) >= max_samples:
                break
    return data


def tokenize_function(sample: Dataset):
    tokenized_inputs = tokenizer(
        sample["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
        return_tensors="pt",
    )
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].clone()
    return tokenized_inputs


# Load pretraining data
data = load_pretrain_data(PRETRAIN_DATASET)
print(f"Loaded {len(data)} samples from {PRETRAIN_DATASET}")


sys_prompt = "請為新聞生成摘要："

prompts = [
    tokenizer.apply_chat_template(
        [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": item["article"]},
            {"role": "assistant", "content": item["summary"]},
        ],
        tokenize=False,
        add_generation_prompt=True
    )
    for item in data
]

prompts = [prompt for prompt in prompts if len(prompt) <= MAX_INPUT_LENGTH]
training_set = Dataset.from_dict({"text": prompts})


training_set = training_set.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
    desc="Tokenizing dataset",
)

training_set.set_format(type="torch",
                        columns=["input_ids", "attention_mask", "labels"])

# make sure the datasets are divisible by the batch size
if len(training_set) % BATCH_SIZE != 0:
    training_set = training_set.select(
        range(len(training_set) - len(training_set) % BATCH_SIZE)
    )

steps_per_epoch = len(training_set) // BATCH_SIZE

training_args = TrainingArguments(
    output_dir="output_pretrain",
    save_strategy="epoch",
    per_device_train_batch_size=BATCH_SIZE,
    logging_steps=max(steps_per_epoch // 50, 1),
    logging_dir="./logs_pretrain",
    report_to=["tensorboard"],
    learning_rate=2e-5,
    dataloader_drop_last=True,
    # save_steps=1000,
    # save_safetensors=False,
    bf16=True,
    # optim="paged_adamw_32bit",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=training_set,
    eval_dataset=None,
    tokenizer=tokenizer,
)

trainer.train()

# Save the model
model.save_pretrained(f"{MODEL_NAME}_pretrained")
tokenizer.save_pretrained(f"{MODEL_NAME}_pretrained")
