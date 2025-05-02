import os
import random
import time

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from .constants import NWR_TRAINING_FILE, MODEL_BASE, MAX_TRAINING_INPUT_LENGTH
from .curriculum_utils import DifficultyLevels, load_curriculum_datasets
from crawler.utils import Logger, TERM_COLORS

USE_GPU = True and torch.cuda.is_available()
MAX_INPUT_LENGTH = MAX_TRAINING_INPUT_LENGTH

if USE_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use only first GPU

training_logger = Logger("training", verbose_level=3)
training_logger.info("curriculum_training.py started.")

TRAINING = True  # Set to False to skip training

# model_name = "Qwen/Qwen2.5-0.5B"
model_path = MODEL_BASE
model_name = model_path.split("/")[-1]
training_logger.info(f"Fine-tuning model: {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.padding_side = "left"

LIMIT_NEWS = True
LIMIT_NEWS_COUNT = 5000

DATASET_NAME = NWR_TRAINING_FILE
# count the number of news in the dataset
with open(DATASET_NAME, "r", encoding="utf-8") as f:
    news_count = sum(1 for _ in f)

if LIMIT_NEWS:
    news_count = min(news_count, LIMIT_NEWS_COUNT)

training_logger.info(f"Total news count: {news_count}")

BATCH_SIZE = 8
EPOCH = 3

if USE_GPU:
    training_logger.info("Using GPU for training.")
    device = torch.device("cuda:0")
else:
    training_logger.info("Using CPU for training.")
    device = torch.device("cpu")

total_tokens = 0


def get_training_args(difficulty_level: DifficultyLevels, train_dataset=None):

    match difficulty_level:
        case DifficultyLevels.TO_ZHT:
            learning_rate = 5e-5
        case DifficultyLevels.ESSENTIAL_ASPECTS:
            learning_rate = 5e-5
        case DifficultyLevels.TRIPLES:
            learning_rate = 2e-5
        case DifficultyLevels.SUMMARY:
            learning_rate = 1e-5
        case DifficultyLevels.DIRECT_SUMMARY:
            learning_rate = 1e-5
        case _:
            raise Exception("Invalid difficulty level")

    steps_per_epoch = len(train_dataset) // BATCH_SIZE

    # Training arguments
    return TrainingArguments(
        output_dir="./qwen2.5-finetuned",
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCH,
        logging_dir="./logs",
        logging_steps=max(steps_per_epoch // 10, 1),
        # report_to="none",
        report_to=["tensorboard"],
        learning_rate=learning_rate,
        dataloader_drop_last=True,
        use_cpu=(not USE_GPU),
    )


# def tokenize_function(sample: Dataset):
#     tokenized_inputs = tokenizer(
#         sample["input"],
#         padding="max_length",
#         truncation=True,
#         max_length=MAX_INPUT_LENGTH,
#         return_tensors="pt",
#     )
#     labels = tokenizer(
#         sample["output"],
#         padding="max_length",
#         truncation=True,
#         max_length=MAX_INPUT_LENGTH,
#         return_tensors="pt",
#     )
#     tokenized_inputs["labels"] = labels["input_ids"]
#     return tokenized_inputs

def tokenize_function(sample: Dataset):
    combined_texts = [
        inp + out for inp, out in zip(sample["input"], sample["output"])
    ]
    # print(f"Combined texts: {combined_texts}")
    tokenized_inputs = tokenizer(
        combined_texts,
        padding="max_length",
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
        return_tensors="pt",
    )
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].clone()
    return tokenized_inputs


def check_to_filter(messages: list[dict]) -> bool:
    """ Check if the input is too long to be processed by the model """
    global total_tokens
    tokenized_inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False
    )
    if len(tokenized_inputs) <= MAX_INPUT_LENGTH:
        total_tokens += len(tokenized_inputs)
    return len(tokenized_inputs) > MAX_INPUT_LENGTH


def check_batch_shape(dataset):
    batch = tokenizer(
        [dataset[i]["input"] for i in range(BATCH_SIZE)],
        padding="max_length",
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
        return_tensors="pt"
    )
    batch["labels"] = tokenizer(
        [dataset[i]["output"] for i in range(BATCH_SIZE)],
        padding="max_length",
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
        return_tensors="pt"
    )["input_ids"]

    for key in batch:
        training_logger.debug(f"{key}: {batch[key].shape}")

    # Check the shape of the batch
    assert batch["input_ids"].shape == (BATCH_SIZE, MAX_INPUT_LENGTH)
    assert batch["labels"].shape == (BATCH_SIZE, MAX_INPUT_LENGTH)
    assert batch["attention_mask"].shape == (BATCH_SIZE, MAX_INPUT_LENGTH)
    training_logger.log(
        "Batch shape check passed!", "SUCCESS", TERM_COLORS.GREEN
    )


def custom_data_collator(features):
    batch = {}
    for key in features[0].keys():
        if key == "text" or key == "difficulty":
            # Skip non-tensor fields
            continue

        # Convert lists to tensors if needed
        values = [f[key] for f in features]
        if isinstance(values[0], list):
            batch[key] = torch.tensor(values)
        else:
            batch[key] = torch.stack(values)

    return batch


def curriculum_training(
    train_datasets, eval_datasets, states: list[DifficultyLevels]
):
    """
    Curriculum training for the model

    Args:
        train_datasets: list of tokenized training datasets
        eval_datasets: list of tokenized evaluation datasets
        states: list of difficulty levels to train on
    """
    start_time = time.time()
    ls = len(states)  # length of the states
    if TRAINING:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
        ).to(device)

    # train progressively on harder datasets
    for difficulty_level in states:
        i = difficulty_level.value
        train_dataset, eval_dataset = train_datasets[i], eval_datasets[i]

        print_str = "Training on difficulty " + DifficultyLevels(i).name
        training_logger.info(print_str.center(80, "-"))

        if not TRAINING:
            training_logger.info("Skipping training...")
            continue

        trainer = Trainer(
            model=model,
            args=get_training_args(DifficultyLevels(i), train_dataset),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        cur_start_time = time.time()
        trainer.train()
        cur_end_time = time.time()

        training_logger.info(
            f"Training time for difficulty {DifficultyLevels(i).name}: "
            f"{cur_end_time - cur_start_time:.2f} sec"
        )

    if TRAINING:
        # save the final model
        savename = f"./{model_name}-curriculum_{news_count}news_{ls}stage_A100"
        model.save_pretrained(savename)
        tokenizer.save_pretrained(savename)
        training_logger.info(f"Model saved to {savename}")

    end_time = time.time()
    training_logger.info(
        f"Total training time ({ls} stage): {end_time - start_time:.2f} sec\n"
    )
    training_logger.log(
        f"Training {ls} stage complete!", "SUCCESS", TERM_COLORS.GREEN
    )


def curriculum_trianing_main() -> None:
    global total_tokens
    curriculum_texts: list[list[dict]] = []

    for difficulty_level in DifficultyLevels:
        dataset = load_curriculum_datasets(DATASET_NAME, difficulty_level)
        random.seed(42)
        random.shuffle(dataset)
        # for demo purpose
        # dataset = dataset[:80]

        texts: list[dict] = []  # a list contains the input and output texts

        training_logger.info(
            f"Difficulty level {difficulty_level.name}: {len(dataset)} samples"
        )

        total_tokens = 0
        cur_news_count = 0

        for i, (sys_prompt, user_prompt, out_str) in enumerate(dataset):

            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": out_str},
            ]

            if check_to_filter(messages):
                continue

            full_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )

            text = full_text.split("<|im_start|>assistant\n")
            text_in, text_out = text[0] + "<|im_start|>assistant\n", text[1]

            texts.append({
                "input": text_in,
                "output": text_out
            })

            cur_news_count += 1

            if LIMIT_NEWS and cur_news_count >= LIMIT_NEWS_COUNT:
                break

        training_logger.info(f"After filtering: {len(texts)} samples")
        curriculum_texts.append(texts)

        print(f"Tokens in {difficulty_level.name}: {total_tokens}")

    curriculum_datasets: list[Dataset] = [
        Dataset.from_dict({
            "input": [data["input"] for data in dataset],
            "output": [data["output"] for data in dataset],
            "difficulty": [i] * len(dataset)  # Repeats difficulty level
        }) for i, dataset in enumerate(curriculum_texts)
    ]
    del curriculum_texts

    # split the dataset into training and evaluation sets
    split_dataset = [
        dataset.train_test_split(test_size=0.2, seed=42)
        for dataset in curriculum_datasets
    ]
    train_datasets = [dataset["train"] for dataset in split_dataset]
    eval_datasets = [dataset["test"] for dataset in split_dataset]

    # make sure the datasets are divisible by the batch size
    for i, dataset in enumerate(train_datasets):
        if len(dataset) % BATCH_SIZE != 0:
            train_datasets[i] = dataset.select(
                range(len(dataset) - len(dataset) % BATCH_SIZE)
            )

    for i, dataset in enumerate(eval_datasets):
        if len(dataset) % BATCH_SIZE != 0:
            eval_datasets[i] = dataset.select(
                range(len(dataset) - len(dataset) % BATCH_SIZE)
            )

    for i in range(len(train_datasets)):
        assert len(train_datasets[i]) % BATCH_SIZE == 0
        assert len(eval_datasets[i]) % BATCH_SIZE == 0

    assert isinstance(train_datasets[0], Dataset)
    assert isinstance(train_datasets[0][0], dict)
    assert isinstance(train_datasets[0][0]["input"], str)
    assert isinstance(train_datasets[0][0]["output"], str)

    # tokenize the datasets
    for i in range(len(train_datasets)):
        train_dataset, eval_dataset = train_datasets[i], eval_datasets[i]
        tokenized_train_dataset = train_dataset.map(
            tokenize_function, batched=True
        )
        tokenized_eval_dataset = eval_dataset.map(
            tokenize_function, batched=True
        )
        train_datasets[i] = tokenized_train_dataset
        eval_datasets[i] = tokenized_eval_dataset

    # check the shape of the datasets
    for difficulty_level, (train_dataset, eval_dataset) in enumerate(
        zip(train_datasets, eval_datasets)
    ):
        training_logger.info(
            f"Difficulty level {DifficultyLevels(difficulty_level).name:<18}: "
            f"{len(train_dataset)} training samples, "
            f"{len(eval_dataset)} evaluation samples"
        )
        check_batch_shape(train_dataset)
        check_batch_shape(eval_dataset)

    total_train_samples = sum(len(dataset) for dataset in train_datasets)
    total_eval_samples = sum(len(dataset) for dataset in eval_datasets)
    training_logger.info(f"Total training   samples: {total_train_samples}")
    training_logger.info(f"Total evaluation samples: {total_eval_samples}")

    """ ---------------- Curriculum Training for 5 stages ---------------- """
    stages = [
        DifficultyLevels.TO_ZHT,
        DifficultyLevels.ESSENTIAL_ASPECTS,
        DifficultyLevels.TRIPLES,
        DifficultyLevels.SUMMARY,
        DifficultyLevels.DIRECT_SUMMARY,
    ]
    curriculum_training(train_datasets, eval_datasets, stages)
    torch.cuda.empty_cache()

    """ ---------------- Curriculum Training for 4 stages ---------------- """
    stages = [
        DifficultyLevels.ESSENTIAL_ASPECTS,
        DifficultyLevels.TRIPLES,
        DifficultyLevels.SUMMARY,
        DifficultyLevels.DIRECT_SUMMARY,
    ]
    curriculum_training(train_datasets, eval_datasets, stages)
    torch.cuda.empty_cache()

    """ ---------------- Curriculum Training for 1 stages ---------------- """
    stages = [
        DifficultyLevels.DIRECT_SUMMARY,
    ]
    curriculum_training(train_datasets, eval_datasets, stages)
    torch.cuda.empty_cache()

    training_logger.log("Training complete!", "SUCCESS", TERM_COLORS.GREEN)


if __name__ == "__main__":
    curriculum_trianing_main()
