# import os
import random
import time
from dataclasses import field

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
)

from .constants import MAX_TRAINING_INPUT_LENGTH, USE_LORA
from .curriculum_utils import DifficultyLevels, load_curriculum_datasets
from crawler.utils import Logger, TERM_COLORS

USE_GPU = True and torch.cuda.is_available()
MAX_INPUT_LENGTH = MAX_TRAINING_INPUT_LENGTH

if USE_LORA:
    from peft import get_peft_model, LoraConfig

training_logger = Logger("training", verbose_level=4)
training_logger.info("curriculum_training.py started.")

if USE_GPU:
    training_logger.info("Using GPU for training.")
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use only first GPU
    device = torch.device("cuda:0")
else:
    training_logger.info("Using CPU for training.")
    device = torch.device("cpu")

TRAINING = True  # Set to False to skip training
BATCH_SIZE = 8
EPOCH = 3

news_count = 0
total_tokens = {
    DifficultyLevels.TO_ZHT: 0,
    DifficultyLevels.ESSENTIAL_ASPECTS: 0,
    DifficultyLevels.TRIPLES: 0,
    DifficultyLevels.SUMMARY: 0,
    DifficultyLevels.DIRECT_SUMMARY: 0,
}
tokenizer: PreTrainedTokenizer = None


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
            # learning_rate = 1e-5  # default
            learning_rate = 5e-5  # lr-adjusted
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


def tokenize_function(sample: Dataset):
    combined_texts = [
        inp + out for inp, out in zip(sample["input"], sample["output"])
    ]
    tokenized_inputs = tokenizer(
        combined_texts,
        padding="max_length",
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
        return_tensors="pt",
    )
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].clone()
    return tokenized_inputs


def check_to_filter(messages: list[dict], df: DifficultyLevels) -> bool:
    """ Check if the input is too long to be processed by the model """
    # global total_tokens
    tokenized_inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False
    )
    token_count = len(tokenized_inputs)
    if token_count <= MAX_INPUT_LENGTH:
        total_tokens[df] += token_count
        return False
    return True


def check_batch_shape(dataset):
    if len(dataset) == 0:
        return
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


def check_batch_shapes(datasets: list[Dataset]):
    """
    Check the shape of the batches in the datasets.
    """
    for dataset in datasets:
        check_batch_shape(dataset)

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
    model_path, train_datasets, eval_datasets, states: list[DifficultyLevels],
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

    model_name = model_path.split("/")[-1]
    training_logger.info(f"Fine-tuning model: {model_name} with {ls} stages")

    if TRAINING:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )

        if USE_LORA:
            qv_config = LoraConfig(
                r=32,  # disable LoRA for q_proj and v_proj
                lora_alpha=32,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, qv_config)
            gud_config = LoraConfig(
                r=160,
                lora_alpha=160,
                target_modules=["gate_proj", "up_proj", "down_proj"],
                # lora_dropout=0.05,
                lora_dropout=0,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model.add_adapter("high_r_proj", gud_config)
            model.set_adapter("high_r_proj")

        # freeze all parameters except the self-attention/mlp layers
        # for name, param in model.named_parameters():
        #     if "self_attn" in name:  # only train self attention layers
        #     # if "mlp" in name:  # only train mlp layers
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False

        model.to(device)

    # train progressively on harder tasks
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
        savename = f"./{model_name}-cl_{news_count}news_{ls}stg_v4-lr_adj"
        if USE_LORA:
            savename += "_lora"
            model = model.merge_and_unload()  # This merges LoRA weights
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


def preprare_dataset(
    dataset_name: str,
    limit_news: int | None = None,
) -> tuple[list[Dataset], list[Dataset]]:
    """
    Prepare the dataset for training

    Args:
        dataset_name: name of the dataset
        limit_news: limit the number of news to process

    Returns:
        tuple of training and evaluation datasets
    """

    training_logger.debug("Preparing dataset: " + dataset_name)
    curriculum_texts: list[list[dict]] = []

    for df in DifficultyLevels:
        dataset = load_curriculum_datasets(dataset_name, df)
        # dataset = dataset[:80]  # for demo purpose
        random.seed(42)
        random.shuffle(dataset)

        training_logger.info(f"{len(dataset)} samples in diff level {df.name}")

        texts: list[dict] = []  # a list contains the input and output texts
        cur_news_count = 0

        for i, (sys_prompt, user_prompt, out_str) in enumerate(dataset):

            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": out_str},
            ]

            if check_to_filter(messages, df):
                continue

            full_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )

            assert isinstance(full_text, str)

            text = full_text.split("<|im_start|>assistant\n")
            text_in, text_out = text[0] + "<|im_start|>assistant\n", text[1]

            texts.append({
                "input": text_in,
                "output": text_out
            })

            cur_news_count += 1

            if limit_news and cur_news_count >= limit_news:
                break

        training_logger.info(f"After filtering: {len(texts)} samples")
        curriculum_texts.append(texts)

    for df in DifficultyLevels:
        training_logger.info(f"Tokens in {df.name}: {total_tokens[df]}")

    curriculum_datasets: list[Dataset] = [
        Dataset.from_dict({
            "input": [data["input"] for data in dataset],
            "output": [data["output"] for data in dataset],
            "difficulty": [i] * len(dataset)  # Repeats difficulty level
        }) for i, dataset in enumerate(curriculum_texts)
    ]

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

    assert isinstance(train_datasets[1], Dataset)
    assert isinstance(train_datasets[1][0], dict)
    assert isinstance(train_datasets[1][0]["input"], str)
    assert isinstance(train_datasets[1][0]["output"], str)

    return train_datasets, eval_datasets


def print_sample_texts(
    train_datasets: list[Dataset],
    num: int = 5,
    difficulty_level: DifficultyLevels = DifficultyLevels.DIRECT_SUMMARY,
) -> None:
    """
    Print sample texts from the training datasets

    Args:
        train_datasets: list of training datasets
        num: number of samples to print
        difficulty_level: difficulty level to print
    """
    dataset = train_datasets[difficulty_level.value]
    for i in range(num):
        input_text = dataset[i]["input"]
        output_text = dataset[i]["output"]
        print(f"Input: {input_text}")
        print(f"Output: {output_text}")
        print("-" * 80)


def adjust_params(model_name: str) -> None:
    """
    Adjust parameters based on the VRAM size and model name.
    Args:
        vram_size: size of the VRAM (e.g., 32 for 32GB)
        model_name: name of the model (e.g., "Qwen/Qwen2.5-0.5B-Instruct")
    """
    global BATCH_SIZE

    # simple setting for 80GB VRAM
    batch_size = BATCH_SIZE
    if "0.5B" in model_name:
        batch_size = 12
    elif not USE_LORA and "1.5B" in model_name:
        batch_size = 6
    elif "3B" in model_name:
        batch_size = 3
    elif "7B" in model_name:
        batch_size = 2

    vram_size = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

    BATCH_SIZE = int(batch_size * vram_size // 80)
    assert BATCH_SIZE > 0, "Batch size must be greater than 0"
    assert isinstance(BATCH_SIZE, int), "Batch size must be an integer"

    training_logger.info(f"Adjusted batch size: {BATCH_SIZE} for VRAM size: "
                         f"{vram_size}GB and model: {model_name}")


def curriculum_trianing_main(
    model_path: str,
    dataset_name: str,
    limit_news: int | None = None,
    stages_list: list[list[DifficultyLevels]] = field(default_factory=list),
    to_train: bool = True,
) -> None:
    global tokenizer, news_count, TRAINING

    news_count = 0
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"
    TRAINING = to_train

    adjust_params(model_name=model_path)

    with open(dataset_name, "r", encoding="utf-8") as f:
        news_count = sum(1 for _ in f)

    if limit_news:
        news_count = min(news_count, limit_news)

    training_logger.info(f"Total news count: {news_count}")

    # load the datasets
    train_datasets, eval_datasets = preprare_dataset(
        dataset_name, limit_news=limit_news
    )

    # print sample texts
    # for df in DifficultyLevels:
    #     print_sample_texts(train_datasets, num=3, difficulty_level=df)

    training_logger.info(f"Loaded {len(train_datasets[0])} datasets")

    # tokenize the datasets
    train_datasets = [
        dataset.map(tokenize_function, batched=True)
        for dataset in train_datasets
    ]
    eval_datasets = [
        dataset.map(tokenize_function, batched=True)
        for dataset in eval_datasets
    ]

    # log dataset sizes and check batch shapes
    for diff_level, datasets in enumerate(zip(train_datasets, eval_datasets)):
        train_dataset, eval_dataset = datasets
        training_logger.info(
            f"Difficulty level {DifficultyLevels(diff_level).name:<18}: "
            f"{len(train_dataset)} train, {len(eval_dataset)} eval samples"
        )

    check_batch_shapes(train_datasets + eval_datasets)

    total_train_samples = sum(len(dataset) for dataset in train_datasets)
    total_eval_samples = sum(len(dataset) for dataset in eval_datasets)
    training_logger.info(f"Total training   samples: {total_train_samples}")
    training_logger.info(f"Total evaluation samples: {total_eval_samples}")

    # curriculum training
    for stages in stages_list:
        curriculum_training(model_path, train_datasets, eval_datasets, stages)
        torch.cuda.empty_cache()

    training_logger.log("Training complete!", "SUCCESS", TERM_COLORS.GREEN)


# sample usage
if __name__ == "__main__":
    curriculum_trianing_main(
        model_path="Qwen/Qwen2.5-0.5B-Instruct",
        dataset_name="formatted_nwr_training.jsonl",
        limit_news=None,
        stages_list=[
            # [DifficultyLevels.TO_ZHT],
            [DifficultyLevels.ESSENTIAL_ASPECTS],
            [DifficultyLevels.TRIPLES],
            [DifficultyLevels.SUMMARY],
            [DifficultyLevels.DIRECT_SUMMARY],
        ],
        to_train=True,
    )
