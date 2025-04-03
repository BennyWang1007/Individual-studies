from datasets import Dataset
from enum import Enum
from opencc import OpenCC
from transformers import AutoTokenizer, Trainer

from curriculum_utils import load_curriculum_datasets, DifficultyLevels

LOAD_MODEL = False
LOAD_MODEL = True

model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

MAX_LENGTH = 1024
BATCH_SIZE = 4

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"

if LOAD_MODEL:
    from transformers import AutoModelForCausalLM, TrainingArguments

    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)


def get_training_args(difficulty_level: DifficultyLevels):

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
        
    # Training arguments
    return TrainingArguments(
        output_dir="./qwen2.5-finetuned",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=3,
        logging_dir="./logs",
        logging_steps=10,
        # report_to="none",
        report_to=["tensorboard"],
        learning_rate=learning_rate,
        use_cpu=True
    )


def tokenize_function(sample: Dataset):
    tokenized_inputs = tokenizer(
        sample["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"]
    return tokenized_inputs


def check_to_filter(messages: dict) -> bool:
    """ Check if the input is too long to be processed by the model """
    tokenized_inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False
    )
    return len(tokenized_inputs) > MAX_LENGTH


def main():
    
    dataset_name = "generated_news_with_rationales_qwen2.5_32b-instruct-q6_K.jsonl"
    curriculum_datasets = []

    for difficulty_level in DifficultyLevels:
        dataset = load_curriculum_datasets(dataset_name, difficulty_level)
        texts = []
        for i, (sys_prompt, user_prompt, out_str) in enumerate(dataset):
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": out_str}
            ]

            if check_to_filter(messages):
                continue

            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                # add_generation_prompt=True
                add_generation_prompt=False
            )

            texts.append(text)
        
        curriculum_datasets.append(texts)

    # print(f"{len(curriculum_datasets)=}")
    # print(f"{len(curriculum_datasets[0])=}")
    # print(f"{curriculum_datasets[0][0]}")

    curriculum_datasets = [
        Dataset.from_dict({
            "text": [text for text in data],  # Correctly format all samples
            "difficulty": [i] * len(data)  # Repeat difficulty level for each sample
        }) for i, data in enumerate(curriculum_datasets)
    ]

    # Split the dataset into training and evaluation sets
    split_dataset = [dataset.train_test_split(test_size=0.2, seed=42) for dataset in curriculum_datasets]
    train_datasets, eval_datasets = [dataset["train"] for dataset in split_dataset], [dataset["test"] for dataset in split_dataset]

    assert isinstance(train_datasets[0], Dataset)
    assert isinstance(train_datasets[0][0], dict)
    assert isinstance(train_datasets[0][0]["text"], str)

    for difficulty_level, (train_dataset, eval_dataset) in enumerate(zip(train_datasets, eval_datasets)):
        print(f"Difficulty level {DifficultyLevels(difficulty_level)}: {len(train_dataset)} training samples, {len(eval_dataset)} evaluation samples")

    print(f"Total training samples: {sum(len(train_dataset) for train_dataset in train_datasets)}")
    print(f"Total evaluation samples: {sum(len(eval_dataset) for eval_dataset in eval_datasets)}")

    # Train progressively on harder datasets
    for i, (train_dataset, eval_dataset) in enumerate(zip(train_datasets, eval_datasets)):
        print(f"Training on difficulty level {DifficultyLevels(i)}:")
        tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
        tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True)
        # print(train_dataset[0])
        # print(eval_dataset[0])

        trainer = Trainer(
            model=model,
            # args=training_args,
            args=get_training_args(DifficultyLevels(i)),
            train_dataset=tokenized_train_dataset,
            eval_dataset=tokenized_eval_dataset
        )
        trainer.train()

    # Save the final model
    model.save_pretrained("./qwen2.5-curriculum-trained4_3")
    tokenizer.save_pretrained("./qwen2.5-curriculum-trained4_3")
    print("Training complete!")

if __name__ == "__main__":
    main()