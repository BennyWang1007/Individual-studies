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
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        logging_dir="./logs",
        logging_steps=10,
        report_to="none",
        learning_rate=learning_rate,
        no_cuda=True
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
    model.save_pretrained("./qwen2.5-curriculum-trained3")
    tokenizer.save_pretrained("./qwen2.5-curriculum-trained3")
    print("Training complete!")

if __name__ == "__main__":
    main()