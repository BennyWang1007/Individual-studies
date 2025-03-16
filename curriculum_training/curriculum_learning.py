from datasets import Dataset
from enum import Enum
from opencc import OpenCC
from transformers import AutoTokenizer, Trainer

from utils import load_generated_new_with_rationale, NewsWithRationale

LOAD_MODEL = False
LOAD_MODEL = True

model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
MAX_LENGTH = 1024

if LOAD_MODEL:
    from transformers import AutoModelForCausalLM, TrainingArguments

    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./qwen2.5-finetuned",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=1,
        logging_dir="./logs",
        logging_steps=10,
        report_to="none"
    )


class DifficultyLevels(Enum):
    TO_ZHT = 0
    ESSENTIAL_ASPECTS = 1
    TRIPLES = 2
    SUMMARY = 3
    DIRECT_SUMMARY = 4

# PREFIX_OF_DIFFICULTY_LEVELS = ["請提取新聞中的核心要素：", "請提取新聞中的三元組：", "請為新聞生成摘要："]
PREFIX_OF_DIFFICULTY_LEVELS = ["將以下文章翻譯成繁體中文：", "請提取新聞中的核心要素：", "請根據提供的核心要素，提取新聞中的三元組：", "請根據提供的核心要素和三元組，為新聞生成摘要：", "請為新聞生成摘要："]


def load_curriculum_datasets(dataset_name, difficulty_levels: DifficultyLevels) -> list[tuple[str, str]]:
    """ Load datasets with increasing difficulty. Returns a list of datasets with [system, user] pairs. """
    data: list[NewsWithRationale] = load_generated_new_with_rationale()
    ret: list[tuple[str, str]] = []

    match difficulty_levels:
        case DifficultyLevels.TO_ZHT:
            cc = OpenCC('s2twp')
            for d in data:
                ret.append((d.summary, cc.convert(d.summary)))
        case DifficultyLevels.ESSENTIAL_ASPECTS:
            for d in data:
                ret.append((d.article, ", ".join(d.essential_aspects)))
        case DifficultyLevels.TRIPLES:
            for d in data:
                ret.append((d.article + "\n\n核心要素：\n\n" + ", ".join(d.essential_aspects), ", ".join(d.triples)))
        case DifficultyLevels.SUMMARY:
            for d in data:
                ret.append((d.article + "\n\n核心要素：\n\n" + ", ".join(d.essential_aspects) + "\n\n三元組：\n\n" + ", ".join(d.triples), d.summary))
        case DifficultyLevels.DIRECT_SUMMARY:
            for d in data:
                ret.append((d.article, d.summary))
        case _:
            raise Exception("Invalid difficulty level")
    
    ret = [(f"{PREFIX_OF_DIFFICULTY_LEVELS[difficulty_levels.value]}\n\n{d[0]}", d[1]) for d in ret]

    return ret


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

    curriculum_datasets_original = [load_curriculum_datasets("generated_news_with_rationales.jsonl", level) for level in DifficultyLevels]
    # for i, dataset in enumerate(curriculum_datasets_original):
    #     print(f"Difficulty level {DifficultyLevels(i)}:")
    #     for j in range(1):
    #         print(dataset[j])
    #     print()

    # curriculum_datasets = []
    # for i, dataset in enumerate(curriculum_datasets_original):
    #     curriculum_datasets.extend([{"text": f"system: {data[0]} user: {data[1]}", "difficulty": i} for data in dataset])

    curriculum_datasets = [
        Dataset.from_dict({
            "text": [f"system: {d[0]} user: {d[1]}" for d in data],  # Correctly format all samples
            "difficulty": [i] * len(data)  # Repeat difficulty level for each sample
        }) for i, data in enumerate(curriculum_datasets_original)
    ]

    split_dataset = [dataset.train_test_split(test_size=0.2, seed=42) for dataset in curriculum_datasets]

    train_datasets = [dataset["train"] for dataset in split_dataset]
    eval_datasets = [dataset["test"] for dataset in split_dataset]

    # print(len(train_datasets[0]))
    # print(len(eval_datasets[0]))


    # Train progressively on harder datasets
    for i, (train_dataset, eval_dataset) in enumerate(zip(train_datasets, eval_datasets)):
        print(f"Training on difficulty level {DifficultyLevels(i)}:")
        tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
        tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True)
        print(train_dataset[0])
        print(eval_dataset[0])
        # continue
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train_dataset,
            eval_dataset=tokenized_eval_dataset,
        )
        trainer.train()

    # Save the final model
    model.save_pretrained("./qwen2.5-curriculum-trained")
    tokenizer.save_pretrained("./qwen2.5-curriculum-trained")
    print("Training complete!")

if __name__ == "__main__":
    main()