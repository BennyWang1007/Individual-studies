
import random

from curriculum_training.constants import (
    FORMATTED_NWR_FILE,
    NWR_TRAINING_FILE,
    NWR_BENCHMARK_FILE,
    BENCHMARK_PERCENTAGE,
)

if __name__ == "__main__":
    # Check if the file exists
    try:
        with open(FORMATTED_NWR_FILE, "r", encoding="utf-8") as f:
            news_count = sum(1 for _ in f)
    except FileNotFoundError:
        print(f"File {FORMATTED_NWR_FILE} not found.")
        exit(1)

    # Print the number of lines in the file
    print(f"Total news count: {news_count}")

    # Split the file into train and eval sets
    with open(FORMATTED_NWR_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Shuffle the lines to ensure randomness
    random.seed(42)  # For reproducibility
    random.shuffle(lines)

    # Split the data into train and eval sets
    split_index = int(len(lines) * (1 - BENCHMARK_PERCENTAGE))
    train_data = lines[:split_index]
    eval_data = lines[split_index:]

    # Write the train and eval data to separate files
    with open(NWR_TRAINING_FILE, "w", encoding="utf-8") as f:
        f.writelines(train_data)

    with open(NWR_BENCHMARK_FILE, "w", encoding="utf-8") as f:
        f.writelines(eval_data)

    print(
        "Data split completed. Train and eval data saved to "
        f"{NWR_TRAINING_FILE} and {NWR_BENCHMARK_FILE}, respectively."
    )
