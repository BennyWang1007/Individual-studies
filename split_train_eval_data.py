from curriculum_training.constants import (
    FORMATTED_NWR_FILE2,
    NWR_TRAINING_FILE,
    NWR_BENCHMARK_FILE,
    BENCHMARK_PERCENTAGE,
)

if __name__ == "__main__":
    # Check if the file exists
    try:
        with open(FORMATTED_NWR_FILE2, "r", encoding="utf-8") as f:
            news_count = sum(1 for _ in f)
    except FileNotFoundError:
        print(f"File {FORMATTED_NWR_FILE2} not found.")
        exit(1)

    # Print the number of lines in the file
    print(f"Total news count: {news_count}")

    train_data = []
    eval_data = []

    # Split the file into train and eval sets
    cummu_num = 0.0
    with open(FORMATTED_NWR_FILE2, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            if cummu_num % (1 / BENCHMARK_PERCENTAGE) < 1:
                eval_data.append(line)
            else:
                train_data.append(line)
            cummu_num += 1
    # Check if the split is correct
    print(f"Train data count: {len(train_data)}")
    print(f"Eval data count: {len(eval_data)}")

    # Write the train and eval data to separate files
    with open(NWR_TRAINING_FILE, "w", encoding="utf-8") as f:
        f.writelines(train_data)

    with open(NWR_BENCHMARK_FILE, "w", encoding="utf-8") as f:
        f.writelines(eval_data)

    print(
        "Data split completed. Train and eval data saved to "
        f"{NWR_TRAINING_FILE} and {NWR_BENCHMARK_FILE}, respectively."
    )
