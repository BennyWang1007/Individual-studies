from curriculum_training.constants import (
    FORMATTED_NWR_FILE2,
    NWR_TRAINING_FILE,
    NWR_BENCHMARK_FILE,
    BENCHMARK_PERCENTAGE,
    BETTER_FORMATTED_NWR_FILE2,
    BETTER_NWR_TRAINING_FILE,
    BETTER_NWR_BENCHMARK_FILE,
)


def split_train_eval_data(nwr_file, train_file, benchmark_file) -> None:
    # Check if the file exists
    try:
        with open(nwr_file, "r", encoding="utf-8") as f:
            news_count = sum(1 for _ in f)
    except FileNotFoundError:
        print(f"File {nwr_file} not found.")
        return

    # Print the number of lines in the file
    print(f"Total news count: {news_count}")

    train_data = []
    eval_data = []

    # Split the file into train and eval sets
    cummu_num = 0.0
    with open(nwr_file, "r", encoding="utf-8") as f:
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
    with open(train_file, "w", encoding="utf-8") as f:
        f.writelines(train_data)

    with open(benchmark_file, "w", encoding="utf-8") as f:
        f.writelines(eval_data)

    print(
        "Data split completed. Train and eval data saved to "
        f"{train_file} and {benchmark_file}, respectively."
    )


if __name__ == "__main__":
    # Split the data into train and eval sets
    split_train_eval_data(
        nwr_file=FORMATTED_NWR_FILE2,
        train_file=NWR_TRAINING_FILE,
        benchmark_file=NWR_BENCHMARK_FILE,
    )

    split_train_eval_data(
        nwr_file=BETTER_FORMATTED_NWR_FILE2,
        train_file=BETTER_NWR_TRAINING_FILE,
        benchmark_file=BETTER_NWR_BENCHMARK_FILE,
    )
