import json
# import hashlib

from curriculum_training.constants import (
    FORMATTED_NWR_FILE2,
    NWR_TRAINING_FILE,
    NWR_BENCHMARK_FILE,
    BENCHMARK_PERCENTAGE,
    FORMATTED_NWR_FILE_V2_2,
    NWR_TRAINING_V2,
    NWR_BENCHMARK_V2,

    NWR_FORMATTED_V3,
    NWR_TRAINING_V3,
    NWR_BENCHMARK_V3,
)

# def id_to_float_hash(identifier: str) -> float:
#     """Convert string ID to a deterministic float in [0, 1) using a hash."""
#     hash_bytes = hashlib.md5(identifier.encode("utf-8")).digest()
#     int_val = int.from_bytes(hash_bytes, "big")
#     return (int_val % 10**8) / 10**8


def split_train_eval_data(nwr_file, train_file, benchmark_file) -> None:
    # Check if the file exists
    try:
        with open(nwr_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        print(f"Split from file {nwr_file}")
    except FileNotFoundError:
        print(f"File {nwr_file} not found.")
        return

    # Print the number of lines in the file
    print(f"Total news count: {len(lines)}")

    train_data = []
    eval_data = []

    # Split the file into train and eval sets
    for line in lines:
        # Extract the ID from the line
        id = json.loads(line)["id"]
        if id % (1 / BENCHMARK_PERCENTAGE) < 1:
            eval_data.append(line)
        else:
            train_data.append(line)

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
    # split the data into train and eval sets

    split_train_eval_data(
        nwr_file=FORMATTED_NWR_FILE2,
        train_file=NWR_TRAINING_FILE,
        benchmark_file=NWR_BENCHMARK_FILE,
    )

    split_train_eval_data(
        nwr_file=FORMATTED_NWR_FILE_V2_2,
        train_file=NWR_TRAINING_V2,
        benchmark_file=NWR_BENCHMARK_V2,
    )

    split_train_eval_data(
        nwr_file=NWR_FORMATTED_V3,
        train_file=NWR_TRAINING_V3,
        benchmark_file=NWR_BENCHMARK_V3,
    )
