"""
This script is used to merge old and new generated data.

1. Create a backup of the file
    'generated_news_with_rationales_qwen2.5_32b-instruct-q6_K.jsonl'
    as 'generated_news_with_rationales_qwen2.5_32b-instruct-q6_K_bak.jsonl'.

2. Merge the contents of
    'generated_news_with_rationales_qwen2.5_32b-instruct-q6_K_old.jsonl'
    and 'generated_news_with_rationales_qwen2.5_32b-instruct-q6_K.jsonl'.

3. Place the old data at the beginning of the merged file.
"""

import shutil

if __name__ == "__main__":
    model_name = "qwen2.5_32b-instruct-q6_K"
    new_file = f"generated_news_with_rationales_{model_name}.jsonl"
    old_file = f"generated_news_with_rationales_{model_name}_old.jsonl"
    backup_file = f"generated_news_with_rationales_{model_name}_bak.jsonl"

    # backup the original file
    shutil.copyfile(new_file, backup_file)

    # merge old and new data
    with open(old_file, "r", encoding="utf-8") as f:
        old_data = f.readlines()

    with open(new_file, "r", encoding="utf-8") as f:
        new_data = f.readlines()

    # print(f"Old data count: {len(old_data)}")
    # print(f"New data count: {len(new_data)}")

    with open(new_file, "w", encoding="utf-8") as f:
        for line in old_data:
            f.write(line)
        for line in new_data:
            f.write(line)

    print("Merge completed.")
