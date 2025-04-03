"""
This script is used to merge old and new generated data.
copy a backup generated_news_with_rationales_qwen2.5_32b-instruct-q6_K.jsonl to generated_news_with_rationales_qwen2.5_32b-instruct-q6_K_bak.jsonl
merge generated_news_with_rationales_qwen2.5_32b-instruct-q6_K_old.jsonl and generated_news_with_rationales_qwen2.5_32b-instruct-q6_K.jsonl
put old data in the front"
"""

import json
import os
import shutil

if __name__ == "__main__":
    # copy a backup generated_news_with_rationales_qwen2.5_32b-instruct-q6_K.jsonl to generated_news_with_rationales_qwen2.5_32b-instruct-q6_K_bak.jsonl
    # shutil.copyfile("generated_news_with_rationales_qwen2.5_32b-instruct-q6_K.jsonl", "generated_news_with_rationales_qwen2.5_32b-instruct-q6_K_bak.jsonl")

    # merge generated_news_with_rationales_qwen2.5_32b-instruct-q6_K_old.jsonl and generated_news_with_rationales_qwen2.5_32b-instruct-q6_K.jsonl
    with open("generated_news_with_rationales_qwen2.5_32b-instruct-q6_K_old.jsonl", "r", encoding="utf-8") as f:
        old_data = f.readlines()

    with open("generated_news_with_rationales_qwen2.5_32b-instruct-q6_K.jsonl", "r", encoding="utf-8") as f:
        new_data = f.readlines()

    # print(f"Old data count: {len(old_data)}")
    # print(f"New data count: {len(new_data)}")

    with open("generated_news_with_rationales_qwen2.5_32b-instruct-q6_K.jsonl", "w", encoding="utf-8") as f:
        for line in old_data:
            f.write(line)
        for line in new_data:
            f.write(line)

    print(f"Merge completed.")