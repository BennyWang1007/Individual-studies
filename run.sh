#!/bin/bash

python local_data_generate.py

for i in {1..5}; do
    python local_data_format.py
done

python split_train_eval_data.py
python curriculum_main.py