#!/bin/bash

python local_data_generate.py &&\
python local_data_format.py &&\
python local_data_format.py &&\
python local_data_format.py &&\
python local_data_format.py &&\
python local_data_format.py &&\
python split_train_eval_data.py &&\
python curriculum_main.py