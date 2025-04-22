#!/bin/bash

python generate_data.py &&\
python python local_data_format.py &&\
python python local_data_format.py &&\
python python local_data_format.py &&\
python python local_data_format.py &&\
python python local_data_format.py &&\
python split_train_eval_data.py &&\
python curriculum_main.py