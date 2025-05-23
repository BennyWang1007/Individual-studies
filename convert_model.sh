#!/bin/bash

MODEL_NAME=Qwen2.5-0.5B-Instruct-cl_24233news_4stg_v3
QUANTIZATION=q4f16_1

OUTPUT_DIR=dist/${MODEL_NAME}-${QUANTIZATION}-MLC

mkdir -p ${OUTPUT_DIR}

mlc_llm convert_weight ./${MODEL_NAME}/ \
    --output ${OUTPUT_DIR} \
    --quantization ${QUANTIZATION} \
    --model-type qwen2 \
    --device cpu

mlc_llm gen_config ./${MODEL_NAME}/ \
    --quantization $QUANTIZATION \
    --model-type qwen2 \
    --conv-template chatml \
    --context-window-size 2048 \
    --output ${OUTPUT_DIR}

# convert the model for android
mkdir dist/libs

mlc_llm compile ./${OUTPUT_DIR}/mlc-chat-config.json \
    --device android \
    -o ./dist/libs/${MODEL_NAME}-${QUANTIZATION}-android.tar