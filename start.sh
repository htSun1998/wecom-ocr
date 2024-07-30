#!/bin/bash
export CUDA_HOME=/data/cuda-11.8-2
export PATH=$PATH:/data/cuda-11.8-2/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/cuda-11.8-2/lib64

for port in $(seq 10901 10910)
do
  kill -9 $(lsof -t -i:$port)
done

nohup /data/anaconda3/envs/ocr/bin/python /data/project/PaddleOCR/app.py > /dev/null 2>&1 &
