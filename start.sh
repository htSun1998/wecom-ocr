#!/bin/bash

for port in $(seq 10901 10910)
do
  kill -9 $(lsof -t -i:$port)
done

nohup /data/Anaconda3/envs/ocr/bin/python /data/PaddleOCR/app.py > /dev/null 2>&1 &
