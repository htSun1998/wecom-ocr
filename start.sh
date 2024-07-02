kill -9 $(lsof -t -i:10900)

nohup /data/Anaconda3/envs/ocr/bin/python /data/PaddleOCR/app.py > /dev/null 2>&1 &