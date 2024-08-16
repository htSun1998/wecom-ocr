from fastapi import FastAPI, File, Form
from fastapi.middleware.cors import CORSMiddleware
import subprocess
import uvicorn

from loguru import logger
from concurrent.futures import ThreadPoolExecutor
from execute import execute_ocr
import json
from temp_demo import *

app = FastAPI()

executor = ThreadPoolExecutor()


logger.add("logs/{time:YYYY-MM-DD}.log",
           rotation="1 day",
           retention="3 days",
           compression="zip")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
    )


@app.post("/ocr/predict/single")
def predict_single(file: bytes = File(),
                   ip: str = Form(),
                   phoneNumber: str = Form()):
    future = executor.submit(execute_ocr, file, ip, phoneNumber)
    return future.result()

@app.post("/change/status")
def change(request: StatusMessage):
    try:
        change_status(request.status)
        return {"code": "0000", "result": "success"}
    except Exception as e:
        return {"code": "1111", "result": str(e)}


if __name__ == "__main__":
    # uvicorn.run(app='app:app', host="0.0.0.0", port=10900)
    for port in range(10901, 10911):  # 10901 ~ 10910
        subprocess.Popen(["uvicorn", "app:app", "--host", "0.0.0.0", "--port", str(port)])
