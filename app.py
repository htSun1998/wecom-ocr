import uvicorn
from fastapi import FastAPI, File, Form
from fastapi.middleware.cors import CORSMiddleware

from loguru import logger
from concurrent.futures import ThreadPoolExecutor
from execute import execute_ocr

app = FastAPI()

executor = ThreadPoolExecutor()


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


if __name__ == "__main__":
    logger.add("logs/{time:YYYY-MM-DD}.log",
           rotation="1 day",
           retention="3 days",
           compression="zip")
    uvicorn.run(app='app:app', host="0.0.0.0", port=10900, workers=10)
