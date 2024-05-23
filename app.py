import os
import uvicorn
from fastapi import FastAPI, File, Form
from fastapi.middleware.cors import CORSMiddleware

import cv2
import numpy as np
from loguru import logger
import copy

import tools.infer.utility as utility
from tools.infer.predict_rec import TextRecognizer
from tools.infer.predict_system import TextSystem

from utils.box_utils import find_title, find_boxes, merge_emoji
from utils.image_utils import compare_images, is_image
from utils.messages import Response
from utils.emoji_utils import EmojiSeacher


app = FastAPI()
args = utility.parse_args()
text_recognizer = TextRecognizer(args)
text_sys = TextSystem(args)
emoji_seacher = EmojiSeacher()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
    )


@app.post("/ocr/predict/single")
async def predict_single(file: bytes = File(),
                         ip: str = Form(),
                         phoneNumber: str = Form()):
    image2 = cv2.imdecode(np.frombuffer(file, np.uint8), cv2.IMREAD_COLOR)
    title_box = find_title(image2)
    h_new = 0
    
    image1 = None
    if os.path.exists(f"/data/PaddleOCR/images/{phoneNumber}.png"):
        image1 = cv2.imread(f"/data/PaddleOCR/images/{phoneNumber}.png")
        h_new = compare_images(image1, image2)
    logger.debug(f"h_new: {h_new}")
    # 查找符合要求的box和roi
    boxes = find_boxes(image2, h_new)
    
    for box in boxes:
        # 第一次检测，直接用文字识别
        res, _ = text_recognizer([box.roi])
        score = res[0][1]
        print(res)
        # 图片类型
        if is_image(box.roi):
            box.set_text("当前类型无法识别")
            logger.info("图片\t当前类型无法识别")
        # 弱文本类型
        # TODO 小表情搜索定位并整合进文本内容
        elif score < 0.87:
            box_copy = copy.deepcopy(box)  # 使用copy，否则会在原图上进行mask
            emoji_list = emoji_seacher.find_emojis(box_copy)
            for emoji in emoji_list:
                box_copy.mask(emoji.relate_location)
                print(emoji.relate_location)
                print(emoji.location)
                print(emoji.text)
            cv2.imwrite("1.png", box_copy.roi)
            _, text_list, _ = text_sys(box_copy.roi)
            print(text_list)
            text = ""
            for t in text_list:
                text += t[0] + "\n"
            box.set_text(text[:-1])
            logger.info(f"{text[:-1]}")
        # 强文本类型
        else:
            box.set_text(res[0][0])
            logger.info(f"强文本\t{res[0][0]}")
    _, title, _ = text_sys(title_box)
    logger.info(f"title\t{title[0][0]}")
    cv2.imwrite(f"/data/PaddleOCR/images/{phoneNumber}.png", image2)
    return Response(code=0,
                    # result=list(zip(boxes, textes)),
                    result=[box.to_list() for box in boxes],
                    ip=ip,
                    phoneNumber=phoneNumber,
                    message="success",
                    title=title[0][0])


if __name__ == "__main__":
    uvicorn.run(app='app:app', host="0.0.0.0", port=10900)
