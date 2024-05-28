import os
import uvicorn
from fastapi import FastAPI, File, Form
from fastapi.middleware.cors import CORSMiddleware

import cv2
import numpy as np
from loguru import logger
import copy

import tools.infer.utility as utility
from tools.infer.predict_system import TextSystem

from utils.box_utils import find_title, find_boxes, merge_emoji
from utils.image_utils import compare_images, is_image
from utils.messages import Response
from utils.emoji_utils import EmojiSeacher, Text


app = FastAPI()
args = utility.parse_args()
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
    logger.info(f"phone_number\t{phoneNumber}")
    image2 = cv2.imdecode(np.frombuffer(file, np.uint8), cv2.IMREAD_COLOR)
    title_box = find_title(image2)
    
    # 查找增量数据
    h_new = 0
    image1 = None
    if os.path.exists(f"/data/PaddleOCR/images/{phoneNumber}.png"):
        image1 = cv2.imread(f"/data/PaddleOCR/images/{phoneNumber}.png")
        h_new = compare_images(image1, image2)
    logger.info(f"是否重叠\t{'yes' if h_new > 0 else 'no'}")

    # 查找符合要求的box和roi
    boxes = find_boxes(image2, h_new)
    for box in boxes:
        # 第一次检测，直接用文字识别
        res, _ = text_sys.text_recognizer([box.roi])
        score = res[0][1]

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
            text_list = []
            locations, textes, _ = text_sys(box_copy.roi)
            for l, t in zip(locations, textes):
                text = Text()
                text.set_location(l.tolist(), box_copy.location)
                text.set_text(t[0])
                text_list.append(text)
            weak_res = merge_emoji(emoji_list, text_list)
            box.set_text(weak_res)
            logger.info(f"弱文本\t{weak_res}")

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
