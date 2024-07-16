from loguru import logger
import cv2
import numpy as np
import copy
import os

from tools.infer.predict_system import TextSystem
from utils.messages import Response
from utils.emoji_utils import EmojiSeacher, Text
from utils.box_utils import find_title, find_boxes, merge_emoji
from utils.image_utils import compare_images, is_image, BOTTOM
from utils.log_utils import timer
from utils.args import Arguments


text_sys = TextSystem(Arguments)
emoji_seacher = EmojiSeacher()

THRESHOLD: float = 0.87


@timer(message="完整程序")
def execute_ocr(file, ip, phoneNumber):
    logger.info(f"phone_number：{phoneNumber}")
    image2 = cv2.imdecode(np.frombuffer(file, np.uint8), cv2.IMREAD_COLOR)
    # 1 搜索标题box
    title_box = find_title(image2)
    
    # 2 查找增量数据
    h_new = 0
    image1 = None
    if os.path.exists(f"/data/PaddleOCR/images/{phoneNumber}.png"):
        image1 = cv2.imread(f"/data/PaddleOCR/images/{phoneNumber}.png")
        h_new = compare_images(image1, image2)
    logger.info(f"是否重叠：{'有重叠' if h_new > 0 else '无重叠'}")

    # 3 查找符合要求的box和roi
    boxes = find_boxes(image2, h_new)
    # boxes = find_boxes(image2, 0)
    for box in boxes:
        # 第一次检测，直接用文字识别，获取结果和得分
        res, _ = text_sys.text_recognizer([box.roi])
        score = res[0][1]

        # 3.1 图片类型
        # 使用函数判断，使用色彩值判断
        if is_image(box.roi):
            box.set_text("当前类型无法识别")
            logger.info("图片：当前类型无法识别")
        # 3.2 强文本类型
        # score在阈值以上，直接使用文字识别的结果
        elif score > THRESHOLD:
            box.set_text(res[0][0])
            logger.info(f"强文本：{res[0][0]}")
        # 3.3 弱文本类型
        # score小于阈值，划定为弱文本，使用文字检测和文字识别重新计算
        else:
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
            logger.info(f"弱文本：{weak_res}")
  
    _, title, _ = text_sys(title_box)
    logger.info(f"标题：{title[0][0]}")
    if h_new != BOTTOM:
        cv2.imwrite(f"/data/PaddleOCR/images/{phoneNumber}.png", image2)
    return Response(code=0,
                    result=[box.to_list() for box in boxes],
                    ip=ip,
                    phoneNumber=phoneNumber,
                    message="success",
                    title=title[0][0])
