import os
os.environ["FLAGS_allocator_strategy"] = 'auto_growth'
import uvicorn
from fastapi import FastAPI, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import numpy as np
from typing import List
from loguru import logger

import tools.infer.utility as utility
from tools.infer.predict_rec import TextRecognizer
from tools.infer.predict_system import TextSystem


app = FastAPI()
args = utility.parse_args()
text_recognizer = TextRecognizer(args)
text_sys = TextSystem(args)


class Response(BaseModel):
    code: int
    result: List
    ip: str
    phoneNumber: str
    message: str
    title: str


def convert_coordinates(coordinates):
    x, y, w, h = coordinates
    x0, y0 = x, y
    x1, y1 = x + w, y
    x2, y2 = x + w, y + h
    x3, y3 = x, y + h
    transformed_coordinates = [[x0, y0], [x1, y1], [x2, y2], [x3, y3]]
    return transformed_coordinates


def find_boxes(image, h_new):
    rois = []
    boxes = []
    target_color = (247, 246, 246)
    lower = np.array(target_color, dtype="uint8")
    upper = np.array(target_color, dtype="uint8")

    # 找到图像中所有不匹配指定颜色的区域
    mask = cv2.inRange(image, lower, upper)
    mask = cv2.bitwise_not(mask)  # 取反掩膜

    # 找到白色区域的轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 定义左上角点筛选的范围
    x_right = 1150
    # x_min, x_max = 310, 1200
    y_min, y_max = 60, 640

    raw_boxes = []
    # 记录每个边界框
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        raw_boxes.append([x, y, w, h])
    # 合并边界框
    raw_boxes = merge_boxes(raw_boxes)
    # 为每个轮廓绘制边界框并进行OCR检测
    for cnt in raw_boxes:
        x, y, w, h = cnt
        if x + w > x_right or w * h < 100 or y < h_new:
            continue
        if y_min < y < y_max:
            # 提取边界框内的图像作为ROI
            roi = image[y:y + h, x:x + w]
            rois.append(roi)
            boxes.append(convert_coordinates([float(x), float(y), float(w), float(h)]))
    return rois, boxes


def get_center(box):
    # 计算并返回边界框的中心点坐标
    return [box[0] + box[2] / 2.0, box[1] + box[3] / 2.0]


def merge_boxes(boxes):
    # 将边界框按照中心点的纵坐标排序
    boxes.sort(key=lambda box: get_center(box)[1])

    merged_boxes = []
    current_group = [boxes[0]]

    for box in boxes[1:]:
        # 如果当前框的中心与上一个框的中心的纵坐标差小于5，则将其添加到当前组
        if abs(get_center(box)[1] - get_center(current_group[-1])[1]) < 5:
            current_group.append(box)
        else:
            # 否则，合并当前组的框并开始一个新组
            merged_boxes.append(current_group)
            current_group = [box]
    # 合并最后一个组
    merged_boxes.append(current_group)

    # 对每个组内的框按照中心点的横坐标排序，并合并
    result = []
    for group in merged_boxes:
        # 对组内的框按横坐标排序
        group.sort(key=lambda box: get_center(box)[0])
        # 计算合并后的边界框
        x_min = min(box[0] for box in group)
        y_min = min(box[1] for box in group)
        x_max = max(box[0] + box[2] for box in group)
        y_max = max(box[1] + box[3] for box in group)
        result.append([x_min, y_min, x_max - x_min, y_max - y_min])

    return result


def find_title(image):
    return image[10:60, 310:620]


def check_image(image, point=(5, 5), target_color=(233, 232, 232)):
    # 检查给定点是否在图片范围内
    if point[0] < image.shape[0] and point[1] < image.shape[1]:
        # 获取给定点的颜色
        color_at_point = image[point[0], point[1]]
        # 比较颜色
        return np.array_equal(color_at_point, np.array(target_color))
    else:
        return False


def compare_images(image1, image2):
    image1 = image1[62:618, 304:1267, :]
    image2 = image2[62:618, 304:1267, :]
    H = image1.shape[0]
    if np.array_equal(image1, image2):
        return 618
    for h in range(H, 25, -15):  # mask的高度从H到10依次递减
        mask = image1[H - h:H, :, :]  # mask部分从H-h到H，即图片的尾部部分，高度从大到小
        for step in range(H - h):
            if np.array_equal(mask, image2[step:step + h, :, :]):
                cv2.imwrite("mask.png", mask)
                return step + h + 62  # 若找到相似部分，返回相似部分的底边纵坐标
    return 0  # 若没找到，则返回0


@app.post("/ocr/predict/single")
async def predict_single(file: bytes = File(),
                         ip: str = Form(),
                         phoneNumber: str = Form()):
    image2 = cv2.imdecode(np.frombuffer(file, np.uint8), cv2.IMREAD_COLOR)
    title_box = find_title(image2)
    textes = []
    h_new = 0
    
    image1 = None
    if os.path.exists(f"/data/PaddleOCR/images/{phoneNumber}.png"):
        image1 = cv2.imread(f"/data/PaddleOCR/images/{phoneNumber}.png")
        h_new = compare_images(image1, image2)
    logger.debug(f"h_new: {h_new}")
    # 查找符合要求的box和roi
    rois, boxes = find_boxes(image2, h_new)
    
    for i in rois:
        # 第一次检测，直接用文字识别
        res, _ = text_recognizer([i])
        # 图片类型
        if res[0][1] < 0.2 and not check_image(i):
            textes.append("当前类型无法识别")
            logger.info("当前类型无法识别")
        # 弱文本类型
        elif res[0][1] < 0.9:
            _, text_result, _ = text_sys(i)
            text = ""
            for t in text_result:
                text += t[0] + "\n"
            textes.append(text[:-1])
            logger.info(f"{text[:-1]}")
        # 强文本类型
        else:
            textes.append(res[0][0])
            logger.info(f"{res[0][0]}")
    _, title, _ = text_sys(title_box)
    logger.info(f"title: {title[0][0]}")
    cv2.imwrite(f"/data/PaddleOCR/images/{phoneNumber}.png", image2)
    return Response(code=0,
                    result=list(zip(boxes, textes)),
                    ip=ip,
                    phoneNumber=phoneNumber,
                    message="success",
                    title=title[0][0])


def api_start(host, port):
    app.add_middleware(CORSMiddleware,
                       allow_origins=["*"],
                       allow_credentials=True,
                       allow_methods=["*"],
                       allow_headers=["*"])

    uvicorn.run(app='api_new:app', host=host, port=port)


if __name__ == "__main__":
    api_start(host='0.0.0.0', port=10800)
