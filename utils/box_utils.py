import numpy as np
import cv2


class Rect:
    def __init__(self) -> None:
        self.location = None
        self.roi = None
        self.text = None
    
    def set_text(self, text):
        self.text = text
    
    def set_location(self, location):
        self.location = location
    
    def set_roi(self, roi):
        self.roi = roi
    
    @property
    def vertical_center(self):
        top_left, _, bottom_right, _ = self.location
        return (top_left[1] + bottom_right[1]) / 2


class Box(Rect):
    def to_list(self):
        return [self.location, self.text]
    
    def mask(self, field):
        left = field[0][0]
        right = field[1][0]
        top = field[0][1]
        bottom = field[2][1]
        # 由于小表情截图不完全，有一些修正
        self.roi[top - 1:bottom + 1, left:right + 2, :] = [233, 232, 232]


def convert_coordinates(coordinates):
    x, y, w, h = coordinates
    x0, y0 = x, y
    x1, y1 = x + w, y
    x2, y2 = x + w, y + h
    x3, y3 = x, y + h
    transformed_coordinates = [[x0, y0], [x1, y1], [x2, y2], [x3, y3]]
    return transformed_coordinates


def find_boxes(image, h_new) -> list[Box]:
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
    y_min, y_max = 60, 610

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
            box = Box()
            box.set_location(convert_coordinates([float(x), float(y), float(w), float(h)]))
            box.set_roi(image[y:y + h, x:x + w])
            boxes.append(box)
    return boxes


def get_center(box):
    # 计算并返回边界框的中心点坐标
    return [box[0] + box[2] / 2.0, box[1] + box[3] / 2.0]


def merge_boxes(boxes: list[int]):
    """
    合并同一行的box
    """
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


def boxes_in_same_row(box1: Rect, box2: Rect, threshold=10):
    return abs(box1.vertical_center - box2.vertical_center) < threshold


def merge_emoji(emoji_list: list[Rect], text_list: list[Rect]):  # TODO
    boxes = emoji_list + text_list
    boxes.sort(key=lambda x: x.location[0][1])
    rows: list[list[Rect]] = []
    current_row = [boxes[0]]

    for current_box in boxes[1:]:
        if boxes_in_same_row(current_box, current_row[-1]):
            current_row.append(current_box)
        else:
            rows.append(current_row)
            current_row = [current_box]
    rows.append(current_row)
    for row in rows:
        row.sort(key=lambda x: x.location[0][0])
    merged_rows = [''.join(box.text for box in row) for row in rows]
    final_text = '\n'.join(merged_rows)
    return final_text
