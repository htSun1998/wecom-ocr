import os
import cv2
import numpy as np
import copy
import json

from .image_utils import load_cv2
from .box_utils import convert_coordinates, Box, Rect
from .log_utils import timer


class Emoji(Rect):
    def __init__(self) -> None:
        self.relate_location = None
    
    def set_location(self, location, roi_location):
        if isinstance(location[0], list):  # 标准化
            self.relate_location = location
        else:  # 非标准化
            self.relate_location = convert_coordinates(location)
        self.location = [[x + roi_location[0][0], y + roi_location[0][1]] for [x, y] in self.relate_location]


class Text(Emoji):
    pass


class EmojiSeacher:
    def __init__(self) -> None:
        self.dir_path: str = "/data/PaddleOCR/assets/emojis"
        self.encode_path = "/data/PaddleOCR/assets/wechat_emoji.json"

    @property
    def emoji_list(self) -> list[Emoji]:
        emojis = []
        for filename in os.listdir(self.dir_path):
            file_path = os.path.join(self.dir_path, filename)
            emoji = Emoji()
            emoji.set_roi(cv2.imread(file_path))
            emoji.set_text(self.encode(filename))
            emojis.append(emoji)
        return emojis


    @property
    def encode_dict(self):
        data = None
        with open(self.encode_path, 'r') as json_file:
            data = json.load(json_file)
        return data


    @timer(message="搜索所有emoji表情")
    def find_emojis(self, box: Box) -> list[Emoji]:
        """
        定位emoji 并转换为图片绝对位置
        """
        roi, roi_location = box.roi, box.location
        results = []
        for emoji in self.emoji_list:
            locations = self.locate(emoji.roi, roi)
            if locations:
                for location in locations:
                    e = copy.deepcopy(emoji)
                    e.set_location(location, roi_location)
                    results.append(e)
        return results


    def encode(self, filename):
        emoji_name = os.path.splitext(filename)[0]
        return f"{self.encode_dict[emoji_name]}"


    def locate(self, needle_image: np.ndarray, haystack_image: np.ndarray, grayscale=True, limit=10000, region=(0, 0), step=1, confidence=0.999):
        results = []
        confidence = float(confidence)

        needle_image = load_cv2(needle_image, grayscale)
        needle_height, needle_width = needle_image.shape[:2]
        haystack_image = load_cv2(haystack_image, grayscale)

        if haystack_image.shape[0] < needle_image.shape[0] or haystack_image.shape[1] < needle_image.shape[1]:
            return results

        result = cv2.matchTemplate(haystack_image, needle_image, cv2.TM_CCOEFF_NORMED)
        match_indices = np.arange(result.size)[(result > confidence).flatten()]
        matches = np.unravel_index(match_indices[:limit], result.shape)

        if len(matches[0]) == 0:
            return results

        matchx = matches[1] * step + region[0]
        matchy = matches[0] * step + region[1]
        for x, y in zip(matchx, matchy):
            results.append([x, y, needle_width, needle_height])
        return results
