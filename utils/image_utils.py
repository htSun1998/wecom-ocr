import numpy as np
import cv2
from PIL import Image
from .log_utils import timer


TOP: int = 62
BOTTOM: int = 618
LEFT: int = 304
RIGHT: int = 1267
CHAT_COLOR: tuple = (233, 232, 232)
CHAT_POINT: tuple = (5, 5)
IMAGE_MIN_HEIGHT: int = 30
SCAN_STEP: int = 15
SCAN_MIN_HEIGHT: int = 25


def is_image(image: np.ndarray, point=CHAT_POINT, target_color=CHAT_COLOR):
    # 检查高度
    if image.shape[0] < IMAGE_MIN_HEIGHT:
        return False
    # 检查给定点是否在图片范围内
    if point[0] < image.shape[0] and point[1] < image.shape[1]:
        # 获取给定点的颜色
        color_at_point = image[point[0], point[1]]
        # 比较颜色
        if np.array_equal(color_at_point, np.array(target_color)):
            return False
        return True
    return False


@timer(message="与上一张图片比较")
def compare_images(image1, image2):
    image1 = image1[TOP:BOTTOM, LEFT:RIGHT, :]
    image2 = image2[TOP:BOTTOM, LEFT:RIGHT, :]
    height = image1.shape[0]
    if np.array_equal(image1, image2):
        return BOTTOM
    for h in range(height, SCAN_MIN_HEIGHT, -SCAN_STEP):  # mask的高度从height到25依次递减
        mask = image1[height - h:height, :, :]  # mask部分从H-h到H，即图片的尾部部分，高度从大到小
        for step in range(height - h):
            if np.array_equal(mask, image2[step:step + h, :, :]):
                return step + h + TOP  # 若找到相似部分，返回相似部分的底边纵坐标
    return 0  # 若没找到，则返回0


def cv2_to_pil(cv2_image):
    # 将 OpenCV 图像从 BGR 转换为 RGB
    cv2_image_rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    # 将 NumPy 数组转换为 PIL 图像
    pil_image = Image.fromarray(cv2_image_rgb)
    return pil_image


def load_cv2(img, grayscale=True):
    if isinstance(img, np.ndarray):
        if grayscale and len(img.shape) == 3:  # 转为灰度图
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img
