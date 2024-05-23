import numpy as np
import cv2
from PIL import Image


def is_image(image: np.ndarray, point=(5, 5), target_color=(233, 232, 232)):
    # 检查高度，图片高度必然超过30
    if image.shape[0] < 30:
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
                return step + h + 62  # 若找到相似部分，返回相似部分的底边纵坐标
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
