import re

import cv2
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR
import matplotlib.pyplot as plt
import matplotlib

# Matplotlib 中文支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 路径配置
model_path = r"D:\code\Python\License plate recognition\License plate recognition\runs\detect\train\weights\best.pt"
img_path = r"D:\code\Python\val150\val\33.jpg"

# 初始化模型
ocr = PaddleOCR(use_angle_cls=True, lang="ch")
yolo_model = YOLO(model_path, task='detect')

# 图像处理函数
def high_reserve(img, ksize, sigm):
    img = img * 1.0
    gauss_out = cv2.GaussianBlur(img, (ksize, ksize), sigm)
    img_out = img - gauss_out + 128
    img_out = img_out / 255.0
    # 饱和处理
    mask_1 = img_out < 0
    mask_2 = img_out > 1
    img_out = img_out * (1 - mask_1)
    img_out = img_out * (1 - mask_2) + mask_2
    return img_out

def usm(img, number):
    blur_img = cv2.GaussianBlur(img, (0, 0), number)
    usm = cv2.addWeighted(img, 1.5, blur_img, -0.5, 0)
    return usm

def Overlay(target, blend):
    mask = blend < 0.5
    img = 2 * target * blend * mask + (1 - mask) * (1 - 2 * (1 - target) * (1 - blend))
    return img

# 加载图像
original_image = cv2.imread(img_path)
if original_image is None:
    print("❌ 图像读取失败")
    exit()

# 车牌颜色识别（1=蓝，0=绿）
def detect_plate_color(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv, (100, 50, 50), (140, 255, 255))
    green_mask = cv2.inRange(hsv, (35, 50, 50), (85, 255, 255))
    blue_ratio = np.sum(blue_mask > 0) / (img.shape[0] * img.shape[1])
    green_ratio = np.sum(green_mask > 0) / (img.shape[0] * img.shape[1])
    return 1 if blue_ratio > green_ratio else 0

# 透视变换
def four_point_transform(image, pts):
    rect = np.array(pts, dtype="float32")
    width = int(np.linalg.norm(rect[1] - rect[0]))
    height = int(np.linalg.norm(rect[3] - rect[0]))
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (width, height))
    return warped

# 执行 YOLO 检测
results = yolo_model(img_path)

# 遍历每个车牌框
for box in results[0].boxes.xyxy:
    x1, y1, x2, y2 = map(int, box[:4])
    padding = 10
    x1, y1 = max(x1 - padding, 0), max(y1 - padding, 0)
    x2, y2 = min(x2 + padding, original_image.shape[1]), min(y2 + padding, original_image.shape[0])

    cropped = original_image[y1:y2, x1:x2]

    # 图像处理（高分辨率增强）
    img_gas = cv2.GaussianBlur(cropped, (3, 3), 1.5)
    high = high_reserve(img_gas, 11, 5)
    usm1 = usm(high, 11)
    enhanced = (Overlay(img_gas / 255, usm1) * 255).astype(np.uint8)

    # 判断颜色
    plate_class = detect_plate_color(cropped)

    # 透视矫正（基于原图坐标）
    pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype="float32")
    warped = four_point_transform(original_image, pts)

    # OCR 识别
    ocr_result = ocr.ocr(enhanced, cls=True)
    plate_text = "".join([word[1][0] for line in ocr_result for word in line])
    plate_text = plate_text.replace("·", "")  # 去除·
    plate_text = plate_text.replace("-", "")  # 去除-
    plate_text = re.sub(r'[iI]', '1', plate_text)  # 将 i 或 I 替换为 1
    plate_text = re.sub(r'[oO]', '0', plate_text)  # 将 o 或 O 替换为 0

    # 保留字符数量
    plate_text = plate_text[:7] if plate_class == 1 else plate_text[:8]

    # 输出信息
    print(f"识别结果: {plate_text}，颜色类别: {'蓝牌(1)' if plate_class == 1 else '绿牌(0)'}")

    # 显示图像
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    plt.title("原始车牌区域")

    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
    plt.title("增强后的图像")

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
    plt.title(f"OCR识别结果: {plate_text}")

    plt.tight_layout()
    plt.show()
