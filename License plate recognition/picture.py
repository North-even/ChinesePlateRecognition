import cv2
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR
import matplotlib.pyplot as plt
import matplotlib
import os
import csv
import re

# 解决 Matplotlib 中文乱码
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 初始化 OCR 和 YOLO 模型
ocr = PaddleOCR(use_angle_cls=True, lang="ch")
model_path = r"D:\code\Python\License plate recognition\License plate recognition\runs\detect\train\weights\best.pt"
model = YOLO(model_path, task='detect')

# 图像目录
img_dir =  r"D:\code\Python\val150\val"

# 输出 CSV 路径
csv_path = os.path.join(img_dir, "识别结果.csv")

# 图像排序
img_files = [f for f in os.listdir(img_dir) if re.match(r'^\d+\.jpg$', f)]
img_files.sort(key=lambda x: int(x.split('.')[0]))

# 四点透视变换
def four_point_transform(image, pts):
    rect = np.array(pts, dtype="float32")
    width = int(np.linalg.norm(rect[1] - rect[0]))
    height = int(np.linalg.norm(rect[3] - rect[0]))
    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (width, height))
    return warped

# 简单车牌颜色识别函数（蓝牌为1，绿牌为0）
def detect_plate_color(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # 蓝色范围
    blue_mask = cv2.inRange(hsv, (100, 43, 46), (124, 255, 255))
    blue_ratio = np.sum(blue_mask > 0) / (img.shape[0] * img.shape[1])

    # 绿色范围
    green_mask = cv2.inRange(hsv, (35, 43, 46), (77, 255, 255))
    green_ratio = np.sum(green_mask > 0) / (img.shape[0] * img.shape[1])

    return 1 if blue_ratio > green_ratio else 0

# 总结果列表
results_list = []

# 遍历图像识别
for img_name in img_files:
    img_path = os.path.join(img_dir, img_name)
    original_image = cv2.imread(img_path)

    if original_image is None:
        print(f"⚠️ 无法读取图像：{img_path}")
        results_list.append([img_name, "读取失败"])
        continue

    results = model(img_path)
    plate_texts = []

    if results and results[0].boxes.xyxy.shape[0] > 0:
        # 遍历所有检测到的车牌
        for result in results[0].boxes.xyxy:
            x1, y1, x2, y2 = map(int, result[:4])
            padding = 10
            x1, y1 = max(x1 - padding, 0), max(y1 - padding, 0)
            x2, y2 = min(x2 + padding, original_image.shape[1] - 1), min(y2 + padding, original_image.shape[0] - 1)

            cropped_image = original_image[y1:y2, x1:x2]
            pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype="float32")
            warped_image = four_point_transform(original_image, pts)

            ocr_result = ocr.ocr(warped_image, cls=True)

            if ocr_result and isinstance(ocr_result, list) and len(ocr_result) > 0 and ocr_result[0]:
                plate_text = "".join([word[1][0] for line in ocr_result for word in line])
                plate_text = plate_text.replace("·", "")  # 删除·符号
                plate_text = plate_text.replace("-", "")  # 删除-符号
                plate_text = re.sub(r'[iI]', '1', plate_text)  # 将 i 或 I 替换为 1
                plate_text = re.sub(r'[oO]', '0', plate_text)  # 将 o 或 O 替换为 0

                # 识别颜色并裁剪字符数
                color_class = detect_plate_color(cropped_image)
                if color_class == 1:
                    plate_text = plate_text[:7]  # 蓝牌
                else:
                    plate_text = plate_text[:8]  # 绿牌
                plate_texts.append(plate_text)
            else:
                plate_texts.append("未识别")

    if not plate_texts:
        plate_texts.append("未检测到车牌")

    # 将所有识别到的车牌结果保存到列表
    for plate_text in plate_texts:
        results_list.append([img_name, plate_text])

# 保存 CSV 文件
# 修改为一个简单路径
csv_path =  r"D:\recognition.csv"

with open(csv_path, mode='w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    writer.writerow(["图片名", "车牌识别结果"])
    writer.writerows(results_list)

print(f"✅ 所有识别结果已保存至：{csv_path}")
