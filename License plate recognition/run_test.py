import re
import os
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
input_folder = r"D:\code\Python\val150\val"
output_folder = r"D:\testResult\150"

# 创建输出文件夹
os.makedirs(output_folder, exist_ok=True)

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
    # 对四个点进行排序：左上，右上，右下，左下
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


# 获取文件夹中的所有图片文件（按文件名排序）
def get_image_files(folder):
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))
             and f.lower().endswith(tuple(image_extensions))]
    # 按文件名排序（确保1,2,3...顺序）
    files.sort(key=lambda x: int(re.findall(r'\d+', x)[0]) if re.findall(r'\d+', x) else 0)
    return files


# 处理单张图片
def process_image(image_path, output_folder, index):
    # 加载图像
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"❌ 图像读取失败: {image_path}")
        return

    file_name = os.path.basename(image_path)

    # 执行 YOLO 检测
    results = yolo_model(original_image)

    # 遍历每个车牌框
    for i, box in enumerate(results[0].boxes.xyxy):
        x1, y1, x2, y2 = map(int, box[:4])
        padding = 5
        x1, y1 = max(x1 - padding, 0), max(y1 - padding, 0)
        x2, y2 = min(x2 + padding, original_image.shape[1]), min(y2 + padding, original_image.shape[0])

        cropped = original_image[y1:y2, x1:x2]

        # --- 透视矫正逻辑 ---
        warped = cropped  # 默认情况下，矫正图就是裁剪图
        try:
            # 1. 图像预处理，寻找轮廓
            gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # 2. 找到最大的轮廓，认为它就是车牌
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                largest_contour = max(contours, key=cv2.contourArea)

                # 3. 计算包围最大轮廓的最小旋转矩形
                rect = cv2.minAreaRect(largest_contour)

                # 4. 获取矩形的四个顶点，并转换回原图坐标系
                box_corners = cv2.boxPoints(rect)
                box_corners = np.intp(box_corners + [x1, y1])  # 转换回原图坐标

                # 5. 使用这四个角点进行透视变换
                warped = four_point_transform(original_image, box_corners)
        except:
            # 如果寻找轮廓或变换失败，则继续使用原始裁剪图
            print(f"⚠️ {file_name} 透视矫正失败，使用原始裁剪区域")
            warped = cropped
        # --- 透视矫正逻辑结束 ---

        # 图像增强（现在对摆正后的图像进行）
        img_gas = cv2.GaussianBlur(warped, (3, 3), 1.5)
        high = high_reserve(img_gas, 11, 5)
        usm1 = usm(high, 11)
        enhanced = (Overlay(img_gas / 255, usm1) * 255).astype(np.uint8)

        # 判断颜色（依然用原始裁剪图判断）
        plate_class = detect_plate_color(cropped)

        # OCR 识别 (现在对增强后的、摆正的图像进行)
        ocr_result = ocr.ocr(enhanced, cls=True)
        plate_text = ""
        # 解决NoneType错误：先判断识别结果是否有效
        if ocr_result is not None:
            try:
                plate_text = "".join([word[1][0] for line in ocr_result for word in line])
                plate_text = plate_text.replace("·", "")
                plate_text = plate_text.replace("-", "")
                plate_text = re.sub(r'[iI]', '1', plate_text)
                plate_text = re.sub(r'[oO]', '0', plate_text)
                # 保留字符数量
                plate_text = plate_text[:7] if plate_class == 1 else plate_text[:8]
            except:
                plate_text = "识别失败"
        else:
            plate_text = "识别失败"

        # 输出信息
        print(f"处理文件: {file_name}, 识别结果: {plate_text}，颜色类别: {'蓝牌(1)' if plate_class == 1 else '绿牌(0)'}")

        # 显示图像
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
        plt.title("原始车牌区域")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
        plt.title("透视矫正后")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
        plt.title(f"增强后 OCR识别: {plate_text}")
        plt.axis('off')

        plt.tight_layout()

        # 保存图像（命名格式：序号-识别.png）
        output_name = f"{index}-识别.png"
        output_path = os.path.join(output_folder, output_name)
        plt.savefig(output_path)
        plt.close()  # 关闭图形以释放内存


# 主函数：遍历文件夹中的所有图片
def main():
    image_files = get_image_files(input_folder)
    print(f"找到 {len(image_files)} 张图片，开始处理...")

    for i, image_file in enumerate(image_files, start=1):  # 从1开始计数
        print(f"\n处理第 {i}/{len(image_files)} 张图片: {image_file}")
        image_path = os.path.join(input_folder, image_file)
        process_image(image_path, output_folder, i)  # 传入序号作为参数

    print("\n所有图片处理完成!")


if __name__ == "__main__":
    main()