import os
import re
import cv2
import numpy as np
import csv
from ultralytics import YOLO
from paddleocr import PaddleOCR
from PIL import ImageFont, ImageDraw, Image
import matplotlib.pyplot as plt
import matplotlib

# --- 1. 配置区域 (请根据您的实际情况修改) ---

# Matplotlib 中文支持 (确保您的系统已安装黑体 'SimHei')
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 路径配置
MODEL_PATH = r"D:\code\Python\License plate recognition\License plate recognition\runs\detect\train\weights\best.pt"
# 第一个测试集
TEST_SET_DIR_1 = r"D:\code\Python\val150\val"
# 第二个测试集 (CLPD)
TEST_SET_DIR_2 = r"D:\code\Python\License plate recognition\CLPD_1200"
# 统一的结果输出文件夹路径
OUTPUT_DIR = r"D:\test_results"
# 第二个测试集的CSV结果文件名
CSV_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "CLPD_recognition_results.csv")

# --- 2. 初始化模型 (只需执行一次) ---

print("正在初始化AI模型，请稍候...")
try:
    ocr = PaddleOCR(use_angle_cls=True, lang="ch")
    yolo_model = YOLO(MODEL_PATH, task='detect')
    print("✅ 模型初始化成功！")
except Exception as e:
    print(f"❌ 模型初始化失败: {e}")
    exit()


# --- 3. 核心处理函数 ---

def draw_chinese_text(img, text, position, font_size=50, color=(0, 255, 0)):
    """在图像上绘制中文字符。"""
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("simhei.ttf", font_size, encoding="utf-8")
    except IOError:
        font = ImageFont.load_default()
    draw.text(position, text, font=font, fill=color[::-1])
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def four_point_transform(image, pts):
    """对图像进行透视变换（摆正）。"""
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
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def process_and_save_image(img_path, output_folder):
    """处理单张图片，保存结果图，并返回识别结果文本。"""
    print(f"--- 正在处理: {img_path} ---")
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    original_image = cv2.imread(img_path)
    if original_image is None:
        print(f"❌ 图像读取失败: {img_path}")
        return "读取失败"

    results = yolo_model(original_image)
    localization_result_img = original_image.copy()

    plate_text_result = "未检测到车牌"

    if len(results[0].boxes) > 0:
        best_box_data = results[0].boxes[0]
        x1, y1, x2, y2 = map(int, best_box_data.xyxy[0])
        cropped_plate = original_image[y1:y2, x1:x2]

        warped_plate = cropped_plate
        try:
            gray = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                rect = cv2.minAreaRect(largest_contour)
                box_corners = np.intp(cv2.boxPoints(rect))
                warped_plate = four_point_transform(cropped_plate, box_corners)
        except Exception as e:
            print(f"   - 透视矫正失败: {e}")

        ocr_result = ocr.ocr(warped_plate, cls=True)
        plate_text = ""
        if ocr_result and ocr_result[0] is not None:
            plate_text = "".join([line[1][0] for line in ocr_result[0] if line])
            plate_text = re.sub(r'[·\-]', '', plate_text).upper()
            plate_text = re.sub(r'I', '1', plate_text)
            plate_text = re.sub(r'O', '0', plate_text)

        plate_text_result = plate_text if plate_text else "未识别出文字"
        print(f"   > 识别结果: {plate_text_result}")

        cv2.rectangle(localization_result_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        localization_result_img = draw_chinese_text(localization_result_img, plate_text, (x1, y1 - 60), font_size=50,
                                                    color=(0, 0, 255))

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2RGB));
        axs[0].set_title("原始车牌区域");
        axs[0].axis('off')
        axs[1].imshow(cv2.cvtColor(warped_plate, cv2.COLOR_BGR2RGB));
        axs[1].set_title("透视矫正后");
        axs[1].axis('off')
        final_display = draw_chinese_text(warped_plate.copy(), plate_text, (10, 10), font_size=40, color=(255, 0, 0))
        axs[2].imshow(cv2.cvtColor(final_display, cv2.COLOR_BGR2RGB));
        axs[2].set_title(f"OCR识别结果: {plate_text}");
        axs[2].axis('off')
        plt.tight_layout()
        recognition_path = os.path.join(output_folder, f"{base_name}-识别.jpg")
        plt.savefig(recognition_path);
        plt.close(fig)
        print(f"   ✔️ 已保存识别过程图: {recognition_path}")

    else:
        print("   - 未检测到任何车牌。")

    localization_path = os.path.join(output_folder, f"{base_name}-定位.jpg")
    cv2.imencode('.jpg', localization_result_img)[1].tofile(localization_path)
    print(f"   ✔️ 已保存定位结果图: {localization_path}")

    return plate_text_result


# --- 4. 主执行逻辑 ---
if __name__ == '__main__':
    test_sets = {
        'test_set_1': TEST_SET_DIR_1,
        'CLPD_1200': TEST_SET_DIR_2  # 使用一个有意义的名字
    }

    csv_results = []

    for set_name, test_dir in test_sets.items():
        if not test_dir or not os.path.isdir(test_dir):
            continue

        print(f"\n===== 开始处理测试集: {set_name} ({test_dir}) =====")

        test_set_output_folder = os.path.join(OUTPUT_DIR, set_name)
        os.makedirs(test_set_output_folder, exist_ok=True)

        image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        try:
            image_files.sort(key=lambda f: int(re.search(r'\d+', f).group()))
        except (AttributeError, ValueError):
            image_files.sort()
            print(f"⚠️ 文件夹 {test_dir} 中的文件名不包含数字，将按默认字母顺序处理。")

        for image_name in image_files:
            image_path = os.path.join(test_dir, image_name)
            result_text = process_and_save_image(image_path, test_set_output_folder)

            # 【新增逻辑】: 如果当前是CLPD测试集，则记录结果
            if set_name == 'CLPD_1200':
                csv_results.append([image_name, result_text])

    # 【新增逻辑】: 在所有图片处理完成后，写入CSV文件
    if csv_results:
        print(f"\n正在将CLPD测试集的结果写入CSV文件: {CSV_OUTPUT_PATH} ...")
        try:
            with open(CSV_OUTPUT_PATH, mode='w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow(["图片名", "车牌识别结果"])  # 写入表头
                writer.writerows(csv_results)
            print(f"✅ CSV文件保存成功！")
        except Exception as e:
            print(f"❌ 保存CSV文件失败: {e}")

    print("\n🎉🎉🎉 全部处理完成！ 🎉🎉🎉")