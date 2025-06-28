import cv2
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR
from collections import defaultdict, Counter
from PIL import ImageFont, ImageDraw, Image

# 模型路径
model_path = r"D:\code\Python\License plate recognition\License plate recognition\runs\detect\train\weights\best.pt"

# OCR 模型（轻量模式）
ocr = PaddleOCR(use_angle_cls=False, lang="ch")

# YOLO 加载
yolo_model = YOLO(model_path, task='detect')

# 绘制中文文字
def draw_chinese_text(img, text, position, font_size=30, color=(0, 255, 0)):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font_path = "C:/Windows/Fonts/simhei.ttf"  # 中文字体
    font = ImageFont.truetype(font_path, font_size)
    draw.text(position, text, font=font, fill=color[::-1])  # BGR to RGB
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# IOU计算
def iou(box1, box2):
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area != 0 else 0

# 视频路径
video_path = "D:\code\Python\License plate recognition\static_stream.mp4"
cap = cv2.VideoCapture(video_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 25.0, (int(cap.get(3)), int(cap.get(4))))

plate_history = defaultdict(list)
prev_boxes = []

frame_id = 0
skip_rate = 5  # 每 5 帧处理一次

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1
    if frame_id % skip_rate != 0:
        out.write(frame)
        continue

    results = yolo_model(frame)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    current_boxes = []

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box[:4])
        cropped = frame[y1:y2, x1:x2]
        if cropped.shape[0] < 10 or cropped.shape[1] < 10:
            continue

        # 简单 resize，取消复杂图像增强
        enhanced = cv2.resize(cropped, (256, 64))

        # OCR
        ocr_result = ocr.ocr(enhanced, cls=False)
        plate_text = ""
        if ocr_result and isinstance(ocr_result[0], list):
            try:
                for line in ocr_result:
                    for word in line:
                        text, conf = word[1]
                        if conf > 0.7:
                            plate_text += text
                plate_text = plate_text.replace("·", "")
            except:
                plate_text = ""

        # ID匹配与投票
        matched = False
        for j, pbox in enumerate(prev_boxes):
            if iou(box, pbox) > 0.5:
                matched = True
                box_id = j
                break
        if not matched:
            box_id = len(plate_history)

        if plate_text:
            plate_history[box_id].append(plate_text)
            if len(plate_history[box_id]) > 10:
                plate_history[box_id].pop(0)

        voted_text = ""
        if plate_history[box_id]:
            voted_text = Counter(plate_history[box_id]).most_common(1)[0][0]
            if voted_text and not voted_text.startswith("川"):
                voted_text = "川" + voted_text

        # 显示识别结果
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if voted_text:
            frame = draw_chinese_text(frame, voted_text, (x1, y1 - 30))

        current_boxes.append(box)

    prev_boxes = current_boxes

    # 显示和写出
    cv2.imshow("License Plate Recognition", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
