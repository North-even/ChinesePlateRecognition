#coding:utf-8
from ultralytics import YOLO
# 加载预训练模型
model = YOLO("yolov8n.pt")
# Use the model
if __name__ == '__main__':
    # Use the model
    results = model.train(data="E:\datesets\PlateData\data.yaml", epochs=300, batch=64)  # 训练模型
    # 将模型转为onnx格
     #success = model.export(format='onnx')