import torch
import cv2

# Modeli yükle
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # YOLOv5s modelini kullan

# Görüntüyü yükle
image_path = 'bisiklet.jpg'  # Buraya görüntü yolunu yaz
img = cv2.imread(image_path)

# Nesne tespiti yap
results = model(img)

# Sonuçları göster
results.show()  # Tespit edilen nesneleri göster
