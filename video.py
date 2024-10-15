import torch
import cv2

# Modeli yükle
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Video yakalama
cap = cv2.VideoCapture('trafik.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Nesne tespiti yap
    results = model(frame)

    # Sonuçları göster
    cv2.imshow('YOLOv5 Tespit', results.render()[0])

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
