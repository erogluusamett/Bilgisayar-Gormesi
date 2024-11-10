# pip install torch  # gpu kullanmak için
# pip install ultralytics # yolo kullanmak için
# pip install roboflow # roboflowa bağlanmak için

import torch
from ultralytics import YOLO
from roboflow import Roboflow

# CUDA ve GPU kullanılabilirliğini kontrol et
print("CUDA Version:", torch.version.cuda)
print("CUDA Available:", torch.cuda.is_available())

# YOLOv8 modelini başlat
model = YOLO('yolov8n.pt')  # YOLOv8 Nano modeli

# Roboflow API anahtarı ile projeyi indir
rf = Roboflow(api_key="T26WaUqWLTMm3FMgmZUI")
project = rf.workspace("kayisi-nga9w").project("kayisi-gsoho")
version = project.version(2)
dataset = version.download("yolov8")

# Eğitim işlemini başlat
if __name__ == "__main__":
    model.train(data=f"{dataset.location}/data.yaml", epochs=20)
