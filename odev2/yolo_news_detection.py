import pandas as pd
import requests
from bs4 import BeautifulSoup
from ultralytics import YOLO
import os
import cv2
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time

# CSV'den linkleri oku
csv_dosya = "linkler.csv"
veriler = pd.read_csv(csv_dosya)

# YOLO modelini yükle
model = YOLO("yolo11n.pt")

# Selenium WebDriver ayarları
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

tespit_sonuclari = []

# Her bir linkteki sayfanın ekran görüntüsünü al ve YOLO ile tespit yap
for indeks, satir in veriler.iterrows():
    url = satir['linkler']  # CSV'deki link sütunu
    print(f"{url} işleniyor...")

    try:
        driver.get(url)

        # Sayfanın tamamen yüklenmesini bekleyin
        driver.implicitly_wait(10)  # 10 saniye bekle

        # Ekran görüntüsünü al
        screenshot_path = f"screenshot_{indeks}.png"
        driver.save_screenshot(screenshot_path)
        print(f"Ekran görüntüsü {screenshot_path} olarak kaydedildi")

        # YOLO ile tespit yap
        results = model(screenshot_path)

        # Sonuçları işle
        for sonuc in results:
            kutular = sonuc.boxes
            if kutular is None or len(kutular) == 0:
                print(f"{screenshot_path} içinde nesne tespit edilmedi")
                continue

            for kutu in kutular:
                sinif = kutu.cls.cpu().numpy()  # tespit edilen sınıf
                sinif_adi = model.names[int(sinif)]  # sınıf ismi (örneğin insan)
                tespit_sonuclari.append({
                    'url': url,
                    'sinif': sinif_adi
                })

                # Tespit edilen kutuyu görselleştir
                x1, y1, x2, y2 = map(int, kutu.xyxy.cpu().numpy())
                cv2.rectangle(results.imgs[0], (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(results.imgs[0], sinif_adi, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Tespit edilen sonuçları kaydet
        sonuc_yolu = f"sonuc_{indeks}.png"
        cv2.imwrite(sonuc_yolu, results.imgs[0])  # sonucu diske kaydet
        print(f"{sonuc_yolu} olarak kaydedildi")

        # Bir süre bekleyin (örneğin 5 saniye)
        time.sleep(5)

    except Exception as e:
        print(f"{url} işlenirken hata oluştu: {str(e)}")

# Tarayıcıyı kapat
driver.quit()
print("Tespit işlemi tamamlandı.")
