import urllib.request
import os

url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
filename = "haarcascade_frontalface_default.xml"

print(f"İndiriliyor: {filename}...")

try:
    urllib.request.urlretrieve(url, filename)
    if os.path.exists(filename):
        print("✅ Başarılı! Dosya indirildi.")
    else:
        print("❌ Hata: Dosya oluşturulamadı.")
except Exception as e:
    print(f"❌ İndirme hatası: {e}")