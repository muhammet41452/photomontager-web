import urllib.request
import os

# İndirilecek dosya
url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
filename = "haarcascade_frontalface_default.xml"

print(f"🌍 Dosya indiriliyor: {filename}...")

try:
    # Dosyayı direkt backend klasörüne indir
    urllib.request.urlretrieve(url, filename)
    
    # Dosyanın inip inmediğini kontrol et
    if os.path.exists(filename):
        print("✅ Başarılı! Dosya projenin yanına kaydedildi.")
        print(f"Konum: {os.path.abspath(filename)}")
    else:
        print("❌ Hata: İndirme başarısız oldu.")
except Exception as e:
    print(f"❌ Kritik Hata: {e}")