import urllib.request
import os

files = {
    "deploy.prototxt": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
    "res10_300x300_ssd_iter_140000.caffemodel": "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
}

print("⏳ Yüz tanıma modelleri indiriliyor...")

for filename, url in files.items():
    try:
        print(f"⬇️ İndiriliyor: {filename}...")
        urllib.request.urlretrieve(url, filename)
        print("✅ Tamamlandı.")
    except Exception as e:
        print(f"❌ Hata ({filename}): {e}")

print("\n🎉 Tüm dosyalar hazır!")