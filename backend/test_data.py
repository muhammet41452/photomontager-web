import os
import matplotlib.pyplot as plt
import cv2

# Veri setinin yolu (Eğer klasör adınız farklıysa burayı düzeltin)
DATA_DIR = "datasets/UTKFace"

print(f"📂 Klasör taranıyor: {DATA_DIR}...")

# Klasördeki tüm dosyaları listele
if not os.path.exists(DATA_DIR):
    print("HATA: Klasör bulunamadı! Lütfen 'datasets/UTKFace' yolunu kontrol edin.")
else:
    files = os.listdir(DATA_DIR)
    print(f"✅ Toplam resim sayısı: {len(files)}")

    # İlk resmi alıp analiz edelim
    sample_file = files[0]
    print(f"\nÖrnek Dosya Adı: {sample_file}")

    # Dosya adını parçalayalım (Format: yas_cinsiyet_irk_tarih.jpg)
    try:
        parts = sample_file.split('_')
        age = parts[0]
        gender = "Erkek" if parts[1] == "0" else "Kadın"
        print(f"📊 Etiket Bilgisi -> Yaş: {age}, Cinsiyet: {gender}")
        
        # Resmi ekrana çizelim
        img_path = os.path.join(DATA_DIR, sample_file)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # OpenCV BGR okur, RGB'ye çevir
        
        plt.imshow(img)
        plt.title(f"Yas: {age} - {gender}")
        plt.axis('off')
        plt.show()
        
        print("✅ Resim başarıyla okundu ve görselleştirildi.")
        
    except Exception as e:
        print(f"⚠️ Dosya adı analiz edilirken hata oluştu: {e}")
        print("Dosya adının 'yas_cinsiyet_irk_tarih.jpg' formatında olduğundan emin olun.")