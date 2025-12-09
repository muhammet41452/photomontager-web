import torch
import cv2
import numpy as np

print("--- Kurulum Testi Başladı ---")

# 1. PyTorch Kontrolü
print(f"PyTorch Versiyonu: {torch.__version__}")

# GPU (CUDA) Kontrolü
if torch.cuda.is_available():
    print("✅ GPU (CUDA) Kullanılabilir! Model eğitimi çok hızlı olacak.")
    print(f"Ekran Kartı: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ GPU bulunamadı veya PyTorch CPU modunda çalışıyor.")
    print("   Model eğitimi biraz yavaş olabilir ama çalışacaktır.")

# 2. OpenCV Kontrolü
print(f"OpenCV Versiyonu: {cv2.__version__}")

# 3. Basit Tensör Testi
x = torch.rand(5, 3)
print("Örnek Tensör Oluşturuldu:")
print(x)

print("--- Test Başarıyla Tamamlandı ---")