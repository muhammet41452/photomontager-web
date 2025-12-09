import torch
import torch.nn as nn
import torch.optim as optim
from model import AgeEstimationModel
from data_loader import get_data_loaders
import os
import time

# --- AYARLAR (GÜNCELLENDİ) ---
EPOCHS = 30           # 5'ten 30'a çıkardık. Daha uzun ama daha zeki olacak.
LEARNING_RATE = 0.001 
# Eğer GPU varsa onu kullan, yoksa CPU'ya dön
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan Cihaz: {DEVICE}") # Ekranda 'cuda' yazmalı

def train_model():
    print(f"🚀 Yaş Tahmin Eğitimi Başlıyor (ResNet)... Cihaz: {DEVICE}")
    print(f"Hedef: {EPOCHS} Epoch")
    
    # 1. Veri ve Modeli Hazırla
    train_loader, test_loader = get_data_loaders()
    model = AgeEstimationModel().to(DEVICE)
    
    # 2. Hata Hesaplayıcı ve Öğretmen
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    if not os.path.exists("models"):
        os.makedirs("models")

    # --- EĞİTİM DÖNGÜSÜ ---
    for epoch in range(EPOCHS):
        start_time = time.time()
        model.train() 
        running_loss = 0.0
        
        for i, (images, ages) in enumerate(train_loader):
            images = images.to(DEVICE)
            ages = ages.to(DEVICE).view(-1, 1)
            
            # Tahmin - Hata - Öğren
            outputs = model(images)
            loss = criterion(outputs, ages)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if (i+1) % 10 == 0:
                print(f"Ep [{epoch+1}/{EPOCHS}] Adım [{i+1}/{len(train_loader)}] Hata: {loss.item():.2f}")
        
        # Epoch Özeti
        duration = time.time() - start_time
        avg_loss = running_loss/len(train_loader)
        print(f"✅ Epoch {epoch+1} Bitti. Süre: {duration:.0f}sn. Ort. Hata: {avg_loss:.4f}")
        
        # Her 5 epochta bir ara kayıt alalım (Elektrik kesilirse vs.)
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"models/yas_tahmin_checkpoint_{epoch+1}.pth")
            print(f"💾 Ara kayıt alındı: models/yas_tahmin_checkpoint_{epoch+1}.pth")

    print("🎉 Eğitim Tamamlandı!")
    
    # --- FİNAL KAYIT ---
    save_path = "models/yas_tahmin_modeli.pth"
    torch.save(model.state_dict(), save_path)
    print(f"💾 Final Model Kaydedildi: {save_path}")

if __name__ == "__main__":
    train_model()