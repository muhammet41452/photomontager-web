import torch
import torch.nn as nn
import torch.optim as optim
from model import AgeEstimationModel
from data_loader import get_data_loaders
import os
import time

# --- AYARLAR ---
EPOCHS = 30
LEARNING_RATE = 0.001 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model():
    print(f"🚀 Yaş Tahmin Eğitimi (Augmentation ile) Başlıyor... Cihaz: {DEVICE}")
    
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
        
        # --- A) EĞİTİM MODU ---
        model.train() 
        running_loss = 0.0
        
        for i, (images, ages) in enumerate(train_loader):
            images = images.to(DEVICE)
            ages = ages.to(DEVICE).view(-1, 1)
            
            outputs = model(images)
            loss = criterion(outputs, ages)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)

        # --- B) TEST MODU (Validation) ---
        # Modelin hiç görmediği resimlerdeki performansı
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, ages in test_loader:
                images = images.to(DEVICE)
                ages = ages.to(DEVICE).view(-1, 1)
                outputs = model(images)
                loss = criterion(outputs, ages)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(test_loader)
        
        duration = time.time() - start_time
        print(f"✅ Epoch {epoch+1}/{EPOCHS} | Süre: {duration:.0f}sn")
        print(f"   📉 Eğitim Hatası: {avg_train_loss:.2f} | 📊 Test Hatası: {avg_val_loss:.2f}")

    print("🎉 Eğitim Tamamlandı!")
    
    # --- KAYDET ---
    save_path = "models/yas_tahmin_modeli.pth"
    torch.save(model.state_dict(), save_path)
    print(f"💾 Güçlendirilmiş Model Kaydedildi: {save_path}")

if __name__ == "__main__":
    train_model()