import torch
import torch.nn as nn
import torch.optim as optim
from model import AgeEstimationModel
from data_loader import get_data_loaders
import os
import time

# --- AYARLAR ---
EPOCHS = 40
START_LR = 0.0001 # Güvenli Hız
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH = "models/yas_tahmin_modeli.pth"

def train_model():
    print(f"🚀 Sıfırdan Yaş Tahmin Eğitimi (L1 Loss) Başlıyor... Cihaz: {DEVICE}")
    
    # 1. Veri Yükle
    loaders = get_data_loaders()
    if loaders is None: return
    train_loader, test_loader = loaders
    
    # 2. Modeli Başlat
    model = AgeEstimationModel().to(DEVICE)
    
    # DİKKAT: Eski modeli yükleme kodunu kaldırdık (Sıfırdan başlasın)
    # Eğer çok isterseniz manuel açabilirsiniz ama şimdilik kapalı kalsın.
    
    # 3. Kayıp Fonksiyonu (L1 LOSS KULLANIYORUZ)
    # MSELoss karesini aldığı için hatayı çok büyütüp 'nan' yapıyordu.
    # L1Loss daha stabildir.
    criterion = nn.L1Loss() 
    
    optimizer = optim.Adam(model.parameters(), lr=START_LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    best_loss = float('inf')

    # --- EĞİTİM DÖNGÜSÜ ---
    for epoch in range(EPOCHS):
        start_time = time.time()
        
        # --- A) EĞİTİM ---
        model.train()
        running_loss = 0.0
        valid_batches = 0
        
        for i, (images, ages) in enumerate(train_loader):
            images = images.to(DEVICE)
            ages = ages.to(DEVICE).view(-1, 1)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, ages)
            
            # NAN Kontrolü
            if torch.isnan(loss):
                # Sadece uyarı ver, atla
                continue
            
            loss.backward()
            
            # Gradient Clipping (Patlama Önleyici)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            valid_batches += 1
        
        if valid_batches > 0:
            avg_train_loss = running_loss / valid_batches
        else:
            avg_train_loss = 0.0 # Hepsi nan ise

        # --- B) TEST ---
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for images, ages in test_loader:
                images = images.to(DEVICE)
                ages = ages.to(DEVICE).view(-1, 1)
                outputs = model(images)
                loss = criterion(outputs, ages)
                if not torch.isnan(loss):
                    val_loss += loss.item()
                    val_batches += 1
        
        if val_batches > 0:
            avg_val_loss = val_loss / val_batches
        else:
            avg_val_loss = float('inf')
        
        # Scheduler
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        duration = time.time() - start_time
        print(f"Epoch {epoch+1}/{EPOCHS} ({duration:.0f}sn) | "
              f"Eğitim Hata (L1): {avg_train_loss:.2f} | Test Hata (L1): {avg_val_loss:.2f} | LR: {current_lr:.6f}")

        # Kaydet
        if avg_val_loss < best_loss and avg_val_loss > 0:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"   💾 Kaydedildi. (En iyi hata: {best_loss:.2f})")

    print(f"🎉 Eğitim Bitti!")

if __name__ == "__main__":
    if not os.path.exists("models"):
        os.makedirs("models")
    train_model()