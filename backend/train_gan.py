import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from gan_model import Generator, Discriminator
from data_loader import get_data_loaders
import time
import os

# --- AYARLAR ---
EPOCHS = 50            
LR = 0.0002            
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "models/yaslandirma_gan.pth"

def train_gan():
    print(f"🚀 GAN (Ressam) Eğitimi Başlıyor... Hedef: {EPOCHS} Epoch")
    print(f"⚙️ Cihaz: {DEVICE}")
    
    # 1. Veri Yükleyici
    train_loader, _ = get_data_loaders()
    
    # 2. Modeller
    generator = Generator().to(DEVICE)
    discriminator = Discriminator().to(DEVICE)
    
    # Kaldığı yerden devam etme kontrolü
    start_epoch = 0
    if os.path.exists(MODEL_PATH):
        try:
            # Dosya bozuk mu diye kontrol et (Boyutu 0 ise sil)
            if os.path.getsize(MODEL_PATH) > 0:
                generator.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
                print("✅ Önceki eğitimden devam ediliyor...")
            else:
                print("⚠️ Model dosyası bozuk (0 byte), sıfırdan başlanıyor.")
        except:
            print("⚠️ Eski model yüklenirken hata oluştu, sıfırdan başlanıyor.")

    # 3. Ayarlar
    criterion_GAN = nn.MSELoss()
    criterion_pixel = nn.L1Loss()
    
    optimizer_G = optim.Adam(generator.parameters(), lr=LR, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=LR, betas=(0.5, 0.999))

    # --- EĞİTİM DÖNGÜSÜ ---
    for epoch in range(start_epoch, EPOCHS):
        start_time = time.time()
        
        for i, (imgs, ages) in enumerate(train_loader):
            
            real_imgs = imgs.to(DEVICE)
            real_age_labels = (ages >= 40).float().view(-1, 1).to(DEVICE)
            target_age_labels = 1 - real_age_labels 

            # --- A) GENERATOR ---
            optimizer_G.zero_grad()
            fake_imgs = generator(real_imgs, target_age_labels)
            pred_fake = discriminator(fake_imgs, target_age_labels)
            valid = torch.ones_like(pred_fake, requires_grad=False).to(DEVICE)
            
            loss_GAN = criterion_GAN(pred_fake, valid)
            loss_pixel = criterion_pixel(fake_imgs, real_imgs)
            loss_G = loss_GAN + (100 * loss_pixel) 
            
            loss_G.backward()
            optimizer_G.step()

            # --- B) DISCRIMINATOR ---
            optimizer_D.zero_grad()
            pred_real = discriminator(real_imgs, real_age_labels)
            loss_real = criterion_GAN(pred_real, torch.ones_like(pred_real))
            pred_fake = discriminator(fake_imgs.detach(), target_age_labels)
            loss_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
            
            loss_D = 0.5 * (loss_real + loss_fake)
            loss_D.backward()
            optimizer_D.step()

            if i % 50 == 0: # Logları biraz azalttık
                print(f"[Epoch {epoch+1}/{EPOCHS}] [Adım {i}] G_Loss: {loss_G.item():.4f}")

        # Her epoch sonu GÜVENLİ KAYIT
        duration = time.time() - start_time
        print(f"✅ Epoch {epoch+1} Bitti ({duration:.0f}sn). Model Kaydediliyor...")
        
        try:
            torch.save(generator.state_dict(), MODEL_PATH)
        except Exception as e:
            print(f"⚠️ DİKKAT: Model kaydedilemedi (OneDrive sorunu olabilir). Eğitim devam ediyor... Hata: {e}")

    print("🎉 Büyük Eğitim Tamamlandı!")

if __name__ == "__main__":
    train_gan()