import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from gan_model import Generator, Discriminator
from data_loader import get_data_loaders
import time
import os

# --- YENİ AYARLAR ---
EPOCHS = 50            # 3'ten 50'ye çıkardık. (Daha gerçekçi yaşlanma için şart)
LR = 0.0002            
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan Cihaz: {DEVICE}") # Ekranda 'cuda' yazmalı
MODEL_PATH = "models/yaslandirma_gan.pth"

def train_gan():
    print(f"🚀 GAN (Ressam) Eğitimi Başlıyor... Hedef: {EPOCHS} Epoch")
    print("⚠️ Bu işlem uzun sürebilir (Örn: Sabaha kadar bırakabilirsiniz).")
    
    # 1. Veri Yükleyici
    train_loader, _ = get_data_loaders()
    
    # 2. Modeller
    generator = Generator().to(DEVICE)
    discriminator = Discriminator().to(DEVICE)
    
    # Eğer önceden eğitilmiş model varsa onu yükle, sıfırdan başlama
    if os.path.exists(MODEL_PATH):
        try:
            generator.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            print("✅ Önceki eğitimden devam ediliyor...")
        except:
            print("⚠️ Eski model yüklenemedi, sıfırdan başlanıyor.")

    # 3. Ayarlar
    criterion_GAN = nn.MSELoss()
    criterion_pixel = nn.L1Loss()
    
    optimizer_G = optim.Adam(generator.parameters(), lr=LR, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=LR, betas=(0.5, 0.999))

    # --- EĞİTİM DÖNGÜSÜ ---
    for epoch in range(EPOCHS):
        start_time = time.time()
        
        for i, (imgs, ages) in enumerate(train_loader):
            
            real_imgs = imgs.to(DEVICE)
            # 40 yaş altı genç (0), üstü yaşlı (1)
            real_age_labels = (ages >= 40).float().view(-1, 1).to(DEVICE)
            target_age_labels = 1 - real_age_labels # Tam tersine dönüştür

            # --- A) GENERATOR ---
            optimizer_G.zero_grad()
            fake_imgs = generator(real_imgs, target_age_labels)
            pred_fake = discriminator(fake_imgs, target_age_labels)
            valid = torch.ones_like(pred_fake, requires_grad=False).to(DEVICE)
            
            # Kayıp: Gerçekçilik + Piksel Benzerliği
            loss_GAN = criterion_GAN(pred_fake, valid)
            loss_pixel = criterion_pixel(fake_imgs, real_imgs)
            loss_G = loss_GAN + (100 * loss_pixel) # 100 katsayısı yüzün kimliğini korur
            
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

            if i % 10 == 0:
                print(f"[Epoch {epoch+1}/{EPOCHS}] [Adım {i}] G_Loss: {loss_G.item():.4f}")

        # Her epoch sonu kaydet
        duration = time.time() - start_time
        print(f"✅ Epoch {epoch+1} Bitti ({duration:.0f}sn). Model Kaydediliyor...")
        torch.save(generator.state_dict(), MODEL_PATH)

    print("🎉 Büyük Eğitim Tamamlandı!")

if __name__ == "__main__":
    train_gan()