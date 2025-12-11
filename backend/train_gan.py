import torch
import torch.nn as nn
import torch.optim as optim
from gan_model import Generator, Discriminator
from model import AgeEstimationModel
from data_loader import get_data_loaders
import time
import os

# --- AYARLAR ---
EPOCHS = 30  # Yeni yapı daha hızlı öğrenir
LR = 0.0002
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GAN_PATH = "models/yaslandirma_gan.pth"
AGE_PATH = "models/yas_tahmin_modeli.pth"

def train_gan():
    print(f"🚀 HAFIZALI GAN Eğitimi Başlıyor... Cihaz: {DEVICE}")
    print("Not: Model yapısı değiştiği için sıfırdan başlanıyor.")
    
    # Eski modeli sil ki hata vermesin (Boyut uyuşmazlığı)
    if os.path.exists(GAN_PATH):
        try:
            os.remove(GAN_PATH)
            print("🗑️ Eski uyumsuz model silindi.")
        except: pass

    train_loader, _ = get_data_loaders()
    
    # Modeller
    generator = Generator().to(DEVICE)
    discriminator = Discriminator().to(DEVICE)
    
    # Öğretmen (Yaş Modeli)
    age_classifier = AgeEstimationModel().to(DEVICE)
    if os.path.exists(AGE_PATH):
        age_classifier.load_state_dict(torch.load(AGE_PATH, map_location=DEVICE))
        print("✅ Yaş Öğretmeni yüklendi.")
    else:
        print("❌ HATA: Yaş tahmin modeli yok!")
        return
    
    for param in age_classifier.parameters():
        param.requires_grad = False
    age_classifier.eval()

    # Kayıp Fonksiyonları
    criterion_GAN = nn.MSELoss()
    criterion_pixel = nn.L1Loss()
    criterion_age = nn.L1Loss()
    
    optimizer_G = optim.Adam(generator.parameters(), lr=LR, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=LR, betas=(0.5, 0.999))

    for epoch in range(EPOCHS):
        start_time = time.time()
        
        for i, (imgs, ages) in enumerate(train_loader):
            real_imgs = imgs.to(DEVICE)
            
            # Etiketler: 0=Genç, 1=Yaşlı
            real_age_labels = (ages >= 40).float().view(-1, 1).to(DEVICE)
            target_labels = 1 - real_age_labels # Tersine çevir
            
            # Hedef Yaşlar: Genç=15, Yaşlı=75 (Uçurum fark olsun)
            target_age_values = torch.where(target_labels == 1, 
                                            torch.tensor(75.0).to(DEVICE), 
                                            torch.tensor(15.0).to(DEVICE))

            # --- A) GENERATOR ---
            optimizer_G.zero_grad()
            
            fake_imgs = generator(real_imgs, target_labels)
            pred_fake = discriminator(fake_imgs, target_labels)
            
            # Kayıplar
            loss_GAN = criterion_GAN(pred_fake, torch.ones_like(pred_fake))
            
            # Piksel Benzerliğini SIFIRLADIK (Özgür olsun, değişsin)
            loss_pixel = criterion_pixel(fake_imgs, real_imgs) * 5.0 
            
            # Yaş Cezasını ARŞA ÇIKARDIK
            predicted_ages = age_classifier(fake_imgs)
            loss_age = criterion_age(predicted_ages, target_age_values) * 50.0 
            
            loss_G = loss_GAN + loss_pixel + loss_age
            loss_G.backward()
            optimizer_G.step()

            # --- B) DISCRIMINATOR ---
            optimizer_D.zero_grad()
            pred_real = discriminator(real_imgs, real_age_labels)
            loss_real = criterion_GAN(pred_real, torch.ones_like(pred_real))
            pred_fake = discriminator(fake_imgs.detach(), target_labels)
            loss_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
            loss_D = 0.5 * (loss_real + loss_fake)
            loss_D.backward()
            optimizer_D.step()

            if i % 100 == 0:
                print(f"Ep {epoch+1} | G_Loss: {loss_G.item():.2f} (AgeErr: {loss_age.item():.2f})")

        torch.save(generator.state_dict(), GAN_PATH)
        print(f"✅ Epoch {epoch+1} bitti. Model kaydedildi.")

    print("🎉 YENİ NESİL EĞİTİM TAMAMLANDI!")

if __name__ == "__main__":
    train_gan()