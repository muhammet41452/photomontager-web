import torch
import torch.nn as nn
import torch.optim as optim
from gan_model import Generator, Discriminator
from model import AgeEstimationModel # Eğittiğimiz yaş modelini çağırıyoruz
from data_loader import get_data_loaders
import time
import os

# --- AYARLAR ---
EPOCHS = 30 # Agresif eğitim olduğu için 30 epoch yeterli olabilir (GPU ile hızlı biter)
LR = 0.0002            
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GAN_PATH = "models/yaslandirma_gan.pth"
AGE_PATH = "models/yas_tahmin_modeli.pth"

def train_gan():
    print(f"🚀 GAN (Ressam) Eğitimi - AGRESİF MOD - Başlıyor... Cihaz: {DEVICE}")
    
    train_loader, _ = get_data_loaders()
    
    # 1. Modelleri Başlat
    generator = Generator().to(DEVICE)
    discriminator = Discriminator().to(DEVICE)
    
    # "Öğretmen" Modeli Yükle (Yaş Tahmin Modeli)
    # Bu model eğitilmeyecek, sadece GAN'a not verecek.
    age_classifier = AgeEstimationModel().to(DEVICE)
    if os.path.exists(AGE_PATH):
        age_classifier.load_state_dict(torch.load(AGE_PATH, map_location=DEVICE))
        print("✅ Öğretmen (Yaş Modeli) yüklendi. GAN'ı denetleyecek.")
    else:
        print("❌ HATA: Yaş tahmin modeli bulunamadı! Lütfen önce onu eğitin.")
        return
    
    # Öğretmenin bilgilerini dondur (Burası bozulmasın)
    for param in age_classifier.parameters():
        param.requires_grad = False
    age_classifier.eval()

    # Önceki GAN eğitiminden devam et
    if os.path.exists(GAN_PATH):
        try:
            generator.load_state_dict(torch.load(GAN_PATH, map_location=DEVICE))
            print("✅ Kaldığı yerden devam ediliyor...")
        except: pass

    # 2. Kayıp Fonksiyonları
    criterion_GAN = nn.MSELoss()
    criterion_pixel = nn.L1Loss()
    criterion_age = nn.L1Loss() # Yaş farkı cezası
    
    optimizer_G = optim.Adam(generator.parameters(), lr=LR, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=LR, betas=(0.5, 0.999))

    # --- EĞİTİM DÖNGÜSÜ ---
    for epoch in range(EPOCHS):
        start_time = time.time()
        
        for i, (imgs, ages) in enumerate(train_loader):
            
            real_imgs = imgs.to(DEVICE)
            
            # Etiketler: 0=Genç, 1=Yaşlı
            # Gerçek yaş 40'tan büyükse 1, küçükse 0
            real_labels = (ages >= 40).float().view(-1, 1).to(DEVICE)
            
            # Hedef: Tam tersi (Gençse yaşlandır, yaşlıysa gençleştir)
            target_labels = 1 - real_labels 
            
            # Hedef Yaş Değeri (Öğretmen için):
            # Eğer hedef "Yaşlı" ise (1), öğretmenden 60 yaş bekle.
            # Eğer hedef "Genç" ise (0), öğretmenden 20 yaş bekle.
            target_age_values = torch.where(target_labels == 1, 
                                            torch.tensor(60.0).to(DEVICE), 
                                            torch.tensor(20.0).to(DEVICE))

            # ==========================
            #  A) GENERATOR EĞİTİMİ
            # ==========================
            optimizer_G.zero_grad()

            # 1. Sahte resim üret
            fake_imgs = generator(real_imgs, target_labels)

            # 2. Eleştirmeni Kandırma Kaybı
            pred_fake = discriminator(fake_imgs, target_labels)
            loss_GAN = criterion_GAN(pred_fake, torch.ones_like(pred_fake)) # "Beni gerçek san"
            
            # 3. Piksel Benzerliği (Kimliği koru)
            # Katsayıyı 100'den 10'a düşürdük! Artık değiştirmeye daha cesur.
            loss_pixel = criterion_pixel(fake_imgs, real_imgs) * 10 
            
            # 4. YAŞ KAYBI (YENİ VE KRİTİK)
            # Üretilen resim gerçekten istenen yaşta görünüyor mu?
            predicted_ages = age_classifier(fake_imgs)
            loss_age = criterion_age(predicted_ages, target_age_values) * 0.5 # Yaş cezası
            
            # Toplam Kayıp
            loss_G = loss_GAN + loss_pixel + loss_age
            
            loss_G.backward()
            optimizer_G.step()

            # ==========================
            #  B) DISCRIMINATOR EĞİTİMİ
            # ==========================
            optimizer_D.zero_grad()

            # Gerçekleri tanı
            pred_real = discriminator(real_imgs, real_labels)
            loss_real = criterion_GAN(pred_real, torch.ones_like(pred_real))

            # Sahteleri yakala
            pred_fake = discriminator(fake_imgs.detach(), target_labels)
            loss_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
            
            loss_D = 0.5 * (loss_real + loss_fake)
            loss_D.backward()
            optimizer_D.step()

            if i % 100 == 0:
                print(f"[Epoch {epoch+1}/{EPOCHS}] [Adım {i}] G_Loss: {loss_G.item():.4f} (Age Loss: {loss_age.item():.4f})")

        # Kaydet
        duration = time.time() - start_time
        print(f"✅ Epoch {epoch+1} Bitti ({duration:.0f}sn). Model Kaydediliyor...")
        torch.save(generator.state_dict(), GAN_PATH)

    print("🎉 Agresif Eğitim Tamamlandı!")

if __name__ == "__main__":
    train_gan()