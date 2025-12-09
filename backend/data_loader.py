import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import torch

# --- AYARLAR ---
DATA_DIR = "datasets/UTKFace" # Veri setinizin olduğu klasör
BATCH_SIZE = 32               # Her seferde kaç resim işlensin? (CPU için 32 iyidir)
IMAGE_SIZE = 128              # Resimlerin küçültüleceği boyut (128x128)

class UTKFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.files = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        img_path = os.path.join(self.root_dir, file_name)
        
        # 1. Resmi Yükle
        image = Image.open(img_path).convert("RGB")
        
        # 2. Etiketi (Yaşı) Dosya Adından Çıkar
        # Dosya adı formatı: yas_cinsiyet_irk_tarih.jpg
        try:
            age = int(file_name.split('_')[0])
        except:
            age = 0 # Hatalı dosya ismi varsa 0 kabul et
            
        # 3. Dönüşümleri Uygula (Tensor'a çevir, boyutlandır vs.)
        if self.transform:
            image = self.transform(image)
            
        # 4. Yaşı da Tensor formatına çevir (Model sayı istiyor, float32)
        age_tensor = torch.tensor(age, dtype=torch.float32)
        
        return image, age_tensor

def get_data_loaders():
    # Resimlere uygulanacak işlemler zinciri
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), # Boyutu sabitle
        transforms.ToTensor(),                       # Sayısal matrise çevir (0-1 arası)
        transforms.Normalize(mean=[0.5, 0.5, 0.5],   # Renkleri dengele
                             std=[0.5, 0.5, 0.5])
    ])

    # Veri Setini Oluştur
    dataset = UTKFaceDataset(DATA_DIR, transform=transform)
    
    # Eğitim (%80) ve Test (%20) Ayırma
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    print(f"✅ Veri Seti Hazır: {len(dataset)} toplam resim.")
    print(f"   - Eğitim Seti: {len(train_dataset)} resim")
    print(f"   - Test Seti:   {len(test_dataset)} resim")

    # DataLoader'ları (Paketleyicileri) Oluştur
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, test_loader

# --- TEST BLOĞU (Dosyayı doğrudan çalıştırırsan burası çalışır) ---
if __name__ == "__main__":
    train_loader, test_loader = get_data_loaders()
    
    # İlk paketi (batch) çekip kontrol edelim
    images, ages = next(iter(train_loader))
    
    print("\n--- Örnek Paket Kontrolü ---")
    print(f"Resim Paketi Boyutu: {images.shape}  -> (Adet, Renk Kanalı, Boy, En)")
    print(f"Yaş Paketi Boyutu:   {ages.shape}")
    print(f"İlk 5 Yaş Etiketi:   {ages[:5]}")
    print("✅ DataLoader başarıyla çalışıyor!")