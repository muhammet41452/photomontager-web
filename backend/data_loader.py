import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import random

# --- AYARLAR ---
DATA_DIR = "datasets/UTKFace"
BATCH_SIZE = 64 # GPU varsa batch size'ı artırabiliriz (daha hızlı olur)
IMAGE_SIZE = 128

class UTKFaceDataset(Dataset):
    def __init__(self, root_dir, file_list, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.files = file_list

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        img_path = os.path.join(self.root_dir, file_name)
        
        try:
            image = Image.open(img_path).convert("RGB")
            age = int(file_name.split('_')[0])
        except:
            # Hatalı dosya varsa siyah resim ve 0 yaş döndür
            image = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE))
            age = 0
            
        if self.transform:
            image = self.transform(image)
            
        age_tensor = torch.tensor(age, dtype=torch.float32)
        return image, age_tensor

def get_data_loaders():
    # Tüm dosya listesini al
    all_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.jpg')]
    
    # Listeyi karıştır (Her eğitimde farklı sıra olsun)
    random.shuffle(all_files)
    
    # %80 Eğitim, %20 Test olarak ayır
    split_idx = int(0.8 * len(all_files))
    train_files = all_files[:split_idx]
    test_files = all_files[split_idx:]

    # --- EĞİTİM TRANSFORMU (Veri Çoğaltma - Augmentation) ---
    # Modelin ezberlemesini önlemek için resimleri zorlaştırıyoruz
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5), # %50 ihtimalle aynala
        transforms.RandomRotation(15),          # -15 ile +15 derece arası döndür
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Işıkla oyna
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # --- TEST TRANSFORMU (Sade) ---
    # Test ederken resmi bozmuyoruz, olduğu gibi soruyoruz
    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # Datasetleri oluştur
    train_dataset = UTKFaceDataset(DATA_DIR, train_files, transform=train_transform)
    test_dataset = UTKFaceDataset(DATA_DIR, test_files, transform=test_transform)
    
    print(f"✅ Veri Seti Hazırlandı (Augmentation Aktif).")
    print(f"   - Eğitim Seti: {len(train_dataset)} resim")
    print(f"   - Test Seti:   {len(test_dataset)} resim")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, test_loader