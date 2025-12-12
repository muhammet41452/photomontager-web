import os
import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import pandas as pd

# --- AYARLAR ---
DATA_DIR = "datasets/Adience"
BATCH_SIZE = 64
IMAGE_SIZE = 128

class AdienceDataset(Dataset):
    def __init__(self, root_dir, fold_files, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        
        # 1. ADIM: ÖNCE TÜM RESİMLERİ BUL VE HARİTALA
        print("🔍 Klasördeki tüm resimler taranıyor (Bu işlem bir kez yapılır)...")
        self.image_map = {}
        faces_dir = os.path.join(root_dir, "faces")
        
        found_count = 0
        for root, dirs, files in os.walk(faces_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    # Dosya adını (küçük harfe çevirerek) yola eşle
                    self.image_map[file.lower()] = os.path.join(root, file)
                    found_count += 1
        
        print(f"✅ Klasörde fiziksel olarak {found_count} resim bulundu.")

        # 2. ADIM: TXT DOSYALARINI OKU VE EŞLEŞTİR
        print("📄 Metin dosyaları okunuyor ve eşleştiriliyor...")
        matched_count = 0
        
        for txt_file in fold_files:
            try:
                # Adience txt dosyaları tab (\t) ile ayrılmıştır
                df = pd.read_csv(txt_file, sep="\t")
                
                for _, row in df.iterrows():
                    face_id = str(row['face_id'])
                    age_raw = str(row['age'])
                    original_image = str(row['original_image'])
                    
                    # Yaş ayrıştırma
                    age = self.parse_age(age_raw)
                    if age is None: continue 
                    
                    # Olası dosya isimlerini türet (Adience formatları)
                    # Hepsini küçük harfe çevirip arayacağız çünkü image_map öyle kaydetti
                    possible_names = [
                        f"coarse_tilt_aligned_face.{face_id}.{original_image}".lower(),
                        original_image.lower(),
                        original_image.replace(".png", ".jpg").lower(),
                        f"landmark_aligned_face.{face_id}.{original_image}".lower()
                    ]
                    
                    final_path = None
                    for name in possible_names:
                        if name in self.image_map:
                            final_path = self.image_map[name]
                            break
                    
                    if final_path:
                        self.data.append((final_path, age))
                        matched_count += 1
                        
            except Exception as e:
                print(f"⚠️ Dosya okuma hatası ({txt_file}): {e}")

        if matched_count == 0:
            print("❌ HATA: Hiçbir resim eşleştirilemedi! Klasör yapısını kontrol edin.")
            if found_count > 0:
                print(f"   İpucu: Klasördeki ilk dosya adı: {list(self.image_map.keys())[0]}")
        else:
            print(f"✅ Toplam {matched_count} resim başarıyla eşleştirildi ve eğitime hazır!")

    def parse_age(self, age_str):
        # Bu fonksiyon try-except bloğunu doğru kullanmalı
        try:
            age_str = str(age_str).strip()
            # "(25, 32)" formatı
            if '(' in age_str:
                parts = age_str.replace('(', '').replace(')', '').split(',')
                if len(parts) >= 2:
                    return (int(parts[0]) + int(parts[1])) / 2.0
            # "35" formatı
            elif age_str.replace('.', '', 1).isdigit():
                return float(age_str)
            return float(age_str)
        except:
            return None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, age = self.data[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except:
            image = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE))
            
        if self.transform:
            image = self.transform(image)
        # Yaşı tensor'a çevir
        return image, torch.tensor(age, dtype=torch.float32)

def get_data_loaders():
    if not os.path.exists(DATA_DIR):
        print(f"❌ HATA: '{DATA_DIR}' klasörü bulunamadı!")
        return None, None

    fold_files = glob.glob(os.path.join(DATA_DIR, "fold_*_data.txt"))
    if not fold_files:
        fold_files = glob.glob(os.path.join(DATA_DIR, "fold_*_data"))
    
    if not fold_files:
        print("❌ HATA: fold dosyaları bulunamadı! 'datasets/Adience' içine attığınızdan emin olun.")
        return None, None

    # Transformlar (Augmentation)
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # Dataset Oluştur
    try:
        full_dataset = AdienceDataset(DATA_DIR, fold_files, transform=train_transform)
    except Exception as e:
        print(f"Dataset oluşturulurken hata: {e}")
        return None, None
    
    if len(full_dataset) == 0:
        print("⚠️ Veri seti boş, DataLoader oluşturulamıyor.")
        return None, None

    # Bölme (%80 Eğitim, %20 Test)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    
    # DataLoader (Windows için num_workers=0)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    return train_loader, test_loader