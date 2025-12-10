from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps, ImageFilter
import io
import os
import cv2
import numpy as np
import uuid
import mediapipe as mp

from model import AgeEstimationModel
from gan_model import Generator

app = FastAPI()

# --- AYARLAR ---
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIRECTORY = "uploaded_images"
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

app.mount("/uploaded_images", StaticFiles(directory=UPLOAD_DIRECTORY), name="uploaded_images")

# --- MODELLER ---
DEVICE = torch.device("cpu") # Render'da CPU kullanılır
age_model = AgeEstimationModel()
gan_model = Generator()
AGE_PATH = "models/yas_tahmin_modeli.pth"
GAN_PATH = "models/yaslandirma_gan.pth"

print("⏳ Sistem Başlatılıyor...")
try:
    if os.path.exists(AGE_PATH):
        age_model.load_state_dict(torch.load(AGE_PATH, map_location=DEVICE))
        age_model.to(DEVICE).eval()
    if os.path.exists(GAN_PATH):
        gan_model.load_state_dict(torch.load(GAN_PATH, map_location=DEVICE))
        gan_model.to(DEVICE).eval()
    print("✅ Modeller Yüklendi.")
except Exception as e:
    print(f"⚠️ Model Hatası: {e}")

# --- YÜZ AĞI (FACE MESH) KURULUMU ---
# Bu teknoloji yüzü kare olarak değil, 468 noktalı bir harita olarak görür.
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# --- YARDIMCI FONKSİYONLAR ---

def get_face_mask_and_bbox(image_np):
    """
    Yüzün tam sınırlarını (Maske) ve kare çerçevesini (BBox) çıkarır.
    """
    results = face_mesh.process(image_np)
    
    if not results.multi_face_landmarks:
        return None, None
    
    landmarks = results.multi_face_landmarks[0].landmark
    h, w, _ = image_np.shape
    
    # Yüzdeki tüm noktaları piksel koordinatına çevir
    points = np.array([(int(l.x * w), int(l.y * h)) for l in landmarks], np.int32)
    
    # 1. MASKE OLUŞTUR (Yüzün gerçek sınırları)
    # Convex Hull: Noktaları saran en dış çemberi bulur
    hull = cv2.convexHull(points)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    
    # 2. KARE ÇERÇEVE OLUŞTUR (GAN için gerekli)
    rect = cv2.boundingRect(hull) # (x, y, w, h)
    
    return mask, rect

def tensor_to_cv2(tensor):
    image = tensor.detach().cpu().squeeze(0)
    image = image * 0.5 + 0.5
    image = image.clamp(0, 1)
    image = image.permute(1, 2, 0).numpy()
    image = (image * 255).astype(np.uint8)
    return image # RGB Formatında döner (PIL kaynaklı olduğu için)

# --- ANA İŞLEM ---

@app.post("/analyze/")
async def analyze_photo(
    file: UploadFile = File(...), 
    target_mode: str = Form("age_estimation") 
):
    try:
        # 1. Dosya Okuma
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_np = np.array(pil_image) # RGB
        
        # 2. Yüz Maskesi ve Koordinatları Al
        mask, rect = get_face_mask_and_bbox(img_np)
        
        final_output = img_np.copy()
        
        if rect:
            (x, y, w, h) = rect
            
            # --- Padding (Biraz genişletelim) ---
            p = int(w * 0.15) 
            x1 = max(0, x - p)
            y1 = max(0, y - int(p*1.5)) # Alın kısmı için ekstra pay
            x2 = min(img_np.shape[1], x + w + p)
            y2 = min(img_np.shape[0], y + h + p)
            
            face_roi = img_np[y1:y2, x1:x2]
            
            # 3. Modelleri Çalıştır
            face_pil = Image.fromarray(face_roi)
            face_input = face_pil.resize((128, 128), Image.Resampling.LANCZOS)
            
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            tensor = transform(face_input).unsqueeze(0).to(DEVICE)
            
            result = {}

            if target_mode == "age_estimation":
                with torch.no_grad():
                    pred = age_model(tensor)
                    age = pred.item()
                result["type"] = "prediction"
                result["age"] = round(age, 1)
                
                # Sadece tahminde kare çizelim (Artık çok daha doğru yerde olacak)
                # OpenCV işlemleri için RGB kopyası üzerinde çalışıyoruz
                cv2.rectangle(final_output, (x1, y1), (x2, y2), (0, 255, 0), 3)

            else:
                # --- PROFESYONEL MONTAJ (SEAMLESS BLENDING) ---
                target_label = 1.0 if target_mode == "make_old" else 0.0
                label_tensor = torch.tensor([target_label]).to(DEVICE)
                
                with torch.no_grad():
                    fake_tensor = gan_model(tensor, label_tensor)
                
                gen_face = tensor_to_cv2(fake_tensor) # (128, 128)
                
                # A) GAN çıktısını orijinal kesim boyutuna büyüt
                gen_face_resized = cv2.resize(gen_face, (x2-x1, y2-y1))
                
                # B) Maskeyi de kesip hazırla (Yumuşak geçiş için blurla)
                mask_roi = mask[y1:y2, x1:x2]
                mask_roi = cv2.GaussianBlur(mask_roi, (15, 15), 10) # Yumuşatma
                
                # C) Alfa Karışımı (Alpha Blending)
                # Orijinal parça
                roi_bg = final_output[y1:y2, x1:x2].astype(float)
                # Yeni yüz
                roi_fg = gen_face_resized.astype(float)
                
                # Maskeyi 0-1 arasına çek ve 3 kanallı yap
                alpha = mask_roi.astype(float) / 255.0
                alpha = np.stack([alpha]*3, axis=-1)
                
                # Formül: (Yeni * Maske) + (Eski * (1-Maske))
                # Bu sayede sadece yüz değişir, arka plan ve saç kenarları orijinal kalır.
                blended = (roi_fg * alpha) + (roi_bg * (1.0 - alpha))
                
                final_output[y1:y2, x1:x2] = blended.astype(np.uint8)
                
                result["type"] = "transformation"
        else:
            # Yüz Bulunamadı (Fallback: Merkez)
            # ... (Merkezi kesim kodları buraya eklenebilir ama FaceMesh genelde bulur)
            result = {"type": "error", "age": 0} # Basit hata yönetimi
            print("⚠️ Yüz bulunamadı.")

        # 4. Kaydet
        # OpenCV BGR kaydeder, elimizdeki RGB. Dönüştür:
        final_output_bgr = cv2.cvtColor(final_output, cv2.COLOR_RGB2BGR)
        out_filename = f"result_{uuid.uuid4().hex[:8]}.jpg"
        save_path = os.path.join(UPLOAD_DIRECTORY, out_filename)
        cv2.imwrite(save_path, final_output_bgr)
        
        result["image_url"] = f"/uploaded_images/{out_filename}"
        return result

    except Exception as e:
        print(f"Kritik Hata: {e}")
        return {"error": str(e)}