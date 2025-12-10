from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps
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
DEVICE = torch.device("cpu")
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

# MediaPipe Yüz Ağı (En hassas yöntem)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# --- YARDIMCI FONKSİYONLAR ---

def enhance_image(image_cv):
    """
    128px'lik resmi büyütürken oluşan bulanıklığı gidermek için
    Keskinleştirme ve Detay Artırma uygular.
    """
    # 1. Detay Artırma (Detail Enhance)
    # Bu filtre, yüzdeki çizgileri ve dokuyu belirginleştirir.
    enhanced = cv2.detailEnhance(image_cv, sigma_s=10, sigma_r=0.15)
    
    # 2. Keskinleştirme (Sharpening Kernel)
    # Bulanıklığı almak için agresif bir filtre
    kernel = np.array([[0, -1, 0], 
                       [-1, 5,-1], 
                       [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    
    return sharpened

def get_face_mask_and_bbox(image_np):
    results = face_mesh.process(image_np)
    if not results.multi_face_landmarks:
        return None, None
    landmarks = results.multi_face_landmarks[0].landmark
    h, w, _ = image_np.shape
    points = np.array([(int(l.x * w), int(l.y * h)) for l in landmarks], np.int32)
    hull = cv2.convexHull(points)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    rect = cv2.boundingRect(hull)
    return mask, rect

def tensor_to_cv2(tensor):
    image = tensor.detach().cpu().squeeze(0)
    image = image * 0.5 + 0.5
    image = image.clamp(0, 1)
    image = image.permute(1, 2, 0).numpy()
    image = (image * 255).astype(np.uint8)
    return image # RGB

# --- ANA İŞLEM ---

@app.post("/analyze/")
async def analyze_photo(
    file: UploadFile = File(...), 
    target_mode: str = Form("age_estimation") 
):
    try:
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_np = np.array(pil_image) # RGB
        
        mask, rect = get_face_mask_and_bbox(img_np)
        final_output = img_np.copy()
        
        if rect:
            (x, y, w, h) = rect
            p = int(w * 0.15)
            x1 = max(0, x - p)
            y1 = max(0, y - int(p*1.5))
            x2 = min(img_np.shape[1], x + w + p)
            y2 = min(img_np.shape[0], y + h + p)
            
            face_roi = img_np[y1:y2, x1:x2]
            
            face_pil = Image.fromarray(face_roi)
            # Modele girerken en kaliteli küçültme
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

            else:
                # --- GAN DÖNÜŞÜMÜ ---
                target_label = 1.0 if target_mode == "make_old" else 0.0
                label_tensor = torch.tensor([target_label]).to(DEVICE)
                
                with torch.no_grad():
                    fake_tensor = gan_model(tensor, label_tensor)
                
                gen_face = tensor_to_cv2(fake_tensor) # 128x128
                
                # --- KALİTE ARTIRMA BÖLÜMÜ ---
                # 1. 128px'lik çıktıyı büyütmeden önce biraz iyileştir
                gen_face = enhance_image(gen_face)

                # 2. Orijinal boyuta büyütürken LANCZOS4 kullan (En kaliteli algoritma)
                gen_face_resized = cv2.resize(gen_face, (x2-x1, y2-y1), interpolation=cv2.INTER_LANCZOS4)
                
                # 3. Büyüttükten sonra tekrar hafif bir netleştirme
                # Büyük boyutta detay kaybını telafi etmek için
                gen_face_resized = cv2.detailEnhance(gen_face_resized, sigma_s=5, sigma_r=0.1)

                # --- MONTAJ ---
                mask_roi = mask[y1:y2, x1:x2]
                mask_roi = cv2.GaussianBlur(mask_roi, (15, 15), 10)
                
                roi_bg = final_output[y1:y2, x1:x2].astype(float)
                roi_fg = gen_face_resized.astype(float)
                
                alpha = mask_roi.astype(float) / 255.0
                alpha = np.stack([alpha]*3, axis=-1)
                
                blended = (roi_fg * alpha) + (roi_bg * (1.0 - alpha))
                final_output[y1:y2, x1:x2] = blended.astype(np.uint8)
                
                result["type"] = "transformation"
        else:
            result = {"type": "error", "age": 0}

        # Kaydet (RGB -> BGR)
        final_output_bgr = cv2.cvtColor(final_output, cv2.COLOR_RGB2BGR)
        out_filename = f"result_{uuid.uuid4().hex[:8]}.jpg"
        save_path = os.path.join(UPLOAD_DIRECTORY, out_filename)
        cv2.imwrite(save_path, final_output_bgr)
        
        result["image_url"] = f"/uploaded_images/{out_filename}"
        return result

    except Exception as e:
        print(f"Hata: {e}")
        return {"error": str(e)}