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

# Modeller
from model import AgeEstimationModel
from gan_model import Generator

app = FastAPI()

# --- 1. AYARLAR ---
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

# --- 2. MODELLER ---
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
        print("✅ Yaş Tahmin Modeli Hazır.")
    
    if os.path.exists(GAN_PATH):
        gan_model.load_state_dict(torch.load(GAN_PATH, map_location=DEVICE))
        gan_model.to(DEVICE).eval()
        print("✅ Yaşlandırma Modeli Hazır.")
except Exception as e:
    print(f"⚠️ Model Hatası: {e}")

# --- 3. YÜZ TANIMA (OPENCV DNN) ---
# MediaPipe yerine bunu kullanıyoruz çünkü "Masaüstü" yolunda hata vermez.
protoPath = "deploy.prototxt"
modelPath = "res10_300x300_ssd_iter_140000.caffemodel"
face_net = None

if os.path.exists(protoPath) and os.path.exists(modelPath):
    face_net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
    print("✅ Güvenli Yüz Tanıma (DNN) Aktif.")
else:
    print("❌ HATA: 'deploy.prototxt' eksik! Lütfen 'model_indir.py' dosyasını çalıştırın.")

# --- YARDIMCI FONKSİYONLAR ---

def detect_face_dnn(img_cv):
    """ OpenCV DNN ile yüz bulur (Türkçe karakter dostu) """
    if face_net is None: return None
    h, w = img_cv.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img_cv, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()
    best_face = None
    max_conf = 0
    
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.3:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            startX, startY = max(0, startX), max(0, startY)
            endX, endY = min(w, endX), min(h, endY)
            if confidence > max_conf:
                max_conf = confidence
                best_face = (startX, startY, endX-startX, endY-startY)
    return best_face

def tensor_to_cv2(tensor):
    image = tensor.detach().cpu().squeeze(0)
    image = image * 0.5 + 0.5
    image = image.clamp(0, 1)
    image = image.permute(1, 2, 0).numpy()
    image = (image * 255).astype(np.uint8)
    # PIL (RGB) -> OpenCV (BGR) dönüşümü GEREKLİ DEĞİL çünkü aşağıda RGB işliyoruz
    return image 

# --- API ENDPOINT ---

@app.post("/analyze/")
async def analyze_photo(
    file: UploadFile = File(...), 
    target_mode: str = Form("age_estimation") 
):
    try:
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_np = np.array(pil_image) # RGB formatında çalışıyoruz
        
        # Yüzü Bul (DNN kullanır, daha sağlamdır)
        # DNN BGR bekler, o yüzden geçici dönüşüm yapıyoruz
        img_bgr_temp = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        face_rect = detect_face_dnn(img_bgr_temp)
        
        final_output = img_np.copy()
        
        if face_rect:
            (x, y, w, h) = face_rect
            # Padding (Çok az pay bırakıyoruz, %10)
            p_w = int(w * 0.10)
            p_h = int(h * 0.10)
            x1 = max(0, x - p_w)
            y1 = max(0, y - int(p_h * 1.5))
            x2 = min(img_np.shape[1], x + w + p_w)
            y2 = min(img_np.shape[0], y + h + p_h)
            
            face_roi = img_np[y1:y2, x1:x2]
            
            # Modele Hazırlık
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

            else:
                # --- OVAL MONTAJLI DÖNÜŞÜM ---
                target_label = 1.0 if target_mode == "make_old" else 0.0
                label_tensor = torch.tensor([target_label]).to(DEVICE)
                
                with torch.no_grad():
                    fake_tensor = gan_model(tensor, label_tensor)
                
                gen_face = tensor_to_cv2(fake_tensor) # 128x128, RGB
                
                # 1. Büyüt
                roi_h, roi_w = (y2-y1), (x2-x1)
                gen_face_resized = cv2.resize(gen_face, (roi_w, roi_h), interpolation=cv2.INTER_LANCZOS4)
                
                # 2. OVAL MASKE OLUŞTUR (Kare görünümü yok etmek için)
                mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
                center = (roi_w // 2, roi_h // 2)
                axes = (int(roi_w * 0.45), int(roi_h * 0.48)) # Hafif içten elips
                cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
                
                # Maskeyi yumuşat (Blur)
                mask = cv2.GaussianBlur(mask, (35, 35), 20)
                
                # 3. MONTAJ (Alpha Blending)
                roi_bg = final_output[y1:y2, x1:x2].astype(float)
                roi_fg = gen_face_resized.astype(float)
                
                # Maskeyi 0-1 arasına çek ve 3 kanala yay
                alpha = mask.astype(float) / 255.0
                alpha = np.stack([alpha]*3, axis=-1)
                
                blended = (roi_fg * alpha) + (roi_bg * (1.0 - alpha))
                final_output[y1:y2, x1:x2] = blended.astype(np.uint8)
                
                result["type"] = "transformation"
        else:
            result = {"type": "error", "age": 0}
            print("⚠️ Yüz bulunamadı.")

        # Kaydet (OpenCV BGR bekler, bizde RGB var, dönüştür)
        final_output_bgr = cv2.cvtColor(final_output, cv2.COLOR_RGB2BGR)
        out_filename = f"result_{uuid.uuid4().hex[:8]}.jpg"
        save_path = os.path.join(UPLOAD_DIRECTORY, out_filename)
        cv2.imwrite(save_path, final_output_bgr)
        
        result["image_url"] = f"/uploaded_images/{out_filename}"
        return result

    except Exception as e:
        print(f"Hata: {e}")
        return {"error": str(e)}