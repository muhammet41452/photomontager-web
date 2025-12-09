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

# Kendi modellerimiz
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

# --- 2. CİHAZ VE MODELLER ---
DEVICE = torch.device("cpu")
age_model = AgeEstimationModel()
gan_model = Generator()
AGE_PATH = "models/yas_tahmin_modeli.pth"
GAN_PATH = "models/yaslandirma_gan.pth"

print("⏳ Sistem Başlatılıyor...")

# Yaş ve GAN Modellerini Yükle
try:
    if os.path.exists(AGE_PATH):
        age_model.load_state_dict(torch.load(AGE_PATH, map_location=DEVICE))
        age_model.to(DEVICE).eval()
        print("✅ Yaş Tahmin Modeli Yüklendi.")
    
    if os.path.exists(GAN_PATH):
        gan_model.load_state_dict(torch.load(GAN_PATH, map_location=DEVICE))
        gan_model.to(DEVICE).eval()
        print("✅ Yaşlandırma Modeli Yüklendi.")
except Exception as e:
    print(f"⚠️ Model Yükleme Hatası: {e}")

# --- 3. OPENCV DNN YÜZ TANIMA ---
protoPath = "deploy.prototxt"
modelPath = "res10_300x300_ssd_iter_140000.caffemodel"
face_net = None

if os.path.exists(protoPath) and os.path.exists(modelPath):
    face_net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
    print("✅ OpenCV DNN Yüz Tanıma Sistemi Aktif.")
else:
    print("❌ HATA: 'deploy.prototxt' veya 'caffemodel' dosyası eksik!")

# --- YARDIMCI FONKSİYONLAR ---

def detect_face_dnn(img_cv):
    """
    OpenCV'nin Derin Öğrenme modelini kullanarak yüzü bulur.
    """
    if face_net is None: return None
    
    h, w = img_cv.shape[:2]
    # Resmi 300x300 boyutuna getirip modele veriyoruz
    blob = cv2.dnn.blobFromImage(cv2.resize(img_cv, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()
    
    best_face = None
    max_confidence = 0
    
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        # %40'tan fazla eminse yüz kabul et
        if confidence > 0.4:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, endY)
            
            face_w = endX - startX
            face_h = endY - startY
            
            if confidence > max_confidence:
                max_confidence = confidence
                best_face = (startX, startY, face_w, face_h)
                
    return best_face

def tensor_to_cv2(tensor):
    image = tensor.detach().cpu().squeeze(0)
    image = image * 0.5 + 0.5
    image = image.clamp(0, 1)
    image = image.permute(1, 2, 0).numpy()
    image = (image * 255).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # Renk düzeltme
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
        img_np = np.array(pil_image) 
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        face_rect = detect_face_dnn(img_bgr)
        
        final_output = img_bgr.copy()
        
        if face_rect:
            (x, y, w, h) = face_rect
            
            # --- DÜZELTME BURADA: Padding'i Azalttık ---
            # Modelimiz yüzü yakından görmeye alışkın.
            # Önceden %20 boşluk bırakıyorduk, şimdi %5'e indirdik.
            # Böylece yüz, resmin tamamını kaplayacak.
            p_w = int(w * 0.05) 
            p_h = int(h * 0.05)
            
            x1 = max(0, x - p_w)
            y1 = max(0, y - p_h) 
            x2 = min(img_bgr.shape[1], x + w + p_w)
            y2 = min(img_bgr.shape[0], y + h + p_h)
            
            face_roi_bgr = img_bgr[y1:y2, x1:x2]
            print(f"✅ Yüz Bulundu: {x1},{y1} - {x2},{y2}")
        else:
            print("⚠️ Yüz Bulunamadı, Merkez Kesiliyor.")
            H, W = img_bgr.shape[:2]
            x1, y1 = int(W*0.25), int(H*0.20)
            x2, y2 = int(W*0.75), int(H*0.80)
            face_roi_bgr = img_bgr[y1:y2, x1:x2]

        # Modele Hazırla (128x128'e Sıkıştır - Resize)
        # Training verisi resize edildiği için burada da resize (squash) ediyoruz.
        face_roi_rgb = cv2.cvtColor(face_roi_bgr, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_roi_rgb)
        face_input = face_pil.resize((128, 128), Image.Resampling.LANCZOS)
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        tensor = transform(face_input).unsqueeze(0).to(DEVICE)
        
        result = {}

        # --- A) YAŞ TAHMİNİ ---
        if target_mode == "age_estimation":
            with torch.no_grad():
                pred = age_model(tensor)
                age = pred.item()
            result["type"] = "prediction"
            result["age"] = round(age, 1)
            
            if face_rect:
                cv2.rectangle(final_output, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # --- B) YAŞLANDIRMA ---
        else:
            target_label = 1.0 if target_mode == "make_old" else 0.0
            label_tensor = torch.tensor([target_label]).to(DEVICE)
            
            with torch.no_grad():
                fake_tensor = gan_model(tensor, label_tensor)
            
            gen_face_bgr = tensor_to_cv2(fake_tensor)
            
            # Montaj
            try:
                gen_face_resized = cv2.resize(gen_face_bgr, (x2-x1, y2-y1))
                final_output[y1:y2, x1:x2] = gen_face_resized
            except:
                pass
            
            result["type"] = "transformation"

        # Kaydet
        out_filename = f"result_{uuid.uuid4().hex[:8]}.jpg"
        save_path = os.path.join(UPLOAD_DIRECTORY, out_filename)
        cv2.imwrite(save_path, final_output)
        
        result["image_url"] = f"http://localhost:8000/uploaded_images/{out_filename}"
        result["message"] = "İşlem Başarılı"
        
        return result

    except Exception as e:
        print(f"Kritik Hata: {e}")
        return {"error": str(e)}