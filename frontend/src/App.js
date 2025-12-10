import React, { useState } from 'react';
import './App.css';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [resultImage, setResultImage] = useState(null);
  const [resultAge, setResultAge] = useState(null);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState('');

  // --- AYARLAR ---
  // Eğer lokalde çalışıyorsanız burayı açın:
  // const BACKEND_URL = "http://localhost:8000"; 
  // Eğer canlı sunucu kullanıyorsanız (Render):
  const BACKEND_URL = "https://photomontager-web.onrender.com";

  // --- DOSYA SEÇME ---
  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResultImage(null);
      setResultAge(null);
      setStatus('');
    }
  };

  // --- İŞLEME FONKSİYONU ---
  const handleProcess = async (mode) => {
    if (!selectedFile) return;
    
    setLoading(true);
    setStatus('Yapay Zeka İşliyor...');
    setResultImage(null);
    setResultAge(null);

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('target_mode', mode);

    try {
      const response = await fetch(`${BACKEND_URL}/analyze/`, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (data.error) {
        setStatus('Hata: ' + data.error);
      } else {
        // URL Düzeltme
        let fullImageUrl = data.image_url;
        if (!fullImageUrl.startsWith('http')) {
            fullImageUrl = `${BACKEND_URL}${data.image_url}`;
        }

        if (data.type === 'prediction') {
          setResultAge(data.age);
          setResultImage(fullImageUrl);
          setStatus(`Tahmin Edilen Yaş: ${data.age}`);
        } else {
          setResultImage(fullImageUrl);
          setStatus(mode === 'make_old' ? 'Yaşlandırma Tamamlandı!' : 'Gençleştirme Tamamlandı!');
        }
      }
    } catch (error) {
      console.error(error);
      setStatus('Sunucuya bağlanılamadı.');
    }
    setLoading(false);
  };

  // --- YENİ ÖZELLİK: Sonucu İndirme ---
  const handleDownload = () => {
    if (resultImage) {
      const link = document.createElement('a');
      link.href = resultImage;
      link.download = `sonuc_${Date.now()}.jpg`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }
  };

  // --- YENİ ÖZELLİK: Sonucu Orijinal Yapma ---
  const handleSetAsOriginal = async () => {
    if (!resultImage) return;

    try {
      setStatus('Resim aktarılıyor...');
      // 1. Resim URL'sini alıp Blob (Veri yığını) haline getiriyoruz
      const response = await fetch(resultImage);
      const blob = await response.blob();
      
      // 2. Blob'dan yeni bir Dosya oluşturuyoruz
      const file = new File([blob], "islenmis_resim.jpg", { type: "image/jpeg" });

      // 3. State'leri güncelliyoruz
      setSelectedFile(file);
      setPreviewUrl(resultImage); // Artık orijinal kısımda bu resim görünecek
      
      // Sağ tarafı temizle
      setResultImage(null);
      setResultAge(null);
      setStatus('İşlenmiş fotoğraf yeni orijinal olarak ayarlandı. Tekrar işlem yapabilirsiniz.');

    } catch (error) {
      console.error("Dönüştürme hatası:", error);
      setStatus("Resim aktarılırken hata oluştu.");
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Yapay Zeka Fotoğraf Stüdyosu</h1>
        
        <div className="upload-container">
          <input type="file" accept="image/*" onChange={handleFileChange} id="fileInput" style={{ display: 'none' }} />
          <label htmlFor="fileInput" className="upload-btn">📷 Fotoğraf Yükle</label>
        </div>

        <div className="main-content">
          
          {/* SOL KUTU: GİRİŞ */}
          {previewUrl && (
            <div className="image-box">
              <h3>Orijinal</h3>
              <img src={previewUrl} alt="Orijinal" className="img-display" />
              
              <div className="button-group">
                <button onClick={() => handleProcess('age_estimation')} disabled={loading} className="action-btn predict-btn">
                  🔍 Yaşı Tahmin Et
                </button>
                <button onClick={() => handleProcess('make_old')} disabled={loading} className="action-btn old-btn">
                  👴 Beni Yaşlandır
                </button>
                <button onClick={() => handleProcess('make_young')} disabled={loading} className="action-btn young-btn">
                  👶 Beni Gençleştir
                </button>
              </div>
            </div>
          )}

          {/* SAĞ KUTU: ÇIKIŞ */}
          {resultImage && (
            <div className="image-box result-box">
              <h3>Sonuç</h3>
              <img key={resultImage} src={resultImage} alt="Sonuç" className="img-display" />
              
              {resultAge !== null && (
                <div className="age-result">{resultAge} <span style={{fontSize:'1rem'}}>YAŞ</span></div>
              )}

              {/* YENİ BUTONLAR */}
              <div className="button-group" style={{ marginTop: '15px' }}>
                <button onClick={handleDownload} className="action-btn download-btn">
                  ⬇️ İndir
                </button>
                <button onClick={handleSetAsOriginal} className="action-btn reuse-btn">
                  u21a9 Bu Resmi Kullan
                </button>
              </div>
            </div>
          )}
        </div>

        <p className="status-text">{status}</p>
      </header>
    </div>
  );
}

export default App;