import React, { useState } from 'react';
import './App.css';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [resultImage, setResultImage] = useState(null);
  const [resultAge, setResultAge] = useState(null);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState('');

  // 🌍 Backend Adresini Kendi Space Adresinle Değiştir
  const BACKEND_URL = "https://muho4145-photomontager-backend.hf.space";

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

  const handleProcess = async (mode) => {
    if (!selectedFile) return;
    
    setLoading(true);
    setStatus('Yapay Zeka İşliyor... Lütfen Bekleyin...');
    setResultImage(null);
    setResultAge(null);

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('target_mode', mode);

    try {
      const response = await fetch(`${BACKEND_URL}/predict`, { 
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      console.log("Sunucu Cevabı:", data);

      // 🔥 KORUMA: Eğer veri NULL gelirse hata ver ama çökme
      if (!data) {
         setStatus("Hata: Sunucudan boş cevap geldi. Tekrar deneyin.");
         setLoading(false);
         return;
      }

      // Hata Kontrolü
      if (data.error || data.detail) {
        setStatus('Hata: ' + (data.error || data.detail));
        setLoading(false);
        return;
      }

      if (!data.image_url) {
        setStatus('Hata: Sunucu resim oluşturamadı.');
        setLoading(false);
        return;
      }

      // URL Düzeltme
      let fullImageUrl = data.image_url;
      if (fullImageUrl.startsWith('http://')) {
          fullImageUrl = fullImageUrl.replace('http://', 'https://');
      }
      else if (fullImageUrl.startsWith('/')) {
          fullImageUrl = `${BACKEND_URL}${fullImageUrl}`;
      }

      if (data.type === 'prediction') {
        setResultAge(data.predicted_age);
        setResultImage(fullImageUrl);
        setStatus(`Tahmin Edilen Yaş: ${data.predicted_age}`);
      } else {
        setResultImage(fullImageUrl);
        setStatus(mode === 'make_old' ? 'Yaşlandırma Tamamlandı!' : 'Gençleştirme Tamamlandı!');
      }

    } catch (error) {
      console.error("Bağlantı Hatası:", error);
      setStatus('Hata: Sunucuya bağlanılamadı. İnternetinizi kontrol edin.');
    }
    setLoading(false);
  };

  const handleDownload = async () => {
    if (resultImage) {
      try {
        setStatus('İndiriliyor...');
        const response = await fetch(resultImage);
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `sonuc_${Date.now()}.jpg`;
        document.body.appendChild(link);
        link.click(); 
        document.body.removeChild(link);
        window.URL.revokeObjectURL(url);
        setStatus('İndirme tamamlandı.');
      } catch (error) {
        console.error("İndirme hatası:", error);
        setStatus("İndirirken hata oluştu.");
      }
    }
  };

  const handleSetAsOriginal = async () => {
    if (!resultImage) return;
    try {
      setStatus('Resim aktarılıyor...');
      const response = await fetch(resultImage);
      const blob = await response.blob();
      const file = new File([blob], "islenmis_resim.jpg", { type: "image/jpeg" });

      setSelectedFile(file);
      setPreviewUrl(resultImage);
      setResultImage(null);
      setResultAge(null);
      setStatus('İşlenmiş fotoğraf yeni orijinal olarak ayarlandı.');
    } catch (error) {
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

          {resultImage && (
            <div className="image-box result-box">
              <h3>Sonuç</h3>
              <img key={resultImage} src={resultImage} alt="Sonuç" className="img-display" />
              
              {resultAge !== null && (
                <div className="age-result">{resultAge} <span style={{fontSize:'1rem'}}>YAŞ</span></div>
              )}

              <div className="button-group" style={{ marginTop: '15px' }}>
                <button onClick={handleDownload} className="action-btn download-btn">
                  ⬇️ İndir
                </button>
                <button onClick={handleSetAsOriginal} className="action-btn reuse-btn">
                  ↩ Bu Resmi Kullan
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