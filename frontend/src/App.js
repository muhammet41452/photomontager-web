import React, { useState } from 'react';
import './App.css';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [resultImage, setResultImage] = useState(null);
  const [resultAge, setResultAge] = useState(null);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState('');

  // SİZİN BACKEND ADRESİNİZ (Sonunda / işareti yok)
  const BACKEND_URL = "https://photomontager-web.onrender.com";

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
    setStatus('Yapay Zeka İşliyor... (İlk işlem 1-2 dk sürebilir)');
    setResultImage(null);
    setResultAge(null);

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('target_mode', mode);

    try {
      // 1. Backend'e İstek At
      const response = await fetch(`${BACKEND_URL}/analyze/`, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (data.error) {
        setStatus('Hata: ' + data.error);
      } else {
        // 2. Gelen Resim Adresini Düzelt (Tam URL Yap)
        // Eğer gelen adres http ile başlamıyorsa, başına backend adresini ekle
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
      setStatus('Sunucuya bağlanılamadı. (Sunucu uyanıyor olabilir, tekrar deneyin)');
    }
    setLoading(false);
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Yapay Zeka Fotoğraf Stüdyosu</h1>
        
        <div className="upload-container">
          <input 
            type="file" 
            accept="image/*" 
            onChange={handleFileChange} 
            id="fileInput" 
            style={{ display: 'none' }} 
          />
          <label htmlFor="fileInput" className="upload-btn">
            📷 Fotoğraf Yükle
          </label>
        </div>

        <div className="main-content">
          {/* Sol: Orijinal */}
          {previewUrl && (
            <div className="image-box">
              <h3>Orijinal</h3>
              <img src={previewUrl} alt="Orijinal" className="img-display" />
              
              <div className="button-group">
                <button onClick={() => handleProcess('age_estimation')} disabled={loading} className="action-btn predict-btn">
                  🔍 Yaş Tahmini
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

          {/* Sağ: Sonuç */}
          {resultImage && (
            <div className="image-box result-box">
              <h3>Sonuç</h3>
              {/* key={resultImage} ekleyerek resim değiştiğinde yeniden yüklenmesini sağlıyoruz */}
              <img key={resultImage} src={resultImage} alt="Sonuç" className="img-display" />
              
              {resultAge !== null && (
                <div className="age-result">
                  {resultAge} <span style={{fontSize:'1rem'}}>YAŞ</span>
                </div>
              )}
            </div>
          )}
        </div>

        <p className="status-text">{status}</p>

      </header>
    </div>
  );
}

export default App;