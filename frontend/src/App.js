import React, { useState } from 'react';
import './App.css';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [resultImage, setResultImage] = useState(null);
  const [resultAge, setResultAge] = useState(null);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState('');

  // Dosya seçilince çalışır
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

  // Butonlara basılınca çalışır
  const handleProcess = async (mode) => {
    if (!selectedFile) return;
    
    setLoading(true);
    setStatus('Yapay Zeka İşliyor...');
    setResultImage(null);
    setResultAge(null);

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('target_mode', mode); // Backend'e ne yapacağını söylüyoruz

    try {
      const response = await fetch('https://photomontager-web.onrender.com/analyze/', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (data.error) {
        setStatus('Hata oluştu: ' + data.error);
      } else {
        if (data.type === 'prediction') {
          // Yaş Tahmini Sonucu
          setResultAge(data.age);
          setResultImage(data.image_url); // Algılanan yüzü göster
          setStatus(`Tahmin Edilen Yaş: ${data.age}`);
        } else {
          // Yaşlandırma/Gençleştirme Sonucu
          setResultImage(data.image_url);
          setStatus(mode === 'make_old' ? 'Yaşlandırma Tamamlandı!' : 'Gençleştirme Tamamlandı!');
        }
      }
    } catch (error) {
      console.error(error);
      setStatus('Sunucuya bağlanılamadı. Backend açık mı?');
    }
    setLoading(false);
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Yapay Zeka Fotoğraf Stüdyosu</h1>
        
        {/* Yükleme Alanı */}
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

        {/* Ana İçerik */}
        <div className="main-content">
          
          {/* Sol: Orijinal */}
          {previewUrl && (
            <div className="image-box">
              <h3>Orijinal</h3>
              <img src={previewUrl} alt="Seçilen" className="img-display" />
              
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
              <img src={resultImage} alt="Sonuç" className="img-display" />
              
              {resultAge !== null && (
                <div className="age-result">
                  {resultAge} <span style={{fontSize:'1rem'}}>YAŞ</span>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Durum Mesajı */}
        <p className="status-text">{status}</p>

      </header>
    </div>
  );
}

export default App;