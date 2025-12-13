import React, { useState } from 'react';
import './App.css';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [resultImage, setResultImage] = useState(null);
  const [resultAge, setResultAge] = useState(null);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState('');

  // SİZİN RENDER ADRESİNİZs
  const BACKEND_URL = "https://muho4145-photomontager-backend.hf.space";

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
    setStatus('Yapay Zeka İşliyor... (1-2 dk sürebilir)');
    setResultImage(null);
    setResultAge(null);

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('target_mode', mode);

    try {
      const response = await fetch(`${BACKEND_URL}/predict/`, {
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

  // --- DÜZELTİLDİ: İndirme Fonksiyonu (Blob Yöntemi) ---
  const handleDownload = async () => {
    if (resultImage) {
      try {
        setStatus('İndiriliyor...');
        // Resmi veri olarak çek
        const response = await fetch(resultImage);
        const blob = await response.blob();
        
        // Geçici bir indirme bağlantısı oluştur
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `sonuc_${Date.now()}.jpg`; // Dosya ismini ayarla
        document.body.appendChild(link);
        link.click(); // Otomatik tıkla
        
        // Temizlik
        document.body.removeChild(link);
        window.URL.revokeObjectURL(url);
        setStatus('İndirme tamamlandı.');
      } catch (error) {
        console.error("İndirme hatası:", error);
        setStatus("İndirirken hata oluştu.");
      }
    }
  };

  // --- SONUCU ORİJİNAL YAPMA ---
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
          
          {/* SOL KUTU */}
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

          {/* SAĞ KUTU */}
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
                {/* DÜZELTİLDİ: Unicode karakteri */}
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