import React, { useState } from 'react';
import './App.css';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [resultImage, setResultImage] = useState(null);
  const [resultAge, setResultAge] = useState(null);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState('');

  // Backend adresi (HTTPS olduğundan emin ol)
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

  // --- İŞLEME FONKSİYONU (HATA DÜZELTMESİ BURADA) ---
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
      // 1. İSTEK GÖNDERME
      const response = await fetch(`${BACKEND_URL}/predict`, { // Sondaki '/' yok
        method: 'POST',
        body: formData,
      });

      // 2. CEVABI OKUMA
      const data = await response.json();
      console.log("Sunucudan Gelen Cevap:", data); // Konsola yazdıralım ki hatayı görelim

      // 3. HATA KONTROLÜ (FastAPI 'detail', biz 'error' kullanıyoruz. İkisini de kontrol et)
      if (data.error || data.detail) {
        setStatus('Hata: ' + (data.error || data.detail));
        setLoading(false);
        return; // Hata varsa aşağıya inme, dur.
      }

      // 4. RESİM VERİSİ VAR MI? (Senin hatanın sebebi burasıydı)
      if (!data.image_url) {
        setStatus('Hata: Sunucu resim adresi göndermedi. Konsolu kontrol et.');
        setLoading(false);
        return;
      }

      // 5. URL DÜZELTME VE GÖSTERME
      let fullImageUrl = data.image_url;

      // URL Güvenlik Kontrolleri
      if (fullImageUrl && fullImageUrl.startsWith('http://')) {
          fullImageUrl = fullImageUrl.replace('http://', 'https://');
      }
      else if (fullImageUrl && fullImageUrl.startsWith('/')) {
          fullImageUrl = `${BACKEND_URL}${fullImageUrl}`;
      }

      if (data.type === 'prediction') {
        setResultAge(data.age);
        setResultImage(fullImageUrl);
        setStatus(`Tahmin Edilen Yaş: ${data.age}`);
      } else {
        setResultImage(fullImageUrl);
        setStatus(mode === 'make_old' ? 'Yaşlandırma Tamamlandı!' : 'Gençleştirme Tamamlandı!');
      }

    } catch (error) {
      console.error("Bağlantı Hatası:", error);
      setStatus('Bir hata oluştu. Lütfen konsolu (F12) kontrol edin.');
    }
    setLoading(false);
  };

  // --- İndirme Fonksiyonu ---
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

  // --- Resmi Orijinal Yapma ---
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
      console.error(error);
      setStatus("Hata oluştu.");
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
                <button onClick={() => handleProcess('age_estimation')} disabled={loading} className="action-btn predict-btn">🔍 Yaşı Tahmin Et</button>
                <button onClick={() => handleProcess('make_old')} disabled={loading} className="action-btn old-btn">👴 Beni Yaşlandır</button>
                <button onClick={() => handleProcess('make_young')} disabled={loading} className="action-btn young-btn">👶 Beni Gençleştir</button>
              </div>
            </div>
          )}

          {resultImage && (
            <div className="image-box result-box">
              <h3>Sonuç</h3>
              <img key={resultImage} src={resultImage} alt="Sonuç" className="img-display" />
              {resultAge !== null && <div className="age-result">{resultAge} <span style={{fontSize:'1rem'}}>YAŞ</span></div>}
              <div className="button-group" style={{ marginTop: '15px' }}>
                <button onClick={handleDownload} className="action-btn download-btn">⬇️ İndir</button>
                <button onClick={handleSetAsOriginal} className="action-btn reuse-btn">↩ Bu Resmi Kullan</button>
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