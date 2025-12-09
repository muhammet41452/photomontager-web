import torch
import torch.nn as nn
from torchvision import models

class AgeEstimationModel(nn.Module):
    def __init__(self):
        super(AgeEstimationModel, self).__init__()
        
        # 1. Dünyaca ünlü ResNet18 modelini (hazır beyin) indiriyoruz.
        # Bu model, kenarları, şekilleri, yüz hatlarını zaten çok iyi biliyor.
        try:
            # Yeni PyTorch sürümleri için
            weights = models.ResNet18_Weights.DEFAULT
            self.net = models.resnet18(weights=weights)
        except:
            # Eski sürümler için yedek yöntem
            self.net = models.resnet18(pretrained=True)
        
        # 2. ResNet'in son katmanını kendi işimize (Yaş Tahmini) göre değiştiriyoruz.
        # Orijinal ResNet 1000 çeşit nesne tanır, biz bunu 1 sayı (YAŞ) üretecek hale getiriyoruz.
        n_features = self.net.fc.in_features
        
        self.net.fc = nn.Sequential(
            nn.Linear(n_features, 512), # 512 nörona indir
            nn.ReLU(),
            nn.Dropout(0.5),            # Ezberlemeyi önle
            nn.Linear(512, 1)           # Sonuç: Tek bir yaş değeri
        )

    def forward(self, x):
        return self.net(x)

if __name__ == "__main__":
    model = AgeEstimationModel()
    print("✅ ResNet18 tabanlı güçlü model oluşturuldu.")
    test_input = torch.randn(1, 3, 128, 128)
    print(f"Test Çıktısı: {model(test_input).item()}")