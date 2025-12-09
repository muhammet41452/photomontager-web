import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features)
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    """ RESSAM: Resmi alır, istenen yaşa dönüştürür. """
    def __init__(self, input_shape=(3, 128, 128), num_residual_blocks=6):
        super(Generator, self).__init__()
        channels = input_shape[0] + 1 # Resim + Hedef Yaş Etiketi

        # 1. Kodlama (Aşağı İndirgeme)
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(channels, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]

        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2

        # 2. Dönüşüm (Residual Bloklar)
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(in_features)]

        # 3. Kod Çözme (Yukarı Çıkarma)
        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features // 2

        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, 7),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x, target_age_label):
        # Etiketi resim boyutuna genişlet
        label = target_age_label.view(x.size(0), 1, 1, 1).repeat(1, 1, x.size(2), x.size(3))
        input_concat = torch.cat((x, label), 1)
        return self.model(input_concat)

class Discriminator(nn.Module):
    """ ELEŞTİRMEN: Resmin gerçek mi sahte mi olduğuna karar verir. """
    def __init__(self, input_shape=(3, 128, 128)):
        super(Discriminator, self).__init__()
        
        # Resim (3) + Hedef Yaş (1) = 4 Kanal
        channels, height, width = input_shape
        channels += 1 

        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1) # Sonuç: Gerçeklik skoru haritası
        )

    def forward(self, img, target_age_label):
        # Etiketi resim boyutuna genişlet
        label = target_age_label.view(img.size(0), 1, 1, 1).repeat(1, 1, img.size(2), img.size(3))
        d_in = torch.cat((img, label), 1)
        return self.model(d_in)

if __name__ == "__main__":
    print("🎨 GAN Modeli Test Ediliyor (Ressam + Eleştirmen)...")
    gen = Generator()
    disc = Discriminator()
    
    dummy_img = torch.randn(1, 3, 128, 128)
    dummy_label = torch.tensor([0.0]) # 0 = Genç, 1 = Yaşlı
    
    fake_img = gen(dummy_img, dummy_label)
    validity = disc(fake_img, dummy_label)
    
    print(f"✅ Generator Çıktısı: {fake_img.shape}")
    print(f"✅ Discriminator Çıktısı: {validity.shape}")