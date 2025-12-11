import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_features, in_features, 3, padding=1, padding_mode='reflect')
        self.in1 = nn.InstanceNorm2d(in_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_features, in_features, 3, padding=1, padding_mode='reflect')
        self.in2 = nn.InstanceNorm2d(in_features)

    def forward(self, x):
        res = x
        out = self.conv1(x)
        out = self.in1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.in2(out)
        return res + out

class Generator(nn.Module):
    """
    GÜÇLENDİRİLMİŞ RESSAM:
    Etiketi (Yaş bilgisini) sadece girişte değil, darboğazda (bottleneck) da alır.
    Böylece 'Ne yapıyordum ben?' diye unutmaz.
    """
    def __init__(self, input_shape=(3, 128, 128), num_residual_blocks=6):
        super(Generator, self).__init__()
        
        # Giriş: Resim(3) + Label(1) = 4 kanal
        channels = input_shape[0] + 1 

        # --- ENCODER (Aşağı İndirme) ---
        self.initial = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(channels, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Downsampling
        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # --- TRANSFORMATION (Değişim) ---
        # Burada etiketi tekrar içeri alacağız. 
        # 256 özellik haritası + 1 etiket haritası = 257 kanal
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(257) for _ in range(num_residual_blocks)]
        )

        # --- DECODER (Yukarı Çıkarma) ---
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(257, 128, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.final = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, 7),
            nn.Tanh()
        )

    def forward(self, x, label):
        # 1. Girişte etiketi birleştir
        label_map = label.view(x.size(0), 1, 1, 1).repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat((x, label_map), 1)
        
        # Encoder
        x = self.initial(x)
        x = self.down1(x)
        x = self.down2(x) # Boyut: 32x32, Kanal: 256
        
        # 2. DARBOĞAZDA ETİKETİ TEKRAR BİRLEŞTİR (Inject Label)
        # Etiketi 32x32 boyutuna küçültüp ekliyoruz
        label_small = label.view(label.size(0), 1, 1, 1).repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat((x, label_small), 1) # Kanal: 256 + 1 = 257
        
        # Residual Bloklar (Değişim burada oluyor)
        x = self.res_blocks(x)
        
        # Decoder
        x = self.up1(x)
        x = self.up2(x)
        return self.final(x)

class Discriminator(nn.Module):
    """ ELEŞTİRMEN (Standart PatchGAN) """
    def __init__(self, input_shape=(3, 128, 128)):
        super(Discriminator, self).__init__()
        channels = input_shape[0] + 1 # Resim + Label

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
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img, label):
        label_map = label.view(img.size(0), 1, 1, 1).repeat(1, 1, img.size(2), img.size(3))
        d_in = torch.cat((img, label_map), 1)
        return self.model(d_in)