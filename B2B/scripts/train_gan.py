#!/usr/bin/env python3
"""
train_gan.py
آموزش GAN (Generative Adversarial Network) برای تولید تصاویر مصنوعی پلاک خودرو
کاربرد: افزایش داده، تطبیق دامنه، تولید داده برای سناریوهای نادر
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.utils as vutils
import cv2
from pathlib import Path
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm

# ==================== تنظیمات ====================
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_status(message, status="info"):
    if status == "success":
        print(f"{GREEN}✅ {message}{RESET}")
    elif status == "warning":
        print(f"{YELLOW}⚠️ {message}{RESET}")
    elif status == "error":
        print(f"{RED}❌ {message}{RESET}")
    else:
        print(f"{BLUE}📌 {message}{RESET}")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print_status(f"دستگاه مورد استفاده: {DEVICE}", "info")


# ==================== مدل Generator ====================
class Generator(nn.Module):
    """
    شبکه Generator برای تولید تصاویر پلاک مصنوعی
    ورودی: نویز تصادفی (latent vector)
    خروجی: تصویر RGB 64x64
    """
    def init(self, latent_dim=100, img_channels=3, img_size=64):
        super(Generator, self).__init__()
        
        self.latent_dim = latent_dim
        self.img_size = img_size
        
        # محاسبه اندازه پس از لایه‌های کانولوشن
        self.init_size = img_size // 4  # 16
        self.l1 = nn.Sequential(
            nn.Linear(latent_dim, 128 * self.init_size ** 2),
            nn.BatchNorm1d(128 * self.init_size  **2),
            nn.ReLU(inplace=True)
        )
        
        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, img_channels, 3, 1, 1),
            nn.Tanh()
        )
    
    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


# ==================== مدل Discriminator ====================
class Discriminator(nn.Module):
    """
    شبکه Discriminator برای تشخیص تصاویر واقعی از مصنوعی
    ورودی: تصویر RGB 64x64
    خروجی: احتمال واقعی بودن (0 تا 1)
    """
    def init(self, img_channels=3, img_size=64):
        super(Discriminator, self).__init__()
        
        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters))
            return block
        
        self.model = nn.Sequential(
            *discriminator_block(img_channels, 32, bn=False),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img):
        return self.model(img)


# ==================== دیتاست تصاویر پلاک ====================
class PlateDataset(Dataset):
    """
    دیتاست تصاویر واقعی پلاک برای آموزش GAN
    """
    def init(self, data_dir, img_size=64):
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.images = []
        
        # جستجوی تصاویر در پوشه
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            self.images.extend(list(self.data_dir.glob(ext)))
        
        print_status(f"{len(self.images)} تصویر در {data_dir} یافت شد", "info")
    
    def len(self):
        return len(self.images)
    
    def getitem(self, idx):
        img_path = self.images[idx]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype(np.float32) / 127.5 - 1  # نرمال‌سازی [-1, 1]
        img = torch.tensor(img).permute(2, 0, 1)
        return img


# ==================== کلاس GAN ====================
class GANTrainer:
    def init(self, latent_dim=100, lr=0.0002, betas=(0.5, 0.999)):
        self.latent_dim = latent_dim
        self.generator = Generator(latent_dim).to(DEVICE)
        self.discriminator = Discriminator().to(DEVICE)
        
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=lr, betas=betas)
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=lr, betas=betas)
        self.criterion = nn.BCELoss()
        
        self.g_losses = []
        self.d_losses = []
    
    def train_step(self, real_imgs):
        batch_size = real_imgs.size(0)
        
        # لیبل‌ها
        real_labels = torch.ones(batch_size, 1).to(DEVICE)
        fake_labels = torch.zeros(batch_size, 1).to(DEVICE)
        
        # ========== آموزش Discriminator ==========
        self.d_optimizer.zero_grad()
        
        # loss روی تصاویر واقعی
        real_output = self.discriminator(real_imgs)
        d_real_loss = self.criterion(real_output, real_labels)
        
        # loss روی تصاویر مصنوعی
        z = torch.randn(batch_size, self.latent_dim).to(DEVICE)
        fake_imgs = self.generator(z)
        fake_output = self.discriminator(fake_imgs.detach())
        d_fake_loss = self.criterion(fake_output, fake_labels)
        
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        self.d_optimizer.step()
        
        # ========== آموزش Generator ==========
        self.g_optimizer.zero_grad()
        
        z = torch.randn(batch_size, self.latent_dim).to(DEVICE)
        fake_imgs = self.generator(z)
        fake_output = self.discriminator(fake_imgs)
        g_loss = self.criterion(fake_output, real_labels)
        
        g_loss.backward()
        self.g_optimizer.step()
        
        return g_loss.item(), d_loss.item()
    
    def train(self, dataloader, epochs=100):
        print_status("شروع آموزش GAN...", "info")
        
        for epoch in range(epochs):
            epoch_g_loss = 0
            epoch_d_loss = 0
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            for real_imgs in pbar:
                real_imgs = real_imgs.to(DEVICE)
                g_loss, d_loss = self.train_step(real_imgs)
                epoch_g_loss += g_loss
                epoch_d_loss += d_loss
                pbar.set_postfix({'G_loss': g_loss, 'D_loss': d_loss})
            
            avg_g_loss = epoch_g_loss / len(dataloader)
            avg_d_loss = epoch_d_loss / len(dataloader)
            self.g_losses.append(avg_g_loss)
            self.d_losses.append(avg_d_loss)
            
            if (epoch + 1) % 10 == 0:
                print_status(f"Epoch {epoch+1}: G_loss={avg_g_loss:.4f}, D_loss={avg_d_loss:.4f}", "info")
                self.generate_samples(epoch + 1)
    
    def generate_samples(self, epoch=None):
        """تولید نمونه تصاویر برای ذخیره"""
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(16, self.latent_dim).to(DEVICE)
            samples = self.generator(z).cpu()
            samples = (samples + 1) / 2  # تبدیل به [0, 1]
            
            # ذخیره تصاویر
            vutils.save_image(samples, f'gan_samples_epoch_{epoch}.png' if epoch else 'gan_samples.png', nrow=4)
    
    def save_models(self, path='models/gan_plate_generator.pt'):
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'g_losses': self.g_losses,
            'd_losses': self.d_losses
        }, path)
        print_status(f"مدل‌ها در {path} ذخیره شدند", "success")
    
    def load_models(self, path='models/gan_plate_generator.pt'):
        checkpoint = torch.load(path, map_location=DEVICE)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        self.g_losses = checkpoint['g_losses']
        self.d_losses = checkpoint['d_losses']
        print_status(f"مدل‌ها از {path} بارگذاری شدند", "success")
    
    def generate_synthetic_plates(self, num_samples=100, output_dir='dataset/synthetic'):
        """تولید تصاویر مصنوعی پلاک"""
        os.makedirs(output_dir, exist_ok=True)
        
        self.generator.eval()
        with torch.no_grad():
            for i in tqdm(range(num_samples), desc="تولید تصاویر مصنوعی"):
                z = torch.randn(1, self.latent_dim).to(DEVICE)
                img = self.generator(z).cpu()
                img = (img + 1) / 2
                img = img.squeeze(0).permute(1, 2, 0).numpy()
                img = (img * 255).astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(f'{output_dir}/synthetic_plate_{i:06d}.jpg', img)
        
        print_status(f"{num_samples} تصویر مصنوعی در {output_dir} تولید شد", "success")


# ==================== رسم نمودار ====================
def plot_losses(g_losses, d_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(g_losses, label='Generator Loss')
    plt.plot(d_losses, label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('GAN Training Losses')
    plt.legend()
    plt.grid(True)
    plt.savefig('models/gan_training_losses.png')
    plt.close()


# ==================== تابع اصلی ====================
def main():
    parser = argparse.ArgumentParser(description='آموزش GAN برای تولید تصاویر مصنوعی پلاک خودرو')
    parser.add_argument('--data', type=str, default='dataset/ILPD', help='مسیر دیتاست تصاویر واقعی')
    parser.add_argument('--epochs', type=int, default=200, help='تعداد دوره‌های آموزش')
    parser.add_argument('--batch-size', type=int, default=32, help='سایز دسته')
    parser.add_argument('--latent-dim', type=int, default=100, help='ابعاد فضای نهان')
    parser.add_argument('--lr', type=float, default=0.0002, help='نرخ یادگیری')
    parser.add_argument('--save', type=str, default='models/gan_plate_generator.pt', help='مسیر ذخیره مدل')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🎨 آموزش GAN برای تولید تصاویر مصنوعی پلاک خودرو")
    print("="*60)
    
    # بارگذاری دیتاست
    dataset = PlateDataset(args.data)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # آموزش GAN
    gan = GANTrainer(latent_dim=args.latent_dim, lr=args.lr)
    gan.train(dataloader, epochs=args.epochs)
    
    # ذخیره مدل
    gan.save_models(args.save)
    
    # رسم نمودار
    plot_losses(gan.g_losses, gan.d_losses)
    
    # تولید نمونه
    gan.generate_samples()
    
    print("\n" + "="*60)
    print_status("آموزش GAN با موفقیت به پایان رسید!", "success")
    print("="*60)


if __name__ == "__main__":
    main()
            