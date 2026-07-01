"""
domain_adaptation.py - تطابق دامنه و تولید داده مصنوعی با GAN
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class Generator(nn.Module):
    def init(self, latent_dim=100, img_channels=3, img_size=64):
        super().__init__()
        self.init_size = img_size // 4  # 16
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128 * self.init_size**  2),
            nn.BatchNorm1d(128 * self.init_size ** 2),
            nn.ReLU()
        )
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, img_channels, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.fc(z)
        out = out.view(out.size(0), 128, self.init_size, self.init_size)
        return self.conv(out)


class Discriminator(nn.Module):
    def init(self, img_channels=3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(img_channels, 32, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(256, 1), nn.Sigmoid()
        )

    def forward(self, img):
        return self.model(img)


class DomainAdaptor:
    def init(self, latent_dim=100, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.generator = Generator(latent_dim).to(self.device)
        self.discriminator = Discriminator().to(self.device)
        self.latent_dim = latent_dim

    def train_gan(self, dataloader, epochs=100, lr=0.0002):
        g_opt = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        d_opt = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        criterion = nn.BCELoss()
        for epoch in range(epochs):
            for real_imgs in dataloader:
                real_imgs = real_imgs.to(self.device)
                batch_size = real_imgs.size(0)
                real_labels = torch.ones(batch_size, 1).to(self.device)
                fake_labels = torch.zeros(batch_size, 1).to(self.device)

                # Train D
                noise = torch.randn(batch_size, self.latent_dim).to(self.device)
                fake_imgs = self.generator(noise)
                d_loss = criterion(self.discriminator(real_imgs), real_labels) + \
                         criterion(self.discriminator(fake_imgs.detach()), fake_labels)
                d_opt.zero_grad()
                d_loss.backward()
                d_opt.step()

                # Train G
                noise = torch.randn(batch_size, self.latent_dim).to(self.device)
                fake_imgs = self.generator(noise)
                g_loss = criterion(self.discriminator(fake_imgs), real_labels)
                g_opt.zero_grad()
                g_loss.backward()
                g_opt.step()
            if (epoch+1) % 20 == 0:
                logger.info(f"GAN Epoch {epoch+1}: D_loss={d_loss.item():.4f}, G_loss={g_loss.item():.4f}")

    def generate_synthetic(self, num_samples=16) -> np.ndarray:
        self.generator.eval()
        with torch.no_grad():
            noise = torch.randn(num_samples, self.latent_dim).to(self.device)
            imgs = self.generator(noise).cpu().numpy()
            imgs = (imgs.transpose(0, 2, 3, 1) + 1) / 2
            return np.clip(imgs, 0, 1)