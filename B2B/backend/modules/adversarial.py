"""
adversarial.py - دفاع در برابر حملات Adversarial (FGSM, PGD, BIM) و آموزش مقاوم
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class AdversarialAttack:
    def init(self, model: nn.Module, epsilon: float = 0.03, targeted: bool = False):
        self.model = model
        self.epsilon = epsilon
        self.targeted = targeted
        self.model.eval()
    
    def _compute_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.targeted:
            return -F.cross_entropy(outputs, targets)
        return F.cross_entropy(outputs, targets)


class FGSM(AdversarialAttack):
    def generate(self, images: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        images = images.clone().detach().requires_grad_(True)
        outputs = self.model(images)
        loss = self._compute_loss(outputs, targets)
        self.model.zero_grad()
        loss.backward()
        grad_sign = images.grad.data.sign()
        perturbed = images + self.epsilon * grad_sign
        return torch.clamp(perturbed, 0, 1).detach()


class PGD(AdversarialAttack):
    def init(self, model: nn.Module, epsilon: float = 0.03, alpha: float = 0.01, num_iter: int = 10, random_start: bool = True, targeted: bool = False):
        super().__init__(model, epsilon, targeted)
        self.alpha = alpha
        self.num_iter = num_iter
        self.random_start = random_start
    
    def generate(self, images: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.random_start:
            perturbed = images + torch.empty_like(images).uniform_(-self.epsilon, self.epsilon)
            perturbed = torch.clamp(perturbed, 0, 1)
        else:
            perturbed = images.clone()
        for _ in range(self.num_iter):
            perturbed.requires_grad_(True)
            outputs = self.model(perturbed)
            loss = self._compute_loss(outputs, targets)
            self.model.zero_grad()
            loss.backward()
            grad_sign = perturbed.grad.data.sign()
            perturbed = perturbed + self.alpha * grad_sign
            eta = torch.clamp(perturbed - images, -self.epsilon, self.epsilon)
            perturbed = torch.clamp(images + eta, 0, 1)
        return perturbed.detach()


class BIM(AdversarialAttack):
    def init(self, model: nn.Module, epsilon: float = 0.03, alpha: float = 0.005, num_iter: int = 10, targeted: bool = False):
        super().__init__(model, epsilon, targeted)
        self.alpha = alpha
        self.num_iter = num_iter
    
    def generate(self, images: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        perturbed = images.clone()
        for _ in range(self.num_iter):
            perturbed.requires_grad_(True)
            outputs = self.model(perturbed)
            loss = self._compute_loss(outputs, targets)
            self.model.zero_grad()
            loss.backward()
            grad_sign = perturbed.grad.data.sign()
            perturbed = perturbed + self.alpha * grad_sign
            eta = torch.clamp(perturbed - images, -self.epsilon, self.epsilon)
            perturbed = torch.clamp(images + eta, 0, 1)
        return perturbed.detach()


class AdversarialDefense:
    def init(self, model: nn.Module, device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device
    
    def generate_adversarial_examples(self, images: torch.Tensor, labels: torch.Tensor, attack_type: str = 'pgd', epsilon: float = 0.03, **kwargs) -> torch.Tensor:
        if attack_type.lower() == 'fgsm':
            attacker = FGSM(self.model, epsilon)
        elif attack_type.lower() == 'pgd':
            alpha = kwargs.get('alpha', 0.01); num_iter = kwargs.get('num_iter', 10); random_start = kwargs.get('random_start', True)
            attacker = PGD(self.model, epsilon, alpha, num_iter, random_start)
        elif attack_type.lower() == 'bim':
            alpha = kwargs.get('alpha', 0.005); num_iter = kwargs.get('num_iter', 10)
            attacker = BIM(self.model, epsilon, alpha, num_iter)
        else:
            raise ValueError(f"Unknown attack: {attack_type}")
        return attacker.generate(images, labels)
    
    def adversarial_training(self, train_loader, optimizer, criterion, epochs: int = 10, attack_type: str = 'pgd', epsilon: float = 0.03, alpha: float = 0.01, num_iter: int = 7, adv_ratio: float = 0.5) -> List[float]:
        losses = []
        for epoch in range(epochs):
            total_loss = 0.0
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                adv_images = self.generate_adversarial_examples(images, labels, attack_type, epsilon, alpha=alpha, num_iter=num_iter)
                batch_size = images.size(0)
                num_adv = int(batch_size * adv_ratio)
                if num_adv > 0:
                    combined = torch.cat([images[num_adv:], adv_images[:num_adv]], dim=0)
                    combined_labels = torch.cat([labels[num_adv:], labels[:num_adv]], dim=0)
                else:
                    combined, combined_labels = images, labels
                optimizer.zero_grad()
                outputs = self.model(combined)
                loss = criterion(outputs, combined_labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)
            losses.append(avg_loss)
            logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
        return losses
    
    def evaluate_robustness(self, test_loader, attack_type: str = 'pgd', epsilon: float = 0.03,** kwargs) -> Dict:
        self.model.eval()
        correct_clean, correct_adv, total = 0, 0, 0
        for images, labels in test_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = self.model(images)
            _, pred = torch.max(outputs, 1)
            correct_clean += (pred == labels).sum().item()
            adv_images = self.generate_adversarial_examples(images, labels, attack_type, epsilon, kwargs)
            outputs_adv = self.model(adv_images)
            _, pred_adv = torch.max(outputs_adv, 1)
            correct_adv += (pred_adv == labels).sum().item()
            total += labels.size(0)
        return {"clean_accuracy": 100.0 * correct_clean / total, "adversarial_accuracy": 100.0 * correct_adv / total, "robustness_degradation": 100.0 * (correct_clean - correct_adv) / total}