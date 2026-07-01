"""
xia.py - توضیح‌پذیری مدل‌ها با GradCAM و SHAP
"""

import torch
import cv2
import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class XAIExplainer:
    def init(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.device = next(model.parameters()).device

    def grad_cam(self, image: np.ndarray, target_class: Optional[int] = None) -> np.ndarray:
        from pytorch_gradcam import GradCAM
        gradcam = GradCAM(model=self.model, target_layers=[self.target_layer])
        tensor = torch.tensor(image).permute(2,0,1).unsqueeze(0).float() / 255.0
        tensor = tensor.to(self.device)
        heatmap = gradcam(input_tensor=tensor, targets=None)
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(image, 0.6, colored, 0.4, 0)
        return overlay

    def shap_explain(self, model, background_images, test_image):
        import shap
        explainer = shap.DeepExplainer(model, background_images)
        shap_values = explainer.shap_values(test_image)
        shap.image_plot(shap_values, test_image)
        return shap_values

    def attention_map(self, image: np.ndarray) -> np.ndarray:
        # Simple gradient-based attention
        tensor = torch.tensor(image).permute(2,0,1).unsqueeze(0).float() / 255.0
        tensor.requires_grad = True
        tensor = tensor.to(self.device)
        output = self.model(tensor)
        output[0, output.argmax()].backward()
        grad = tensor.grad.squeeze().cpu().numpy()
        att_map = np.mean(np.abs(grad), axis=0)
        att_map = cv2.resize(att_map, (image.shape[1], image.shape[0]))
        att_map = np.uint8(255 * (att_map - att_map.min()) / (att_map.max() - att_map.min() + 1e-8))
        colored = cv2.applyColorMap(att_map, cv2.COLORMAP_JET)
        return cv2.addWeighted(image, 0.6, colored, 0.4, 0)