"""
feature_engineering.py - استخراج ویژگی‌های دستی (HOG، LBP، Color Histogram، Moments)
"""

import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from typing import Tuple, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class FeatureExtractor:
    @staticmethod
    def extract_hog(image: np.ndarray, orientations: int = 9, pixels_per_cell: Tuple[int, int] = (8, 8),
                    cells_per_block: Tuple[int, int] = (2, 2), visualize: bool = False) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        features, hog_img = hog(gray, orientations=orientations, pixels_per_cell=pixels_per_cell,
                                cells_per_block=cells_per_block, visualize=True, feature_vector=True)
        return (features, hog_img) if visualize else features

    @staticmethod
    def extract_lbp(image: np.ndarray, radius: int = 1, n_points: int = 8, method: str = 'uniform') -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lbp = local_binary_pattern(gray, n_points, radius, method)
        n_bins = n_points + 2 if method == 'uniform' else 256
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
        hist = hist.astype(np.float32)
        hist /= (hist.sum() + 1e-6)
        return hist

    @staticmethod
    def extract_color_histogram(image: np.ndarray, bins: Tuple[int, int, int] = (8, 8, 8), normalize: bool = True) -> np.ndarray:
        hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
        if normalize:
            hist = cv2.normalize(hist, hist).flatten()
        else:
            hist = hist.flatten()
        return hist

    @staticmethod
    def extract_hsv_histogram(image: np.ndarray, bins: Tuple[int, int] = (50, 60), normalize: bool = True) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, bins, [0, 180, 0, 256])
        if normalize:
            hist = cv2.normalize(hist, hist).flatten()
        else:
            hist = hist.flatten()
        return hist

    @staticmethod
    def extract_edge_density(image: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_pixels = np.count_nonzero(edges)
        total_pixels = edges.shape[0] * edges.shape[1]
        return edge_pixels / total_pixels

    @staticmethod
    def extract_moment_features(image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        moments = cv2.moments(gray)
        hu_moments = cv2.HuMoments(moments).flatten()
        hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
        return hu_moments

    @staticmethod
    def extract_all_features(image: np.ndarray) -> Dict[str, Any]:
        return {
            "hog": FeatureExtractor.extract_hog(image).tolist(),
            "lbp": FeatureExtractor.extract_lbp(image).tolist(),
            "color_hist_rgb": FeatureExtractor.extract_color_histogram(image).tolist(),
            "color_hist_hsv": FeatureExtractor.extract_hsv_histogram(image).tolist(),
            "edge_density": FeatureExtractor.extract_edge_density(image),
            "hu_moments": FeatureExtractor.extract_moment_features(image).tolist()
        }

    @staticmethod
    def combine_features(image: np.ndarray) -> np.ndarray:
        hog_feat = FeatureExtractor.extract_hog(image)
        lbp_feat = FeatureExtractor.extract_lbp(image)
        color_feat = FeatureExtractor.extract_color_histogram(image)
        return np.concatenate([hog_feat, lbp_feat, color_feat])