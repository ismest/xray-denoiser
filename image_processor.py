"""
Core image processing class for X-ray image denoising.
Improved with multi-depth support and stability fixes.
"""

import cv2
import numpy as np
import os
from typing import Tuple, Optional, Dict, Any
from denoise_algorithms import hybrid_denoise, adaptive_denoise, normalize_image, denormalize_image, safe_resize_for_display
from neural_denoise import NeuralDenoiser
from metrics import evaluate_denoising_quality


class ImageProcessor:
    """
    Core image processing class for X-ray image denoising.
    Supports multiple bit depths and formats.
    """
    
    def __init__(self):
        self.original_image = None
        self.denoised_image = None
        self.metrics = None
        self.image_info = {}
        self.neural_denoiser = NeuralDenoiser()
        
    def load_image(self, file_path: str) -> bool:
        """
        Load an image from file path with automatic depth detection.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                return False
            
            # Get file extension
            ext = os.path.splitext(file_path)[1].lower()
            
            # Try different loading methods based on format
            if ext in ['.tif', '.tiff']:
                # For TIFF, try to load with all channels
                image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            elif ext in ['.png']:
                # For PNG, preserve original depth
                image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            else:
                # For other formats, standard loading
                image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            
            if image is None:
                print(f"Failed to load image: {file_path}")
                return False
            
            # Store image info
            self.image_info = {
                'path': file_path,
                'filename': os.path.basename(file_path),
                'shape': image.shape,
                'dtype': str(image.dtype),
                'depth': image.dtype.itemsize * 8,
                'size_mb': image.nbytes / (1024 * 1024)
            }
            
            print(f"Loaded image: {self.image_info}")
            
            self.original_image = image
            self.denoised_image = None
            self.metrics = None
            
            return True
            
        except Exception as e:
            print(f"Error loading image: {e}")
            return False
    
    def process_image(self, method: str = 'hybrid', **kwargs) -> bool:
        """
        Apply denoising to the loaded image.
        
        Args:
            method: Denoising method ('hybrid', 'nlm', 'bilateral', 'wavelet', 'neural', 'gaussian')
            **kwargs: Additional parameters for the denoising algorithm
            
        Returns:
            bool: True if successful, False otherwise
        """
        if self.original_image is None:
            print("No image loaded")
            return False
        
        try:
            print(f"Processing with method: {method}")
            
            if method == 'hybrid':
                strength = kwargs.get('strength', 'medium')
                self.denoised_image = hybrid_denoise(self.original_image, strength=strength)
            
            elif method == 'neural':
                patch_size = kwargs.get('patch_size', 256)
                stride = kwargs.get('stride', 128)
                self.denoised_image = self.neural_denoiser.denoise(
                    self.original_image, 
                    patch_size=patch_size, 
                    stride=stride
                )
            
            elif method in ['nlm', 'bilateral', 'wavelet', 'gaussian']:
                self.denoised_image = adaptive_denoise(self.original_image, method=method)
            
            else:
                # Default to hybrid
                self.denoised_image = hybrid_denoise(self.original_image)
            
            # Verify output
            if self.denoised_image is None:
                print("Denoising returned None")
                return False
            
            if self.denoised_image.shape != self.original_image.shape:
                print(f"Warning: Output shape mismatch. Resizing...")
                h, w = self.original_image.shape[:2]
                if len(self.denoised_image.shape) == 2:
                    self.denoised_image = cv2.resize(self.denoised_image, (w, h))
                else:
                    self.denoised_image = cv2.resize(self.denoised_image, (w, h))
            
            print(f"Processing complete. Output dtype: {self.denoised_image.dtype}")
            return True
            
        except Exception as e:
            print(f"Error processing image: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def calculate_metrics(self) -> dict:
        """
        Calculate denoising quality metrics.
        
        Returns:
            dict: Dictionary containing PSNR and SSIM values
        """
        if self.original_image is None or self.denoised_image is None:
            return {'psnr': 0, 'ssim': 0, 'mse': 0}
        
        try:
            self.metrics = evaluate_denoising_quality(self.original_image, self.denoised_image)
            return self.metrics
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            return {'psnr': 0, 'ssim': 0, 'mse': 0}
    
    def save_result(self, output_path: str) -> bool:
        """
        Save the denoised image to file.
        
        Args:
            output_path: Path to save the denoised image
            
        Returns:
            bool: True if successful, False otherwise
        """
        if self.denoised_image is None:
            print("No denoised image to save")
            return False
        
        try:
            # Ensure the output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            # Save image
            success = cv2.imwrite(output_path, self.denoised_image)
            
            if success:
                print(f"Saved to: {output_path}")
            else:
                print(f"Failed to save to: {output_path}")
            
            return success
            
        except Exception as e:
            print(f"Error saving image: {e}")
            return False
    
    def get_original_image(self):
        """Get original image."""
        return self.original_image
    
    def get_denoised_image(self):
        """Get denoised image."""
        return self.denoised_image
    
    def get_metrics(self):
        """Get current metrics."""
        return self.metrics
    
    def get_image_info(self) -> Dict[str, Any]:
        """Get information about the loaded image."""
        return self.image_info
    
    def reset(self):
        """Reset processor state."""
        self.original_image = None
        self.denoised_image = None
        self.metrics = None
        self.image_info = {}
    
    def get_supported_methods(self) -> list:
        """Get list of supported denoising methods."""
        methods = [
            ('hybrid', 'Hybrid (Recommended)'),
            ('nlm', 'Non-local Means'),
            ('bilateral', 'Bilateral Filter'),
            ('wavelet', 'Wavelet'),
            ('gaussian', 'Gaussian Filter'),
        ]
        
        # Add neural if available
        if self.neural_denoiser.is_available():
            methods.append(('neural', 'Neural Network'))
        
        return methods
