import cv2
import numpy as np
import os
from typing import Tuple, Optional
from denoise_algorithms import hybrid_denoise
from metrics import evaluate_denoising_quality

class ImageProcessor:
    """
    Core image processing class for X-ray image denoising.
    """
    
    def __init__(self):
        self.original_image = None
        self.denoised_image = None
        self.metrics = None
    
    def load_image(self, file_path: str) -> bool:
        """
        Load an image from file path.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                return False
            
            # Load image
            image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            if image is None:
                return False
            
            self.original_image = image
            return True
            
        except Exception as e:
            print(f"Error loading image: {e}")
            return False
    
    def process_image(self, method: str = 'hybrid') -> bool:
        """
        Apply denoising to the loaded image.
        
        Args:
            method: Denoising method to use ('hybrid', 'nlm', 'bilateral', 'wavelet')
            
        Returns:
            bool: True if successful, False otherwise
        """
        if self.original_image is None:
            return False
        
        try:
            if method == 'hybrid':
                self.denoised_image = hybrid_denoise(self.original_image)
            else:
                # For other methods, you can extend this
                from denoise_algorithms import adaptive_denoise
                self.denoised_image = adaptive_denoise(self.original_image, method)
            
            return True
            
        except Exception as e:
            print(f"Error processing image: {e}")
            return False
    
    def calculate_metrics(self) -> dict:
        """
        Calculate denoising quality metrics.
        
        Returns:
            dict: Dictionary containing PSNR and SSIM values
        """
        if self.original_image is None or self.denoised_image is None:
            return {'psnr': 0, 'ssim': 0}
        
        self.metrics = evaluate_denoising_quality(self.original_image, self.denoised_image)
        return self.metrics
    
    def save_result(self, output_path: str) -> bool:
        """
        Save the denoised image to file.
        
        Args:
            output_path: Path to save the denoised image
            
        Returns:
            bool: True if successful, False otherwise
        """
        if self.denoised_image is None:
            return False
        
        try:
            # Ensure the output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save image
            success = cv2.imwrite(output_path, self.denoised_image)
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