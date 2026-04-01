"""
Core image processing class for X-ray image denoising.
Improved with multi-depth support and stability fixes.
"""

import cv2
import numpy as np
import os
from typing import Tuple, Optional, Dict, Any
from denoise_algorithms import (
    hybrid_denoise, adaptive_denoise, normalize_image, denormalize_image,
    safe_resize_for_display, non_local_means_denoise, bilateral_filter_denoise,
    wavelet_denoise, gaussian_denoise, bm3d_denoise, anisotropic_diffusion_denoise,
    iterative_reconstruction_denoise
)
from neural_denoise import NeuralDenoiser
from metrics import evaluate_denoising_quality, evaluate_super_resolution, compare_sr_with_reference
from super_resolution import super_resolution_denoised_image, get_supported_sr_methods

# Try to import PIL for better format support
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("Warning: PIL not available. Some image formats may not load properly.")


class ImageProcessor:
    """
    Core image processing class for X-ray image denoising.
    Supports multiple bit depths and formats.
    """

    def __init__(self):
        self.original_image = None
        self.denoised_image = None
        self.sr_image = None  # Super-resolution result
        self.metrics = None
        self.image_info = {}
        self.neural_denoiser = NeuralDenoiser()
        self.sr_metrics = None  # Metrics after super-resolution

    def load_image(self, file_path: str) -> bool:
        """
        Load an image from file path with automatic depth detection.
        Uses PIL for better format support, falls back to OpenCV.

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

            image = None
            use_pil = False

            # Try PIL first for better format support
            if HAS_PIL:
                try:
                    with Image.open(file_path) as img:
                        # Convert to numpy array
                        # PIL loads as RGBA for PNG with alpha, RGB for color, L for grayscale
                        image = np.array(img)
                        use_pil = True
                        print(f"Loaded with PIL: {img.mode}, {img.size}")
                except Exception as pil_error:
                    print(f"PIL loading failed: {pil_error}")
                    # Fall through to OpenCV

            # If PIL failed or not available, try OpenCV
            if image is None:
                # Try multiple OpenCV flags for robust loading
                image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

                # If still failed, try with grayscale
                if image is None:
                    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

                if image is None:
                    print(f"Failed to load image with OpenCV: {file_path}")
                    return False

            # Handle PNG with alpha channel (RGBA) - convert to RGB
            if use_pil and len(image.shape) == 3 and image.shape[2] == 4:
                # Remove alpha channel, keep RGB
                image = image[:, :, :3]

            # Ensure image is in proper numpy format
            if image is None or image.size == 0:
                print(f"Loaded image is empty: {file_path}")
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
            self.sr_image = None
            self.metrics = None
            self.sr_metrics = None

            return True

        except Exception as e:
            print(f"Error loading image: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def process_image(self, method: str = 'hybrid', **kwargs) -> bool:
        """
        Step 1: Apply denoising to the loaded image.

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
            print(f"Step 1 - Denoising with method: {method}, params: {kwargs}")

            # Store original dtype for verification
            original_dtype = self.original_image.dtype

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

            elif method == 'nlm':
                # Pass NLM parameters from UI
                h = kwargs.get('h', 10)
                patch_size = kwargs.get('patch_size', 7)
                self.denoised_image = non_local_means_denoise(
                    self.original_image,
                    h=h,
                    patch_size=patch_size
                )

            elif method == 'bilateral':
                # Pass bilateral parameters from UI
                d = kwargs.get('d', 9)
                sigma_color = kwargs.get('sigma_color', 75)
                sigma_space = kwargs.get('sigma_space', 75)
                self.denoised_image = bilateral_filter_denoise(
                    self.original_image,
                    d=d,
                    sigma_color=sigma_color,
                    sigma_space=sigma_space
                )

            elif method == 'wavelet':
                self.denoised_image = wavelet_denoise(self.original_image)

            elif method == 'gaussian':
                self.denoised_image = gaussian_denoise(self.original_image)

            elif method == 'bm3d':
                sigma = kwargs.get('sigma', 20)
                self.denoised_image = bm3d_denoise(self.original_image, sigma_psd=sigma)

            elif method == 'anisotropic':
                niter = kwargs.get('niter', 10)
                kappa = kwargs.get('kappa', 50)
                self.denoised_image = anisotropic_diffusion_denoise(
                    self.original_image,
                    niter=niter,
                    kappa=kappa
                )

            elif method == 'iterative':
                niter = kwargs.get('niter', 5)
                regularization = kwargs.get('regularization', 0.1)
                method_type = kwargs.get('recon_method', 'tv')
                self.denoised_image = iterative_reconstruction_denoise(
                    self.original_image,
                    niter=niter,
                    regularization=regularization,
                    method=method_type
                )

            else:
                # Default to hybrid
                self.denoised_image = hybrid_denoise(self.original_image)

            # Verify output
            if self.denoised_image is None:
                print("Denoising returned None")
                return False

            # Handle shape mismatch with dtype preservation
            if self.denoised_image.shape != self.original_image.shape:
                print(f"Warning: Output shape mismatch. Resizing...")
                h, w = self.original_image.shape[:2]
                resized = cv2.resize(self.denoised_image, (w, h))

                # Restore original dtype after resize
                if original_dtype == np.uint8:
                    self.denoised_image = np.clip(resized * 255.0, 0, 255).round().astype(np.uint8)
                elif original_dtype == np.uint16:
                    self.denoised_image = np.clip(resized * 65535.0, 0, 65535).round().astype(np.uint16)
                else:
                    self.denoised_image = resized.astype(original_dtype)
            else:
                self.denoised_image = self.denoised_image

            # Calculate denoising metrics
            self.metrics = evaluate_denoising_quality(self.original_image, self.denoised_image)

            print(f"Denoising complete. Output dtype: {self.denoised_image.dtype}")
            return True

        except Exception as e:
            print(f"Error processing image: {e}")
            import traceback
            traceback.print_exc()
            return False

    def apply_super_resolution(self, scale: float = 2.0, method: str = 'lanczos',
                                enhance_edges: bool = True, enhance_contrast: bool = True) -> bool:
        """
        Step 2: Apply super-resolution to the denoised image.

        Args:
            scale: Upscaling factor (1.5, 2.0, 3.0, 4.0)
            method: SR method ('bicubic', 'lanczos', 'edge_preserving')
            enhance_edges: Whether to apply edge enhancement
            enhance_contrast: Whether to apply contrast enhancement

        Returns:
            bool: True if successful, False otherwise
        """
        if self.denoised_image is None:
            print("No denoised image available for super-resolution")
            return False

        try:
            print(f"Step 2 - Super-resolution: scale={scale}, method={method}")

            self.sr_image = super_resolution_denoised_image(
                self.denoised_image,
                scale=scale,
                method=method,
                enhance_edges=enhance_edges,
                enhance_contrast=enhance_contrast
            )

            if self.sr_image is None:
                print("Super-resolution returned None")
                return False

            # Calculate comprehensive SR metrics
            # 1. Resolution verification and quality metrics
            self.sr_metrics = compare_sr_with_reference(
                sr_image=self.sr_image,
                original_lr_image=self.denoised_image,
                expected_scale=scale
            )

            # 2. Additional no-reference quality assessment
            sr_quality = evaluate_super_resolution(self.sr_image, scale_factor=scale)
            self.sr_metrics.update(sr_quality)

            print(f"Super-resolution complete. Output shape: {self.sr_image.shape}")
            print(f"SR Metrics: sharpness={self.sr_metrics.get('sharpness', 0):.2f}, "
                  f"edge_strength={self.sr_metrics.get('edge_strength', 0):.4f}, "
                  f"entropy={self.sr_metrics.get('entropy', 0):.2f}")
            return True

        except Exception as e:
            print(f"Error applying super-resolution: {e}")
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
    
    def save_result(self, output_path: str, use_sr: bool = False) -> bool:
        """
        Save the processed image to file.

        Args:
            output_path: Path to save the image
            use_sr: If True, save super-resolution image; if False, save denoised image

        Returns:
            bool: True if successful, False otherwise
        """
        image_to_save = self.sr_image if use_sr and self.sr_image is not None else self.denoised_image

        if image_to_save is None:
            print("No processed image to save")
            return False

        try:
            # Ensure the output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            # Save image
            success = cv2.imwrite(output_path, image_to_save)

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

    def get_sr_image(self):
        """Get super-resolution image."""
        return self.sr_image

    def get_metrics(self):
        """Get current metrics."""
        return self.metrics

    def get_sr_metrics(self):
        """Get super-resolution metrics."""
        return self.sr_metrics

    def get_image_info(self) -> Dict[str, Any]:
        """Get information about the loaded image."""
        return self.image_info

    def reset(self):
        """Reset processor state."""
        self.original_image = None
        self.denoised_image = None
        self.sr_image = None
        self.metrics = None
        self.sr_metrics = None
        self.image_info = {}
    
    def get_supported_methods(self) -> list:
        """Get list of supported denoising methods."""
        methods = [
            ('hybrid', 'Hybrid (Recommended)'),
            ('bm3d', 'BM3D (Block-Matching 3D)'),
            ('anisotropic', 'Anisotropic Diffusion'),
            ('iterative', 'Iterative Reconstruction'),
            ('nlm', 'Non-local Means'),
            ('bilateral', 'Bilateral Filter'),
            ('wavelet', 'Wavelet'),
            ('gaussian', 'Gaussian Filter'),
        ]

        # Add neural if available
        if self.neural_denoiser.is_available():
            methods.append(('neural', 'Neural Network'))

        return methods
