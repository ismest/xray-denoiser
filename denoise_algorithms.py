"""
Improved denoising algorithms with multi-depth support and stability fixes.
"""

import numpy as np
import cv2
from skimage.restoration import denoise_nl_means, denoise_wavelet
from skimage.util import img_as_float, img_as_ubyte
import warnings

# Suppress skimage warnings
warnings.filterwarnings('ignore', category=UserWarning)


def normalize_image(image):
    """
    Normalize any image depth to float64 [0, 1] range.
    
    Args:
        image: Input image (uint8, uint16, float32, float64)
        
    Returns:
        tuple: (normalized_image, original_dtype, original_max)
    """
    original_dtype = image.dtype
    
    if original_dtype == np.uint8:
        original_max = 255.0
        normalized = image.astype(np.float64) / 255.0
    elif original_dtype == np.uint16:
        original_max = 65535.0
        normalized = image.astype(np.float64) / 65535.0
    elif original_dtype == np.float32:
        original_max = image.max() if image.max() > 0 else 1.0
        normalized = image.astype(np.float64)
        if original_max > 1.0:
            normalized = normalized / original_max
    elif original_dtype == np.float64:
        original_max = image.max() if image.max() > 0 else 1.0
        normalized = image.copy()
        if original_max > 1.0:
            normalized = normalized / original_max
    else:
        # Fallback for other types
        original_max = 255.0
        normalized = image.astype(np.float64) / 255.0
    
    return normalized, original_dtype, original_max


def denormalize_image(normalized, original_dtype, original_max):
    """
    Convert normalized float64 image back to original depth.
    
    Args:
        normalized: Float64 image in [0, 1] range
        original_dtype: Target dtype
        original_max: Original maximum value
        
    Returns:
        Image in original dtype
    """
    # Clip to valid range
    normalized = np.clip(normalized, 0.0, 1.0)
    
    if original_dtype == np.uint8:
        return (normalized * 255.0).round().astype(np.uint8)
    elif original_dtype == np.uint16:
        return (normalized * 65535.0).round().astype(np.uint16)
    elif original_dtype == np.float32:
        if original_max > 1.0:
            return (normalized * original_max).astype(np.float32)
        return normalized.astype(np.float32)
    elif original_dtype == np.float64:
        if original_max > 1.0:
            return normalized * original_max
        return normalized
    else:
        return (normalized * 255.0).round().astype(np.uint8)


def safe_resize_for_display(image, max_size=4000):
    """
    Safely resize very large images to prevent memory issues.
    
    Args:
        image: Input image
        max_size: Maximum dimension size
        
    Returns:
        Resized image and scale factor
    """
    h, w = image.shape[:2]
    max_dim = max(h, w)
    
    if max_dim > max_size:
        scale = max_size / max_dim
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Use area interpolation for downscaling (better for images)
        if len(image.shape) == 2:
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        return resized, scale
    
    return image, 1.0


def non_local_means_denoise(image, h=10, patch_size=7, patch_distance=21):
    """
    Non-local means denoising with multi-depth support.
    
    Args:
        image: Input image (any depth)
        h: Filter strength
        patch_size: Size of patches
        patch_distance: Max distance for patch comparison
    
    Returns:
        Denoised image in original depth
    """
    try:
        # Handle very small images
        min_dim = min(image.shape[:2])
        if min_dim < 5:
            # For extremely small images, use simple Gaussian blur
            return gaussian_denoise(image, kernel_size=3)
        elif min_dim < patch_size:
            # Fall back to bilateral for tiny images
            return bilateral_filter_denoise(image)
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            if image.shape[2] == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image[:,:,0]
        else:
            gray = image
        
        # Normalize
        normalized, original_dtype, original_max = normalize_image(gray)
        
        # Adjust h for normalized image
        h_normalized = h / 255.0
        
        # Apply non-local means with safe bounds
        min_dim = min(gray.shape)
        safe_patch_size = max(3, min(patch_size, min_dim // 2))
        safe_patch_distance = min(patch_distance, min_dim // 2)
        
        # Ensure patch_distance is at least patch_size
        safe_patch_distance = max(safe_patch_distance, safe_patch_size)
        
        denoised = denoise_nl_means(
            normalized, 
            h=h_normalized,
            fast_mode=True,
            patch_size=safe_patch_size,
            patch_distance=safe_patch_distance
        )
        
        # Convert back to original depth
        return denormalize_image(denoised, original_dtype, original_max)
        
    except Exception as e:
        print(f"NLM denoising failed: {e}")
        # Fallback to bilateral filter
        return bilateral_filter_denoise(image)


def bilateral_filter_denoise(image, d=9, sigma_color=75, sigma_space=75):
    """
    Bilateral filter denoising with multi-depth support.
    
    Args:
        image: Input image (any depth)
        d: Diameter of pixel neighborhood
        sigma_color: Filter sigma in color space
        sigma_space: Filter sigma in coordinate space
    
    Returns:
        Denoised image in original depth
    """
    try:
        # Ensure d is odd and reasonable
        d = max(3, d if d % 2 == 1 else d + 1)
        d = min(d, min(image.shape[:2]) - 1)
        
        # Normalize to uint8 for OpenCV (most stable)
        normalized, original_dtype, original_max = normalize_image(image)
        uint8_img = (normalized * 255).round().astype(np.uint8)
        
        if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
            # Grayscale
            denoised_uint8 = cv2.bilateralFilter(uint8_img, d, sigma_color, sigma_space)
        else:
            # Color image
            denoised_uint8 = cv2.bilateralFilter(uint8_img, d, sigma_color, sigma_space)
        
        # Convert back to normalized float
        denoised = denoised_uint8.astype(np.float64) / 255.0
        
        # Convert back to original depth
        return denormalize_image(denoised, original_dtype, original_max)
        
    except Exception as e:
        print(f"Bilateral filter failed: {e}")
        # Last resort: Gaussian blur
        return gaussian_denoise(image)


def wavelet_denoise(image, wavelet='db1', threshold='BayesShrink', mode='soft'):
    """
    Wavelet-based denoising with multi-depth support.
    
    Args:
        image: Input image (any depth)
        wavelet: Wavelet to use
        threshold: Thresholding method
        mode: Thresholding mode
    
    Returns:
        Denoised image in original depth
    """
    try:
        # Handle very small images
        min_dim = min(image.shape[:2])
        if min_dim < 8:
            return bilateral_filter_denoise(image)
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            if image.shape[2] == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image[:,:,0]
        else:
            gray = image
        
        # Normalize
        normalized, original_dtype, original_max = normalize_image(gray)
        
        # Apply wavelet denoising
        denoised = denoise_wavelet(
            normalized,
            wavelet=wavelet,
            method=threshold,
            mode=mode,
            rescale_sigma=True
        )
        
        # Convert back to original depth
        return denormalize_image(denoised, original_dtype, original_max)
        
    except Exception as e:
        print(f"Wavelet denoising failed: {e}")
        return bilateral_filter_denoise(image)


def gaussian_denoise(image, kernel_size=5, sigma=1.0):
    """
    Simple Gaussian denoising as fallback.
    
    Args:
        image: Input image
        kernel_size: Gaussian kernel size
        sigma: Gaussian sigma
        
    Returns:
        Denoised image
    """
    try:
        # Ensure kernel size is odd
        kernel_size = max(3, kernel_size if kernel_size % 2 == 1 else kernel_size + 1)
        
        # Normalize to uint8 for OpenCV
        normalized, original_dtype, original_max = normalize_image(image)
        uint8_img = (normalized * 255).round().astype(np.uint8)
        
        if len(image.shape) == 2:
            denoised_uint8 = cv2.GaussianBlur(uint8_img, (kernel_size, kernel_size), sigma)
        else:
            denoised_uint8 = cv2.GaussianBlur(uint8_img, (kernel_size, kernel_size), sigma)
        
        denoised = denoised_uint8.astype(np.float64) / 255.0
        return denormalize_image(denoised, original_dtype, original_max)
        
    except Exception as e:
        print(f"Gaussian denoising failed: {e}")
        # Ultimate fallback: return original
        return image


def adaptive_denoise(image, method='auto', noise_estimate=None):
    """
    Adaptive denoising that selects the best method based on image characteristics.
    
    Args:
        image: Input image (any depth)
        method: 'auto', 'nlm', 'bilateral', 'wavelet', 'gaussian'
        noise_estimate: Estimated noise level (optional)
    
    Returns:
        Denoised image in original depth
    """
    try:
        if method == 'auto':
            # Auto-select based on image size and characteristics
            h, w = image.shape[:2]
            
            if h < 50 or w < 50:
                # Very small image - use bilateral
                return bilateral_filter_denoise(image)
            elif h > 4000 or w > 4000:
                # Very large image - resize first to prevent memory issues
                max_size = 2000
                if h > w:
                    scale = max_size / h
                    new_h, new_w = max_size, int(w * scale)
                else:
                    scale = max_size / w
                    new_h, new_w = int(h * scale), max_size
                
                # Ensure minimum dimensions
                new_h = max(new_h, 100)
                new_w = max(new_w, 100)
                
                resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
                denoised_resized = non_local_means_denoise(resized, h=8)
                # Resize back with proper interpolation
                denoised = cv2.resize(denoised_resized, (w, h), interpolation=cv2.INTER_CUBIC)
                return denoised
            else:
                # Normal size - use NLM
                return non_local_means_denoise(image)
        
        elif method == 'nlm':
            return non_local_means_denoise(image)
        elif method == 'bilateral':
            return bilateral_filter_denoise(image)
        elif method == 'wavelet':
            return wavelet_denoise(image)
        elif method == 'gaussian':
            return gaussian_denoise(image)
        else:
            return non_local_means_denoise(image)
            
    except Exception as e:
        print(f"Adaptive denoising failed: {e}")
        return gaussian_denoise(image)


def hybrid_denoise(image, strength='medium'):
    """
    Hybrid denoising approach combining multiple methods.
    
    Args:
        image: Input image (any depth)
        strength: 'low', 'medium', 'high' - controls denoising strength
    
    Returns:
        Denoised image in original depth
    """
    try:
        # Handle very small images
        if image.shape[0] < 20 or image.shape[1] < 20:
            return bilateral_filter_denoise(image)
        
        # Set parameters based on strength
        if strength == 'low':
            h, patch_size, d = 6, 5, 5
            sigma_color, sigma_space = 50, 50
        elif strength == 'high':
            h, patch_size, d = 15, 9, 13
            sigma_color, sigma_space = 100, 100
        else:  # medium
            h, patch_size, d = 10, 7, 9
            sigma_color, sigma_space = 75, 75
        
        # First pass: Non-local means
        denoised_nlm = non_local_means_denoise(image, h=h, patch_size=patch_size, patch_distance=15)
        
        # Second pass: Bilateral filter to preserve edges
        denoised_final = bilateral_filter_denoise(denoised_nlm, d=d, sigma_color=sigma_color, sigma_space=sigma_space)
        
        return denoised_final
        
    except Exception as e:
        print(f"Hybrid denoising failed: {e}")
        return adaptive_denoise(image, method='auto')
