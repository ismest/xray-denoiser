import numpy as np
from skimage.metrics import structural_similarity as ssim
import cv2

def calculate_psnr(original, processed):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) between two images.
    
    Args:
        original: Original image (numpy array)
        processed: Processed/filtered image (numpy array)
    
    Returns:
        float: PSNR value in dB
    """
    if original.dtype != processed.dtype:
        processed = processed.astype(original.dtype)
    
    mse = np.mean((original - processed) ** 2)
    if mse == 0:
        return float('inf')
    
    max_pixel = 255.0 if original.dtype == np.uint8 else 1.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def calculate_ssim(original, processed, multichannel=True):
    """
    Calculate Structural Similarity Index (SSIM) between two images.
    
    Args:
        original: Original image (numpy array)
        processed: Processed/filtered image (numpy array)
        multichannel: Whether to treat the last dimension as channels
    
    Returns:
        float: SSIM value (0-1, higher is better)
    """
    if original.shape != processed.shape:
        # Resize processed image to match original
        processed = cv2.resize(processed, (original.shape[1], original.shape[0]))
    
    if original.dtype != processed.dtype:
        processed = processed.astype(original.dtype)
    
    # Ensure images are in range [0, 1] for SSIM calculation
    if original.dtype == np.uint8:
        original_norm = original.astype(np.float64) / 255.0
        processed_norm = processed.astype(np.float64) / 255.0
    else:
        original_norm = original
        processed_norm = processed
    
    try:
        if len(original_norm.shape) == 2:
            # Grayscale image
            ssim_score = ssim(original_norm, processed_norm, multichannel=False)
        else:
            # Color image
            ssim_score = ssim(original_norm, processed_norm, multichannel=multichannel)
    except Exception as e:
        # Fallback for edge cases
        ssim_score = 0.0
    
    return ssim_score

def evaluate_denoising_quality(original, denoised):
    """
    Comprehensive evaluation of denoising quality.
    
    Args:
        original: Original noisy image
        denoised: Denoised image
    
    Returns:
        dict: Dictionary containing PSNR and SSIM values
    """
    psnr = calculate_psnr(original, denoised)
    ssim_score = calculate_ssim(original, denoised)
    
    return {
        'psnr': psnr,
        'ssim': ssim_score
    }