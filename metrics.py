"""
Image quality metrics with multi-depth support.
"""

import numpy as np
from skimage.metrics import structural_similarity as ssim
import cv2


def normalize_for_comparison(img1, img2):
    """
    Normalize two images to the same format for comparison.
    
    Args:
        img1, img2: Input images
        
    Returns:
        tuple: (normalized_img1, normalized_img2, max_value)
    """
    # Convert to float64 for calculation
    if img1.dtype == np.uint8:
        img1_norm = img1.astype(np.float64) / 255.0
        max_val = 255.0
    elif img1.dtype == np.uint16:
        img1_norm = img1.astype(np.float64) / 65535.0
        max_val = 65535.0
    elif img1.dtype == np.float32 or img1.dtype == np.float64:
        img1_norm = img1.astype(np.float64)
        max_val = img1.max() if img1.max() > 0 else 1.0
        if max_val > 1.0:
            img1_norm = img1_norm / max_val
    else:
        img1_norm = img1.astype(np.float64) / 255.0
        max_val = 255.0
    
    # Normalize second image to match
    if img2.dtype != img1.dtype:
        if img2.dtype == np.uint8:
            img2_norm = img2.astype(np.float64) / 255.0
        elif img2.dtype == np.uint16:
            img2_norm = img2.astype(np.float64) / 65535.0
        else:
            img2_norm = img2.astype(np.float64)
            if img2.max() > 1.0:
                img2_norm = img2_norm / img2.max()
    else:
        img2_norm = img1_norm.copy()
        if img2.dtype == np.float32 or img2.dtype == np.float64:
            if img2.max() > 1.0:
                img2_norm = img2.astype(np.float64) / max_val
    
    return img1_norm, img2_norm, max_val


def calculate_psnr(original, processed):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) between two images.
    Supports multiple bit depths.
    
    Args:
        original: Original image (numpy array)
        processed: Processed/filtered image (numpy array)
    
    Returns:
        float: PSNR value in dB
    """
    try:
        # Normalize images for comparison
        orig_norm, proc_norm, max_val = normalize_for_comparison(original, processed)
        
        # Ensure same shape
        if orig_norm.shape != proc_norm.shape:
            # Resize processed to match original
            h, w = orig_norm.shape[:2]
            if len(proc_norm.shape) == 2:
                proc_norm = cv2.resize(proc_norm, (w, h))
            else:
                proc_norm = cv2.resize(proc_norm, (w, h))
        
        # Calculate MSE
        mse = np.mean((orig_norm - proc_norm) ** 2)
        
        if mse == 0:
            return float('inf')
        
        # PSNR calculation
        psnr = 20 * np.log10(1.0 / np.sqrt(mse))
        
        return psnr
        
    except Exception as e:
        print(f"PSNR calculation failed: {e}")
        return 0.0


def calculate_ssim(original, processed):
    """
    Calculate Structural Similarity Index (SSIM) between two images.
    Supports multiple bit depths.
    
    Args:
        original: Original image (numpy array)
        processed: Processed/filtered image (numpy array)
    
    Returns:
        float: SSIM value (0-1, higher is better)
    """
    try:
        # Normalize images for comparison
        orig_norm, proc_norm, _ = normalize_for_comparison(original, processed)
        
        # Ensure same shape
        if orig_norm.shape != proc_norm.shape:
            h, w = orig_norm.shape[:2]
            if len(proc_norm.shape) == 2:
                proc_norm = cv2.resize(proc_norm, (w, h))
            else:
                proc_norm = cv2.resize(proc_norm, (w, h))
        
        # Calculate SSIM
        if len(orig_norm.shape) == 2:
            # Grayscale
            ssim_score = ssim(orig_norm, proc_norm, data_range=1.0)
        else:
            # Color - convert to grayscale for SSIM
            if orig_norm.shape[2] == 3:
                orig_gray = cv2.cvtColor((orig_norm * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
                proc_gray = cv2.cvtColor((proc_norm * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
            else:
                orig_gray = orig_norm[:,:,0]
                proc_gray = proc_norm[:,:,0]
            
            ssim_score = ssim(orig_gray, proc_gray, data_range=1.0)
        
        return ssim_score
        
    except Exception as e:
        print(f"SSIM calculation failed: {e}")
        return 0.0


def calculate_mse(original, processed):
    """
    Calculate Mean Squared Error between two images.
    
    Args:
        original: Original image
        processed: Processed image
        
    Returns:
        float: MSE value
    """
    try:
        orig_norm, proc_norm, _ = normalize_for_comparison(original, processed)
        
        # Ensure same shape
        if orig_norm.shape != proc_norm.shape:
            h, w = orig_norm.shape[:2]
            if len(proc_norm.shape) == 2:
                proc_norm = cv2.resize(proc_norm, (w, h))
            else:
                proc_norm = cv2.resize(proc_norm, (w, h))
        
        mse = np.mean((orig_norm - proc_norm) ** 2)
        return mse
        
    except Exception as e:
        print(f"MSE calculation failed: {e}")
        return 0.0


def evaluate_denoising_quality(original, denoised):
    """
    Comprehensive evaluation of denoising quality.
    
    Args:
        original: Original noisy image
        denoised: Denoised image
    
    Returns:
        dict: Dictionary containing PSNR, SSIM, and MSE values
    """
    psnr = calculate_psnr(original, denoised)
    ssim_score = calculate_ssim(original, denoised)
    mse = calculate_mse(original, denoised)
    
    return {
        'psnr': psnr,
        'ssim': ssim_score,
        'mse': mse
    }
