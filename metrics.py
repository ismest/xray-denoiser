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


def calculate_laplacian_variance(image):
    """
    Calculate Laplacian variance as a blur/no-blur metric.
    Higher values indicate sharper images with more edges.

    Args:
        image: Input image (numpy array)

    Returns:
        float: Laplacian variance value
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # Convert to float for calculation
    if gray.dtype == np.uint16:
        gray_float = gray.astype(np.float64) / 65535.0
    elif gray.dtype == np.uint8:
        gray_float = gray.astype(np.float64) / 255.0
    else:
        gray_float = gray.astype(np.float64)

    # Calculate Laplacian
    laplacian = cv2.Laplacian(gray_float, cv2.CV_64F)
    variance = laplacian.var()

    return variance


def calculate_edge_strength(image):
    """
    Calculate edge strength using Canny edge detector.
    Higher values indicate more defined edges.

    Args:
        image: Input image

    Returns:
        float: Edge strength (ratio of edge pixels)
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # Convert to uint8 for Canny
    if gray.dtype == np.uint16:
        gray_uint8 = (gray.astype(np.float64) / 65535.0 * 255).astype(np.uint8)
    elif gray.dtype == np.uint8:
        gray_uint8 = gray
    else:
        gray_uint8 = (gray / gray.max() * 255).astype(np.uint8)

    # Apply Canny edge detector
    edges = cv2.Canny(gray_uint8, 50, 150)

    # Calculate edge ratio
    edge_ratio = np.sum(edges > 0) / edges.size

    return edge_ratio


def calculate_brightness_contrast(image):
    """
    Calculate brightness and contrast metrics.

    Args:
        image: Input image

    Returns:
        tuple: (brightness, contrast)
    """
    # Convert to float
    if image.dtype == np.uint16:
        img_float = image.astype(np.float64) / 65535.0
    elif image.dtype == np.uint8:
        img_float = image.astype(np.float64) / 255.0
    else:
        img_float = image.astype(np.float64)
        if img_float.max() > 1.0:
            img_float = img_float / img_float.max()

    brightness = np.mean(img_float)
    contrast = np.std(img_float)

    return brightness, contrast


def calculate_histogram_entropy(image):
    """
    Calculate histogram entropy as a measure of texture complexity.
    Higher entropy indicates more texture details.

    Args:
        image: Input image

    Returns:
        float: Histogram entropy
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # Convert to uint8 for histogram
    if gray.dtype == np.uint16:
        gray_uint8 = (gray.astype(np.float64) / 65535.0 * 255).astype(np.uint8)
    elif gray.dtype == np.uint8:
        gray_uint8 = gray
    else:
        gray_uint8 = (gray / gray.max() * 255).astype(np.uint8)

    # Calculate histogram
    hist = cv2.calcHist([gray_uint8], [0], None, [256], [0, 256])
    hist = hist.flatten()

    # Normalize to probability
    total_pixels = gray_uint8.size
    prob = hist / total_pixels

    # Remove zeros to avoid log(0)
    prob = prob[prob > 0]

    # Calculate entropy
    entropy = -np.sum(prob * np.log2(prob))

    return entropy


def evaluate_super_resolution(sr_image, scale_factor=None):
    """
    Comprehensive evaluation of super-resolution quality.
    Compares SR output against expected properties rather than
    a ground-truth high-resolution image.

    Args:
        sr_image: Super-resolved image (high resolution output)
        scale_factor: Expected scale factor (optional, for resolution verification)

    Returns:
        dict: Dictionary containing SR quality metrics
    """
    # Resolution verification
    result = {}

    if scale_factor is not None:
        result['scale_verification'] = 'Provided - needs reference image dimensions'

    # Sharpness metric (Laplacian variance)
    result['sharpness'] = calculate_laplacian_variance(sr_image)

    # Edge strength
    result['edge_strength'] = calculate_edge_strength(sr_image)

    # Histogram entropy (texture complexity)
    result['entropy'] = calculate_histogram_entropy(sr_image)

    # Brightness and contrast
    brightness, contrast = calculate_brightness_contrast(sr_image)
    result['brightness'] = brightness
    result['contrast'] = contrast

    return result


def compare_sr_with_reference(sr_image, reference_hr_image=None,
                               original_lr_image=None, expected_scale=None):
    """
    Evaluate super-resolution quality with optional reference images.

    Args:
        sr_image: Super-resolved image (output)
        reference_hr_image: Ground truth high-resolution image (if available)
        original_lr_image: Original low-resolution image (for consistency check)
        expected_scale: Expected upscaling factor

    Returns:
        dict: Comprehensive SR evaluation metrics
    """
    metrics = {}

    # 1. Resolution verification
    if original_lr_image is not None and expected_scale is not None:
        lr_h, lr_w = original_lr_image.shape[:2]
        sr_h, sr_w = sr_image.shape[:2]
        expected_h = int(lr_h * expected_scale)
        expected_w = int(lr_w * expected_scale)

        metrics['resolution'] = {
            'lr_size': (lr_w, lr_h),
            'sr_size': (sr_w, sr_h),
            'expected_sr_size': (expected_w, expected_h),
            'width_match': sr_w == expected_w,
            'height_match': sr_h == expected_h,
            'actual_scale_w': sr_w / lr_w,
            'actual_scale_h': sr_h / lr_h
        }
    else:
        metrics['resolution'] = {
            'sr_size': (sr_image.shape[1], sr_image.shape[0])
        }

    # 2. SR quality metrics (no-reference)
    metrics['sharpness'] = calculate_laplacian_variance(sr_image)
    metrics['edge_strength'] = calculate_edge_strength(sr_image)
    metrics['entropy'] = calculate_histogram_entropy(sr_image)

    brightness, contrast = calculate_brightness_contrast(sr_image)
    metrics['brightness'] = brightness
    metrics['contrast'] = contrast

    # 3. Full-reference metrics (if ground truth HR image available)
    if reference_hr_image is not None:
        metrics['psnr'] = calculate_psnr(reference_hr_image, sr_image)
        metrics['ssim'] = calculate_ssim(reference_hr_image, sr_image)
        metrics['mse'] = calculate_mse(reference_hr_image, sr_image)

    # 4. Reduced-reference: consistency check with LR image
    if original_lr_image is not None:
        # Downscale SR to LR size and compare
        import cv2
        lr_h, lr_w = original_lr_image.shape[:2]
        sr_downscaled = cv2.resize(sr_image, (lr_w, lr_h), interpolation=cv2.INTER_AREA)

        # Normalize dtypes for comparison
        if sr_downscaled.dtype != original_lr_image.dtype:
            if original_lr_image.dtype == np.uint8:
                sr_downscaled = (sr_downscaled.astype(np.float64) /
                                (65535 if sr_downscaled.dtype == np.uint16 else sr_downscaled.max())
                                * 255).astype(np.uint8)

        metrics['consistency_psnr'] = calculate_psnr(original_lr_image, sr_downscaled)
        metrics['consistency_ssim'] = calculate_ssim(original_lr_image, sr_downscaled)

    return metrics
