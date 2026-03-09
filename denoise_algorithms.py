import numpy as np
import cv2
from skimage.restoration import denoise_nl_means, denoise_bilateral, denoise_wavelet
from skimage.util import img_as_float, img_as_ubyte

def non_local_means_denoise(image, h=10, patch_size=7, patch_distance=21):
    """
    Non-local means denoising algorithm.
    
    Args:
        image: Input image (numpy array)
        h: Filter strength (higher = more smoothing)
        patch_size: Size of patches used for comparison
        patch_distance: Max distance for patch comparison
    
    Returns:
        Denoised image
    """
    if len(image.shape) == 3:
        # Convert to grayscale if needed
        if image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image[:,:,0]
    else:
        gray = image
    
    # Ensure proper data type
    if gray.dtype != np.float64:
        gray_float = img_as_float(gray)
    else:
        gray_float = gray
    
    # Apply non-local means denoising
    denoised = denoise_nl_means(
        gray_float, 
        h=h/255.0,  # Normalize h for float images
        fast_mode=True,
        patch_size=patch_size,
        patch_distance=patch_distance
    )
    
    # Convert back to original dtype
    if image.dtype == np.uint8:
        denoised = img_as_ubyte(denoised)
    
    return denoised

def bilateral_filter_denoise(image, d=9, sigma_color=75, sigma_space=75):
    """
    Bilateral filter denoising.
    
    Args:
        image: Input image
        d: Diameter of pixel neighborhood
        sigma_color: Filter sigma in color space
        sigma_space: Filter sigma in coordinate space
    
    Returns:
        Denoised image
    """
    if len(image.shape) == 2:
        # Grayscale image
        denoised = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    else:
        # Color image
        denoised = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    
    return denoised

def wavelet_denoise(image, wavelet='db1', threshold='BayesShrink', mode='soft'):
    """
    Wavelet-based denoising.
    
    Args:
        image: Input image
        wavelet: Wavelet to use
        threshold: Thresholding method
        mode: Thresholding mode ('soft' or 'hard')
    
    Returns:
        Denoised image
    """
    if len(image.shape) == 3:
        if image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image[:,:,0]
    else:
        gray = image
    
    # Ensure proper data type
    if gray.dtype != np.float64:
        gray_float = img_as_float(gray)
    else:
        gray_float = gray
    
    # Apply wavelet denoising
    denoised = denoise_wavelet(
        gray_float,
        wavelet=wavelet,
        threshold=threshold,
        mode=mode,
        rescale_sigma=True
    )
    
    # Convert back to original dtype
    if image.dtype == np.uint8:
        denoised = img_as_ubyte(denoised)
    
    return denoised

def adaptive_denoise(image, method='auto'):
    """
    Adaptive denoising that selects the best method based on image characteristics.
    
    Args:
        image: Input image
        method: 'auto', 'nlm', 'bilateral', or 'wavelet'
    
    Returns:
        Denoised image using the specified or automatically selected method
    """
    if method == 'auto':
        # Simple heuristic: use NLM for most cases
        return non_local_means_denoise(image)
    elif method == 'nlm':
        return non_local_means_denoise(image)
    elif method == 'bilateral':
        return bilateral_filter_denoise(image)
    elif method == 'wavelet':
        return wavelet_denoise(image)
    else:
        return non_local_means_denoise(image)

def hybrid_denoise(image):
    """
    Hybrid denoising approach combining multiple methods.
    
    Args:
        image: Input image
    
    Returns:
        Denoised image using hybrid approach
    """
    # First pass: Non-local means
    denoised_nlm = non_local_means_denoise(image, h=8, patch_size=5, patch_distance=15)
    
    # Second pass: Bilateral filter to preserve edges
    if len(denoised_nlm.shape) == 2:
        denoised_final = bilateral_filter_denoise(denoised_nlm, d=5, sigma_color=50, sigma_space=50)
    else:
        denoised_final = denoised_nlm
    
    return denoised_final