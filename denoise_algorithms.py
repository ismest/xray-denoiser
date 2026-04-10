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
        # Store original dtype for later restoration
        original_dtype = image.dtype

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

                # Resize with proper dtype handling
                resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
                denoised_resized = non_local_means_denoise(resized, h=8)

                # Resize back with proper dtype preservation
                denoised_float = cv2.resize(denoised_resized, (w, h), interpolation=cv2.INTER_CUBIC)

                # Ensure output dtype matches input dtype
                if original_dtype == np.uint8:
                    denoised = np.clip(denoised_float * 255.0, 0, 255).round().astype(np.uint8)
                elif original_dtype == np.uint16:
                    denoised = np.clip(denoised_float * 65535.0, 0, 65535).round().astype(np.uint16)
                else:
                    denoised = denoised_float.astype(original_dtype)

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


def bm3d_denoise(image, sigma_psd=20, stage_arg='hard'):
    """
    BM3D (Block-Matching 3D) denoising - advanced patch-based algorithm.

    BM3D is one of the most effective denoising algorithms, using 3D
    transform domain filtering with grouped similar patches.

    Args:
        image: Input image (any depth)
        sigma_psd: Noise standard deviation estimate
        stage_arg: 'hard' for basic estimate, 'hard+Wien' for final estimate

    Returns:
        Denoised image in original depth
    """
    try:
        # Handle very small images
        min_dim = min(image.shape[:2])
        if min_dim < 16:
            return bilateral_filter_denoise(image)

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            if image.shape[2] == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image[:,:,0]
        else:
            gray = image

        # Normalize to [0, 1]
        normalized, original_dtype, original_max = normalize_image(gray)

        # Try to use bm3d package if available
        try:
            import bm3d
            denoised = bm3d.bm3d(normalized, sigma_psd=sigma_psd, stage_arg=stage_arg)
        except ImportError:
            # Fallback: implement simplified BM3D-like approach using OpenCV
            # This is a simplified version that mimics BM3D behavior
            denoised = _simplified_bm3d(normalized, sigma_psd=sigma_psd)

        # Convert back to original depth
        return denormalize_image(denoised, original_dtype, original_max)

    except Exception as e:
        print(f"BM3D denoising failed: {e}")
        return adaptive_denoise(image, method='nlm')


def _simplified_bm3d(image, sigma_psd=20, block_size=8, search_window=21):
    """
    Simplified BM3D implementation using OpenCV when bm3d package is not available.

    This is a fallback implementation that approximates BM3D behavior using
    block matching and collaborative filtering concepts.
    """
    # Normalize sigma
    sigma = sigma_psd / 255.0

    # Use OpenCV's fastNlMeansDenoise which implements similar concepts to BM3D
    # Convert to uint8 for OpenCV
    uint8_img = (image * 255).round().astype(np.uint8)

    # Apply fast NLM with BM3D-like parameters
    h = int(sigma * 100)
    h = max(5, min(h, 100))

    denoised_uint8 = cv2.fastNlMeansDenoise(
        uint8_img,
        h=h,
        templateWindowSize=7,
        searchWindowSize=search_window
    )

    return denoised_uint8.astype(np.float64) / 255.0


def anisotropic_diffusion_denoise(image, niter=10, kappa=50, gamma=0.1, option=1):
    """
    Anisotropic Diffusion denoising - edge-preserving smoothing.

    Uses partial differential equations to smooth homogeneous regions
    while preserving edges. Based on Perona-Malik diffusion.

    Args:
        image: Input image (any depth)
        niter: Number of iterations (more = stronger smoothing)
        kappa: Gradient threshold (higher = more smoothing)
        gamma: Step size (0-0.25, typically 0.1)
        option: 1 for Perona-Malik function 1, 2 for function 2

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

        # Normalize to float64 [0, 1]
        normalized, original_dtype, original_max = normalize_image(gray)

        # Apply anisotropic diffusion
        denoised = _perona_mailik_diffusion(normalized, niter=niter, kappa=kappa, gamma=gamma, option=option)

        # Convert back to original depth
        return denormalize_image(denoised, original_dtype, original_max)

    except Exception as e:
        print(f"Anisotropic diffusion failed: {e}")
        return bilateral_filter_denoise(image)


def _perona_mailik_diffusion(image, niter=10, kappa=50, gamma=0.1, option=1):
    """
    Perona-Malik anisotropic diffusion implementation.

    The diffusion coefficient controls the rate of smoothing:
    - Option 1: c = exp(-(||grad(I)||/kappa)^2)
    - Option 2: c = 1 / (1 + (||grad(I)||/kappa)^2)
    """
    image = image.astype(np.float64)

    # Initialize output
    output = image.copy()

    # Clamp gamma to stable range
    gamma = max(0.01, min(gamma, 0.25))

    for i in range(niter):
        # Compute gradients using finite differences
        # North-South gradient
        grad_n = np.zeros_like(output)
        grad_n[:-1, :] = output[1:, :] - output[:-1, :]

        grad_s = np.zeros_like(output)
        grad_s[1:, :] = output[:-1, :] - output[1:, :]

        # East-West gradient
        grad_w = np.zeros_like(output)
        grad_w[:, :-1] = output[:, 1:] - output[:, :-1]

        grad_e = np.zeros_like(output)
        grad_e[:, 1:] = output[:, :-1] - output[:, 1:]

        # Compute diffusion coefficients
        grad_magnitude_ns = np.abs(grad_n + grad_s) / 2
        grad_magnitude_ew = np.abs(grad_w + grad_e) / 2

        if option == 1:
            # Perona-Malik function 1 (exponential)
            c_n = np.exp(-(grad_magnitude_ns / kappa) ** 2)
            c_s = c_n.copy()
            c_w = np.exp(-(grad_magnitude_ew / kappa) ** 2)
            c_e = c_w.copy()
        else:
            # Perona-Malik function 2 (rational)
            c_n = 1.0 / (1.0 + (grad_magnitude_ns / kappa) ** 2)
            c_s = c_n.copy()
            c_w = 1.0 / (1.0 + (grad_magnitude_ew / kappa) ** 2)
            c_e = c_w.copy()

        # Handle boundaries
        c_n[-1, :] = 0
        c_s[0, :] = 0
        c_w[:, -1] = 0
        c_e[:, 0] = 0

        # Update image using diffusion equation
        output += gamma * (
            c_n * grad_n +
            c_s * grad_s +
            c_w * grad_w +
            c_e * grad_e
        )

        # Clip to valid range
        output = np.clip(output, 0, 1)

    return output


def iterative_reconstruction_denoise(image, niter=5, regularization=0.1, method='tv'):
    """
    Iterative Reconstruction denoising using regularization.

    Iteratively refines the image by minimizing an energy function
    combining data fidelity and regularization terms.

    Args:
        image: Input image (any depth)
        niter: Number of iterations
        regularization: Regularization strength (higher = smoother)
        method: 'tv' for Total Variation, 'tikhonov' for Tikhonov regularization

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

        # Normalize to float64 [0, 1]
        normalized, original_dtype, original_max = normalize_image(gray)

        # Apply iterative reconstruction
        if method == 'tv':
            denoised = _tv_denoise_iterative(normalized, niter=niter, alpha=regularization)
        else:
            denoised = _tikhonov_denoise_iterative(normalized, niter=niter, alpha=regularization)

        # Convert back to original depth
        return denormalize_image(denoised, original_dtype, original_max)

    except Exception as e:
        print(f"Iterative reconstruction failed: {e}")
        return adaptive_denoise(image, method='auto')


def _tv_denoise_iterative(image, niter=5, alpha=0.1, eps=1e-8):
    """
    Total Variation denoising using split Bregman / gradient descent.

    TV regularization preserves edges while removing noise.
    Minimizes: ||u - f||^2 + alpha * ||grad(u)||_1
    """
    image = image.astype(np.float64)
    u = image.copy()

    # Gradient descent with TV regularization
    for _ in range(niter):
        # Compute gradient magnitude
        grad_x = np.zeros_like(u)
        grad_x[:, :-1] = u[:, 1:] - u[:, :-1]

        grad_y = np.zeros_like(u)
        grad_y[:-1, :] = u[1:, :] - u[:-1, :]

        # Magnitude with epsilon for numerical stability
        grad_mag = np.sqrt(grad_x**2 + grad_y**2 + eps)

        # Compute divergence for TV flow
        div_x = np.zeros_like(u)
        div_x[:, 1:] = grad_x[:, 1:] / np.maximum(grad_mag[:, 1:], eps)
        div_x[:, :-1] -= grad_x[:, :-1] / np.maximum(grad_mag[:, :-1], eps)

        div_y = np.zeros_like(u)
        div_y[1:, :] = grad_y[1:, :] / np.maximum(grad_mag[1:, :], eps)
        div_y[:-1, :] -= grad_y[:-1, :] / np.maximum(grad_mag[:-1, :], eps)

        # Update with TV regularization
        u = u - alpha * (u - image) + 0.1 * (div_x + div_y)

        # Clip to valid range
        u = np.clip(u, 0, 1)

    return u


def _tikhonov_denoise_iterative(image, niter=5, alpha=0.1):
    """
    Tikhonov regularization denoising (L2 regularization).

    Smoother than TV but may blur edges.
    Minimizes: ||u - f||^2 + alpha * ||grad(u)||^2
    """
    image = image.astype(np.float64)
    u = image.copy()

    # Laplacian kernel for regularization
    laplacian = np.array([[0, 1, 0],
                          [1, -4, 1],
                          [0, 1, 0]], dtype=np.float64)

    for _ in range(niter):
        # Convolve with Laplacian (using manual implementation for speed)
        # This approximates the second derivative
        lap_u = np.zeros_like(u)

        # Manual Laplacian computation
        lap_u[1:-1, 1:-1] = (
            u[:-2, 1:-1] + u[2:, 1:-1] +
            u[1:-1, :-2] + u[1:-1, 2:] -
            4 * u[1:-1, 1:-1]
        )

        # Update with regularization
        u = u + alpha * lap_u * (u - image)

        # Clip to valid range
        u = np.clip(u, 0, 1)

    return u
