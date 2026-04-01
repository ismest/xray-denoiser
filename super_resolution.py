"""
Super-resolution reconstruction module for X-ray images.
Provides multiple upsampling methods for image enhancement.
"""

import numpy as np
import cv2
from typing import Tuple, Optional


def bicubic_upscale(image: np.ndarray, scale: float) -> np.ndarray:
    """
    Bicubic interpolation upscaling.

    Args:
        image: Input image
        scale: Upscaling factor (e.g., 2.0 for 2x)

    Returns:
        Upscaled image
    """
    h, w = image.shape[:2]
    new_w, new_h = int(w * scale), int(h * scale)

    # Ensure dimensions are valid
    new_w = max(new_w, 1)
    new_h = max(new_h, 1)

    upscaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    return upscaled


def lanczos_upscale(image: np.ndarray, scale: float) -> np.ndarray:
    """
    Lanczos interpolation upscaling (high quality).

    Args:
        image: Input image
        scale: Upscaling factor

    Returns:
        Upscaled image
    """
    h, w = image.shape[:2]
    new_w, new_h = int(w * scale), int(h * scale)

    new_w = max(new_w, 1)
    new_h = max(new_h, 1)

    upscaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    return upscaled


def edge_preserving_upscale(image: np.ndarray, scale: float, alpha: float = 0.5) -> np.ndarray:
    """
    Edge-preserving upscaling using guided filter approach.

    Args:
        image: Input image
        scale: Upscaling factor
        alpha: Edge preservation strength (0-1)

    Returns:
        Upscaled image with enhanced edges
    """
    # First upscale with bicubic
    upscaled = bicubic_upscale(image, scale)

    # Convert to float for processing
    if upscaled.dtype == np.uint16:
        img_float = upscaled.astype(np.float64) / 65535.0
    elif upscaled.dtype == np.uint8:
        img_float = upscaled.astype(np.float64) / 255.0
    else:
        img_float = upscaled.astype(np.float64)

    # Apply edge-preserving smoothing
    # Use bilateral filter to enhance edges
    if len(img_float.shape) == 2:
        # Grayscale
        uint8_img = (np.clip(img_float, 0, 1) * 255).astype(np.uint8)
        enhanced = cv2.bilateralFilter(uint8_img, d=9, sigmaColor=50, sigmaSpace=50)
        enhanced = enhanced.astype(np.float64) / 255.0
    else:
        # Color
        uint8_img = (np.clip(img_float, 0, 1) * 255).astype(np.uint8)
        enhanced = cv2.bilateralFilter(uint8_img, d=9, sigmaColor=50, sigmaSpace=50)
        enhanced = enhanced.astype(np.float64) / 255.0

    # Blend original and enhanced based on alpha
    result = (1 - alpha) * img_float + alpha * enhanced
    result = np.clip(result, 0, 1)

    # Convert back to original dtype
    if image.dtype == np.uint16:
        return (result * 65535).round().astype(np.uint16)
    else:
        return (result * 255).round().astype(np.uint8)


def adaptive_hist_equalization(image: np.ndarray, clip_limit: float = 2.0,
                                tile_grid_size: int = 8) -> np.ndarray:
    """
    Adaptive histogram equalization for contrast enhancement.

    Args:
        image: Input image
        clip_limit: Contrast limit for local enhancements
        tile_grid_size: Size of grid for histogram equalization

    Returns:
        Contrast-enhanced image
    """
    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))

    # Store original dtype
    original_dtype = image.dtype

    # Convert to appropriate format for CLAHE
    if image.dtype == np.uint16:
        # Scale to 16-bit range that CLAHE can handle
        img_normalized = ((image.astype(np.float64) / 65535.0) * 255).astype(np.uint8)
        enhanced = clahe.apply(img_normalized)
        # Scale back to 16-bit
        return (enhanced.astype(np.float64) / 255.0 * 65535).round().astype(np.uint16)
    elif image.dtype == np.uint8:
        return clahe.apply(image)
    else:
        # For float images, normalize first
        img_max = image.max()
        img_normalized = ((image / img_max) * 255).astype(np.uint8)
        enhanced = clahe.apply(img_normalized)
        return (enhanced.astype(np.float64) / 255.0 * img_max).astype(image.dtype)


def super_resolution_denoised_image(image: np.ndarray,
                                     scale: float = 2.0,
                                     method: str = 'lanczos',
                                     enhance_edges: bool = True,
                                     enhance_contrast: bool = True) -> np.ndarray:
    """
    Complete super-resolution pipeline for denoised images.

    Args:
        image: Denoised input image
        scale: Upscaling factor (1.5, 2.0, 3.0, 4.0)
        method: Upscaling method ('bicubic', 'lanczos', 'edge_preserving')
        enhance_edges: Whether to apply edge enhancement
        enhance_contrast: Whether to apply contrast enhancement

    Returns:
        Super-resolved image
    """
    original_dtype = image.dtype

    # Step 1: Upscale
    if method == 'bicubic':
        upscaled = bicubic_upscale(image, scale)
    elif method == 'lanczos':
        upscaled = lanczos_upscale(image, scale)
    elif method == 'edge_preserving':
        upscaled = edge_preserving_upscale(image, scale)
    else:
        upscaled = lanczos_upscale(image, scale)

    # Step 2: Optional edge enhancement (skip if already edge-preserving method)
    if enhance_edges and method != 'edge_preserving':
        upscaled = edge_preserving_upscale(upscaled, scale=1.0, alpha=0.3)

    # Step 3: Optional contrast enhancement
    if enhance_contrast:
        upscaled = adaptive_hist_equalization(upscaled, clip_limit=2.0, tile_grid_size=8)

    # Ensure output dtype matches input
    if upscaled.dtype != original_dtype:
        if original_dtype == np.uint16:
            upscaled = (upscaled.astype(np.float64) / 255.0 * 65535).round().astype(np.uint16)
        elif original_dtype == np.uint8:
            upscaled = (upscaled.astype(np.float64) / 255.0 * 255).round().astype(np.uint8)

    return upscaled


def get_supported_sr_methods() -> list:
    """Get list of supported super-resolution methods."""
    return [
        ('bicubic', '双三次插值 (Bicubic)'),
        ('lanczos', '兰索斯插值 (Lanczos) - 推荐'),
        ('edge_preserving', '保边增强 (Edge Preserving)'),
    ]
