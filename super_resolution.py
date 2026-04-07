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
        method: Upscaling method ('bicubic', 'lanczos', 'edge_preserving', 'trained_sr')
        enhance_edges: Whether to apply edge enhancement
        enhance_contrast: Whether to apply contrast enhancement

    Returns:
        Super-resolved image
    """
    import os
    import torch
    from denoise_algorithms import normalize_image, denormalize_image

    original_dtype = image.dtype

    # 使用训练的超分辨率模型（支持带时间戳的模型，任意文件名）
    if method and method.startswith('trained_sr_'):
        # 从方法名中提取时间戳
        timestamp = method.replace('trained_sr_', '')

        # 从特定子目录加载模型
        sr_model_dir = os.path.join(os.path.dirname(__file__), 'integrated_model', 'super_resolution', timestamp)

        if os.path.isdir(sr_model_dir):
            # 查找模型文件（任意文件名，只限制扩展名）
            model_file_onnx = None
            model_file_pth = None

            for filename in os.listdir(sr_model_dir):
                if filename.startswith('.'):
                    continue
                if filename.lower().endswith('.onnx') and model_file_onnx is None:
                    model_file_onnx = os.path.join(sr_model_dir, filename)
                elif (filename.lower().endswith('.pth') or filename.lower().endswith('.pt')) and model_file_pth is None:
                    model_file_pth = os.path.join(sr_model_dir, filename)

            model_file = model_file_onnx or model_file_pth

            if model_file and os.path.exists(model_file):
                try:
                    # 加载模型
                    if model_file.endswith('.onnx'):
                        # ONNX 模型处理
                        from neural_denoise import NeuralDenoiser
                        denoiser = NeuralDenoiser(model_path=model_file)
                        upscaled = denoiser.denoise(image)
                    else:
                        # PyTorch 模型
                        model = torch.load(model_file, map_location='cpu')
                        model.eval()

                        # 归一化
                        img_norm = normalize_image(image)
                        img_tensor = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0).float()

                        # 推理
                        with torch.no_grad():
                            result = model(img_tensor)
                            upscaled = denormalize_image(result.squeeze().numpy())
                except Exception as e:
                    print(f"Failed to load trained SR model: {e}")
                    # Fallback to lanczos
                    upscaled = lanczos_upscale(image, scale)
            else:
                # Fallback to lanczos
                upscaled = lanczos_upscale(image, scale)
        else:
            # Fallback to lanczos if directory doesn't exist
            upscaled = lanczos_upscale(image, scale)

    elif method == 'trained_sr':
        # Legacy: 从 integrated_model/super_resolution 目录加载（不带时间戳）
        sr_model_dir = os.path.join(os.path.dirname(__file__), 'integrated_model', 'super_resolution')
        model_file = None
        for filename in os.listdir(sr_model_dir):
            if filename.lower().endswith('.pth') or filename.lower().endswith('.pt'):
                model_file = os.path.join(sr_model_dir, f)
                break

        if model_file and os.path.exists(model_file):
            try:
                # 加载模型
                model = torch.load(model_file, map_location='cpu')
                model.eval()

                # 归一化
                img_norm = normalize_image(image)
                img_tensor = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0).float()

                # 推理
                with torch.no_grad():
                    result = model(img_tensor)
                    upscaled = denormalize_image(result.squeeze().numpy())
            except Exception as e:
                print(f"Failed to load trained SR model: {e}")
                # Fallback to lanczos
                upscaled = lanczos_upscale(image, scale)
        else:
            # Fallback to lanczos
            upscaled = lanczos_upscale(image, scale)

    # Step 1: Upscale
    elif method == 'bicubic':
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
    import os
    from algorithm_config import get_sr_algorithms

    # 从配置读取算法列表
    config_methods = get_sr_algorithms()
    methods = [(item["key"], item["name"]) for item in config_methods]

    # 检查是否有集成的超分辨率训练模型（扫描所有子目录）
    sr_model_dir = os.path.join(os.path.dirname(__file__), 'integrated_model', 'super_resolution')
    if os.path.isdir(sr_model_dir):
        # 扫描所有子目录获取训练模型
        for entry in os.listdir(sr_model_dir):
            model_subdir = os.path.join(sr_model_dir, entry)
            if os.path.isdir(model_subdir):
                marker_file = os.path.join(model_subdir, 'model_ready.marker')
                if os.path.exists(marker_file):
                    # 读取时间戳
                    timestamp = entry  # 使用目录名作为时间戳
                    try:
                        with open(marker_file, 'r', encoding='utf-8') as f:
                            first_line = f.readline().strip()
                            if 'Integrated at' in first_line:
                                timestamp = first_line.replace('Integrated at', '').strip()
                    except Exception:
                        pass

                    # 生成唯一的算法 key
                    algo_key = f'trained_sr_{entry}'
                    algo_name = f'神经网络 (训练模型) [{timestamp}]'
                    methods.append((algo_key, algo_name))

    return methods
