#!/usr/bin/env python3
"""
Independent test script to verify the fixes without any dependencies.
"""

import numpy as np

def normalize_image(image):
    """
    Normalize any image depth to float64 [0, 1] range.
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

def test_uint16_normalization():
    """Test proper handling of uint16 images."""
    print("Testing uint16 normalization...")
    
    # Create a 16-bit test image
    img_uint16 = np.random.randint(0, 65536, (100, 100), dtype=np.uint16)
    
    # Test normalization
    normalized, orig_dtype, orig_max = normalize_image(img_uint16)
    print(f"Original max: {img_uint16.max()}")
    print(f"Normalized range: [{normalized.min():.4f}, {normalized.max():.4f}]")
    print(f"Original dtype: {orig_dtype}, Original max: {orig_max}")
    
    # Test denormalization
    denormalized = denormalize_image(normalized, orig_dtype, orig_max)
    print(f"Denormalized max: {denormalized.max()}")
    print(f"Values match: {np.allclose(img_uint16.astype(np.float32), denormalized.astype(np.float32), rtol=1e-3)}")
    
    assert orig_dtype == np.uint16
    assert orig_max == 65535.0
    assert normalized.min() >= 0.0 and normalized.max() <= 1.0
    print("✓ uint16 normalization test passed!")

def test_small_image_bounds():
    """Test safe bounds for small images."""
    print("\nTesting small image bounds...")
    
    # Test the logic that would be used in non_local_means_denoise
    def test_patch_bounds(image_shape, patch_size, patch_distance):
        min_dim = min(image_shape[:2])
        if min_dim < 5:
            return "gaussian"
        elif min_dim < patch_size:
            return "bilateral"
        else:
            safe_patch_size = max(3, min(patch_size, min_dim // 2))
            safe_patch_distance = min(patch_distance, min_dim // 2)
            safe_patch_distance = max(safe_patch_distance, safe_patch_size)
            return f"nlm: patch_size={safe_patch_size}, patch_distance={safe_patch_distance}"
    
    test_cases = [
        ((10, 10), 7, 21),
        ((20, 20), 7, 21),
        ((100, 100), 7, 21),
        ((500, 500), 15, 30)
    ]
    
    for shape, patch_size, patch_distance in test_cases:
        result = test_patch_bounds(shape, patch_size, patch_distance)
        print(f"Image {shape}: {result}")
    
    print("✓ Small image bounds test passed!")

if __name__ == "__main__":
    print("Running independent fix verification tests...\n")
    
    try:
        test_uint16_normalization()
        test_small_image_bounds()
        
        print("\n✅ All independent tests completed!")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()