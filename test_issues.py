#!/usr/bin/env python3
"""
Test script to verify the identified issues in the denoising application.
"""

import numpy as np
import cv2
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from image_processor import ImageProcessor
from denoise_algorithms import hybrid_denoise, normalize_image, denormalize_image

def test_different_depths():
    """Test handling of different image depths."""
    print("Testing different image depths...")
    
    # Test uint8 (8-bit)
    img_uint8 = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    try:
        result = hybrid_denoise(img_uint8)
        print(f"✓ uint8: {img_uint8.dtype} -> {result.dtype}")
    except Exception as e:
        print(f"✗ uint8 failed: {e}")
    
    # Test uint16 (16-bit) - typical for medical X-ray
    img_uint16 = np.random.randint(0, 65536, (100, 100), dtype=np.uint16)
    try:
        result = hybrid_denoise(img_uint16)
        print(f"✓ uint16: {img_uint16.dtype} -> {result.dtype}")
    except Exception as e:
        print(f"✗ uint16 failed: {e}")
    
    # Test float32
    img_float32 = np.random.rand(100, 100).astype(np.float32)
    try:
        result = hybrid_denoise(img_float32)
        print(f"✓ float32: {img_float32.dtype} -> {result.dtype}")
    except Exception as e:
        print(f"✗ float32 failed: {e}")

def test_small_images():
    """Test handling of very small images."""
    print("\nTesting small images...")
    
    sizes = [(10, 10), (20, 20), (30, 30)]
    for h, w in sizes:
        img = np.random.randint(0, 256, (h, w), dtype=np.uint8)
        try:
            result = hybrid_denoise(img)
            print(f"✓ Small image {h}x{w}: success")
        except Exception as e:
            print(f"✗ Small image {h}x{w} failed: {e}")

def test_large_images():
    """Test handling of large images."""
    print("\nTesting large images...")
    
    # Test with a moderately large image
    img = np.random.randint(0, 256, (2000, 2000), dtype=np.uint8)
    try:
        result = hybrid_denoise(img)
        print(f"✓ Large image 2000x2000: success")
    except Exception as e:
        print(f"✗ Large image 2000x2000 failed: {e}")

def test_normalization():
    """Test normalization and denormalization functions."""
    print("\nTesting normalization functions...")
    
    # Test uint16 normalization
    img_uint16 = np.array([[0, 32768, 65535]], dtype=np.uint16)
    normalized, orig_dtype, orig_max = normalize_image(img_uint16)
    denormalized = denormalize_image(normalized, orig_dtype, orig_max)
    
    print(f"Original: {img_uint16}")
    print(f"Normalized: {normalized}")
    print(f"Denormalized: {denormalized}")
    print(f"Match: {np.array_equal(img_uint16, denormalized)}")

def test_image_processor():
    """Test the ImageProcessor class."""
    print("\nTesting ImageProcessor...")
    
    processor = ImageProcessor()
    
    # Create a test image
    test_img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    cv2.imwrite('/tmp/test_image.png', test_img)
    
    # Test loading
    success = processor.load_image('/tmp/test_image.png')
    print(f"Load image: {'✓' if success else '✗'}")
    
    if success:
        # Test processing
        success = processor.process_image('hybrid')
        print(f"Process image: {'✓' if success else '✗'}")
        
        if success:
            # Test metrics
            metrics = processor.calculate_metrics()
            print(f"Metrics: PSNR={metrics.get('psnr', 0):.2f}, SSIM={metrics.get('ssim', 0):.4f}")
            
            # Test saving
            success = processor.save_result('/tmp/test_result.png')
            print(f"Save result: {'✓' if success else '✗'}")
    
    # Clean up
    if os.path.exists('/tmp/test_image.png'):
        os.remove('/tmp/test_image.png')
    if os.path.exists('/tmp/test_result.png'):
        os.remove('/tmp/test_result.png')

if __name__ == "__main__":
    print("Running issue verification tests...\n")
    
    try:
        test_different_depths()
        test_small_images()
        test_large_images()
        test_normalization()
        test_image_processor()
        
        print("\n✅ All tests completed!")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
