#!/usr/bin/env python3
"""
Test script to verify the fixes for the identified issues.
"""

import numpy as np
import cv2
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from denoise_algorithms import hybrid_denoise, normalize_image, denormalize_image
from image_processor import ImageProcessor

def test_uint16_handling():
    """Test proper handling of uint16 images."""
    print("Testing uint16 image handling...")
    
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
    
    # Test full denoising pipeline
    try:
        result = hybrid_denoise(img_uint16)
        print(f"✓ uint16 denoising successful: {result.dtype}, shape: {result.shape}")
    except Exception as e:
        print(f"✗ uint16 denoising failed: {e}")

def test_small_images():
    """Test handling of very small images."""
    print("\nTesting small image handling...")
    
    sizes = [(10, 10), (20, 20), (30, 30), (40, 40)]
    for h, w in sizes:
        img = np.random.randint(0, 256, (h, w), dtype=np.uint8)
        try:
            result = hybrid_denoise(img)
            print(f"✓ Small image {h}x{w}: success, output shape: {result.shape}")
        except Exception as e:
            print(f"✗ Small image {h}x{w} failed: {e}")

def test_large_images():
    """Test handling of large images."""
    print("\nTesting large image handling...")
    
    # Test with a moderately large image (not too big to avoid memory issues)
    img = np.random.randint(0, 256, (1000, 1000), dtype=np.uint8)
    try:
        result = hybrid_denoise(img)
        print(f"✓ Large image 1000x1000: success, output shape: {result.shape}")
    except Exception as e:
        print(f"✗ Large image 1000x1000 failed: {e}")

def test_image_processor():
    """Test the ImageProcessor class with different depths."""
    print("\nTesting ImageProcessor with different depths...")
    
    processor = ImageProcessor()
    
    # Test uint8
    img_uint8 = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    cv2.imwrite('/tmp/test_uint8.png', img_uint8)
    success = processor.load_image('/tmp/test_uint8.png')
    print(f"uint8 load: {'✓' if success else '✗'}")
    if success:
        success = processor.process_image('hybrid')
        print(f"uint8 process: {'✓' if success else '✗'}")
    
    # Test uint16
    img_uint16 = np.random.randint(0, 65536, (100, 100), dtype=np.uint16)
    cv2.imwrite('/tmp/test_uint16.tiff', img_uint16)
    success = processor.load_image('/tmp/test_uint16.tiff')
    print(f"uint16 load: {'✓' if success else '✗'}")
    if success:
        success = processor.process_image('hybrid')
        print(f"uint16 process: {'✓' if success else '✗'}")
    
    # Clean up
    if os.path.exists('/tmp/test_uint8.png'):
        os.remove('/tmp/test_uint8.png')
    if os.path.exists('/tmp/test_uint16.tiff'):
        os.remove('/tmp/test_uint16.tiff')

if __name__ == "__main__":
    print("Running fix verification tests...\n")
    
    try:
        test_uint16_handling()
        test_small_images()
        test_large_images()
        test_image_processor()
        
        print("\n✅ All tests completed!")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()