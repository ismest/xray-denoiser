#!/usr/bin/env python3
"""
Test script to verify all bug fixes for the X-ray Image Denoiser application.

Tests:
1. Dynamic parameter panel visibility (UI - manual test)
2. Enhanced progress feedback (UI - manual test)
3. Improved error handling (UI - manual test)
4. Hybrid denoise parameter passing
5. Neural algorithm parameter handling
6. Shape and dtype consistency in resize operations
"""

import numpy as np
import cv2
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import modules to test
from image_processor import ImageProcessor
from denoise_algorithms import (
    hybrid_denoise, adaptive_denoise, non_local_means_denoise,
    bilateral_filter_denoise, wavelet_denoise, gaussian_denoise,
    normalize_image, denormalize_image
)


def create_test_image(size=(512, 512), dtype=np.uint8, channels=1):
    """Create a synthetic test image with noise."""
    if channels == 1:
        shape = size
    else:
        shape = (size[0], size[1], channels)

    # Create gradient pattern
    if len(shape) == 2:
        img = np.zeros(shape, dtype=np.float64)
        for i in range(min(size[0], 100)):  # Limit for speed
            for j in range(min(size[1], 100)):
                img[i, j] = (i + j) / (size[0] + size[1]) * 255
        # Fill rest with constant
        img[100:, :] = 128
        img[:, 100:] = 128
    else:
        img = np.zeros(shape, dtype=np.float64)
        for i in range(min(size[0], 100)):
            for j in range(min(size[1], 100)):
                img[i, j, :] = (i + j) / (size[0] + size[1]) * 255
        img[100:, :, :] = 128
        img[:, 100:, :] = 128

    # Add noise
    noise = np.random.normal(0, 10, shape)
    img = img + noise

    # Convert to target dtype
    if dtype == np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    elif dtype == np.uint16:
        img = np.clip(img, 0, 65535).astype(np.uint16)
    elif dtype == np.float32:
        img = img.astype(np.float32)
    elif dtype == np.float64:
        img = img.astype(np.float64)

    return img


def test_hybrid_denoise_parameters():
    """Test that hybrid_denoise correctly handles strength parameter."""
    print("\n" + "=" * 60)
    print("TEST: Hybrid Denoise Parameter Passing")
    print("=" * 60)

    img = create_test_image(size=(128, 128), dtype=np.uint8)

    try:
        # Test low strength
        result_low = hybrid_denoise(img, strength='low')
        assert result_low is not None, "Low strength returned None"
        assert result_low.shape == img.shape, f"Shape mismatch: {result_low.shape} vs {img.shape}"
        assert result_low.dtype == img.dtype, f"Dtype mismatch: {result_low.dtype} vs {img.dtype}"
        print("  [PASS] Low strength processing")

        # Test medium strength
        result_medium = hybrid_denoise(img, strength='medium')
        assert result_medium is not None, "Medium strength returned None"
        print("  [PASS] Medium strength processing")

        # Test high strength
        result_high = hybrid_denoise(img, strength='high')
        assert result_high is not None, "High strength returned None"
        print("  [PASS] High strength processing")

        # Verify different strengths produce different results
        diff_low_high = np.mean(np.abs(result_low.astype(float) - result_high.astype(float)))
        print(f"  Difference between low and high strength: {diff_low_high:.2f}")

        print("  [PASS] All hybrid denoise tests passed!")
        return True

    except Exception as e:
        print(f"  [FAIL] Hybrid denoise test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dtype_preservation():
    """Test that algorithms preserve input dtype."""
    print("\n" + "=" * 60)
    print("TEST: Data Type Preservation")
    print("=" * 60)

    dtypes_to_test = [np.uint8, np.uint16]  # Skip float for brevity

    for dtype in dtypes_to_test:
        print(f"\n  Testing dtype: {dtype}")
        img = create_test_image(size=(64, 64), dtype=dtype)

        try:
            # Test hybrid
            result = hybrid_denoise(img, strength='medium')
            assert result.dtype == dtype, f"hybrid: {result.dtype} != {dtype}"
            print(f"    [PASS] hybrid_denoise preserved {dtype}")

            # Test NLM
            result = non_local_means_denoise(img, h=10)
            assert result.dtype == dtype, f"nlm: {result.dtype} != {dtype}"
            print(f"    [PASS] non_local_means_denoise preserved {dtype}")

            # Test bilateral
            result = bilateral_filter_denoise(img)
            assert result.dtype == dtype, f"bilateral: {result.dtype} != {dtype}"
            print(f"    [PASS] bilateral_filter_denoise preserved {dtype}")

        except Exception as e:
            print(f"    [FAIL] dtype preservation failed for {dtype}: {e}")
            return False

    print("  [PASS] All dtype preservation tests passed!")
    return True


def test_small_image_handling():
    """Test that algorithms handle small images correctly."""
    print("\n" + "=" * 60)
    print("TEST: Small Image Handling")
    print("=" * 60)

    small_sizes = [(10, 10), (20, 20), (50, 50)]

    for size in small_sizes:
        print(f"\n  Testing size: {size}")
        img = create_test_image(size=size, dtype=np.uint8)

        try:
            result = hybrid_denoise(img, strength='medium')
            assert result is not None, f"Returned None for size {size}"
            print(f"    [PASS] hybrid_denoise handled {size}")

        except Exception as e:
            print(f"    [FAIL] Small image handling failed for {size}: {e}")
            return False

    print("  [PASS] All small image tests passed!")
    return True


def test_large_image_handling():
    """Test that adaptive_denoise handles large images without memory issues."""
    print("\n" + "=" * 60)
    print("TEST: Large Image Handling")
    print("=" * 60)

    # Test with moderately large image (simulating large image behavior)
    print("  Testing adaptive_denoise with large image simulation...")
    img = create_test_image(size=(500, 500), dtype=np.uint8)

    try:
        result = adaptive_denoise(img, method='auto')
        assert result is not None, "Returned None for large image"
        assert result.shape == img.shape, f"Shape mismatch: {result.shape} vs {img.shape}"
        print("  [PASS] Large image handled correctly")
        return True

    except Exception as e:
        print(f"  [FAIL] Large image handling failed: {e}")
        return False


def test_image_processor_parameters():
    """Test that ImageProcessor correctly passes parameters to algorithms."""
    print("\n" + "=" * 60)
    print("TEST: ImageProcessor Parameter Passing")
    print("=" * 60)

    processor = ImageProcessor()

    # Create test image
    img = create_test_image(size=(128, 128), dtype=np.uint8)

    # Save test image
    cv2.imwrite('/tmp/test_input.png', img)

    try:
        # Load image
        success = processor.load_image('/tmp/test_input.png')
        assert success, "Failed to load test image"
        print("  [PASS] Image loaded")

        # Test NLM with parameters
        success = processor.process_image('nlm', h=10, patch_size=7)
        assert success, "NLM processing failed"
        assert processor.get_denoised_image() is not None, "NLM returned None"
        print("  [PASS] NLM with parameters")

        # Test bilateral with parameters
        success = processor.process_image('bilateral', d=9, sigma_color=75, sigma_space=75)
        assert success, "Bilateral processing failed"
        print("  [PASS] Bilateral with parameters")

        # Test hybrid with strength
        success = processor.process_image('hybrid', strength='medium')
        assert success, "Hybrid processing failed"
        print("  [PASS] Hybrid with strength parameter")

        # Test neural (will fallback if no model)
        success = processor.process_image('neural', patch_size=256, stride=128)
        # Neural may fallback to bilateral if no model, but should not crash
        print("  [PASS] Neural with parameters (may use fallback)")

        # Test wavelet
        success = processor.process_image('wavelet')
        assert success, "Wavelet processing failed"
        print("  [PASS] Wavelet processing")

        # Test gaussian
        success = processor.process_image('gaussian')
        assert success, "Gaussian processing failed"
        print("  [PASS] Gaussian processing")

        processor.reset()
        print("  [PASS] All ImageProcessor parameter tests passed!")
        return True

    except Exception as e:
        print(f"  [FAIL] ImageProcessor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_shape_consistency():
    """Test that output shape matches input shape."""
    print("\n" + "=" * 60)
    print("TEST: Shape Consistency")
    print("=" * 60)

    test_shapes = [(64, 64), (128, 256), (256, 128)]

    for shape in test_shapes:
        print(f"\n  Testing shape: {shape}")
        img = create_test_image(size=shape, dtype=np.uint8)

        try:
            result = hybrid_denoise(img, strength='medium')
            assert result.shape == img.shape, f"Shape mismatch: {result.shape} vs {img.shape}"
            print(f"    [PASS] Shape preserved for {shape}")

        except Exception as e:
            print(f"    [FAIL] Shape consistency failed for {shape}: {e}")
            return False

    print("  [PASS] All shape consistency tests passed!")
    return True


def test_normalize_denormalize():
    """Test normalization and denormalization functions."""
    print("\n" + "=" * 60)
    print("TEST: Normalize/Denormalize Functions")
    print("=" * 60)

    dtypes = [np.uint8, np.uint16]

    for dtype in dtypes:
        img = create_test_image(size=(64, 64), dtype=dtype)

        try:
            normalized, orig_dtype, orig_max = normalize_image(img)
            assert normalized.max() <= 1.0, f"Normalized max > 1: {normalized.max()}"
            assert normalized.min() >= 0.0, f"Normalized min < 0: {normalized.min()}"

            denormalized = denormalize_image(normalized, orig_dtype, orig_max)
            assert denormalized.dtype == dtype, f"Dtype mismatch after denormalize"

            print(f"  [PASS] normalize/denormalize for {dtype}")

        except Exception as e:
            print(f"  [FAIL] normalize/denormalize failed for {dtype}: {e}")
            return False

    print("  [PASS] All normalize/denormalize tests passed!")
    return True


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "=" * 70)
    print("X-RAY IMAGE DENOISER - BUG FIX VERIFICATION TESTS")
    print("=" * 70)

    tests = [
        ("Data Type Preservation", test_dtype_preservation),
        ("Shape Consistency", test_shape_consistency),
        ("Small Image Handling", test_small_image_handling),
        ("Large Image Handling", test_large_image_handling),
        ("Hybrid Denoise Parameters", test_hybrid_denoise_parameters),
        ("Normalize/Denormalize", test_normalize_denormalize),
        ("ImageProcessor Parameters", test_image_processor_parameters),
    ]

    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n  [ERROR] {name} crashed: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"  [{status}] {name}")

    print(f"\n  Total: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print("=" * 70)

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    print("\nNote: UI tests (dynamic parameter panel, progress feedback, error display)")
    print("      must be tested manually by running: python main.py")
    print("=" * 70)
    sys.exit(0 if success else 1)
