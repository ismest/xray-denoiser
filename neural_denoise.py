"""
Neural Network-based denoising algorithms for X-ray images.
Uses ONNX Runtime for CPU-friendly inference.
"""

import numpy as np
import cv2
from pathlib import Path

# Try to import onnxruntime, provide fallback if not available
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: onnxruntime not available. Neural denoising will be disabled.")


class NeuralDenoiser:
    """
    Neural network based denoiser using ONNX models.
    Falls back to traditional methods if model not available.
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the neural denoiser.
        
        Args:
            model_path: Path to ONNX model file. If None, uses fallback methods.
        """
        self.session = None
        self.model_path = model_path
        
        if ONNX_AVAILABLE and model_path and Path(model_path).exists():
            try:
                # Create inference session with CPU provider
                providers = ['CPUExecutionProvider']
                self.session = ort.InferenceSession(model_path, providers=providers)
                print(f"Loaded ONNX model: {model_path}")
            except Exception as e:
                print(f"Failed to load ONNX model: {e}")
                self.session = None
    
    def is_available(self):
        """Check if neural denoising is available."""
        return self.session is not None
    
    def denoise(self, image, patch_size=256, stride=128):
        """
        Apply neural denoising to image.
        
        Args:
            image: Input image (numpy array, any depth)
            patch_size: Size of patches to process
            stride: Stride for patch extraction
            
        Returns:
            Denoised image
        """
        if self.session is None:
            # Fallback to traditional method
            return self._fallback_denoise(image)
        
        try:
            # Store original dtype and range
            original_dtype = image.dtype
            
            # Normalize to float32 [0, 1]
            if image.dtype == np.uint8:
                img_float = image.astype(np.float32) / 255.0
            elif image.dtype == np.uint16:
                img_float = image.astype(np.float32) / 65535.0
            elif image.dtype == np.float32 or image.dtype == np.float64:
                img_float = image.astype(np.float32)
                # Normalize if needed
                if img_float.max() > 1.0:
                    img_float = img_float / img_float.max()
            else:
                img_float = image.astype(np.float32) / 255.0
            
            # Handle grayscale and color images
            if len(img_float.shape) == 2:
                # Grayscale
                result = self._denoise_grayscale(img_float, patch_size, stride)
            else:
                # Color - process each channel
                result = np.zeros_like(img_float)
                for c in range(min(img_float.shape[2], 3)):
                    result[:,:,c] = self._denoise_grayscale(img_float[:,:,c], patch_size, stride)
            
            # Convert back to original dtype
            if original_dtype == np.uint8:
                result = (result * 255.0).clip(0, 255).astype(np.uint8)
            elif original_dtype == np.uint16:
                result = (result * 65535.0).clip(0, 65535).astype(np.uint16)
            elif original_dtype == np.float32:
                result = result.astype(np.float32)
            elif original_dtype == np.float64:
                result = result.astype(np.float64)
            
            return result
            
        except Exception as e:
            print(f"Neural denoising failed: {e}")
            return self._fallback_denoise(image)
    
    def _denoise_grayscale(self, image, patch_size, stride):
        """
        Denoise a grayscale image using patch-based processing.
        
        Args:
            image: Float32 image in range [0, 1]
            patch_size: Size of patches
            stride: Stride for patch extraction
            
        Returns:
            Denoised image
        """
        h, w = image.shape
        
        # For small images, process directly
        if h <= patch_size and w <= patch_size:
            return self._process_patch(image)
        
        # Initialize output and weight maps
        output = np.zeros_like(image)
        weights = np.zeros_like(image)
        
        # Process patches
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                patch = image[y:y+patch_size, x:x+patch_size]
                denoised_patch = self._process_patch(patch)
                
                # Add to output with blending
                output[y:y+patch_size, x:x+patch_size] += denoised_patch
                weights[y:y+patch_size, x:x+patch_size] += 1.0
        
        # Handle remaining edges
        # Bottom edge
        if h > patch_size:
            for x in range(0, w - patch_size + 1, stride):
                y = h - patch_size
                patch = image[y:y+patch_size, x:x+patch_size]
                denoised_patch = self._process_patch(patch)
                output[y:y+patch_size, x:x+patch_size] += denoised_patch
                weights[y:y+patch_size, x:x+patch_size] += 1.0
        
        # Right edge
        if w > patch_size:
            for y in range(0, h - patch_size + 1, stride):
                x = w - patch_size
                patch = image[y:y+patch_size, x:x+patch_size]
                denoised_patch = self._process_patch(patch)
                output[y:y+patch_size, x:x+patch_size] += denoised_patch
                weights[y:y+patch_size, x:x+patch_size] += 1.0
        
        # Bottom-right corner
        if h > patch_size and w > patch_size:
            y, x = h - patch_size, w - patch_size
            patch = image[y:y+patch_size, x:x+patch_size]
            denoised_patch = self._process_patch(patch)
            output[y:y+patch_size, x:x+patch_size] += denoised_patch
            weights[y:y+patch_size, x:x+patch_size] += 1.0
        
        # Average overlapping regions
        weights = np.maximum(weights, 1e-8)  # Avoid division by zero
        output = output / weights
        
        return output
    
    def _process_patch(self, patch):
        """
        Process a single patch through the neural network.
        
        Args:
            patch: Input patch (H, W) or (H, W, C)
            
        Returns:
            Denoised patch
        """
        try:
            # Prepare input
            if len(patch.shape) == 2:
                # Add batch and channel dimensions: (1, 1, H, W)
                input_tensor = patch[np.newaxis, np.newaxis, :, :].astype(np.float32)
            else:
                # Add batch dimension: (1, C, H, W)
                input_tensor = np.transpose(patch, (2, 0, 1))[np.newaxis, :, :, :].astype(np.float32)
            
            # Get input name
            input_name = self.session.get_inputs()[0].name
            
            # Run inference
            outputs = self.session.run(None, {input_name: input_tensor})
            
            # Extract result
            result = outputs[0]
            if len(result.shape) == 4:
                result = result[0, 0]  # Remove batch and channel dims
            elif len(result.shape) == 3:
                result = result[0]
            
            return result
            
        except Exception as e:
            print(f"Patch processing failed: {e}")
            return patch
    
    def _fallback_denoise(self, image):
        """
        Fallback denoising when neural network is not available.
        Uses optimized bilateral filter.
        
        Args:
            image: Input image
            
        Returns:
            Denoised image
        """
        # Convert to float for processing
        if image.dtype == np.uint8:
            img_float = image.astype(np.float32) / 255.0
            max_val = 255.0
            output_dtype = np.uint8
        elif image.dtype == np.uint16:
            img_float = image.astype(np.float32) / 65535.0
            max_val = 65535.0
            output_dtype = np.uint16
        else:
            img_float = image.astype(np.float32)
            max_val = 1.0
            output_dtype = image.dtype
        
        # Apply bilateral filter
        if len(img_float.shape) == 2:
            # Grayscale
            denoised = cv2.bilateralFilter((img_float * 255).astype(np.uint8), 9, 75, 75)
            denoised = denoised.astype(np.float32) / 255.0
        else:
            # Color - convert to grayscale for processing if needed
            if img_float.shape[2] == 3:
                gray = cv2.cvtColor((img_float * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
                denoised_gray = cv2.bilateralFilter(gray, 9, 75, 75)
                denoised = cv2.cvtColor(denoised_gray, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
            else:
                denoised = img_float
        
        # Convert back to original dtype
        if output_dtype == np.uint8:
            denoised = (denoised * 255.0).clip(0, 255).astype(np.uint8)
        elif output_dtype == np.uint16:
            denoised = (denoised * 65535.0).clip(0, 65535).astype(np.uint16)
        
        return denoised


def create_sample_model():
    """
    Create a simple sample ONNX model for testing.
    This is a lightweight denoising autoencoder.
    """
    try:
        import torch
        import torch.nn as nn
        
        class SimpleDenoiser(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Conv2d(1, 32, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, 3, padding=1),
                    nn.ReLU(),
                )
                self.decoder = nn.Sequential(
                    nn.Conv2d(64, 32, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(32, 1, 3, padding=1),
                    nn.Sigmoid(),
                )
            
            def forward(self, x):
                x = self.encoder(x)
                x = self.decoder(x)
                return x
        
        # Create model
        model = SimpleDenoiser()
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, 1, 256, 256)
        
        # Export to ONNX
        model_path = Path(__file__).parent / "models" / "denoiser.onnx"
        model_path.parent.mkdir(exist_ok=True)
        
        torch.onnx.export(
            model, dummy_input, model_path,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                'output': {0: 'batch_size', 2: 'height', 3: 'width'}
            }
        )
        
        print(f"Created sample model: {model_path}")
        return str(model_path)
        
    except ImportError:
        print("PyTorch not available. Cannot create sample model.")
        return None


def neural_denoise(image, model_path=None):
    """
    Convenience function for neural denoising.
    
    Args:
        image: Input image
        model_path: Path to ONNX model (optional)
        
    Returns:
        Denoised image
    """
    denoiser = NeuralDenoiser(model_path)
    return denoiser.denoise(image)
