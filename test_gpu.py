#!/usr/bin/env python3
"""
GPU availability test script for BundleFlow
コンテナ内でGPUが正しく認識されているかテストします
"""

import torch
import sys


def test_gpu():
    """GPUの状態をテストして表示"""
    print("=" * 60)
    print("GPU Availability Test")
    print("=" * 60)
    
    # PyTorchバージョン
    print(f"\n✓ PyTorch version: {torch.__version__}")
    
    # CUDA availability
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print(f"✓ CUDA is available: YES")
    else:
        print(f"✗ CUDA is available: NO")
        print("\n⚠️  WARNING: CUDA is not available!")
        print("   Please check:")
        print("   1. NVIDIA drivers are installed on host")
        print("   2. NVIDIA Container Toolkit is installed")
        print("   3. Docker is configured to use nvidia runtime")
        return False
    
    # CUDA version
    print(f"✓ CUDA version: {torch.version.cuda}")
    
    # cuDNN version
    if torch.backends.cudnn.is_available():
        print(f"✓ cuDNN version: {torch.backends.cudnn.version()}")
        print(f"✓ cuDNN enabled: {torch.backends.cudnn.enabled}")
    
    # GPU count
    gpu_count = torch.cuda.device_count()
    print(f"✓ Number of GPUs: {gpu_count}")
    
    # GPU details
    if gpu_count > 0:
        print("\nGPU Details:")
        print("-" * 60)
        for i in range(gpu_count):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            
            # Memory info
            total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"         Memory: {total_memory:.2f} GB")
            
            # Current memory usage
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"         Allocated: {allocated:.2f} GB")
            print(f"         Reserved: {reserved:.2f} GB")
    
    # Simple computation test
    print("\n" + "=" * 60)
    print("Running simple computation test...")
    print("=" * 60)
    
    try:
        # Create a tensor on GPU
        x = torch.randn(1000, 1000, device='cuda')
        y = torch.randn(1000, 1000, device='cuda')
        
        # Perform computation
        z = torch.matmul(x, y)
        
        print("✓ Matrix multiplication on GPU: SUCCESS")
        print(f"  Result shape: {z.shape}")
        print(f"  Result device: {z.device}")
        
        # Check memory usage after computation
        allocated = torch.cuda.memory_allocated(0) / 1024**2
        print(f"  GPU memory used: {allocated:.2f} MB")
        
        # Cleanup
        del x, y, z
        torch.cuda.empty_cache()
        
        print("\n✓ All GPU tests passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ GPU computation test failed: {e}")
        return False


if __name__ == "__main__":
    success = test_gpu()
    sys.exit(0 if success else 1)

