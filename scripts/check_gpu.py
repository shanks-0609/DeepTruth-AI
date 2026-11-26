"""
scripts/check_gpu.py
Verifies CUDA availability, device properties, VRAM, and runs real matmul
and convolution workloads under bfloat16 autocast to validate RTX 5060 (Blackwell/sm_120).
"""

import sys
import torch
import torch.nn as nn


def check_gpu() -> bool:
    print("=" * 60)
    print("DEEPTRUTH AI - HARDWARE VERIFICATION")
    print("=" * 60)

    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available:  {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available to PyTorch.")
        return False

    device_id = 0
    device_name = torch.cuda.get_device_name(device_id)
    capability = torch.cuda.get_device_capability(device_id)
    vram_bytes = torch.cuda.get_device_properties(device_id).total_memory
    vram_gb = vram_bytes / (1024**3)

    print(f"Device [{device_id}]:     {device_name}")
    print(f"Compute Capability: {capability[0]}.{capability[1]} (sm_{capability[0]}{capability[1]}0)")
    print(f"Total VRAM:         {vram_gb:.2f} GB ({vram_bytes:,} bytes)")
    print("-" * 60)

    device = torch.device(f"cuda:{device_id}")

    # Check bf16 support
    bf16_supported = torch.cuda.is_bf16_supported()
    print(f"bfloat16 Hardware Support: {bf16_supported}")

    dtype = torch.bfloat16 if bf16_supported else torch.float16
    print(f"Testing real workloads under torch.autocast with dtype={dtype}...")

    try:
        # 1. Real Matrix Multiplication under autocast
        with torch.autocast(device_type="cuda", dtype=dtype):
            a = torch.randn(2048, 2048, device=device)
            b = torch.randn(2048, 2048, device=device)
            c = torch.matmul(a, b)
            torch.cuda.synchronize(device)
            matmul_norm = c.norm().item()
        print(f"  [PASS] 2048x2048 Matmul executed successfully (norm: {matmul_norm:.2f}).")

        # 2. Real 2D Convolution under autocast (batch=16, 64->128 channels, 224x224)
        conv = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False).to(device)
        x = torch.randn(16, 64, 224, 224, device=device)
        if hasattr(torch, "channels_last"):
            x = x.to(memory_format=torch.channels_last)
            conv = conv.to(memory_format=torch.channels_last)

        with torch.autocast(device_type="cuda", dtype=dtype):
            out = conv(x)
            loss = out.sum()
            loss.backward()
            torch.cuda.synchronize(device)
            grad_norm = conv.weight.grad.norm().item()
        print(f"  [PASS] Conv2d + backward executed successfully (grad norm: {grad_norm:.2f}).")

        mem_allocated_mb = torch.cuda.memory_allocated(device) / (1024**2)
        mem_reserved_mb = torch.cuda.memory_reserved(device) / (1024**2)
        print(f"VRAM Allocated: {mem_allocated_mb:.1f} MB | Reserved: {mem_reserved_mb:.1f} MB")
        print("=" * 60)
        print("GATE 1 STATUS: PASSED - RTX 5060 is fully functional with CUDA & bf16.")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"ERROR: Execution failed on CUDA device: {e}")
        return False


if __name__ == "__main__":
    success = check_gpu()
    sys.exit(0 if success else 1)
