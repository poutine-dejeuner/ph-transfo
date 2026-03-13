import resource

import torch


def print_peak_memory():
    # RAM
    # On Linux, ru_maxrss is in Kilobytes
    peak_ram = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"Peak RAM Usage: {peak_ram:.2f} MB")

    # VRAM
    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024) # MB
        print(f"Peak VRAM Usage: {peak_vram:.2f} MB")
        torch.cuda.reset_peak_memory_stats()
    else:
        print("CUDA not available, skipping VRAM check.")
