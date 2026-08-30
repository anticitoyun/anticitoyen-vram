"""anticitoyen VRAM/RAM -- tiered, per-GPU-quantized LLM inference.

Two ideas hold the project together:

1. A GPU should store weights in the format its own silicon can read best.
   Blackwell has FP4 tensor cores, Ampere does not, so the same checkpoint is
   written as NVFP4 for one card and INT4 for the other rather than levelling
   both down to a common denominator.

2. Memory is a hierarchy, not a wall. VRAM on the fast card, then VRAM on the
   slow one, then host RAM reached over PCIe -- with the placement chosen by
   measuring what each tier costs rather than by hoping the model fits.

The public surface is the CLI (``acvram``) and the OpenAI-compatible server.
"""

__version__ = "0.2.0"

__all__ = ["__version__"]
