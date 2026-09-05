import os


def resolve_sensevoice_device(requested=None, cuda_available=None):
    """Map SENSEVOICE_DEVICE to a FunASR device string.

    The image defaults to ``auto``. FunASR/torch want ``cuda:0`` or ``cpu``.
    Passing ``auto`` through would fail on hosts without a GPU, which is the
    path the original Docker issue was hitting.
    """
    if requested is None:
        requested = os.getenv("SENSEVOICE_DEVICE", "cuda:0")
    if requested != "auto":
        return requested
    if cuda_available is None:
        import torch

        cuda_available = torch.cuda.is_available()
    return "cuda:0" if cuda_available else "cpu"
