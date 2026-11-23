

from __future__ import annotations

from contextlib import nullcontext

import torch


VALID_SDPA_BACKENDS = ("auto", "math")


def validate_sdpa_backend(backend: str) -> str:

    normalized = str(backend).strip().lower()
    if normalized not in VALID_SDPA_BACKENDS:
        expected = ", ".join(VALID_SDPA_BACKENDS)
        raise ValueError(
            f"sdpa_backend must be one of [{expected}], got {backend!r}"
        )
    return normalized


def sdpa_kernel_context(backend: str):

    backend = validate_sdpa_backend(backend)
    if backend == "auto":
        return nullcontext()
    return torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH)


def sdpa_policy_metadata(backend: str) -> dict[str, object]:

    backend = validate_sdpa_backend(backend)
    return {
        "requested": backend,
        "effective_policy": (
            "pytorch_auto_dispatch" if backend == "auto" else "math_only"
        ),
        "torch_version": torch.__version__,
    }
