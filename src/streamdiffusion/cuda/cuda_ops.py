"""
CUDA Operations for StreamDiffusion

Pure CUDA implementations of tensor operations to replace PyTorch.
"""

import ctypes
import torch
from pathlib import Path
from typing import Optional

# Global library handle
_cuda_lib = None


def _load_cuda_lib():
    """Load the CUDA library (lazy initialization)."""
    global _cuda_lib
    if _cuda_lib is None:
        lib_path = Path(__file__).parent / "libstreamdiffusion_cuda.so"
        if not lib_path.exists():
            # Try parent directory (for development)
            lib_path = Path(__file__).parent.parent.parent.parent / "lib" / "libstreamdiffusion_cuda.so"

        if not lib_path.exists():
            raise FileNotFoundError(
                f"CUDA library not found. Please run 'make' to build it.\n"
                f"Expected location: {lib_path}"
            )

        _cuda_lib = ctypes.CDLL(str(lib_path))

        # Define function signatures
        _cuda_lib.launch_scheduler_step_fp16.argtypes = [
            ctypes.c_void_p,  # model_pred
            ctypes.c_void_p,  # x_t_latent
            ctypes.c_void_p,  # output
            ctypes.c_float,   # alpha
            ctypes.c_float,   # beta
            ctypes.c_float,   # c_skip
            ctypes.c_float,   # c_out
            ctypes.c_int,     # batch
            ctypes.c_int,     # channels
            ctypes.c_int,     # height
            ctypes.c_int,     # width
            ctypes.c_void_p,  # stream
        ]

        _cuda_lib.launch_add_noise_fp16.argtypes = [
            ctypes.c_void_p,  # original_samples
            ctypes.c_void_p,  # noise
            ctypes.c_void_p,  # noisy_samples
            ctypes.c_float,   # alpha
            ctypes.c_float,   # beta
            ctypes.c_int,     # N
            ctypes.c_void_p,  # stream
        ]

        _cuda_lib.launch_apply_cfg_fp16.argtypes = [
            ctypes.c_void_p,  # noise_pred_uncond
            ctypes.c_void_p,  # noise_pred_text
            ctypes.c_void_p,  # output
            ctypes.c_float,   # guidance_scale
            ctypes.c_int,     # N
            ctypes.c_void_p,  # stream
        ]

        _cuda_lib.launch_scalar_mul_inplace_fp16.argtypes = [
            ctypes.c_void_p,  # data
            ctypes.c_float,   # scalar
            ctypes.c_int,     # N
            ctypes.c_void_p,  # stream
        ]

        _cuda_lib.launch_scalar_div_inplace_fp16.argtypes = [
            ctypes.c_void_p,  # data
            ctypes.c_float,   # scalar
            ctypes.c_int,     # N
            ctypes.c_void_p,  # stream
        ]

        _cuda_lib.launch_scalar_div_fp16.argtypes = [
            ctypes.c_void_p,  # data
            ctypes.c_void_p,  # output
            ctypes.c_float,   # scalar
            ctypes.c_int,     # N
            ctypes.c_void_p,  # stream
        ]

        _cuda_lib.launch_tensor_sub_fp16.argtypes = [
            ctypes.c_void_p,  # a
            ctypes.c_void_p,  # b
            ctypes.c_void_p,  # output
            ctypes.c_int,     # N
            ctypes.c_void_p,  # stream
        ]

        _cuda_lib.launch_tensor_clone.argtypes = [
            ctypes.c_void_p,  # src
            ctypes.c_void_p,  # dst
            ctypes.c_size_t,  # num_bytes
            ctypes.c_void_p,  # stream
        ]

        _cuda_lib.launch_randn_fp16.argtypes = [
            ctypes.c_void_p,  # output
            ctypes.c_ulonglong,  # seed
            ctypes.c_int,     # N
            ctypes.c_void_p,  # stream
        ]

        _cuda_lib.launch_concat.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),  # input_ptrs
            ctypes.c_int,     # num_inputs
            ctypes.POINTER(ctypes.c_size_t),  # input_byte_sizes
            ctypes.c_void_p,  # output
            ctypes.c_void_p,  # stream
        ]

        _cuda_lib.launch_concat_fp16.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),  # input_ptrs
            ctypes.c_int,     # num_inputs
            ctypes.POINTER(ctypes.c_int),  # input_sizes
            ctypes.c_void_p,  # output
            ctypes.c_void_p,  # stream
        ]

        _cuda_lib.launch_ones_like_fp16.argtypes = [
            ctypes.c_void_p,  # output
            ctypes.c_int,     # N
            ctypes.c_void_p,  # stream
        ]

        _cuda_lib.launch_randn_like_fp16.argtypes = [
            ctypes.c_void_p,  # output
            ctypes.c_ulonglong,  # seed
            ctypes.c_int,     # N
            ctypes.c_void_p,  # stream
        ]

        _cuda_lib.get_last_cuda_error.restype = ctypes.c_char_p

    return _cuda_lib


def scheduler_step_cuda(
    model_pred: torch.Tensor,
    x_t_latent: torch.Tensor,
    alpha_prod_t_sqrt: float,
    beta_prod_t_sqrt: float,
    c_skip: float,
    c_out: float,
) -> torch.Tensor:
    """
    CUDA implementation of scheduler step.

    Replaces PyTorch code:
        F_theta = (x_t_latent - beta_prod_t_sqrt * model_pred) / alpha_prod_t_sqrt
        denoised = c_out * F_theta + c_skip * x_t_latent

    Args:
        model_pred: Model prediction tensor [B, C, H, W]
        x_t_latent: Current latent tensor [B, C, H, W]
        alpha_prod_t_sqrt: Alpha coefficient
        beta_prod_t_sqrt: Beta coefficient
        c_skip: Skip coefficient
        c_out: Output coefficient

    Returns:
        Denoised tensor [B, C, H, W]
    """
    lib = _load_cuda_lib()

    # Validate inputs
    assert model_pred.is_cuda, "model_pred must be on CUDA"
    assert x_t_latent.is_cuda, "x_t_latent must be on CUDA"
    assert model_pred.shape == x_t_latent.shape, "Shape mismatch"
    assert model_pred.is_contiguous(), "model_pred must be contiguous"
    assert x_t_latent.is_contiguous(), "x_t_latent must be contiguous"

    # Get dimensions
    batch, channels, height, width = model_pred.shape

    # Allocate output
    output = torch.empty_like(model_pred)

    # Get data pointers
    model_pred_ptr = model_pred.data_ptr()
    x_t_latent_ptr = x_t_latent.data_ptr()
    output_ptr = output.data_ptr()

    # Get CUDA stream (use default stream = NULL = 0)
    stream_ptr = 0

    # Launch kernel (FP16 version)
    if model_pred.dtype == torch.float16:
        lib.launch_scheduler_step_fp16(
            model_pred_ptr,
            x_t_latent_ptr,
            output_ptr,
            ctypes.c_float(alpha_prod_t_sqrt),
            ctypes.c_float(beta_prod_t_sqrt),
            ctypes.c_float(c_skip),
            ctypes.c_float(c_out),
            batch,
            channels,
            height,
            width,
            stream_ptr
        )
    else:
        raise NotImplementedError(f"Unsupported dtype: {model_pred.dtype}. Only float16 is supported.")

    # Check for errors
    error = lib.get_last_cuda_error()
    if error != b"no error":
        raise RuntimeError(f"CUDA error: {error.decode()}")

    return output


def add_noise_cuda(
    original_samples: torch.Tensor,
    noise: torch.Tensor,
    alpha_prod_t_sqrt: float,
    beta_prod_t_sqrt: float,
) -> torch.Tensor:
    """
    CUDA implementation of add_noise.

    Replaces PyTorch code:
        noisy_samples = alpha_prod_t_sqrt * original_samples + beta_prod_t_sqrt * noise

    Args:
        original_samples: Original latent samples [B, C, H, W]
        noise: Noise tensor [B, C, H, W]
        alpha_prod_t_sqrt: Alpha coefficient
        beta_prod_t_sqrt: Beta coefficient

    Returns:
        Noisy samples [B, C, H, W]
    """
    lib = _load_cuda_lib()

    # Validate inputs
    assert original_samples.is_cuda, "original_samples must be on CUDA"
    assert noise.is_cuda, "noise must be on CUDA"
    assert original_samples.shape == noise.shape, "Shape mismatch"
    assert original_samples.is_contiguous(), "original_samples must be contiguous"
    assert noise.is_contiguous(), "noise must be contiguous"

    # Calculate total elements
    N = original_samples.numel()

    # Allocate output
    noisy_samples = torch.empty_like(original_samples)

    # Get data pointers
    original_ptr = original_samples.data_ptr()
    noise_ptr = noise.data_ptr()
    output_ptr = noisy_samples.data_ptr()

    # Get CUDA stream (use default stream = NULL = 0)
    stream_ptr = 0

    # Launch kernel (FP16 version)
    if original_samples.dtype == torch.float16:
        lib.launch_add_noise_fp16(
            original_ptr,
            noise_ptr,
            output_ptr,
            ctypes.c_float(alpha_prod_t_sqrt),
            ctypes.c_float(beta_prod_t_sqrt),
            N,
            stream_ptr
        )
    else:
        raise NotImplementedError(f"Unsupported dtype: {original_samples.dtype}. Only float16 is supported.")

    # Check for errors
    error = lib.get_last_cuda_error()
    if error != b"no error":
        raise RuntimeError(f"CUDA error: {error.decode()}")

    return noisy_samples


def apply_cfg_cuda(
    noise_pred_uncond: torch.Tensor,
    noise_pred_text: torch.Tensor,
    guidance_scale: float,
) -> torch.Tensor:
    """
    CUDA implementation of CFG (Classifier-Free Guidance).

    Replaces PyTorch code:
        model_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    Args:
        noise_pred_uncond: Unconditional noise prediction [B, C, H, W]
        noise_pred_text: Text-conditional noise prediction [B, C, H, W]
        guidance_scale: Guidance scale coefficient

    Returns:
        Combined noise prediction [B, C, H, W]
    """
    lib = _load_cuda_lib()

    # Validate inputs
    assert noise_pred_uncond.is_cuda, "noise_pred_uncond must be on CUDA"
    assert noise_pred_text.is_cuda, "noise_pred_text must be on CUDA"
    assert noise_pred_uncond.shape == noise_pred_text.shape, "Shape mismatch"
    assert noise_pred_uncond.is_contiguous(), "noise_pred_uncond must be contiguous"
    assert noise_pred_text.is_contiguous(), "noise_pred_text must be contiguous"

    # Calculate total elements
    N = noise_pred_uncond.numel()

    # Allocate output
    output = torch.empty_like(noise_pred_uncond)

    # Get data pointers
    uncond_ptr = noise_pred_uncond.data_ptr()
    text_ptr = noise_pred_text.data_ptr()
    output_ptr = output.data_ptr()

    # Get CUDA stream (use default stream = NULL = 0)
    stream_ptr = 0

    # Launch kernel (FP16 version)
    if noise_pred_uncond.dtype == torch.float16:
        lib.launch_apply_cfg_fp16(
            uncond_ptr,
            text_ptr,
            output_ptr,
            ctypes.c_float(guidance_scale),
            N,
            stream_ptr
        )
    else:
        raise NotImplementedError(f"Unsupported dtype: {noise_pred_uncond.dtype}. Only float16 is supported.")

    # Check for errors
    error = lib.get_last_cuda_error()
    if error != b"no error":
        raise RuntimeError(f"CUDA error: {error.decode()}")

    return output


def scalar_mul_inplace_cuda(
    data: torch.Tensor,
    scalar: float,
) -> None:
    """
    CUDA implementation of in-place scalar multiplication.

    Replaces PyTorch code:
        data.mul_(scalar)

    Args:
        data: Input tensor [any shape]
        scalar: Scalar multiplier

    Returns:
        None (modifies data in-place)
    """
    lib = _load_cuda_lib()

    # Validate inputs
    assert data.is_cuda, "data must be on CUDA"
    assert data.is_contiguous(), "data must be contiguous"

    # Calculate total elements
    N = data.numel()

    # Get data pointer
    data_ptr = data.data_ptr()

    # Get CUDA stream (use default stream = NULL = 0)
    stream_ptr = 0

    # Launch kernel (FP16 version)
    if data.dtype == torch.float16:
        lib.launch_scalar_mul_inplace_fp16(
            data_ptr,
            ctypes.c_float(scalar),
            N,
            stream_ptr
        )
    else:
        raise NotImplementedError(f"Unsupported dtype: {data.dtype}. Only float16 is supported.")

    # Check for errors
    error = lib.get_last_cuda_error()
    if error != b"no error":
        raise RuntimeError(f"CUDA error: {error.decode()}")


def scalar_div_inplace_cuda(
    data: torch.Tensor,
    scalar: float,
) -> None:
    """
    CUDA implementation of in-place scalar division.

    Replaces PyTorch code:
        data.div_(scalar)

    Args:
        data: Input tensor [any shape]
        scalar: Scalar divisor

    Returns:
        None (modifies data in-place)
    """
    lib = _load_cuda_lib()

    # Validate inputs
    assert data.is_cuda, "data must be on CUDA"
    assert data.is_contiguous(), "data must be contiguous"

    # Calculate total elements
    N = data.numel()

    # Get data pointer
    data_ptr = data.data_ptr()

    # Get CUDA stream (use default stream = NULL = 0)
    stream_ptr = 0

    # Launch kernel (FP16 version)
    if data.dtype == torch.float16:
        lib.launch_scalar_div_inplace_fp16(
            data_ptr,
            ctypes.c_float(scalar),
            N,
            stream_ptr
        )
    else:
        raise NotImplementedError(f"Unsupported dtype: {data.dtype}. Only float16 is supported.")

    # Check for errors
    error = lib.get_last_cuda_error()
    if error != b"no error":
        raise RuntimeError(f"CUDA error: {error.decode()}")


def tensor_sub_cuda(
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """
    CUDA implementation of element-wise tensor subtraction.

    Replaces PyTorch code:
        output = a.sub(b)  or  output = a - b

    Args:
        a: First tensor [any shape]
        b: Second tensor [any shape, must match a]

    Returns:
        Result tensor [same shape as a and b]
    """
    lib = _load_cuda_lib()

    # Validate inputs
    assert a.is_cuda, "a must be on CUDA"
    assert b.is_cuda, "b must be on CUDA"
    assert a.shape == b.shape, "Shape mismatch"
    assert a.is_contiguous(), "a must be contiguous"
    assert b.is_contiguous(), "b must be contiguous"

    # Calculate total elements
    N = a.numel()

    # Allocate output
    output = torch.empty_like(a)

    # Get data pointers
    a_ptr = a.data_ptr()
    b_ptr = b.data_ptr()
    output_ptr = output.data_ptr()

    # Get CUDA stream (use default stream = NULL = 0)
    stream_ptr = 0

    # Launch kernel (FP16 version)
    if a.dtype == torch.float16:
        lib.launch_tensor_sub_fp16(
            a_ptr,
            b_ptr,
            output_ptr,
            N,
            stream_ptr
        )
    else:
        raise NotImplementedError(f"Unsupported dtype: {a.dtype}. Only float16 is supported.")

    # Check for errors
    error = lib.get_last_cuda_error()
    if error != b"no error":
        raise RuntimeError(f"CUDA error: {error.decode()}")

    return output


def scalar_div_cuda(
    data: torch.Tensor,
    scalar: float,
) -> torch.Tensor:
    """
    CUDA implementation of scalar division (non-in-place).

    Replaces PyTorch code:
        output = data / scalar

    Args:
        data: Input tensor [any shape]
        scalar: Scalar divisor

    Returns:
        Result tensor [same shape as data]
    """
    lib = _load_cuda_lib()

    # Validate inputs
    assert data.is_cuda, "data must be on CUDA"
    assert data.is_contiguous(), "data must be contiguous"

    # Calculate total elements
    N = data.numel()

    # Allocate output
    output = torch.empty_like(data)

    # Get data pointers
    data_ptr = data.data_ptr()
    output_ptr = output.data_ptr()

    # Get CUDA stream (use default stream = NULL = 0)
    stream_ptr = 0

    # Launch kernel (FP16 version)
    if data.dtype == torch.float16:
        lib.launch_scalar_div_fp16(
            data_ptr,
            output_ptr,
            ctypes.c_float(scalar),
            N,
            stream_ptr
        )
    else:
        raise NotImplementedError(f"Unsupported dtype: {data.dtype}. Only float16 is supported.")

    # Check for errors
    error = lib.get_last_cuda_error()
    if error != b"no error":
        raise RuntimeError(f"CUDA error: {error.decode()}")

    return output


def randn_cuda(
    tensor: torch.Tensor,
    seed: int = 42,
) -> torch.Tensor:
    """CUDA implementation of random normal generation (in-place).

    Fills the tensor with random values from a normal distribution (mean=0, std=1).
    This replaces PyTorch's .normal_() operation using cuRAND.

    Args:
        tensor: The tensor to fill with random values (must be on CUDA, float16)
        seed: Random seed for reproducibility

    Returns:
        The input tensor (modified in-place)
    """
    lib = _load_cuda_lib()

    # Validate inputs
    assert tensor.is_cuda, "tensor must be on CUDA"
    assert tensor.is_contiguous(), "tensor must be contiguous"

    N = tensor.numel()
    tensor_ptr = tensor.data_ptr()
    stream_ptr = 0  # Default stream

    # Call CUDA kernel
    if tensor.dtype == torch.float16:
        lib.launch_randn_fp16(
            tensor_ptr,
            ctypes.c_ulonglong(seed),
            N,
            stream_ptr
        )
    else:
        raise NotImplementedError(f"Unsupported dtype: {tensor.dtype}. Only float16 is supported.")

    # Check for errors
    error = lib.get_last_cuda_error()
    if error != b"no error":
        raise RuntimeError(f"CUDA error: {error.decode()}")

    return tensor


def tensor_clone_cuda(
    src: torch.Tensor,
) -> torch.Tensor:
    """
    CUDA implementation of tensor clone.

    Replaces PyTorch code:
        output = src.clone()

    Args:
        src: Source tensor [any shape]

    Returns:
        Cloned tensor [same shape as src]
    """
    lib = _load_cuda_lib()

    # Validate inputs
    assert src.is_cuda, "src must be on CUDA"
    assert src.is_contiguous(), "src must be contiguous"

    # Calculate total bytes
    num_bytes = src.numel() * src.element_size()

    # Allocate output
    output = torch.empty_like(src)

    # Get data pointers
    src_ptr = src.data_ptr()
    output_ptr = output.data_ptr()

    # Get CUDA stream (use default stream = NULL = 0)
    stream_ptr = 0

    # Launch memory copy
    lib.launch_tensor_clone(
        src_ptr,
        output_ptr,
        num_bytes,
        stream_ptr
    )

    # Check for errors
    error = lib.get_last_cuda_error()
    if error != b"no error":
        raise RuntimeError(f"CUDA error: {error.decode()}")

    return output


def is_cuda_available() -> bool:
    """Check if CUDA library is available."""
    try:
        _load_cuda_lib()
        return True
    except (FileNotFoundError, OSError):
        return False


def concat_cuda(
    tensors: list[torch.Tensor],
    dim: int = 0
) -> torch.Tensor:
    """CUDA implementation of torch.concat/torch.cat.

    Concatenates tensors along the specified dimension. Works with any dtype.

    Args:
        tensors: List of tensors to concatenate (must be on CUDA, contiguous, same dtype)
        dim: Dimension to concatenate along (only dim=0 is supported for now)

    Returns:
        Concatenated tensor
    """
    lib = _load_cuda_lib()

    # Validate inputs
    assert len(tensors) > 0, "Need at least one tensor"
    assert dim == 0, "Only dim=0 is currently supported"

    ref_dtype = tensors[0].dtype
    ref_device = tensors[0].device

    for i, t in enumerate(tensors):
        assert t.is_cuda, f"tensor {i} must be on CUDA"
        assert t.is_contiguous(), f"tensor {i} must be contiguous"
        assert t.dtype == ref_dtype, f"tensor {i} has dtype {t.dtype}, expected {ref_dtype}"
        assert t.device == ref_device, f"tensor {i} must be on same device"

    # All tensors must have same shape except for dim=0
    ref_shape = list(tensors[0].shape)
    for i, t in enumerate(tensors[1:], 1):
        t_shape = list(t.shape)
        for d in range(len(ref_shape)):
            if d != dim:
                assert ref_shape[d] == t_shape[d], f"Shape mismatch at dim {d}"

    # Calculate output shape
    output_shape = ref_shape.copy()
    output_shape[dim] = sum(t.shape[dim] for t in tensors)

    # Allocate output tensor
    output = torch.empty(
        output_shape,
        dtype=ref_dtype,
        device=ref_device
    )

    # Prepare input pointers and byte sizes
    input_ptrs = (ctypes.c_void_p * len(tensors))()
    input_byte_sizes = (ctypes.c_size_t * len(tensors))()

    for i, t in enumerate(tensors):
        input_ptrs[i] = t.data_ptr()
        input_byte_sizes[i] = t.numel() * t.element_size()

    output_ptr = output.data_ptr()
    stream_ptr = 0  # Default stream

    # Call CUDA kernel
    lib.launch_concat(
        input_ptrs,
        len(tensors),
        input_byte_sizes,
        output_ptr,
        stream_ptr
    )

    # Check for errors
    error = lib.get_last_cuda_error()
    if error != b"no error":
        raise RuntimeError(f"CUDA error: {error.decode()}")

    return output


def ones_like_cuda(
    tensor: torch.Tensor
) -> torch.Tensor:
    """CUDA implementation of torch.ones_like.

    Creates a tensor filled with 1.0, with the same shape/dtype/device as input.

    Args:
        tensor: Reference tensor (must be on CUDA, float16)

    Returns:
        Tensor filled with 1.0
    """
    lib = _load_cuda_lib()

    # Validate inputs
    assert tensor.is_cuda, "tensor must be on CUDA"

    # Allocate output
    output = torch.empty_like(tensor)

    N = tensor.numel()
    output_ptr = output.data_ptr()
    stream_ptr = 0  # Default stream

    # Call CUDA kernel
    if tensor.dtype == torch.float16:
        lib.launch_ones_like_fp16(
            output_ptr,
            N,
            stream_ptr
        )
    else:
        raise NotImplementedError(f"Unsupported dtype: {tensor.dtype}. Only float16 is supported.")

    # Check for errors
    error = lib.get_last_cuda_error()
    if error != b"no error":
        raise RuntimeError(f"CUDA error: {error.decode()}")

    return output


def randn_like_cuda(
    tensor: torch.Tensor,
    seed: int = 42
) -> torch.Tensor:
    """CUDA implementation of torch.randn_like.

    Creates a tensor filled with random normal values, with the same shape/dtype/device as input.

    Args:
        tensor: Reference tensor (must be on CUDA, float16)
        seed: Random seed

    Returns:
        Tensor filled with random normal values
    """
    lib = _load_cuda_lib()

    # Validate inputs
    assert tensor.is_cuda, "tensor must be on CUDA"

    # Allocate output
    output = torch.empty_like(tensor)

    N = tensor.numel()
    output_ptr = output.data_ptr()
    stream_ptr = 0  # Default stream

    # Call CUDA kernel
    if tensor.dtype == torch.float16:
        lib.launch_randn_like_fp16(
            output_ptr,
            ctypes.c_ulonglong(seed),
            N,
            stream_ptr
        )
    else:
        raise NotImplementedError(f"Unsupported dtype: {tensor.dtype}. Only float16 is supported.")

    # Check for errors
    error = lib.get_last_cuda_error()
    if error != b"no error":
        raise RuntimeError(f"CUDA error: {error.decode()}")

    return output
