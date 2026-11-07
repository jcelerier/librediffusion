"""Load a torch distributed checkpoint or a safetensors file (bf16, FP8 or NVFP4) into a module."""

from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from torch import nn
from torch.distributed.checkpoint import FileSystemReader, load

from flux3 import fp8, nvfp4

FLOW_MAP_KEY = "ref_time_in.in_layer.weight"
QUANT_SUFFIXES = (".weight_scale", ".weight_scale_2", ".input_scale")


def safetensors_file(path: str) -> Path | None:
    path = Path(path)
    if path.suffix == ".safetensors":
        return path
    return path / "model.safetensors" if (path / "model.safetensors").exists() else None


def checkpoint_keys(path: str) -> set[str]:
    if file := safetensors_file(path):
        with safe_open(file, "pt") as f:
            return set(f.keys())
    return {k.removeprefix("model.") for k in FileSystemReader(path).read_metadata().state_dict_metadata}


def is_flow_map(path: str) -> bool:
    return FLOW_MAP_KEY in checkpoint_keys(path)


def swap_linears(module: nn.Module, keys) -> type[nn.Module] | None:
    """Every nn.Linear with scales in ``keys`` becomes the quantized linear of its format, in place. NVFP4 is checked
    first: its block scales are also named ``.weight_scale``."""
    for suffix, linear_type in ((nvfp4.SCALE_SUFFIX, nvfp4.NVFP4Linear), (fp8.SCALE_SUFFIX, fp8.FP8Linear)):
        names = [n for n, m in module.named_modules() if isinstance(m, nn.Linear) and n + suffix in keys]
        if names:
            for name in names:
                linear = module.get_submodule(name)
                parent, _, child = name.rpartition(".")
                setattr(module.get_submodule(parent), child,
                        linear_type(linear.in_features, linear.out_features, linear.weight.device))
            return linear_type
    return None


def load_checkpoint(module: nn.Module, path: str, device: str, dtype: torch.dtype) -> nn.Module:
    """Reads only the keys the module has. A safetensors file must match them exactly; quantized weights and
    their scales load as stored."""
    if file := safetensors_file(path):
        return _load_safetensors(module, file, device, dtype)
    state = {
        f"model.{name}": torch.empty(tensor.shape, device=device, dtype=dtype if tensor.is_floating_point() else tensor.dtype)
        for name, tensor in module.state_dict().items()
    }
    load(state, checkpoint_id=path)
    module.load_state_dict({name[len("model."):]: tensor for name, tensor in state.items()}, assign=True)
    return module.to(device).eval()


def _load_safetensors(module: nn.Module, file: Path, device: str, dtype: torch.dtype) -> nn.Module:
    state = load_file(str(file), device=str(device))
    quantized = swap_linears(module, state.keys())
    if quantized is nvfp4.NVFP4Linear and not str(device).startswith("cuda"):
        raise ValueError("NVFP4 weights need the GPU at load time; drop --offload")
    expected = module.state_dict()
    missing, unexpected = sorted(set(expected) - set(state)), sorted(set(state) - set(expected))
    if missing or unexpected:
        raise KeyError(f"{file}: missing {missing[:5]}, unexpected {unexpected[:5]}")
    keep = lambda k, v: k.endswith(QUANT_SUFFIXES) or v.dtype in (fp8.FP8, nvfp4.PACKED) or not v.is_floating_point()
    state = {k: v if keep(k, v) else v.to(dtype) for k, v in state.items()}
    module.load_state_dict(state, strict=True, assign=True)
    if quantized is nvfp4.NVFP4Linear:
        nvfp4.swizzle_weight_scales(module)
    return module.to(device).eval()
