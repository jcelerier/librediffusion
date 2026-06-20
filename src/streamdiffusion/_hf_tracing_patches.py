"""
Trace-safe monkey-patches for HF diffusers when building ONNX/TensorRT engines.
Activated explicitly from StreamDiffusionWrapper when acceleration=='tensorrt'.
"""

import torch

_ALREADY = False  # idempotence guard
# --------------------------------------------------------------------------- #
# 1. UNet2DConditionModel: guard in_channels % up_factor
# --------------------------------------------------------------------------- #
def _patch_unet():
    from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel

    orig_fwd = UNet2DConditionModel.forward

    def patched(self, sample, *args, **kwargs):
        if torch.jit.is_tracing():
            dim       = torch.as_tensor(getattr(self.config, "in_channels", self.in_channels))
            up_factor = torch.as_tensor(getattr(self.config, "default_overall_up_factor", 1))
            torch._assert(
                torch.remainder(dim, up_factor) == 0,
                f"in_channels={dim} not divisible by default_overall_up_factor={up_factor}"
            )
        return orig_fwd(self, sample, *args, **kwargs)

    UNet2DConditionModel.forward = patched


# --------------------------------------------------------------------------- #
# 2. Downsample2D: replace python assert with torch._assert
# --------------------------------------------------------------------------- #
def _patch_downsample():
    import diffusers.models.downsampling as d
    orig_fwd = d.Downsample2D.forward

    def patched(self, hidden_states, *args, **kwargs):
        torch._assert(
            hidden_states.shape[1] == self.channels,
            f"[Downsample2D] channels mismatch: {hidden_states.shape[1]} vs {self.channels}"
        )
        return orig_fwd(self, hidden_states, *args, **kwargs)

    d.Downsample2D.forward = patched


# --------------------------------------------------------------------------- #
# 3. Upsample2D: same shape check, **keep full signature**
# --------------------------------------------------------------------------- #
def _patch_upsample():
    import diffusers.models.upsampling as u
    orig_fwd = u.Upsample2D.forward

    def patched(self, hidden_states, *args, **kwargs):
        torch._assert(
            hidden_states.shape[1] == self.channels,
            f"[Upsample2D] channels mismatch: {hidden_states.shape[1]} vs {self.channels}"
        )
        return orig_fwd(self, hidden_states, *args, **kwargs)

    u.Upsample2D.forward = patched


# --------------------------------------------------------------------------- #
# master toggle
# --------------------------------------------------------------------------- #
def apply_all_patches():
    global _ALREADY
    if _ALREADY:
        return
    _ALREADY = True
    _patch_unet()
    _patch_downsample()
    _patch_upsample()
    print("[StreamDiffusion] trace-safe patches applied for TensorRT build")
