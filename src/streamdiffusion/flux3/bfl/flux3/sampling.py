"""Euler sampling: CFG on a two-sample pack [unconditional, conditional], or a flow map on one sample."""

import torch
from torch import Tensor

from flux3.dit import DiT, Packed


def timesteps(steps: int, alpha: float) -> Tensor:
    t = torch.linspace(1.0, 0.0, steps + 1)
    return alpha * t / (1.0 + (alpha - 1.0) * t)


@torch.inference_mode()
@torch.autocast("cuda", dtype=torch.bfloat16)
def denoise(dit: DiT, packed: Packed, schedule: Tensor, guidance: float) -> Tensor:
    """Returns the denoised image tokens of the conditional sample."""
    for step in range(len(schedule) - 1):
        t, dt = schedule[step].item(), (schedule[step + 1] - schedule[step]).item()
        packed.timesteps["image"].fill_(t)
        unconditional, conditional = dit(packed).chunk(2)
        velocity = unconditional + guidance * (conditional - unconditional)
        x = packed.data["image"].chunk(2)[0] + dt * velocity
        packed.data["image"] = torch.cat((x, x))
    return packed.data["image"].chunk(2)[0]


@torch.inference_mode()
@torch.autocast("cuda", dtype=torch.bfloat16)
def denoise_flow_map(dit: DiT, packed: Packed, schedule: Tensor) -> Tensor:
    for step in range(len(schedule) - 1):
        t, dt = schedule[step].item(), (schedule[step + 1] - schedule[step]).item()
        packed.timesteps["image"].fill_(t)
        target = schedule[step + 1].reshape(1).to(packed.data["image"].device)
        packed.data["image"] = packed.data["image"] + dt * dit(packed, timesteps_ref=target)
    return packed.data["image"]
