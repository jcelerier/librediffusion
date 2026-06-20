import importlib
import importlib.util
import os
import shutil
import subprocess
import sys
from typing import Dict, Literal, Optional

from packaging.version import Version


python = sys.executable
index_url = os.environ.get("INDEX_URL", "")
uv = shutil.which("uv")


def _check_torch_installed():
    try:
        import torch
        import torchvision  # type: ignore
    except Exception:
        msg = (
            "Missing required pre-installed packages: torch, torchvision\n"
            "Install the PyTorch CUDA wheels from the appropriate index first, e.g.:\n"
            "  pip install --index-url https://download.pytorch.org/whl/cu12x torch torchvision\n"
            "Replace the index URL and versions to match your CUDA runtime."
        )
        raise RuntimeError(msg)

    if not torch.version.cuda:
        raise RuntimeError("Detected CPU-only PyTorch. Install CUDA-enabled torch/vision/audio before installing this package.")


def get_cuda_version() -> str | None:
    _check_torch_installed()

    import torch
    return torch.version.cuda


def get_cuda_major() -> Optional[Literal["11", "12"]]:
    version = get_cuda_version()
    if not version:
        return None

    major = version.split(".")[0]
    if major not in ("11", "12"):
        return None
    return major


def version(package: str) -> Optional[Version]:
    try:
        return Version(importlib.import_module(package).__version__)
    except ModuleNotFoundError:
        return None


def is_installed(package: str) -> bool:
    try:
        spec = importlib.util.find_spec(package)
    except ModuleNotFoundError:
        return False

    return spec is not None


def run_python(command: str, env: Dict[str, str] | None = None) -> str:
    run_kwargs = {
        "args": f"\"{python}\" {command}",
        "shell": True,
        "env": os.environ if env is None else env,
        "encoding": "utf8",
        "errors": "ignore",
    }

    print(run_kwargs["args"])

    result = subprocess.run(**run_kwargs)

    if result.returncode != 0:
        print(f"Error running command: {command}", file=sys.stderr)
        raise RuntimeError(f"Error running command: {command}")

    return result.stdout or ""


def run_pip(command: str, env: Dict[str, str] | None = None) -> str:
    # Only use uv pip if there's an active virtual environment.
    # uv pip is much faster and doesn't require pip to be installed.
    has_venv = os.environ.get("VIRTUAL_ENV") is not None
    if uv and has_venv:
        run_kwargs = {
            "args": f'"{uv}" pip {command}',
            "shell": True,
            "env": os.environ if env is None else env,
            "encoding": "utf8",
            "errors": "ignore",
        }
        print(run_kwargs["args"])
        result = subprocess.run(**run_kwargs)

        if result.returncode != 0:
            print(f"Error running uv pip command: {command}", file=sys.stderr)
            raise RuntimeError(f"Error running uv pip command: {command}")

        return result.stdout or ""

    # Fallback: ensure pip is available - needed for some venvs which don't include pip by default
    if not is_installed("pip"):
        if not is_installed("ensurepip"):
            raise RuntimeError(
                "Neither pip nor ensurepip is available. "
                "Install pip manually: uv pip install pip (or apt install python3-venv)"
            )
        print("pip not found, bootstrapping via ensurepip...")
        run_python("-m ensurepip", env)
    return run_python(f"-m pip {command}", env)
