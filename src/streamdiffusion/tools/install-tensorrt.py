from typing import Literal, Optional
import os

import fire
from packaging.version import Version

from ..pip_utils import is_installed, run_pip, version
import platform


def get_cuda_version_from_torch() -> Optional[Literal["11", "12"]]:
    try:
        import torch
    except ImportError:
        return None

    return torch.version.cuda.split(".")[0]


def install(
    cu: Optional[Literal["11", "12"]] = get_cuda_version_from_torch(),
    use_rtx: bool = False,
):
    if cu is None or cu not in ["11", "12"]:
        print("Could not detect CUDA version. Please specify manually.")
        return

    if use_rtx:
        print("Installing TensorRT-RTX requirements...")
        print("\n" + "="*80)
        print("IMPORTANT: TensorRT-RTX Installation Instructions")
        print("="*80)
        print("\n1. TensorRT-RTX is not available on PyPI and must be installed manually.")
        print("2. Download TensorRT-RTX 1.0.0.21 from:")
        print("   https://developer.nvidia.com/tensorrt-rtx")
        print("\n3. After downloading, install the wheel:")
        if platform.system() == "Windows":
            print("   python -m pip install C:\\path\\to\\TensorRT-RTX-1.0.0.21\\python\\tensorrt_rtx-1.0.0.21-cp3X-none-win_amd64.whl")
            print("\n4. Add TensorRT-RTX to your PATH:")
            print("   set PATH=\"%PATH%;C:\\path\\to\\TensorRT-RTX-1.0.0.21\\lib\"")
        else:
            print("   python -m pip install /path/to/TensorRT-RTX-1.0.0.21/python/tensorrt_rtx-1.0.0.21-cp3X-none-linux_x86_64.whl")
            print("\n4. Add TensorRT-RTX to your LD_LIBRARY_PATH:")
            print("   export LD_LIBRARY_PATH=/path/to/TensorRT-RTX-1.0.0.21/lib:$LD_LIBRARY_PATH")
        print("\n5. Install Bazel (required for building with TensorRT-RTX):")
        if platform.system() == "Windows":
            print("   choco install bazelisk -y")
        else:
            print("   curl -L https://github.com/bazelbuild/bazelisk/releases/download/v1.26.0/bazelisk-linux-amd64 \\")
            print("     -o bazelisk && mv bazelisk /usr/bin/bazel && chmod +x /usr/bin/bazel")
        print("\n6. Set USE_TRT_RTX=true when running StreamDiffusion:")
        print("   USE_TRT_RTX=true python your_script.py")
        print("="*80 + "\n")

        # Check if tensorrt_rtx is already installed
        if is_installed("tensorrt_rtx"):
            print("✓ tensorrt_rtx is already installed")
        else:
            print("⚠ tensorrt_rtx is NOT installed. Please follow the instructions above.")
    else:
        print("Installing TensorRT requirements...")

        if is_installed("tensorrt"):
            if version("tensorrt") < Version("9.0.0"):
                run_pip("uninstall -y tensorrt")

        cudnn_name = f"nvidia-cudnn-cu{cu}==8.9.4.25"

        if not is_installed("tensorrt"):
            run_pip(f"install {cudnn_name} --no-cache-dir")
            run_pip(
                "install --pre --extra-index-url https://pypi.nvidia.com tensorrt==9.0.1.post11.dev4 --no-cache-dir"
            )

    if not is_installed("polygraphy"):
        run_pip(
            "install polygraphy==0.47.1 --extra-index-url https://pypi.ngc.nvidia.com"
        )
    if not is_installed("onnx_graphsurgeon"):
        run_pip(
            "install onnx-graphsurgeon==0.3.26 --extra-index-url https://pypi.ngc.nvidia.com"
        )
    if platform.system() == 'Windows' and not is_installed("pywin32"):
        run_pip(
            "install pywin32"
        )

    pass


if __name__ == "__main__":
    fire.Fire(install)
