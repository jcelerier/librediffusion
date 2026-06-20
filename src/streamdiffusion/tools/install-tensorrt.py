from typing import Literal, Optional

import fire
from packaging.version import Version

from ..pip_utils import is_installed, run_pip, version, get_cuda_major
import platform


def install(cu: Optional[Literal["11", "12"]] = get_cuda_major()):
    if cu not in ("11", "12"):
        raise RuntimeError("CUDA major version not detected. Pass --cu 11 or --cu 12 explicitly.")

    print("Installing TensorRT requirements...")

    min_trt_version = Version("10.12.0") if cu == "12" else Version("9.0.0")
    trt_version = version("tensorrt")
    if trt_version and trt_version < min_trt_version:
        run_pip("uninstall -y tensorrt")

    cudnn_package, trt_package = (
        ("nvidia-cudnn-cu12==9.7.1.26", "tensorrt==10.12.0.36")
        if cu == "12" else
        ("nvidia-cudnn-cu11==8.9.7.29", "tensorrt==9.0.1.post11.dev4")
    )
    if not is_installed(trt_package):
        run_pip(f"install {cudnn_package} --no-cache-dir")
        run_pip(f"install --extra-index-url https://pypi.nvidia.com {trt_package} --no-cache-dir")

    if not is_installed("polygraphy"):
        run_pip(
            "install polygraphy==0.49.24 --extra-index-url https://pypi.ngc.nvidia.com"
        )
    if not is_installed("onnx_graphsurgeon"):
        run_pip(
            "install onnx-graphsurgeon==0.5.8 --extra-index-url https://pypi.ngc.nvidia.com"
        )
    if platform.system() == 'Windows' and not is_installed("pywin32"):
        run_pip(
            "install pywin32==306"
        )
    if platform.system() == 'Windows' and not is_installed("triton"):
        run_pip(
            "install triton-windows==3.4.0.post21"
        )


if __name__ == "__main__":
    fire.Fire(install)
