import gc
import os
from typing import *

import torch

from .models.models import BaseModel
from .utilities import (
    build_engine,
    export_onnx,
    optimize_onnx,
)


def create_onnx_path(name, onnx_dir, opt=True):
    return os.path.join(onnx_dir, name + (".opt" if opt else "") + ".onnx")


class EngineBuilder:
    def __init__(
        self,
        model: BaseModel,
        network: Any,
        device=torch.device("cuda"),
    ):
        self.device = device

        self.model = model
        self.network = network

    def build(
        self,
        onnx_path: str,
        onnx_opt_path: str,
        engine_path: str,
        opt_image_height: int = 512,
        opt_image_width: int = 512,
        opt_batch_size: int = 1,
        min_image_resolution: int = 256,
        max_image_resolution: int = 1024,
        # librediffusion fork: independent W/H ranges so non-square engines (e.g. 512x768)
        # are first-class. When None, fall back to the square min/max_image_resolution
        # for backward compat (square callers produce identical engines).
        min_image_height: int = None,
        max_image_height: int = None,
        min_image_width: int = None,
        max_image_width: int = None,
        build_enable_refit: bool = False,
        build_static_batch: bool = False,
        build_dynamic_shape: bool = True,
        build_all_tactics: bool = False,
        onnx_opset: int = 19,  # librediffusion fork: 17 -> 19 for better graph optimizations
        force_engine_build: bool = False,
        force_onnx_export: bool = False,
        force_onnx_optimize: bool = False,
        builder_optimization_level: int = 5,  # 0-5; surfaced as a user toggle upstream
        hardware_compatibility=None,  # None/'none' | 'ampere_plus' | 'same_cc' — portable engines
    ):
        if not force_onnx_export and os.path.exists(onnx_path):
            print(f"Found cached model: {onnx_path}")
        else:
            print(f"Exporting model: {onnx_path}")
            export_onnx(
                self.network,
                onnx_path=onnx_path,
                model_data=self.model,
                opt_image_height=opt_image_height,
                opt_image_width=opt_image_width,
                opt_batch_size=opt_batch_size,
                onnx_opset=onnx_opset,
            )
            self.network = self.network.to("cpu")
            del self.network
            gc.collect()
            torch.cuda.empty_cache()
        if not force_onnx_optimize and os.path.exists(onnx_opt_path):
            print(f"Found cached model: {onnx_opt_path}")
        else:
            print(f"Generating optimizing model: {onnx_opt_path}")
            optimize_onnx(
                onnx_path=onnx_path,
                onnx_opt_path=onnx_opt_path,
                model_data=self.model,
            )
        # librediffusion fork: independent W/H latent bounds. Fall back to the square
        # min/max_image_resolution when the W/H-specific args aren't supplied (backward compat).
        _min_h = min_image_height if min_image_height is not None else min_image_resolution
        _max_h = max_image_height if max_image_height is not None else max_image_resolution
        _min_w = min_image_width if min_image_width is not None else min_image_resolution
        _max_w = max_image_width if max_image_width is not None else max_image_resolution
        self.model.min_latent_height = _min_h // 8
        self.model.max_latent_height = _max_h // 8
        self.model.min_latent_width = _min_w // 8
        self.model.max_latent_width = _max_w // 8
        # Keep the shared scalars consistent (some legacy readers reference them); use the
        # overall min/max across both axes so any remaining square consumer stays in-range.
        self.model.min_latent_shape = min(_min_h, _min_w) // 8
        self.model.max_latent_shape = max(_max_h, _max_w) // 8
        if not force_engine_build and os.path.exists(engine_path):
            print(f"Found cached engine: {engine_path}")
        else:
            build_engine(
                engine_path=engine_path,
                onnx_opt_path=onnx_opt_path,
                model_data=self.model,
                opt_image_height=opt_image_height,
                opt_image_width=opt_image_width,
                opt_batch_size=opt_batch_size,
                build_static_batch=build_static_batch,
                build_dynamic_shape=build_dynamic_shape,
                build_all_tactics=build_all_tactics,
                build_enable_refit=build_enable_refit,
                builder_optimization_level=builder_optimization_level,
                hardware_compatibility=hardware_compatibility,
            )
        # librediffusion fork: keep .timing.cache (speeds rebuilds) — upstream deleted everything
        # in the dir that wasn't a .engine, which nuked the timing cache we now write.
        for file in os.listdir(os.path.dirname(engine_path)):
            if file.endswith('.engine') or file.endswith('.timing.cache'):
                continue
            os.remove(os.path.join(os.path.dirname(engine_path), file))

        gc.collect()
        torch.cuda.empty_cache()
