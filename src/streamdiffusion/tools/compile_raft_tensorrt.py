import torch
import logging
from pathlib import Path
from typing import Optional
import fire

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    logger.error("TensorRT not available. Please install it first.")

try:
    from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False
    logger.error("torchvision not available. Please install it first.")


def export_raft_to_onnx(
    onnx_path: Path,
    min_height: int = 512,
    min_width: int = 512,
    max_height: int = 512,
    max_width: int = 512,
    device: str = "cuda"
) -> bool:
    """
    Export RAFT model to ONNX format
    
    Args:
        onnx_path: Path to save the ONNX model
        min_height: Minimum input height for the model
        min_width: Minimum input width for the model
        max_height: Maximum input height for the model
        max_width: Maximum input width for the model
        device: Device to use for export
        
    Returns:
        True if successful, False otherwise
    """
    if not TORCHVISION_AVAILABLE:
        logger.error("torchvision is required but not installed")
        return False
    
    logger.info(f"Exporting RAFT model to ONNX: {onnx_path}")
    logger.info(f"Resolution range: {min_height}x{min_width} - {max_height}x{max_width}")
    
    try:
        # Load RAFT model
        logger.info("Loading RAFT Small model...")
        raft_model = raft_small(weights=Raft_Small_Weights.DEFAULT, progress=True)
        raft_model = raft_model.to(device=device)
        raft_model.eval()
        
        # Create dummy inputs using max resolution for export
        dummy_frame1 = torch.randn(1, 3, max_height, max_width).to(device)
        dummy_frame2 = torch.randn(1, 3, max_height, max_width).to(device)
        
        # Apply RAFT preprocessing if available
        weights = Raft_Small_Weights.DEFAULT
        if hasattr(weights, 'transforms') and weights.transforms is not None:
            transforms = weights.transforms()
            dummy_frame1, dummy_frame2 = transforms(dummy_frame1, dummy_frame2)
        
        # Make batch, height, and width dimensions dynamic
        dynamic_axes = {
            "frame1": {0: "batch_size", 2: "height", 3: "width"},
            "frame2": {0: "batch_size", 2: "height", 3: "width"},
            "flow": {0: "batch_size", 2: "height", 3: "width"},
        }
        
        logger.info("Exporting to ONNX...")
        with torch.no_grad():
            torch.onnx.export(
                raft_model,
                (dummy_frame1, dummy_frame2),
                str(onnx_path),
                verbose=False,
                input_names=['frame1', 'frame2'],
                output_names=['flow'],
                opset_version=17,
                export_params=True,
                dynamic_axes=dynamic_axes,
            )
        
        del raft_model
        torch.cuda.empty_cache()
        
        logger.info(f"Successfully exported ONNX model to {onnx_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to export ONNX model: {e}")
        import traceback
        traceback.print_exc()
        return False


def build_tensorrt_engine(
    onnx_path: Path,
    engine_path: Path,
    min_height: int = 512,
    min_width: int = 512,
    max_height: int = 512,
    max_width: int = 512,
    fp16: bool = True,
    workspace_size_gb: int = 4
) -> bool:
    """
    Build TensorRT engine from ONNX model
    
    Args:
        onnx_path: Path to the ONNX model
        engine_path: Path to save the TensorRT engine
        min_height: Minimum input height for optimization
        min_width: Minimum input width for optimization
        max_height: Maximum input height for optimization
        max_width: Maximum input width for optimization
        fp16: Enable FP16 precision mode
        workspace_size_gb: Maximum workspace size in GB
        
    Returns:
        True if successful, False otherwise
    """
    if not TENSORRT_AVAILABLE:
        logger.error("TensorRT is required but not installed")
        return False
    
    if not onnx_path.exists():
        logger.error(f"ONNX model not found: {onnx_path}")
        return False
    
    logger.info(f"Building TensorRT engine from ONNX model: {onnx_path}")
    logger.info(f"Output path: {engine_path}")
    logger.info(f"Resolution range: {min_height}x{min_width} - {max_height}x{max_width}")
    logger.info(f"FP16 mode: {fp16}")
    logger.info("This may take several minutes...")
    
    try:
        builder = trt.Builder(trt.Logger(trt.Logger.INFO))
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, trt.Logger(trt.Logger.WARNING))
        
        logger.info("Parsing ONNX model...")
        with open(onnx_path, 'rb') as model:
            if not parser.parse(model.read()):
                logger.error("Failed to parse ONNX model")
                for error in range(parser.num_errors):
                    logger.error(f"Parser error: {parser.get_error(error)}")
                return False
        
        logger.info("Configuring TensorRT builder...")
        config = builder.create_builder_config()
        
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size_gb * (1 << 30))
        
        if fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            logger.info("FP16 mode enabled")
        
        # Calculate optimal resolution (middle point)
        opt_height = (min_height + max_height) // 2
        opt_width = (min_width + max_width) // 2
        
        profile = builder.create_optimization_profile()
        min_shape = (1, 3, min_height, min_width)
        opt_shape = (1, 3, opt_height, opt_width)
        max_shape = (1, 3, max_height, max_width)
        
        profile.set_shape("frame1", min_shape, opt_shape, max_shape)
        profile.set_shape("frame2", min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)
        
        logger.info("Building TensorRT engine... (this will take a while)")
        engine = builder.build_serialized_network(network, config)
        
        if engine is None:
            logger.error("Failed to build TensorRT engine")
            return False
        
        logger.info(f"Saving engine to {engine_path}")
        engine_path.parent.mkdir(parents=True, exist_ok=True)
        with open(engine_path, 'wb') as f:
            f.write(engine)
        
        logger.info(f"Successfully built and saved TensorRT engine: {engine_path}")
        logger.info(f"Engine size: {engine_path.stat().st_size / (1024*1024):.2f} MB")
        
        # Delete ONNX file after successful engine creation
        try:
            if onnx_path.exists():
                onnx_path.unlink()
                logger.info(f"Deleted ONNX file: {onnx_path}")
        except Exception as e:
            logger.warning(f"Failed to delete ONNX file: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to build TensorRT engine: {e}")
        import traceback
        traceback.print_exc()
        return False


def compile_raft(
    min_resolution: str = "512x512",
    max_resolution: str = "512x512",
    output_dir: str = "./models/temporal_net",
    device: str = "cuda",
    fp16: bool = True,
    workspace_size_gb: int = 4,
    force_rebuild: bool = False
):
    """
    Main function to compile RAFT model to TensorRT engine
    
    Args:
        min_resolution: Minimum input resolution as "HxW" (e.g., "512x512") (default: "512x512")
        max_resolution: Maximum input resolution as "HxW" (e.g., "1024x1024") (default: "512x512")
        output_dir: Directory to save the models (default: ./models/temporal_net)
        device: Device to use for export (default: cuda)
        fp16: Enable FP16 precision mode (default: True)
        workspace_size_gb: Maximum workspace size in GB (default: 4)
        force_rebuild: Force rebuild even if engine exists (default: False)
    """
    if not TENSORRT_AVAILABLE:
        logger.error("TensorRT is not available. Please install it first using:")
        logger.error("  python -m streamdiffusion.tools.install-tensorrt")
        return
    
    if not TORCHVISION_AVAILABLE:
        logger.error("torchvision is not available. Please install it first using:")
        logger.error("  pip install torchvision")
        return
    
    # Parse resolution strings
    try:
        min_height, min_width = map(int, min_resolution.split('x'))
    except:
        logger.error(f"Invalid min_resolution format: {min_resolution}. Expected format: HxW (e.g., 512x512)")
        return
    
    try:
        max_height, max_width = map(int, max_resolution.split('x'))
    except:
        logger.error(f"Invalid max_resolution format: {max_resolution}. Expected format: HxW (e.g., 1024x1024)")
        return
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Add resolution suffix to filenames
    onnx_path = output_path / f"raft_small_min_{min_resolution}_max_{max_resolution}.onnx"
    engine_path = output_path / f"raft_small_min_{min_resolution}_max_{max_resolution}.engine"
    
    logger.info("="*80)
    logger.info("RAFT TensorRT Compilation")
    logger.info("="*80)
    logger.info(f"Output directory: {output_path.absolute()}")
    logger.info(f"Resolution range: {min_resolution} - {max_resolution}")
    logger.info(f"ONNX path: {onnx_path}")
    logger.info(f"Engine path: {engine_path}")
    logger.info("="*80)
    
    if engine_path.exists() and not force_rebuild:
        logger.info(f"TensorRT engine already exists: {engine_path}")
        logger.info("Use --force_rebuild to rebuild it")
        return
    
    if not onnx_path.exists() or force_rebuild:
        logger.info("\n[Step 1/2] Exporting RAFT to ONNX...")
        if not export_raft_to_onnx(onnx_path, min_height, min_width, max_height, max_width, device):
            logger.error("Failed to export ONNX model")
            return
    else:
        logger.info(f"\n[Step 1/2] ONNX model already exists: {onnx_path}")
    
    logger.info("\n[Step 2/2] Building TensorRT engine...")
    if not build_tensorrt_engine(onnx_path, engine_path, min_height, min_width, max_height, max_width, fp16, workspace_size_gb):
        logger.error("Failed to build TensorRT engine")
        return
    
    logger.info("\n" + "="*80)
    logger.info("✓ Compilation completed successfully!")
    logger.info("="*80)
    logger.info(f"Engine path: {engine_path.absolute()}")
    logger.info("\nYou can now use this engine in TemporalNetTensorRTPreprocessor:")
    logger.info(f'  engine_path="{engine_path.absolute()}"')
    logger.info("="*80)


if __name__ == "__main__":
    fire.Fire(compile_raft)

