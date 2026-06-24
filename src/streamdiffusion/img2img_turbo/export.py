"""ONNX export + TRT build for the img2img-turbo skip-VAE pipeline.

Produces the three skip-VAE engines the C++ Img2ImgTurboPipeline loads (the CLIP engine is built
separately by train-lora.py via the shared compile_clip, since sd-turbo's text encoder is a plain
CLIPTextModel). I/O contract (validated 69 dB through TRT vs the Python golden):

  unet.onnx         in: latent[1,4,h,w], ehs[1,77,1024]   out: model_pred[1,4,h,w]  (t=999 baked)
  vae_encoder.onnx  in: image[1,3,H,W]                     out: latent(=mean*sf), enc0..enc3 skips
  vae_decoder.onnx  in: latent_scaled, s0..s3 (skips reversed enc3..enc0)  out: image[1,3,H,W] in [-1,1]
                    (skip_conv_1..4 + adds baked inside; gamma=1; /sf baked on the latent input)

The closed-form 1-step DDPM x0 = (latent - sqrt(1-acp999)*model_pred)/sqrt(acp999) and the decoder's
post_quant_conv/sf scaling live on the C++ side; the engines here are self-contained otherwise.
"""
import os
import torch
import tensorrt as trt

from .pix2pix_turbo import Pix2Pix_Turbo


def fuse_loras(unet, vae):
    """Merge PEFT/diffusers LoRA adapters into base weights so the export is adapter-free (exact)."""
    for m, name in ((unet, "unet"), (vae, "vae")):
        fused = False
        if hasattr(m, "fuse_lora"):
            try:
                m.fuse_lora(); fused = True
            except Exception as e:
                print(f"[fuse] {name}.fuse_lora failed ({e}); trying peft merge")
        if not fused:
            from peft.tuners.lora import LoraLayer
            for mod in m.modules():
                if isinstance(mod, LoraLayer):
                    mod.merge()
            fused = True
        print(f"[fuse] {name}: merged LoRA adapters")


class UNetExport(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet
        self.register_buffer("t", torch.tensor([999], dtype=torch.long))

    def forward(self, latent, ehs):
        return self.unet(latent, self.t, encoder_hidden_states=ehs).sample


class EncoderExport(torch.nn.Module):
    """Inlined my_vae_encoder_fwd: returns (mean-latent*sf, enc0..enc3) as explicit outputs."""
    def __init__(self, vae):
        super().__init__()
        self.enc = vae.encoder
        self.quant_conv = vae.quant_conv     # AutoencoderKL.encode applies this after the encoder
        self.sf = float(vae.config.scaling_factor)

    def forward(self, x):
        e = self.enc
        sample = e.conv_in(x)
        skips = []
        for down_block in e.down_blocks:
            skips.append(sample)            # capture BEFORE downsample (enc0..enc3)
            sample = down_block(sample)
        sample = e.mid_block(sample)
        sample = e.conv_norm_out(sample)
        sample = e.conv_act(sample)
        h = e.conv_out(sample)
        moments = self.quant_conv(h)        # [1,8,h,w] mean||logvar (post quant_conv)
        mean = moments[:, :4]               # .mode() == mean of the diagonal gaussian
        latent = mean * self.sf
        return latent, skips[0], skips[1], skips[2], skips[3]


class DecoderExport(torch.nn.Module):
    """Inlined my_vae_decoder_fwd: skip inputs passed already-reversed (s0=enc3 .. s3=enc0).
    Bakes /sf on the latent input and gamma=1."""
    def __init__(self, vae):
        super().__init__()
        self.dec = vae.decoder
        self.post_quant_conv = vae.post_quant_conv  # AutoencoderKL.decode applies this before the decoder
        self.sf = float(vae.config.scaling_factor)

    def forward(self, latent_scaled, s0, s1, s2, s3):
        d = self.dec
        skips = [s0, s1, s2, s3]
        skip_convs = [d.skip_conv_1, d.skip_conv_2, d.skip_conv_3, d.skip_conv_4]
        sample = d.conv_in(self.post_quant_conv(latent_scaled / self.sf))
        up_dtype = next(iter(d.up_blocks.parameters())).dtype
        sample = d.mid_block(sample)
        sample = sample.to(up_dtype)
        for idx, up_block in enumerate(d.up_blocks):
            sample = sample + skip_convs[idx](skips[idx])   # gamma=1 baked
            sample = up_block(sample)
        sample = d.conv_norm_out(sample)
        sample = d.conv_act(sample)
        return d.conv_out(sample).clamp(-1, 1)


def _onnx_export(mod, args_tuple, names_in, names_out, path, dynamic=None):
    # Force the legacy TorchScript exporter (dynamo=False): torch>=2.9 defaults torch.onnx.export to the
    # dynamo/torch.export path, which is stricter and fails to trace some pix2pix-turbo UNets (e.g.
    # sketch_to_image_stochastic -> "unsupported operand -: int and NoneType"). The TorchScript path
    # handles them and is what the edge_to_image bundle was built with.
    torch.onnx.export(mod, args_tuple, path, input_names=names_in, output_names=names_out,
                      dynamic_axes=dynamic, opset_version=17, do_constant_folding=True, dynamo=False)
    print(f"  wrote {path}")


def export_onnx(model, out_dir, res):
    """Trace the 3 skip-VAE graphs. model = a set_eval'd, LoRA-fused Pix2Pix_Turbo."""
    os.makedirs(out_dir, exist_ok=True)
    dev = "cuda"
    R, h = res, res // 8
    dyn_hw = {0: "B", 2: "H", 3: "W"}
    with torch.no_grad():
        _onnx_export(UNetExport(model.unet).eval().to(dev),
                     (torch.randn(1, 4, h, h, device=dev), torch.randn(1, 77, 1024, device=dev)),
                     ["latent", "ehs"], ["model_pred"], os.path.join(out_dir, "unet.onnx"),
                     dynamic={"latent": dyn_hw, "model_pred": dyn_hw})
        _onnx_export(EncoderExport(model.vae).eval().to(dev),
                     (torch.randn(1, 3, R, R, device=dev),),
                     ["image"], ["latent", "enc0", "enc1", "enc2", "enc3"],
                     os.path.join(out_dir, "vae_encoder.onnx"),
                     dynamic={"image": dyn_hw, "latent": dyn_hw,
                              "enc0": dyn_hw, "enc1": dyn_hw, "enc2": dyn_hw, "enc3": dyn_hw})
        _onnx_export(DecoderExport(model.vae).eval().to(dev),
                     (torch.randn(1, 4, h, h, device=dev),
                      torch.randn(1, 512, h, h, device=dev),         # s0 enc3
                      torch.randn(1, 256, h * 2, h * 2, device=dev), # s1 enc2
                      torch.randn(1, 128, h * 4, h * 4, device=dev), # s2 enc1
                      torch.randn(1, 128, R, R, device=dev)),        # s3 enc0
                     ["latent_scaled", "s0", "s1", "s2", "s3"], ["image"],
                     os.path.join(out_dir, "vae_decoder.onnx"))
    print(f"  ONNX written to {out_dir}/")


def build_engines(onnx_dir, out_dir, res, hw_compat="none", builder_optimization_level=5):
    """Build the 3 static engines from ONNX. TRT-11 strongly-typed: precision follows ONNX dtypes."""
    os.makedirs(out_dir, exist_ok=True)
    h = res // 8
    LOG = trt.Logger(trt.Logger.WARNING)
    # name -> {input_name: shape} static (min==opt==max)
    SHAPES = {
        "unet": {"latent": (1, 4, h, h), "ehs": (1, 77, 1024)},
        "vae_encoder": {"image": (1, 3, res, res)},
        "vae_decoder": {"latent_scaled": (1, 4, h, h), "s0": (1, 512, h, h),
                        "s1": (1, 256, 2 * h, 2 * h), "s2": (1, 128, 4 * h, 4 * h),
                        "s3": (1, 128, res, res)},
    }
    HW = {"none": None,
          "ampere_plus": getattr(trt.HardwareCompatibilityLevel, "AMPERE_PLUS", None),
          "same_cc": getattr(trt.HardwareCompatibilityLevel, "SAME_COMPUTE_CAPABILITY", None)}

    def build(name):
        builder = trt.Builder(LOG)
        net = builder.create_network(0)  # TRT 10/11: explicit batch is default; EXPLICIT_BATCH removed
        parser = trt.OnnxParser(net, LOG)
        # parse_from_file resolves external weight data (.onnx.data) relative to the onnx dir
        if not parser.parse_from_file(os.path.join(onnx_dir, f"{name}.onnx")):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError(f"parse failed {name}")
        cfg = builder.create_builder_config()
        cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 12 << 30)
        cfg.builder_optimization_level = builder_optimization_level
        if HW.get(hw_compat) is not None:
            cfg.hardware_compatibility_level = HW[hw_compat]
        prof = builder.create_optimization_profile()
        for inp, shp in SHAPES[name].items():
            prof.set_shape(inp, shp, shp, shp)
        cfg.add_optimization_profile(prof)
        eng = builder.build_serialized_network(net, cfg)
        if eng is None:
            raise RuntimeError(f"build failed {name}")
        with open(os.path.join(out_dir, f"{name}.engine"), "wb") as f:
            f.write(eng)
        print(f"  built {out_dir}/{name}.engine ({eng.nbytes // (1024 * 1024)} MB)")

    for n in ["unet", "vae_encoder", "vae_decoder"]:
        build(n)


def load_model(pretrained_name, pretrained_path, ckpt_folder):
    m = Pix2Pix_Turbo(pretrained_name=pretrained_name, pretrained_path=pretrained_path,
                      ckpt_folder=ckpt_folder)
    m.set_eval()
    fuse_loras(m.unet, m.vae)
    # Stochastic variants (e.g. sketch_to_image_stochastic) replace unet.conv_in with a TwinConv that
    # blends a pretrained and a trained conv by a runtime ratio `r` (default None -> crashes export).
    # Bake r=1.0: at r=1 the model collapses to the deterministic skip-VAE flow our pipeline implements
    # (unet_input = encoded_control*r + noise_map*(1-r) = encoded_control; TwinConv -> the trained conv;
    # LoRA already fused at 1.0; VAE skip gamma baked at 1.0). Harmless no-op for non-TwinConv models.
    if hasattr(m.unet, "conv_in") and hasattr(m.unet.conv_in, "r"):
        m.unet.conv_in.r = 1.0
    m.unet.eval(); m.vae.eval()
    return m
