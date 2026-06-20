"""Build the RIFE (IFNet) frame-interpolation TRT engine from the vendored ONNX.

Generic single-input builder: frames[B,6,H,W] fp16 -> dynamic profile matching the deployed engine
(min 1x6x64x64, opt 2x6x512x512, max 7x6x1024x1024 — covers exp up to 3 at up to 1024px). Honors the
KLEIN_HW_COMPAT env (none|ampere_plus|same_cc) for portable cross-GPU engines, like the other klein builders.

Usage: python build_rife.py <onnx_path> <engine_path>
"""
import sys, os
import tensorrt as trt

onnx_path = sys.argv[1]
eng_path = sys.argv[2]
LOG = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(LOG)
flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
net = builder.create_network(flags)
parser = trt.OnnxParser(net, LOG)
ok = parser.parse(open(onnx_path, "rb").read(), path=onnx_path)
if not ok:
    for i in range(parser.num_errors):
        print("ERR", parser.get_error(i))
    raise SystemExit(1)

cfg = builder.create_builder_config()
cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)

# KLEIN_HW_COMPAT env (none|ampere_plus|same_cc): portable engines across GPU archs (~5-15% slower).
_hc = os.environ.get("KLEIN_HW_COMPAT", "none").lower()
if _hc in ("ampere_plus", "ampere"):
    cfg.hardware_compatibility_level = trt.HardwareCompatibilityLevel.AMPERE_PLUS
    print(f"[I] rife hw compat: {_hc} (PORTABLE)")
elif _hc in ("same_cc", "same"):
    cfg.hardware_compatibility_level = trt.HardwareCompatibilityLevel.SAME_COMPUTE_CAPABILITY
    print(f"[I] rife hw compat: {_hc} (PORTABLE)")

prof = builder.create_optimization_profile()
# single input 'frames' [B,6,H,W]; profile mirrors the deployed engine.
prof.set_shape("frames", (1, 6, 64, 64), (2, 6, 512, 512), (7, 6, 1024, 1024))
cfg.add_optimization_profile(prof)

print(f"building rife engine -> {eng_path} ...")
ser = builder.build_serialized_network(net, cfg)
assert ser is not None, "rife engine build returned None"
open(eng_path, "wb").write(ser)
print(f"WROTE {eng_path} ({os.path.getsize(eng_path)/1e6:.0f} MB)")
