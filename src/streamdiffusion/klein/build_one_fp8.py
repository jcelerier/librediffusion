"""Build one klein transformer FP8 engine from a given ONNX dir -> given engine name."""
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
cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 14 << 30)
# KLEIN_HW_COMPAT env (none|ampere_plus|same_cc): portable engines across GPU archs (~5-15% slower).
_hc = os.environ.get("KLEIN_HW_COMPAT", "none").lower()
if _hc in ("ampere_plus", "ampere"):
    cfg.hardware_compatibility_level = trt.HardwareCompatibilityLevel.AMPERE_PLUS
    print(f"[I] klein FP8 hw compat: {_hc} (PORTABLE)")
elif _hc in ("same_cc", "same"):
    cfg.hardware_compatibility_level = trt.HardwareCompatibilityLevel.SAME_COMPUTE_CAPABILITY
    print(f"[I] klein FP8 hw compat: {_hc} (PORTABLE)")
# Geometry: same env contract as export_klein.py / build_klein_engines.py.
_W = int(os.environ.get("KLEIN_WIDTH", "320"))
_H = int(os.environ.get("KLEIN_HEIGHT", "576"))
LT = int(os.environ.get("KLEIN_TEXT_LEN", "512"))
LP = (_H // 16) * (_W // 16)

prof = builder.create_optimization_profile()
profiles = {
    "hidden_states": ((1, LP, 128), (1, LP, 128), (1, 2 * LP, 128)),
    "encoder_hidden_states": ((1, LT, 7680), (1, LT, 7680), (1, LT, 7680)),
    "timestep": ((1,), (1,), (1,)),
    "img_ids": ((1, LP, 4), (1, LP, 4), (1, 2 * LP, 4)),
    "txt_ids": ((1, LT, 4), (1, LT, 4), (1, LT, 4)),
}
for n, (mn, op, mx) in profiles.items():
    prof.set_shape(n, mn, op, mx)
cfg.add_optimization_profile(prof)
print("building", eng_path, "...")
ser = builder.build_serialized_network(net, cfg)
assert ser is not None, "build returned None"
open(eng_path, "wb").write(ser)
print("WROTE", eng_path, f"{os.path.getsize(eng_path)/1e6:.0f} MB")
