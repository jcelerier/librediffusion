"""Fix the bf16 klein transformer ONNX so the joint-attention sequence axis stays dynamic.

ROOT CAUSE: torch.onnx tracing baked the concatenated sequence length (Lp+Lt = 720+512 = 1232) as a
LITERAL inline Constant in the RoPE reshape target [1, 1232, 24, -1, 2] of every attention block
(50 nodes: 5 double blocks x2 + 20 single blocks x2). The Reshape data input has the dynamic sequence
on axis 1, so TRT solves 1232 -> pins Lp = 720 static, freezing the hidden_states input. The ONNX *IO*
stays symbolic ('Lp'), so the parser reports (-1,-1,128), but the built engine comes out static-720.

FIX: rewrite each inline 1232 -> 0. With Reshape allowzero=0 (default, confirmed on all these nodes),
a 0 in the target shape means "copy this dim from the input tensor" = the dynamic sequence length.
This changes ONLY a handful of 8-byte inline shape constants; NO weights, NO compute, NO precision
change. At Lp=720 the reshape is numerically identical (input axis-1 IS 1232). External weight files
are NOT loaded, NOT modified, and are referenced from the new model.onnx via hardlink.

This is the transformer analogue of fix_qwen_complex.py and MUST run after export_klein.py and before
build_klein_engines.py (which now builds the transformer from transformer_dynseq/model.onnx).
Path overrides (env KLEIN_ONNX_DIR) for train-lora --type klein; standalone use unchanged.
"""
import os as _os
import shutil
from pathlib import Path
import onnx
from onnx import numpy_helper
import numpy as np

import sys as _sys
_BASE = Path(_os.environ.get("KLEIN_ONNX_DIR", "./onnx-klein"))
# Optional argv: <src_subdir> <dst_subdir> so the same fix applies to the fp8-calib ONNX too.
# Defaults reproduce the bf16 transformer fix (transformer -> transformer_dynseq).
_SRC_SUB = _sys.argv[1] if len(_sys.argv) > 1 else "transformer"
_DST_SUB = _sys.argv[2] if len(_sys.argv) > 2 else "transformer_dynseq"
SRC_DIR = _BASE / _SRC_SUB
DST_DIR = _BASE / _DST_SUB
SRC = SRC_DIR / "model.onnx"
DST = DST_DIR / "model.onnx"

# The literal baked into the RoPE reshapes = Lp + Lt. Defaults: Lp=720 (320x576), Lt=512.
BAKED = int(_os.environ.get("KLEIN_BAKED_SEQ", "1232"))
EXPECT = int(_os.environ.get("KLEIN_BAKED_COUNT", "50"))

DST_DIR.mkdir(parents=True, exist_ok=True)
# hardlink the external weight files so the new model.onnx resolves them (basename refs, same dir).
for f in SRC_DIR.iterdir():
    if f.name == "model.onnx":
        continue
    dst = DST_DIR / f.name
    if dst.exists():
        dst.unlink()
    try:
        dst.hardlink_to(f)
    except Exception:
        shutil.copy2(f, dst)

print(f"loading {SRC} WITHOUT external data (only inline tensors materialized)...")
m = onnx.load(str(SRC), load_external_data=False)
g = m.graph

n_fixed = 0
for node in g.node:
    if node.op_type != "Constant":
        continue
    for attr in node.attribute:
        if attr.name != "value":
            continue
        t = attr.t
        if t.data_location != onnx.TensorProto.DEFAULT:
            continue  # never touch external tensors
        try:
            arr = numpy_helper.to_array(t)
        except Exception:
            continue
        if arr.dtype.kind in "iu" and arr.size == 1 and int(arr.flatten()[0]) == BAKED:
            new = np.zeros_like(arr)  # 0 => Reshape copies input dim (allowzero=0)
            nt = numpy_helper.from_array(new, name=t.name)
            t.CopyFrom(nt)
            n_fixed += 1

print(f"rewrote {n_fixed} inline Constant({BAKED}) -> 0")
# n_fixed==0 is OK: the modelopt fp8-calib export is already fully dynamic (no baked 1232) -> passthrough
# (the saved model == input, which build_one_fp8 builds into a dynamic [720..1440] engine). The bf16
# legacy export bakes exactly EXPECT(=50). Any OTHER count means Lp/Lt changed -> fail loudly.
assert n_fixed in (0, EXPECT), \
    f"expected 0 (already-dynamic) or {EXPECT} baked Constant({BAKED}), got {n_fixed} (Lp/Lt changed? set KLEIN_BAKED_SEQ/COUNT)"
if n_fixed == 0:
    print(f"  (0 baked literals -> ONNX already dynamic; passthrough copy to {DST_DIR.name})")

# Sanity: every reshape is allowzero=0 (0 means copy-dim, not literal-zero)
bad = sum(1 for node in g.node if node.op_type == "Reshape"
          and next((a.i for a in node.attribute if a.name == "allowzero"), 0) != 0)
print(f"Reshape nodes with allowzero!=0: {bad} (must be 0)")
assert bad == 0

print(f"saving -> {DST} (model proto only; external weights referenced via hardlink)...")
onnx.save_model(m, str(DST), save_as_external_data=False)
print("checking model (no external data load)...")
onnx.checker.check_model(str(DST), full_check=False)
print("DONE")
