"""Rewrite spurious COMPLEX128 casts in the Qwen3 ONNX to FLOAT.

The legacy TorchScript ONNX exporter mis-types the attention-scaling `sqrt`->cast->mul chain as
COMPLEX128 (data_type 15). The underlying math is real; the cast is a no-op artifact. TRT rejects
COMPLEX128. We rewrite Cast(to=15)->Cast(to=1=FLOAT) and any complex constant tensors to float.
"""
import sys
import onnx
from onnx import TensorProto

import os as _os
# Path overrides (env) for train-lora --type klein; standalone use unchanged.
_BASE = _os.environ.get("KLEIN_ONNX_DIR", "./onnx-klein")
SRC = _os.path.join(_BASE, "qwen3_encoder", "model.onnx")
DST = _os.path.join(_BASE, "qwen3_encoder", "model_fixed.onnx")

COMPLEX128 = 15
COMPLEX64 = 14
FLOAT = TensorProto.FLOAT

m = onnx.load(SRC, load_external_data=True)
g = m.graph

n_cast = 0
for node in g.node:
    if node.op_type == "Cast":
        for a in node.attribute:
            if a.name == "to" and a.i in (COMPLEX128, COMPLEX64):
                a.i = TensorProto.BFLOAT16
                n_cast += 1

n_init = 0
for init in g.initializer:
    if init.data_type in (COMPLEX128, COMPLEX64):
        init.data_type = FLOAT
        n_init += 1

# also fix any value_info / io tensor types
def fix_vi(vis):
    c = 0
    for vi in vis:
        if vi.type.tensor_type.elem_type in (COMPLEX128, COMPLEX64):
            vi.type.tensor_type.elem_type = FLOAT
            c += 1
    return c

n_vi = fix_vi(g.value_info) + fix_vi(g.input) + fix_vi(g.output)

print(f"rewrote {n_cast} complex casts, {n_init} complex initializers, {n_vi} value_infos")
onnx.save(m, DST, save_as_external_data=True, all_tensors_to_one_file=True,
          location="model_fixed.onnx_data")
print("saved", DST)
