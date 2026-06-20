### Monkey Patching Diffusers

* Build ONNX/TRT engines cleanly

* Catch invalid configs during tracing

* Avoid TracerWarnings where possible

TODO: Open upstream PR to DiffUsers

## 1. `unet_2d_condition.py:1109`

| file / line | original | why it warns |
| --- | --- | --- |
| `diffusers/models/unets/unet_2d_condition.py:1109` | `python\nif dim % default_overall_up_factor != 0:\n    raise ValueError(...)\n` | `dim` is a **dynamic SymInt** during tracing; `%` produces a SymInt → comparing with `!=` returns a **tensor-derived bool**. |

**Patch**

```python
python
CopyEdit
# safe for tracing
if (dim % default_overall_up_factor).item() != 0:
    raise ValueError(...)
# or (faster, no .item()):
if torch.remainder(dim, default_overall_up_factor) != 0:
    ...

```

---

## 2–3. `downsampling.py:136` and `:145`

```python
python
CopyEdit
assert hidden_states.shape[1] == self.channels

```

The left side is `SymInt`; the equality returns a tensor-bool.

**Patch**

```python
python
CopyEdit
torch._assert(
    hidden_states.shape[1] == self.channels,
    f"Expected channels={self.channels}, got {hidden_states.shape[1]}",
)

```

`torch._assert` is tracing-friendly (PyTorch ≥ 2.1).

---

## 4. `upsampling.py:147`

Same as above (use `torch._assert`).

---

## 5. `upsampling.py:162`

```python
python
CopyEdit
if hidden_states.shape[0] >= 64:
    ...

```

**Patch**

```python
python
CopyEdit
if hidden_states.shape[0].item() >= 64:
    ...

```

or

```python
python
CopyEdit
if torch.ge(hidden_states.shape[0], 64):
    ...

```

---

## 6. `upsampling.py:173`

```python
python
CopyEdit
if hidden_states.numel() * scale_factor > 2**31:

```

`hidden_states.numel()` returns a **tensor int** in graph mode.

**Patch**

```python
python
CopyEdit
numel = hidden_states.numel()
if (numel * scale_factor).item() > 2**31:
    ...

```

or replace with `torch.numel(hidden_states)`

---

## 7. `unet_2d_condition.py:1307`

```python
python
CopyEdit
if not return_dict:

```

`return_dict` is a real Python bool (comes from kwargs), **not** a tensor — the tracer warns only because earlier ops produced a tensor-bool inside the same block.  **No change needed**.