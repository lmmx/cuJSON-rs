# cujson

Parse JSON and JSON Lines on an NVIDIA GPU from Python, using the
[cuJSON](https://github.com/AutomataLab/cuJSON) parser (ASPLOS '26) packaged by
[cuJSON-rs](https://github.com/lmmx/cuJSON-rs).

```
pip install cujson
```

## Requirements

- Linux x86_64, Python 3.10 or newer
- An NVIDIA GPU of compute capability 7.5 or newer (Turing onwards)
- An NVIDIA driver that supports CUDA 13 (R580 or newer)

The wheel links the CUDA runtime statically, so the driver is the only CUDA component you need
to install. On a CUDA 12 system, build from a
[checkout](https://github.com/lmmx/cuJSON-rs) with `pip install ./crates/cujson-py`.

## Usage

```python
import cujson

doc = cujson.parse('{"user": {"name": "Ada", "langs": ["en", "fr"]}}')
doc.to_python()                  # {'user': {'name': 'Ada', 'langs': ['en', 'fr']}}
doc.pointer("/user/langs/1")     # 'fr'  (RFC 6901 JSON Pointer)

lines = cujson.parse_lines('{"id": 1}\n{"id": 2}\n')
lines.to_python()                # [{'id': 1}, {'id': 2}]
lines.pointer("/1/id")           # 2  (the first token selects the line)
len(lines)                       # 2

cujson.parse_file("data.json")   # read and parse a file
cujson.cuda_info()               # runtime, driver, compiled architectures, devices
cujson.trim_pinned_cache()       # release the pinned buffer kept between parses
```

`parse` and `parse_lines` accept `str`, `bytes`, `bytearray` or `memoryview`. The GIL is
released while the GPU parses.

## Errors

All exceptions derive from `cujson.CujsonError`:

| Exception | Also a | Raised when |
|---|---|---|
| `InvalidUtf8Error` | `ValueError` | the input is not valid UTF-8 |
| `UnbalancedError` | `ValueError` | brackets or braces do not match |
| `InputError` | `ValueError` | the input is empty, too large, or (for `parse`) not exactly one JSON value |
| `CudaError` | `RuntimeError` | CUDA is unusable, e.g. no driver or a driver too old for CUDA 13 |
| `NoDeviceError` | `CudaError` | the driver reports no visible GPU |

## Limitations

- cuJSON checks UTF-8 validity and bracket balance, not the full JSON grammar: malformed scalars
  such as `1.2.3` are not rejected.
- Inputs are limited to just under 2 GiB (cuJSON uses 32-bit sizes).
- Parses are serialised within a process: cuJSON uses the default CUDA stream.
- After a parse, dropping its `Document` keeps at most one tape-sized pinned host buffer (roughly
  a third of the input size) allocated so the next parse can reuse it. It is released by
  `cujson.trim_pinned_cache()` or when the process exits.

## Credits

cuJSON is by Ashkan Vedadi Gargary, Soroosh Safari Loaliyan and Zhijia Zhao (MIT licensed). If
you use it in research, cite
[CuJSON: A Highly Parallel JSON Parser for GPUs](https://doi.org/10.1145/3760250.3762222)
(ASPLOS '26). cuJSON-rs is MIT licensed.
