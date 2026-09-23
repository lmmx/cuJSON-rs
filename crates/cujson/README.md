# cujson

Parse JSON and JSON Lines on an NVIDIA GPU from Rust, using the
[cuJSON](https://github.com/AutomataLab/cuJSON) parser (ASPLOS '26). Part of
[cuJSON-rs](https://github.com/lmmx/cuJSON-rs), which also ships a CLI and a Python package
(`pip install cujson`).

```toml
[dependencies]
cujson = { version = "0.1", features = ["cuda"] }
```

## Requirements

Building with the `cuda` feature needs the CUDA toolkit (12.1 or newer): `nvcc` on `PATH`, or
`CUDA_HOME` set. The kernels are compiled in `cujson-sys`'s build script into a fat binary for
compute capability 7.5 onwards plus a PTX fallback, and the CUDA runtime is linked statically,
so the resulting binary needs only the NVIDIA driver. Set `CUJSON_CUDA_ARCHS=86` (for example)
to build for a single GPU generation.

Without the `cuda` feature the crate still compiles (for docs.rs, or dependents that make the GPU
optional), and every parse returns `Error::CudaNotCompiled`.

## Usage

```rust,no_run
fn main() -> Result<(), cujson::Error> {
    let doc = cujson::parse(br#"{"user": {"name": "Ada", "langs": ["en", "fr"]}}"#)?;
    let lang = doc.pointer("/user/langs/1").and_then(|n| n.as_str().ok());
    assert_eq!(lang.as_deref(), Some("fr"));

    let lines = cujson::parse_lines(b"{\"id\": 1}\n{\"id\": 2}\n", Default::default())?;
    for line in lines.lines() {
        println!("{:?}", line.get("id").and_then(|n| n.as_i64().ok()));
    }
    Ok(())
}
```

`Node` also has `kind`, `index`, `iter_array`, `iter_object`, `raw`, `as_f64`, `as_bool` and
`is_null`; with the `serde` feature, `to_value` converts to a `serde_json::Value`.

## Features

| Feature | Effect |
|---|---|
| `cuda` | Compile the GPU kernels (needs `nvcc`) |
| `serde` | `Node::to_value` into `serde_json::Value` |
| `cpu-reference` | A CPU implementation of the same output format (`cujson::cpu::parse`), for testing |

## Limitations

- cuJSON checks UTF-8 validity and bracket balance, not the full JSON grammar: malformed scalars
  such as `1.2.3` are not rejected. `parse` does reject input that is not exactly one value.
- Inputs are limited to just under 2 GiB (cuJSON uses 32-bit sizes).
- Parses are serialised within a process: cuJSON uses the default CUDA stream.

## Credits

cuJSON is by Ashkan Vedadi Gargary, Soroosh Safari Loaliyan and Zhijia Zhao (MIT licensed); cite
[CuJSON: A Highly Parallel JSON Parser for GPUs](https://doi.org/10.1145/3760250.3762222)
(ASPLOS '26) if you use it in research. cuJSON-rs is MIT licensed.
