#[cfg(not(feature = "cuda"))]
fn main() {
    println!("cargo:rerun-if-changed=build.rs");
}

#[cfg(feature = "cuda")]
fn main() {
    build::run();
}

#[cfg(feature = "cuda")]
mod build {
    use cudaforge::{CudaToolkit, KernelBuilder};
    use std::env;
    use std::path::PathBuf;

    /// One SASS target, or a PTX-only fallback target (embedded IR, no
    /// machine code, JIT-compiled by the driver on unlisted GPUs).
    #[derive(Clone, Copy)]
    enum ArchEntry {
        Sass(u32),
        Ptx(u32),
    }

    // Shared decision (docs/plan/README.md): one fat binary per CUDA major,
    // not one build per SM. No 'a'/'f' suffix on any compiled arch - see
    // resolve_archs's comment on why we build -gencode strings ourselves
    // instead of going through cudaforge's GpuArch::auto_suffix.
    const CUDA12_DEFAULT: &[ArchEntry] = &[
        ArchEntry::Sass(75),
        ArchEntry::Sass(80),
        ArchEntry::Sass(86),
        ArchEntry::Sass(89),
        ArchEntry::Sass(90),
        ArchEntry::Ptx(90),
    ];
    const CUDA13_DEFAULT: &[ArchEntry] = &[
        ArchEntry::Sass(75),
        ArchEntry::Sass(80),
        ArchEntry::Sass(86),
        ArchEntry::Sass(89),
        ArchEntry::Sass(90),
        ArchEntry::Sass(100),
        ArchEntry::Sass(120),
        ArchEntry::Ptx(120),
    ];

    pub fn run() {
        println!("cargo:rerun-if-changed=cuda");
        println!("cargo:rerun-if-env-changed=CUJSON_CUDA_ARCHS");
        println!("cargo:rerun-if-env-changed=NVCC");
        println!("cargo:rerun-if-env-changed=CUDA_HOME");

        let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set by cargo"));

        let toolkit = CudaToolkit::detect().unwrap_or_else(|e| {
            panic!(
                "cujson-sys's `cuda` feature needs a CUDA toolkit with nvcc. \
                 Set the NVCC environment variable to the nvcc binary, or \
                 CUDA_HOME to the toolkit root (so $CUDA_HOME/bin/nvcc exists). \
                 Detection error: {e}"
            )
        });

        let archs = resolve_archs(&toolkit);
        let archs_display = archs_display_string(&archs);
        println!("cargo:archs={archs_display}");

        let mut gencode_args: Vec<String> = archs
            .iter()
            .map(|a| match a {
                // No 'a'/'f' suffix (task 04 brief): cudaforge's GpuArch
                // auto-suffixes any numeric cap >= 90 (e.g. 90 -> sm_90a),
                // and sm_90a code doesn't run on later GPUs. Building the
                // -gencode string ourselves keeps every compiled arch
                // plain (sm_90, sm_100, sm_120, not *_a).
                ArchEntry::Sass(n) => format!("-gencode=arch=compute_{n},code=sm_{n}"),
                ArchEntry::Ptx(n) => format!("-gencode=arch=compute_{n},code=compute_{n}"),
            })
            .collect();
        gencode_args.push(format!("-DCUJSON_COMPILED_ARCHS=\"{archs_display}\""));
        for extra in ["-std=c++17", "-O3", "-w", "-Xcompiler", "-fPIC"] {
            gencode_args.push(extra.to_string());
        }

        let lib_path = out_dir.join("libcujson.a");
        KernelBuilder::new()
            .source_files([
                "cuda/capi_standard.cu",
                "cuda/capi_lines.cu",
                "cuda/capi_common.cu",
            ])
            .watch(["cuda"])
            // Forced low base purely so cudaforge's own per-file default
            // -gencode (added automatically, one per file, see builder.rs)
            // is always valid and never runs nvidia-smi (no GPU in this
            // container - and CI/most build machines building this crate
            // won't have one attached either). The full arch list above is
            // what actually ends up in the fat binary; this one is
            // redundant with (or a superset-neutral addition to) it.
            .compute_cap_arch("75")
            .args(&gencode_args)
            .build_lib(&lib_path)
            .expect("cuJSON CUDA kernel build failed");

        let cuda_root = toolkit
            .nvcc_path
            .parent()
            .and_then(|p| p.parent())
            .expect("nvcc path has no toolkit root")
            .to_path_buf();
        for candidate in ["lib64", "lib", "targets/x86_64-linux/lib"] {
            let dir = cuda_root.join(candidate);
            if dir.is_dir() {
                println!("cargo:rustc-link-search=native={}", dir.display());
            }
        }

        println!("cargo:rustc-link-search=native={}", out_dir.display());
        println!("cargo:rustc-link-lib=static=cujson");
        println!("cargo:rustc-link-lib=static=cudart_static");
        println!("cargo:rustc-link-lib=dylib=stdc++");
        println!("cargo:rustc-link-lib=dylib=rt");
        println!("cargo:rustc-link-lib=dylib=dl");
        println!("cargo:rustc-link-lib=dylib=pthread");
    }

    /// CUJSON_CUDA_ARCHS overrides the default list; format is a comma list
    /// of plain numbers (SASS) and/or "ptxNN" entries (PTX-only), e.g.
    /// "89" or "80,90,ptx90". Otherwise picks CUDA12_DEFAULT/CUDA13_DEFAULT
    /// from the detected toolkit's major version.
    fn resolve_archs(toolkit: &CudaToolkit) -> Vec<ArchEntry> {
        // An empty value (e.g. `CUJSON_CUDA_ARCHS: ""` in a CI matrix) means unset.
        if let Some(raw) = env::var("CUJSON_CUDA_ARCHS")
            .ok()
            .filter(|s| !s.trim().is_empty())
        {
            let mut archs = Vec::new();
            for tok in raw.split(',') {
                let tok = tok.trim();
                if tok.is_empty() {
                    continue;
                }
                if let Some(rest) = tok.strip_prefix("ptx") {
                    let n: u32 = rest
                        .parse()
                        .unwrap_or_else(|_| panic!("CUJSON_CUDA_ARCHS: bad ptx entry '{tok}'"));
                    archs.push(ArchEntry::Ptx(n));
                } else {
                    let n: u32 = tok
                        .parse()
                        .unwrap_or_else(|_| panic!("CUJSON_CUDA_ARCHS: bad arch '{tok}'"));
                    archs.push(ArchEntry::Sass(n));
                }
            }
            assert!(
                !archs.is_empty(),
                "CUJSON_CUDA_ARCHS is set but empty after parsing"
            );
            return archs;
        }

        let major = toolkit
            .version
            .as_deref()
            .and_then(|v| v.split('.').next())
            .and_then(|s| s.parse::<u32>().ok());
        match major {
            Some(13) => CUDA13_DEFAULT.to_vec(),
            // Default to the CUDA 12 list for 12.x and for anything we
            // couldn't parse a version out of - nvcc --version's "release
            // X.Y" line is what CudaToolkit::detect() parses, and every
            // toolkit this crate targets prints it.
            _ => CUDA12_DEFAULT.to_vec(),
        }
    }

    fn archs_display_string(archs: &[ArchEntry]) -> String {
        let sass: Vec<String> = archs
            .iter()
            .filter_map(|a| match a {
                ArchEntry::Sass(n) => Some(n.to_string()),
                ArchEntry::Ptx(_) => None,
            })
            .collect();
        let ptx: Vec<String> = archs
            .iter()
            .filter_map(|a| match a {
                ArchEntry::Ptx(n) => Some(format!("ptx{n}")),
                ArchEntry::Sass(_) => None,
            })
            .collect();
        if ptx.is_empty() {
            sass.join(",")
        } else {
            format!("{};{}", sass.join(","), ptx.join(","))
        }
    }
}
