# 10 — GPU validation runbook

Written alongside 06 and updated by 07/08. **Executed by the user on a machine with an NVIDIA GPU**; results go back into a journal entry. This is the only tier-3 evidence the project has.

Output file: `docs/GPU_VALIDATION.md`, a copy-pasteable script of steps. Each step names the claim it confirms (link to the journal entry that marked the claim unverified).

## Steps the runbook must contain

1. Environment capture: `nvidia-smi`, `nvcc --version`, `rustc --version`, GPU compute capability
2. Fast local build: `CUJSON_CUDA_ARCHS=<cc> cargo build --release -p cujson-cli --features cuda`, then the full-arch build, recording both compile times
3. `cujson info`: devices listed, compiled archs include the GPU (or PTX fallback)
4. `cargo test --workspace --features cuda -- --include-ignored`
5. `cujson verify` on both fixtures (`--lines` for the small-records file). Expected: zero tape diffs and a serde_json match. **This confirms or refutes task 05's FORMAT.md**, so on mismatch capture the `cujson tape` diff output for the first failing index
6. Error-path recovery: invalid UTF-8 and unbalanced inputs return errors, then a valid parse succeeds, with device memory from `nvidia-smi --query-gpu=memory.used` stable across 1000 invalid parses
7. `cujson bench` on the fixtures and on a large file (the runbook gives a download command for a public multi-hundred-MB JSON dataset, e.g. one from cuJSON's paper_reproduced scripts; cite the script)
8. Python: `maturin develop --features cuda` in a venv, `CUJSON_TEST_GPU=1 pytest crates/cujson-py`
9. PTX fallback (only if a GPU newer than the SASS list is available): otherwise note as untested

## Reporting

The user pastes the outputs; the orchestrating agent writes `docs/journal/YYYY-MM-DD-gpu-validation.md` moving each claim from "unverified" to confirmed/refuted, and files fixes as new tasks. Following the one-change-at-a-time rule: each fix is one commit the user can re-test in isolation.
