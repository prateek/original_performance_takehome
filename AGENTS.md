# Agent Guide (Performance Take‑Home)

This repo is a performance‑optimization exercise for a toy VLIW + SIMD machine
simulator. Your job is to make the kernel run in as few simulated cycles as
possible **without changing the tests**.

## Goal

- Optimize `KernelBuilder.build_kernel` in `perf_takehome.py` for minimum cycle
  count (reported by `python tests/submission_tests.py`).

## Integrity / Non‑Goals

- **Do not change anything under `tests/`.** Any “solution” that modifies the
  harness is invalid.
- Don’t “cheat” by changing the simulator: the submission harness uses the
  frozen simulator in `tests/frozen_problem.py`, so edits to `problem.py` won’t
  affect your score.
- Keep `N_CORES = 1` (multicore is intentionally disabled in this version).

## Quick Commands

- Verify your submission (authoritative): `python tests/submission_tests.py`
- Local cycle test: `python perf_takehome.py Tests.test_kernel_cycles`
- Produce a trace: `python perf_takehome.py Tests.test_kernel_trace`
- Watch the trace (Chrome recommended): `python watch_trace.py`

Before submitting/sharing results:

- Ensure the test folder is untouched: `git diff origin/main tests/` (must be empty)

## Repo Map

- `perf_takehome.py`: starter kernel + where you implement optimizations
  (`KernelBuilder.build_kernel`). Includes local debug tests and trace helpers.
- `problem.py`: the “live” simulator + ISA reference used for local debugging.
- `tests/frozen_problem.py`: frozen simulator/reference used by submission tests.
- `tests/submission_tests.py`: correctness + cycle measurement used for scoring.
- `watch_trace.py`, `watch_trace.html`: trace viewer helper for `trace.json`.

## Notes for Implementing a Faster Kernel

- An instruction is a dict mapping engine name → list of slots, e.g.
  `{"alu": [...], "load": [...], "flow": [...]}`. One instruction bundle ≈ one
  cycle (if it contains any non‑debug work).
- Pack as much work per cycle as possible within `SLOT_LIMITS` (see `problem.py`
  / `tests/frozen_problem.py`).
- Prefer vector ops (`valu`, `vload`, `vstore`, `vbroadcast`) when applicable;
  `VLEN = 8`.
- Reuse constants and scratch locations (`KernelBuilder.scratch_const`,
  `KernelBuilder.alloc_scratch`). Scratch space is finite (`SCRATCH_SIZE = 1536`).
- The submission harness disables pause/debug (`enable_pause = False`,
  `enable_debug = False`), so debug slots may help locally but must not be
  required for correctness.

## Requirements

- Python 3.10+ (the simulator uses `match`/`case`).
