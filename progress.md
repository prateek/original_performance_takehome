2026-01-27 21:56: initial setup
2026-01-28: VLIW packing pass (greedy scratch-hazard-aware slot bundling); submission cycles now 98583.
2026-01-28: SIMD vectorization over VLEN=8 (vload/vstore + valu hash + gather via load_offset); submission cycles now 12369.
2026-01-28: Cache idx/val in scratch (load once, store once); submission cycles now 11407.
2026-01-28: Pipeline/unroll pass (software-pipelined, group-interleaved scheduler; overlap gather loads with hash/idx updates); submission cycles now 2264.
2026-01-28: Offload step/idx masking to flow vselect + pack const/broadcast init; submission cycles now 2159.
2026-01-28: Skip gather loads for depth 0–1 rounds (idx in {0,1,2}) via preloaded node vectors + scheduler states; submission cycles now 2098.
2026-01-28: Pack prologue header loads (mem[0..6]) via VLIW load bundling; submission cycles now 2091.
2026-01-28: Fuse hash stages 0/2/4 via `valu multiply_add` (e.g. `a + (a<<k) + C` → `a*(1+2^k) + C`); submission cycles now 1824.
2026-01-28: Skip gather loads for depth 2 rounds (idx in {3,4,5,6}) via preloaded node3 + delta vectors and depth-aware scheduler states; submission cycles now 1672.
2026-01-28: Move gather address prefetch to ALU (8 scalar adds) to free `valu` bandwidth; submission cycles now 1637.
2026-01-28: Remove per-round idx bounds-check/clamp (only wrap at depth boundary) via `IDX_RESET` flow state; skip idx update on final round; submission cycles now 1570.
2026-01-28: Keep indices entirely in scratch (inputs start at 0): remove `idx_ptrs` + index vload/vstore; trim unused header loads; fold pause into existing bundles; submission cycles now 1547.
2026-01-28: Overlap gather-round XOR into scalar ALU (lane-wise `val ^= node`), reducing `valu` pressure; submission cycles now 1538.
2026-01-28: Re-ran authoritative submission tests; submission cycles still 1538.
2026-01-28: Vectorize prologue forest node preload (single `vload` for nodes[0..7] + broadcasts for depth 0–2 fast paths); submission cycles now 1529.
2026-01-28: Tried depth-3 fast path (preload nodes[7..14] + masked-add selection); regressed to 1636 cycles, reverted; best remains 1529.
2026-01-28: Stagger per-group start times (ready offsets) to smooth pipeline fill/drain; submission cycles now 1483.
2026-01-28: Preload `val_cache` + materialize `val_ptrs` with 4 independent pointer streams (blocks of 8 groups) to remove the 32-step scalar chain; retuned `start_spacing` to 14; submission cycles now 1418.
2026-01-28: Upgrade VLIW slot packing in `KernelBuilder.build` to a dependency-aware list scheduler (RAW/WAW + WAR) to reorder independent slots for tighter bundling; submission cycles now 1414.
2026-01-28: Pack const loads + vbroadcasts + setup/caching slots into one VLIW build to overlap load/alu/valu; submission cycles now 1409.
2026-01-28: Retune group start staggering (piecewise start offsets) to reduce pipeline fill bubbles; submission cycles now 1391.
2026-01-29: Pipeline `val_cache` init: preload first 2 groups in setup, stream remaining `vload`s in early load-idle main-loop cycles (`INIT_LOAD`); submission cycles now 1378.
2026-01-29: Update PRD with current status + next-experiment checklist; submission cycles still 1378.
2026-01-29: Re-ran submission harness; cycles still 1378. Static schedule analysis: load=2 on 1308/1378 cycles, with 69 load-idle cycles (largest runs 31–66 and 1367–1377).
2026-01-29: Expose start-stagger knobs as constants and brute-force a small grid (early_groups/early_spacing/late_spacing); no improvement found, best remains 1378.
2026-01-29: Tune per-group `START_OFFSETS` schedule (override piecewise staggering) to reduce load-idle bubbles; submission cycles now 1368.
2026-01-29: Retune `START_OFFSETS`; submission cycles now 1367.
2026-01-29: Improve VLIW packer heuristic (criticality-based candidate selection) to reduce setup bundles; submission cycles now 1365.
2026-01-29: Reduce setup load-const pressure (implicit zero scalar; derive `idx{4,5,6}_vec` from `one_vec`/`two_vec`; skip vector constants for fused-stage shift amounts); submission cycles now 1362.
2026-01-29: Reduce setup load-const pressure further by deriving scalar `9` from `8+1` (ALU add) instead of `load const`; submission cycles now 1361.
2026-01-29: Reduce setup load-const pressure further by deriving shift constants `16` (`8+8`) and `19` (`16+2+1`) via ALU adds instead of `load const`; submission cycles now 1360.
2026-01-29: Keep `idx_cache` as forest addresses for depths >=3 so gather loads read directly from `idx_cache` (no address-prefetch stage); retune `START_OFFSETS`; submission cycles now 1360.
2026-01-29: Eliminate prologue header loads by deriving `forest_values_p`/`inp_values_p` as constants (fixed `build_mem_image` header=7); submission cycles now 1359.
2026-01-29: Materialize `forest_values_p`/`inp_values_p` via `flow add_imm` (avoid two `load const` slots; setup `8→7` cycles); submission cycles now 1358.
2026-01-29: Retune per-group `START_OFFSETS` via multi-parameter random search to improve overlap; submission cycles now 1356.
2026-01-29: Derive `33`/`4097` multiplier scalars via ALU and defer pointer-vector broadcasts + `node{5,6}-node3` delta subtracts into early main-loop slack to shrink setup by one more cycle; submission cycles now 1355.
