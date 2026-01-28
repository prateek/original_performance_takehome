2026-01-27 21:56: initial setup
2026-01-28: VLIW packing pass (greedy scratch-hazard-aware slot bundling); submission cycles now 98583.
2026-01-28: SIMD vectorization over VLEN=8 (vload/vstore + valu hash + gather via load_offset); submission cycles now 12369.
2026-01-28: Cache idx/val in scratch (load once, store once); submission cycles now 11407.
2026-01-28: Pipeline/unroll pass (software-pipelined, group-interleaved scheduler; overlap gather loads with hash/idx updates); submission cycles now 2264.
2026-01-28: Offload step/idx masking to flow vselect + pack const/broadcast init; submission cycles now 2159.
2026-01-28: Skip gather loads for depth 0–1 rounds (idx in {0,1,2}) via preloaded node vectors + scheduler states; submission cycles now 2098.
2026-01-28: Pack prologue header loads (mem[0..6]) via VLIW load bundling; submission cycles now 2091.
