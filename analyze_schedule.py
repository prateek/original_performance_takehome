#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter

import perf_takehome


def n_nodes_for_height(forest_height: int) -> int:
    # Tree.generate(height) creates a full binary tree with 2^(h+1)-1 nodes.
    return (1 << (forest_height + 1)) - 1


def _non_debug_instrs(kb: perf_takehome.KernelBuilder) -> list[dict[str, list[tuple]]]:
    out: list[dict[str, list[tuple]]] = []
    for instr in kb.instrs:
        if set(instr.keys()) == {"debug"}:
            continue
        out.append(instr)
    return out


def idle_runs(counts: list[int]) -> list[tuple[int, int, int]]:
    runs: list[tuple[int, int, int]] = []
    run_start: int | None = None
    for i, c in enumerate(counts):
        if c == 0:
            if run_start is None:
                run_start = i
            continue
        if run_start is not None:
            runs.append((run_start, i - 1, i - run_start))
            run_start = None
    if run_start is not None:
        runs.append((run_start, len(counts) - 1, len(counts) - run_start))
    runs.sort(key=lambda r: r[2], reverse=True)
    return runs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Static schedule analysis for perf_takehome.KernelBuilder.build_kernel"
    )
    parser.add_argument("--forest-height", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--top-idle",
        type=int,
        default=10,
        help="Number of longest idle runs to print per engine",
    )
    args = parser.parse_args()

    n_nodes = n_nodes_for_height(args.forest_height)
    kb = perf_takehome.KernelBuilder()
    kb.build_kernel(args.forest_height, n_nodes, args.batch_size, args.rounds)

    instrs = _non_debug_instrs(kb)
    cycles = len(instrs)
    print(
        f"{args.forest_height=}, {args.rounds=}, {args.batch_size=}, {n_nodes=}, {cycles=}"
    )

    engines = [e for e in ("load", "store", "flow", "valu", "alu") if e in perf_takehome.SLOT_LIMITS]
    for eng in engines:
        slots_per_cycle = [len(instr.get(eng, ())) for instr in instrs]
        used_cycles = sum(1 for c in slots_per_cycle if c)
        total_slots = sum(slots_per_cycle)
        dist = Counter(slots_per_cycle)
        max_per_cycle = perf_takehome.SLOT_LIMITS[eng]
        unused = sum((max_per_cycle - c) for c in slots_per_cycle)

        print(
            f"- {eng}: used_cycles={used_cycles}/{cycles} total_slots={total_slots} "
            f"avg={total_slots / cycles:.2f}/{max_per_cycle} unused_slots={unused}"
        )
        print(f"  slots_dist={dict(sorted(dist.items()))}")

        runs = idle_runs(slots_per_cycle)
        if runs:
            top = runs[: args.top_idle]
            print(f"  idle_runs_top{len(top)}={top}")
        else:
            print("  idle_runs_top0=[]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

