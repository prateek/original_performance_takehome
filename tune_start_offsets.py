#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from dataclasses import dataclass

import perf_takehome


def n_nodes_for_height(forest_height: int) -> int:
    # Tree.generate(height) creates a full binary tree with 2^(h+1)-1 nodes.
    return (1 << (forest_height + 1)) - 1


def kernel_cycles(
    forest_height: int, n_nodes: int, batch_size: int, rounds: int, start_offsets: list[int] | None
) -> int:
    perf_takehome.START_OFFSETS = start_offsets
    kb = perf_takehome.KernelBuilder()
    kb.build_kernel(forest_height, n_nodes, batch_size, rounds)
    # Cycle count is the number of non-debug instruction bundles (submission harness
    # runs to completion with enable_debug=False).
    return sum(1 for instr in kb.instrs if set(instr.keys()) != {"debug"})


@dataclass
class SearchResult:
    cycles: int
    offsets: list[int] | None


def fmt_offsets(offsets: list[int] | None) -> str:
    if offsets is None:
        return "None"
    inner = ",\n    ".join(str(x) for x in offsets)
    return f"[\n    {inner},\n]"


def main() -> int:
    parser = argparse.ArgumentParser(description="Tune perf_takehome.START_OFFSETS via local search")
    parser.add_argument("--forest-height", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--iters", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-delta", type=int, default=32)
    parser.add_argument("--multi", type=int, default=2, help="Mutations per candidate")
    args = parser.parse_args()

    if args.batch_size % perf_takehome.VLEN != 0:
        raise SystemExit(
            f"{args.batch_size=} must be a multiple of {perf_takehome.VLEN=}"
        )

    n_nodes = n_nodes_for_height(args.forest_height)
    base = None if perf_takehome.START_OFFSETS is None else list(perf_takehome.START_OFFSETS)
    best = SearchResult(
        cycles=kernel_cycles(args.forest_height, n_nodes, args.batch_size, args.rounds, base),
        offsets=base,
    )
    print(f"baseline cycles={best.cycles}")

    random.seed(args.seed)
    if best.offsets is None:
        raise SystemExit("START_OFFSETS is None; set it to a list before tuning")

    for it in range(args.iters):
        cand = best.offsets.copy()
        for _ in range(args.multi):
            gi = random.randrange(len(cand))
            delta = random.randint(-args.max_delta, args.max_delta)
            cand[gi] = max(0, cand[gi] + delta)
        c = kernel_cycles(args.forest_height, n_nodes, args.batch_size, args.rounds, cand)
        if c < best.cycles:
            best = SearchResult(cycles=c, offsets=cand)
            print(f"improved cycles={best.cycles} at iter={it}")

    print(f"best cycles={best.cycles}")
    print("Suggested START_OFFSETS:")
    print(fmt_offsets(best.offsets))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
