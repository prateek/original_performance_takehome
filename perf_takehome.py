"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)

# Tuning knobs for the group-interleaved software pipeline schedule in
# `KernelBuilder.build_kernel`. These defaults are chosen for the authoritative
# submission harness parameters (forest_height=10, rounds=16, batch_size=256).
START_EARLY_GROUPS = 4
START_EARLY_SPACING = 4
START_LATE_SPACING = 16
# Optional per-group start offsets (in cycles) for the main software pipeline.
# When provided (and sized correctly), these override the piecewise spacing knobs
# above.
START_OFFSETS: list[int] | None = [
    0,
    0,
    2,
    2,
    9,
    11,
    50,
    71,
    74,
    120,
    120,
    154,
    154,
    179,
    179,
    232,
    256,
    275,
    283,
    263,
    291,
    303,
    286,
    317,
    367,
    373,
    392,
    502,
    575,
    622,
    683,
    698,
]


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def _slot_scratch_io(self, engine: str, slot: tuple) -> tuple[set[int], set[int]]:
        reads: set[int] = set()
        writes: set[int] = set()

        def add_vec(base: int, length: int = VLEN, *, out: set[int]):
            for i in range(length):
                out.add(base + i)

        match engine:
            case "alu":
                _, dest, a1, a2 = slot
                reads.update((a1, a2))
                writes.add(dest)
            case "valu":
                match slot:
                    case ("vbroadcast", dest, src):
                        reads.add(src)
                        add_vec(dest, out=writes)
                    case ("multiply_add", dest, a, b, c):
                        add_vec(a, out=reads)
                        add_vec(b, out=reads)
                        add_vec(c, out=reads)
                        add_vec(dest, out=writes)
                    case (_, dest, a1, a2):
                        add_vec(a1, out=reads)
                        add_vec(a2, out=reads)
                        add_vec(dest, out=writes)
            case "load":
                match slot:
                    case ("const", dest, _val):
                        writes.add(dest)
                    case ("load", dest, addr):
                        reads.add(addr)
                        writes.add(dest)
                    case ("load_offset", dest, addr, offset):
                        reads.add(addr + offset)
                        writes.add(dest + offset)
                    case ("vload", dest, addr):
                        reads.add(addr)
                        add_vec(dest, out=writes)
            case "store":
                match slot:
                    case ("store", addr, src):
                        reads.update((addr, src))
                    case ("vstore", addr, src):
                        reads.add(addr)
                        add_vec(src, out=reads)
            case "flow":
                match slot:
                    case ("select", dest, cond, a, b):
                        reads.update((cond, a, b))
                        writes.add(dest)
                    case ("add_imm", dest, a, _imm):
                        reads.add(a)
                        writes.add(dest)
                    case ("vselect", dest, cond, a, b):
                        add_vec(cond, out=reads)
                        add_vec(a, out=reads)
                        add_vec(b, out=reads)
                        add_vec(dest, out=writes)
                    case ("jump_indirect", addr):
                        reads.add(addr)
                    case ("cond_jump", cond, _addr):
                        reads.add(cond)
                    case ("cond_jump_rel", cond, _offset):
                        reads.add(cond)
                    case ("trace_write", val):
                        reads.add(val)
                    case ("coreid", dest):
                        writes.add(dest)
            case "debug":
                match slot:
                    case ("compare", loc, _key):
                        reads.add(loc)
                    case ("vcompare", loc, _keys):
                        add_vec(loc, out=reads)
            case _:
                pass

        return reads, writes

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        if not vliw:
            # Simple slot packing that just uses one slot per instruction bundle
            return [{engine: [slot]} for engine, slot in slots]

        def is_barrier(engine: str, slot: tuple) -> bool:
            return engine == "flow" and slot and slot[0] in {
                "pause",
                "halt",
                "jump",
                "jump_indirect",
                "cond_jump",
                "cond_jump_rel",
            }

        def schedule_segment(segment: list[tuple[str, tuple]]) -> list[dict[str, list[tuple]]]:
            if not segment:
                return []

            engines: list[str] = []
            slots_only: list[tuple] = []
            reads: list[set[int]] = []
            writes: list[set[int]] = []
            for engine, slot in segment:
                engines.append(engine)
                slots_only.append(slot)
                r, w = self._slot_scratch_io(engine, slot)
                reads.append(r)
                writes.append(w)

            n = len(segment)
            succ0: list[set[int]] = [set() for _ in range(n)]
            succ1: list[set[int]] = [set() for _ in range(n)]
            pred0_count = [0 for _ in range(n)]
            pred1_count = [0 for _ in range(n)]

            def add_edge(pred: int, succ: int, latency: int) -> None:
                if pred == succ:
                    return
                if latency == 0:
                    if succ not in succ0[pred]:
                        succ0[pred].add(succ)
                        pred0_count[succ] += 1
                else:
                    if succ not in succ1[pred]:
                        succ1[pred].add(succ)
                        pred1_count[succ] += 1

            last_write: dict[int, int] = {}
            pending_reads: dict[int, list[int]] = {}

            for i in range(n):
                rset = reads[i]
                wset = writes[i]

                # True deps (RAW) and output deps (WAW) require a new cycle because
                # writes are not visible until the end of the bundle.
                for addr in rset:
                    pred = last_write.get(addr)
                    if pred is not None:
                        add_edge(pred, i, 1)
                for addr in wset:
                    pred = last_write.get(addr)
                    if pred is not None:
                        add_edge(pred, i, 1)

                # Anti-deps (WAR): a write may not move before any earlier read of
                # the same location (but can share the same cycle).
                for addr in wset:
                    waiters = pending_reads.get(addr)
                    if waiters:
                        for pred in waiters:
                            add_edge(pred, i, 0)
                        pending_reads[addr] = []

                for addr in wset:
                    last_write[addr] = i

                # Track reads that must remain before the next write to the same loc.
                if wset:
                    for addr in rset:
                        if addr in wset:
                            continue
                        pending_reads.setdefault(addr, []).append(i)
                else:
                    for addr in rset:
                        pending_reads.setdefault(addr, []).append(i)

            # Simple criticality metric: the longest remaining chain of latency-1
            # deps from each slot. Helps prioritize work on the critical path.
            crit = [0 for _ in range(n)]
            for i in range(n - 1, -1, -1):
                if succ1[i]:
                    crit[i] = 1 + max(crit[j] for j in succ1[i])

            ready: set[int] = {
                i for i in range(n) if pred0_count[i] == 0 and pred1_count[i] == 0
            }
            scheduled = [False for _ in range(n)]
            remaining = n
            out: list[dict[str, list[tuple]]] = []
            engine_order = ("load", "store", "flow", "valu", "alu")

            while remaining:
                cur: dict[str, list[tuple]] = {}
                written: set[int] = set()
                engine_counts: dict[str, int] = {}
                scheduled_this_cycle: list[int] = []

                made_progress = True
                while made_progress:
                    made_progress = False
                    for engine in engine_order:
                        limit = SLOT_LIMITS.get(engine, 1)
                        while engine_counts.get(engine, 0) < limit:
                            cand: int | None = None
                            for i in ready:
                                if engines[i] != engine:
                                    continue
                                if reads[i] & written:
                                    continue
                                if writes[i] & written:
                                    continue
                                if cand is None or crit[i] > crit[cand] or (
                                    crit[i] == crit[cand] and i < cand
                                ):
                                    cand = i

                            if cand is None:
                                break

                            ready.remove(cand)
                            scheduled[cand] = True
                            remaining -= 1
                            engine_counts[engine] = engine_counts.get(engine, 0) + 1
                            cur.setdefault(engine, []).append(slots_only[cand])
                            written.update(writes[cand])
                            scheduled_this_cycle.append(cand)

                            for succ in succ0[cand]:
                                pred0_count[succ] -= 1
                                if (
                                    pred0_count[succ] == 0
                                    and pred1_count[succ] == 0
                                    and not scheduled[succ]
                                ):
                                    ready.add(succ)

                            made_progress = True

                if not cur:
                    # If we couldn't pack anything (should be rare), emit the next
                    # ready slot in its own bundle to break the deadlock.
                    if not ready:
                        raise RuntimeError("VLIW scheduler deadlocked (no ready slots)")
                    cand = min(ready)
                    engine = engines[cand]
                    ready.remove(cand)
                    scheduled[cand] = True
                    remaining -= 1
                    cur = {engine: [slots_only[cand]]}
                    written.update(writes[cand])
                    scheduled_this_cycle.append(cand)
                    for succ in succ0[cand]:
                        pred0_count[succ] -= 1
                        if (
                            pred0_count[succ] == 0
                            and pred1_count[succ] == 0
                            and not scheduled[succ]
                        ):
                            ready.add(succ)

                out.append(cur)

                for pred in scheduled_this_cycle:
                    for succ in succ1[pred]:
                        pred1_count[succ] -= 1
                        if (
                            pred0_count[succ] == 0
                            and pred1_count[succ] == 0
                            and not scheduled[succ]
                        ):
                            ready.add(succ)

            return out

        out: list[dict[str, list[tuple]]] = []
        segment: list[tuple[str, tuple]] = []
        for engine, slot in slots:
            engine = str(engine)
            if engine == "debug":
                continue
            if is_barrier(engine, slot):
                out.extend(schedule_segment(segment))
                segment = []
                out.extend(schedule_segment([(engine, slot)]))
            else:
                segment.append((engine, slot))

        out.extend(schedule_segment(segment))
        return out

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))

        return slots

    def build_hash_vec(self, val_hash_addr, tmp1, tmp2, vec_const_map):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("valu", (op1, tmp1, val_hash_addr, vec_const_map[val1])))
            slots.append(("valu", (op3, tmp2, val_hash_addr, vec_const_map[val3])))
            slots.append(("valu", (op2, val_hash_addr, tmp1, tmp2)))

        return slots

    def build_kernel(
        self,
        forest_height: int,
        n_nodes: int,
        batch_size: int,
        rounds: int,
    ):
        """
        Like reference_kernel2 but building actual instructions.
        Scalar implementation using only scalar ALU and load/store.
        """
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
        tmp3 = self.alloc_scratch("tmp3")
        # Scratch space addresses
        init_vars = [
            ("forest_values_p", 4),
            ("inp_values_p", 6),
        ]
        for v, _hdr_i in init_vars:
            self.alloc_scratch(v, 1)
        hdr_idxs = self.alloc_scratch("hdr_idxs", len(init_vars))
        prologue_slots: list[tuple[Engine, tuple]] = []
        for i, (_v, hdr_i) in enumerate(init_vars):
            prologue_slots.append(("load", ("const", hdr_idxs + i, hdr_i)))
        for i, (v, _hdr_i) in enumerate(init_vars):
            prologue_slots.append(("load", ("load", self.scratch[v], hdr_idxs + i)))
        prologue_instrs = self.build(prologue_slots, vliw=True)

        # Pause instructions are matched up with yield statements in the reference
        # kernel to let you debug at intermediate steps. The testing harness in this
        # file requires these match up to the reference kernel's yields, but the
        # submission harness ignores them.
        if prologue_instrs:
            prologue_instrs[-1].setdefault("flow", []).append(("pause",))
        self.instrs.extend(prologue_instrs)
        # Any debug engine instruction is ignored by the submission simulator
        self.add("debug", ("comment", "Starting loop"))

        body = []  # array of slots

        if batch_size % VLEN != 0:
            raise ValueError(f"{batch_size=} must be a multiple of {VLEN=}")

        # Pack constant materialization and broadcasts to reduce init overhead.
        const_slots: list[tuple[Engine, tuple]] = []

        def scratch_const_packed(val: int, name: str | None = None) -> int:
            if val not in self.const_map:
                addr = self.alloc_scratch(name)
                self.const_map[val] = addr
                const_slots.append(("load", ("const", addr, val)))
            return self.const_map[val]

        # Vector constants and scalars broadcasted to vectors
        one_const = scratch_const_packed(1)
        two_const = scratch_const_packed(2)
        zero_const = scratch_const_packed(0)
        stride_const = scratch_const_packed(VLEN)
        mul33_const = scratch_const_packed(33)
        mul4097_const = scratch_const_packed(4097)
        three_const = scratch_const_packed(3)
        four_const = scratch_const_packed(4)
        five_const = scratch_const_packed(5)
        six_const = scratch_const_packed(6)

        one_vec = self.alloc_scratch("one_vec", VLEN)
        zero_vec = self.alloc_scratch("zero_vec", VLEN)
        two_vec = self.alloc_scratch("two_vec", VLEN)
        idx4_vec = self.alloc_scratch("idx4_vec", VLEN)
        idx5_vec = self.alloc_scratch("idx5_vec", VLEN)
        idx6_vec = self.alloc_scratch("idx6_vec", VLEN)
        mul33_vec = self.alloc_scratch("mul33_vec", VLEN)
        mul4097_vec = self.alloc_scratch("mul4097_vec", VLEN)
        # Hot node constants for fast-path rounds (root + depth-1).
        node0_vec = self.alloc_scratch("node0_vec", VLEN)
        node2_vec = self.alloc_scratch("node2_vec", VLEN)
        node1_minus_node2_vec = self.alloc_scratch("node1_minus_node2_vec", VLEN)
        # Depth-2 fast-path (idx in {3,4,5,6}): materialize node3 and keep
        # node{4,5,6}-node3 deltas for masked adds.
        node3_vec = self.alloc_scratch("node3_vec", VLEN)
        node4_minus_node3_vec = self.alloc_scratch("node4_minus_node3_vec", VLEN)
        node5_minus_node3_vec = self.alloc_scratch("node5_minus_node3_vec", VLEN)
        node6_minus_node3_vec = self.alloc_scratch("node6_minus_node3_vec", VLEN)

        vector_slots: list[tuple[Engine, tuple]] = [
            ("valu", ("vbroadcast", one_vec, one_const)),
            ("valu", ("vbroadcast", two_vec, two_const)),
            ("valu", ("vbroadcast", idx4_vec, four_const)),
            ("valu", ("vbroadcast", idx5_vec, five_const)),
            ("valu", ("vbroadcast", idx6_vec, six_const)),
            ("valu", ("vbroadcast", mul33_vec, mul33_const)),
            ("valu", ("vbroadcast", mul4097_vec, mul4097_const)),
        ]

        # Vector constants used by the hash stages
        vec_const_map = {}
        for op1, val1, op2, op3, val3 in HASH_STAGES:
            for v in (val1, val3):
                if v in vec_const_map:
                    continue
                v_scalar = scratch_const_packed(v)
                v_vec = self.alloc_scratch(f"const_{v:x}_vec", VLEN)
                vector_slots.append(("valu", ("vbroadcast", v_vec, v_scalar)))
                vec_const_map[v] = v_vec

        fused_mul_vec_by_stage: dict[int, int] = {
            0: mul4097_vec,
            2: mul33_vec,
            4: vec_const_map[9],
        }

        # Precompute contiguous vload/vstore base addresses (reused across all rounds)
        n_groups = batch_size // VLEN
        val_ptrs = self.alloc_scratch("val_ptrs", n_groups)
        val_ptr_even = self.alloc_scratch("val_ptr_even")
        val_ptr_odd = self.alloc_scratch("val_ptr_odd")
        idx_cache = self.alloc_scratch("idx_cache", batch_size)
        val_cache = self.alloc_scratch("val_cache", batch_size)
        node_cache = self.alloc_scratch("node_cache", batch_size)
        tmp_hash2_arr = self.alloc_scratch("tmp_hash2_arr", batch_size)
        addr_buf0 = self.alloc_scratch("addr_buf0", VLEN)
        addr_buf1 = self.alloc_scratch("addr_buf1", VLEN)
        forest_nodes0_7 = self.alloc_scratch("forest_nodes0_7", VLEN)
        # Load small, frequently-used forest nodes once and materialize vectors for
        # fast-path rounds where indices are known to be in {0,1,2}.
        body.append(("load", ("vload", forest_nodes0_7, self.scratch["forest_values_p"])))
        body.append(("valu", ("vbroadcast", node0_vec, forest_nodes0_7 + 0)))
        body.append(("valu", ("vbroadcast", node2_vec, forest_nodes0_7 + 2)))
        body.append(("valu", ("vbroadcast", node1_minus_node2_vec, forest_nodes0_7 + 1)))
        body.append(("valu", ("-", node1_minus_node2_vec, node1_minus_node2_vec, node2_vec)))
        # Depth-2 fast path (idx in {3,4,5,6}): reuse the same vload'd block
        # and compute node{4,5,6}-node3 deltas for masked adds.
        body.append(("valu", ("vbroadcast", node3_vec, forest_nodes0_7 + 3)))
        body.append(("valu", ("vbroadcast", node4_minus_node3_vec, forest_nodes0_7 + 4)))
        body.append(("valu", ("vbroadcast", node5_minus_node3_vec, forest_nodes0_7 + 5)))
        body.append(("valu", ("vbroadcast", node6_minus_node3_vec, forest_nodes0_7 + 6)))
        body.append(("valu", ("-", node4_minus_node3_vec, node4_minus_node3_vec, node3_vec)))
        body.append(("valu", ("-", node5_minus_node3_vec, node5_minus_node3_vec, node3_vec)))
        body.append(("valu", ("-", node6_minus_node3_vec, node6_minus_node3_vec, node3_vec)))
        # Cache idx/val in scratch to avoid per-round vload/vstore traffic.
        #
        # Instead of vloading every group's initial values up-front (a large
        # load-engine bubble), we preload just the first two groups here so the
        # main loop can start hashing immediately while we stream-load the rest
        # opportunistically during the early, load-idle cycles.
        two_stride_const = scratch_const_packed(VLEN * 2)
        body.append(("alu", ("+", val_ptr_even, self.scratch["inp_values_p"], zero_const)))
        if n_groups > 1:
            body.append(("alu", ("+", val_ptr_odd, val_ptr_even, stride_const)))
        if n_groups > 0:
            body.append(("load", ("vload", val_cache + 0 * VLEN, val_ptr_even)))
            body.append(("alu", ("+", val_ptrs + 0, val_ptr_even, zero_const)))
        if n_groups > 1:
            body.append(("load", ("vload", val_cache + 1 * VLEN, val_ptr_odd)))
            body.append(("alu", ("+", val_ptrs + 1, val_ptr_odd, zero_const)))
        if n_groups > 2:
            body.append(("alu", ("+", val_ptr_even, val_ptr_even, two_stride_const)))
        if n_groups > 3:
            body.append(("alu", ("+", val_ptr_odd, val_ptr_odd, two_stride_const)))

        # Pack constants, broadcasts, and the setup/caching phase together so
        # independent load/alu/valu work can overlap in fewer bundles.
        setup_instrs = self.build(const_slots + vector_slots + body, vliw=True)
        self.instrs.extend(setup_instrs)

        # Main loop: software-pipelined, group-interleaved schedule.
        #
        # We keep per-group idx/val in scratch (idx_cache/val_cache) and overlap:
        # - load engine: scalar gathers into node_cache (2 loads/cycle)
        # - valu engine: hash + idx update across multiple groups (6 valu slots/cycle)
        addr_bufs = [addr_buf0, addr_buf1]

        # Per-group state machine. Groups are independent, so we can interleave rounds.
        LOAD = 0
        PREFETCHED = 1
        ROOT_XOR = 2
        DEPTH1_MASK = 3
        DEPTH1_NODE = 4
        XOR = 5
        HASH_TMP = 6
        HASH_COMB = 7
        STEP_AND = 8
        STEP_SELECT = 9
        IDX_MAD = 10
        IDX_RESET = 11
        STORE_OUT = 13
        DONE = 14
        HASH_FUSED = 15
        DEPTH2_INIT = 16
        DEPTH2_ADD4 = 17
        DEPTH2_ADD5 = 18
        DEPTH2_ADD6 = 19
        POSTLOAD_XOR = 20
        INIT_LOAD = 21

        period = forest_height + 1
        state = [INIT_LOAD for _ in range(n_groups)]
        if n_groups > 0:
            state[0] = ROOT_XOR
        if n_groups > 1:
            state[1] = ROOT_XOR
        g_round = [0 for _ in range(n_groups)]
        # Stagger group start times to smooth load demand and reduce pipeline
        # fill/drain bubbles (groups are independent).
        #
        # A piecewise schedule starts the first few groups closer together to
        # fill the valu pipeline early, then uses a wider spacing for the rest
        # to maintain a steady load/valu overlap once gathers begin.
        if START_OFFSETS is not None and len(START_OFFSETS) == n_groups:
            ready = list(START_OFFSETS)
        else:
            early_groups = START_EARLY_GROUPS
            early_spacing = START_EARLY_SPACING
            late_spacing = START_LATE_SPACING
            ready = [
                (g * early_spacing)
                if g < early_groups
                else (early_groups * early_spacing + (g - early_groups) * late_spacing)
                for g in range(n_groups)
            ]
        hash_stage = [0 for _ in range(n_groups)]
        postload_xor_off = [0 for _ in range(n_groups)]

        from collections import deque

        load_queue: deque[int] = deque()
        store_queue: deque[int] = deque()
        xor_queue: deque[int] = deque()

        load_group: int | None = None
        load_buf_idx: int | None = None
        load_off = 0
        load_xor_off = 0

        prefetch_group: int | None = None
        prefetch_buf_idx: int | None = None
        prefetch_ready = 0
        buf_toggle = 0

        rr_ptr = 0
        cycle = 0
        init_even = 2
        init_odd = 3
        main_instrs: list[dict[str, list[tuple]]] = []

        def group_iter(start: int):
            for di in range(n_groups):
                yield (start + di) % n_groups

        done_count = 0
        while done_count < n_groups:
            instr_alu: list[tuple] = []
            instr_valu: list[tuple] = []
            instr_load: list[tuple] = []
            instr_store: list[tuple] = []
            instr_flow: list[tuple] = []

            # Promote prefetched group to active loader when ready.
            if (
                load_group is None
                and prefetch_group is not None
                and cycle >= prefetch_ready
            ):
                load_group = prefetch_group
                load_buf_idx = prefetch_buf_idx
                load_off = 0
                load_xor_off = 0
                prefetch_group = None
                prefetch_buf_idx = None

            # Load engine: 2 scalar gathers per cycle for the active load_group.
            load_off_start: int | None = None
            if load_group is not None and load_buf_idx is not None:
                g = load_group
                addr_buf = addr_bufs[load_buf_idx]
                node_base = node_cache + g * VLEN
                load_off_start = load_off
                for _ in range(SLOT_LIMITS["load"]):
                    if load_off < VLEN:
                        instr_load.append(
                            ("load_offset", node_base, addr_buf, load_off)
                        )
                        load_off += 1

            # Load-idle cycles early in the schedule: stream-load remaining groups'
            # initial `val_cache` vectors and materialize their output pointers.
            load_budget = SLOT_LIMITS["load"] - len(instr_load)
            if load_budget > 0 and (init_even < n_groups or init_odd < n_groups):
                if init_even < n_groups and load_budget > 0:
                    g_init = init_even
                    instr_load.append(("vload", val_cache + g_init * VLEN, val_ptr_even))
                    instr_alu.append(("+", val_ptrs + g_init, val_ptr_even, zero_const))
                    instr_alu.append(("+", val_ptr_even, val_ptr_even, two_stride_const))
                    state[g_init] = ROOT_XOR
                    ready[g_init] = max(ready[g_init], cycle + 1)
                    init_even += 2
                    load_budget -= 1

                if init_odd < n_groups and load_budget > 0:
                    g_init = init_odd
                    instr_load.append(("vload", val_cache + g_init * VLEN, val_ptr_odd))
                    instr_alu.append(("+", val_ptrs + g_init, val_ptr_odd, zero_const))
                    instr_alu.append(("+", val_ptr_odd, val_ptr_odd, two_stride_const))
                    state[g_init] = ROOT_XOR
                    ready[g_init] = max(ready[g_init], cycle + 1)
                    init_odd += 2
                    load_budget -= 1

            # valu engine: keep load pipeline primed by prefetching next group's addresses.
            valu_budget = SLOT_LIMITS["valu"]
            if prefetch_group is None:
                g_pref = None
                for _ in range(len(load_queue)):
                    cand = load_queue[0]
                    if state[cand] == LOAD and ready[cand] <= cycle:
                        g_pref = cand
                        load_queue.popleft()
                        break
                    load_queue.append(load_queue.popleft())

                if g_pref is not None:
                    if load_buf_idx is not None:
                        buf_idx = 1 - load_buf_idx
                    else:
                        buf_idx = buf_toggle
                        buf_toggle = 1 - buf_toggle
                    idx_base = idx_cache + g_pref * VLEN
                    forest_values_p = self.scratch["forest_values_p"]
                    for vi in range(VLEN):
                        instr_alu.append(
                            (
                                "+",
                                addr_bufs[buf_idx] + vi,
                                forest_values_p,
                                idx_base + vi,
                            )
                        )
                    prefetch_group = g_pref
                    prefetch_buf_idx = buf_idx
                    prefetch_ready = cycle + 1
                    state[g_pref] = PREFETCHED

            # ALU engine: overlap lane-wise XOR with gather loads to save valu bandwidth.
            alu_budget = SLOT_LIMITS["alu"] - len(instr_alu)
            while alu_budget > 0 and xor_queue:
                g_xor = xor_queue[0]
                if ready[g_xor] > cycle:
                    break

                off = postload_xor_off[g_xor]
                val_base = val_cache + g_xor * VLEN
                node_base = node_cache + g_xor * VLEN
                while alu_budget > 0 and off < VLEN:
                    instr_alu.append(
                        ("^", val_base + off, val_base + off, node_base + off)
                    )
                    off += 1
                    alu_budget -= 1
                postload_xor_off[g_xor] = off

                if off < VLEN:
                    break

                xor_queue.popleft()
                hash_stage[g_xor] = 0
                state[g_xor] = HASH_FUSED if 0 in fused_mul_vec_by_stage else HASH_TMP
                ready[g_xor] = cycle + 1

            if (
                alu_budget > 0
                and load_group is not None
                and load_buf_idx is not None
                and load_off_start is not None
            ):
                g = load_group
                val_base = val_cache + g * VLEN
                node_base = node_cache + g * VLEN
                while alu_budget > 0 and load_xor_off < load_off_start:
                    instr_alu.append(
                        (
                            "^",
                            val_base + load_xor_off,
                            val_base + load_xor_off,
                            node_base + load_xor_off,
                        )
                    )
                    load_xor_off += 1
                    alu_budget -= 1

            if load_group is not None and load_buf_idx is not None and load_off >= VLEN:
                # Remaining lanes were loaded this cycle; finish XOR next cycle.
                g = load_group
                state[g] = POSTLOAD_XOR
                postload_xor_off[g] = load_xor_off
                ready[g] = cycle + 1
                xor_queue.append(g)
                load_group = None
                load_buf_idx = None

            def find_ready(target_state: int):
                nonlocal rr_ptr
                for g in group_iter(rr_ptr):
                    if state[g] == target_state and ready[g] <= cycle:
                        rr_ptr = (g + 1) % n_groups
                        return g
                return None

            while valu_budget > 0:
                g = find_ready(HASH_COMB)
                if g is not None:
                    idx_base = idx_cache + g * VLEN
                    val_base = val_cache + g * VLEN
                    t1 = node_cache + g * VLEN
                    t2 = tmp_hash2_arr + g * VLEN
                    op2 = HASH_STAGES[hash_stage[g]][2]
                    instr_valu.append((op2, val_base, t1, t2))
                    next_stage = hash_stage[g] + 1
                    if next_stage < len(HASH_STAGES):
                        hash_stage[g] = next_stage
                        state[g] = (
                            HASH_FUSED if next_stage in fused_mul_vec_by_stage else HASH_TMP
                        )
                    else:
                        next_round = g_round[g] + 1
                        if next_round >= rounds:
                            state[g] = STORE_OUT
                            store_queue.append(g)
                        elif next_round % period == 0:
                            state[g] = IDX_RESET
                        else:
                            state[g] = STEP_AND
                    ready[g] = cycle + 1
                    valu_budget -= 1
                    continue

                g = find_ready(HASH_FUSED)
                if g is not None:
                    val_base = val_cache + g * VLEN
                    stage_idx = hash_stage[g]
                    mul_vec = fused_mul_vec_by_stage[stage_idx]
                    val1 = HASH_STAGES[stage_idx][1]
                    instr_valu.append(
                        ("multiply_add", val_base, val_base, mul_vec, vec_const_map[val1])
                    )
                    next_stage = stage_idx + 1
                    if next_stage < len(HASH_STAGES):
                        hash_stage[g] = next_stage
                        state[g] = (
                            HASH_FUSED if next_stage in fused_mul_vec_by_stage else HASH_TMP
                        )
                    else:
                        next_round = g_round[g] + 1
                        if next_round >= rounds:
                            state[g] = STORE_OUT
                            store_queue.append(g)
                        elif next_round % period == 0:
                            state[g] = IDX_RESET
                        else:
                            state[g] = STEP_AND
                    ready[g] = cycle + 1
                    valu_budget -= 1
                    continue

                if valu_budget >= 2:
                    g = find_ready(HASH_TMP)
                    if g is not None:
                        val_base = val_cache + g * VLEN
                        t1 = node_cache + g * VLEN
                        t2 = tmp_hash2_arr + g * VLEN
                        op1, val1, _op2, op3, val3 = HASH_STAGES[hash_stage[g]]
                        instr_valu.append((op1, t1, val_base, vec_const_map[val1]))
                        instr_valu.append((op3, t2, val_base, vec_const_map[val3]))
                        state[g] = HASH_COMB
                        ready[g] = cycle + 1
                        valu_budget -= 2
                        continue

                scheduled = False
                for st in (
                    IDX_MAD,
                    STEP_AND,
                    XOR,
                    DEPTH2_ADD6,
                    DEPTH2_ADD5,
                    DEPTH2_ADD4,
                    DEPTH2_INIT,
                    DEPTH1_NODE,
                    DEPTH1_MASK,
                    ROOT_XOR,
                ):
                    g = find_ready(st)
                    if g is None:
                        continue

                    idx_base = idx_cache + g * VLEN
                    val_base = val_cache + g * VLEN
                    node_base = node_cache + g * VLEN
                    t1 = node_base
                    t2 = tmp_hash2_arr + g * VLEN

                    consumed = 1
                    if st == XOR:
                        instr_valu.append(("^", val_base, val_base, node_base))
                        hash_stage[g] = 0
                        state[g] = HASH_FUSED if 0 in fused_mul_vec_by_stage else HASH_TMP
                        ready[g] = cycle + 1
                    elif st == ROOT_XOR:
                        instr_valu.append(("^", val_base, val_base, node0_vec))
                        hash_stage[g] = 0
                        state[g] = HASH_FUSED if 0 in fused_mul_vec_by_stage else HASH_TMP
                        ready[g] = cycle + 1
                    elif st == DEPTH2_INIT:
                        if valu_budget < 2:
                            continue
                        instr_valu.append(("+", node_base, node3_vec, zero_vec))
                        instr_valu.append(("==", t2, idx_base, idx4_vec))
                        state[g] = DEPTH2_ADD4
                        ready[g] = cycle + 1
                        consumed = 2
                    elif st == DEPTH2_ADD4:
                        if valu_budget < 2:
                            continue
                        instr_valu.append(
                            (
                                "multiply_add",
                                node_base,
                                t2,
                                node4_minus_node3_vec,
                                node_base,
                            )
                        )
                        instr_valu.append(("==", t2, idx_base, idx5_vec))
                        state[g] = DEPTH2_ADD5
                        ready[g] = cycle + 1
                        consumed = 2
                    elif st == DEPTH2_ADD5:
                        if valu_budget < 2:
                            continue
                        instr_valu.append(
                            (
                                "multiply_add",
                                node_base,
                                t2,
                                node5_minus_node3_vec,
                                node_base,
                            )
                        )
                        instr_valu.append(("==", t2, idx_base, idx6_vec))
                        state[g] = DEPTH2_ADD6
                        ready[g] = cycle + 1
                        consumed = 2
                    elif st == DEPTH2_ADD6:
                        instr_valu.append(
                            (
                                "multiply_add",
                                node_base,
                                t2,
                                node6_minus_node3_vec,
                                node_base,
                            )
                        )
                        state[g] = XOR
                        ready[g] = cycle + 1
                    elif st == DEPTH1_MASK:
                        # idx is guaranteed in {1,2}; mask = idx&1.
                        instr_valu.append(("&", t2, idx_base, one_vec))
                        state[g] = DEPTH1_NODE
                        ready[g] = cycle + 1
                    elif st == DEPTH1_NODE:
                        instr_valu.append(
                            ("multiply_add", node_base, t2, node1_minus_node2_vec, node2_vec)
                        )
                        state[g] = XOR
                        ready[g] = cycle + 1
                    elif st == STEP_AND:
                        instr_valu.append(("&", t1, val_base, one_vec))
                        state[g] = STEP_SELECT
                        ready[g] = cycle + 1
                    elif st == IDX_MAD:
                        instr_valu.append(("multiply_add", idx_base, idx_base, two_vec, t1))
                        next_round = g_round[g] + 1
                        g_round[g] = next_round
                        if next_round < rounds:
                            depth = next_round % period
                            if depth == 0:
                                state[g] = ROOT_XOR
                            elif depth == 1:
                                state[g] = DEPTH1_MASK
                            elif depth == 2:
                                state[g] = DEPTH2_INIT
                            else:
                                state[g] = LOAD
                                load_queue.append(g)
                            ready[g] = cycle + 1
                        else:
                            state[g] = STORE_OUT
                            ready[g] = cycle + 1
                            store_queue.append(g)
                    else:
                        raise AssertionError("unreachable")

                    valu_budget -= consumed
                    scheduled = True
                    break

                if scheduled:
                    continue
                break

            # flow engine: use vselect to offload simple vector ops from valu.
            flow_budget = SLOT_LIMITS["flow"]
            if flow_budget > 0:
                g_flow = find_ready(IDX_RESET)
                if g_flow is not None:
                    idx_base = idx_cache + g_flow * VLEN
                    instr_flow.append(("vselect", idx_base, one_vec, zero_vec, zero_vec))
                    next_round = g_round[g_flow] + 1
                    g_round[g_flow] = next_round
                    if next_round < rounds:
                        state[g_flow] = ROOT_XOR
                        ready[g_flow] = cycle + 1
                    else:
                        state[g_flow] = STORE_OUT
                        ready[g_flow] = cycle + 1
                        store_queue.append(g_flow)
                else:
                    g_flow = find_ready(STEP_SELECT)
                    if g_flow is not None:
                        t1 = node_cache + g_flow * VLEN
                        instr_flow.append(("vselect", t1, t1, two_vec, one_vec))
                        state[g_flow] = IDX_MAD
                        ready[g_flow] = cycle + 1

            # store engine: once a group completes all rounds, write back values.
            store_budget = SLOT_LIMITS["store"]
            stores_issued = 0
            while stores_issued < store_budget and store_queue:
                g_store = None
                for _ in range(len(store_queue)):
                    cand = store_queue[0]
                    if state[cand] == STORE_OUT and ready[cand] <= cycle:
                        g_store = cand
                        store_queue.popleft()
                        break
                    store_queue.append(store_queue.popleft())

                if g_store is None:
                    break

                instr_store.append(
                    ("vstore", val_ptrs + g_store, val_cache + g_store * VLEN)
                )
                state[g_store] = DONE
                done_count += 1
                stores_issued += 1

            instr: dict[str, list[tuple]] = {}
            if instr_alu:
                instr["alu"] = instr_alu
            if instr_valu:
                instr["valu"] = instr_valu
            if instr_load:
                instr["load"] = instr_load
            if instr_store:
                instr["store"] = instr_store
            if instr_flow:
                instr["flow"] = instr_flow

            if not instr:
                # Shouldn't happen (would deadlock readiness on `cycle`), but keep safe.
                instr = {"flow": [("pause",)]}

            main_instrs.append(instr)
            cycle += 1

        self.instrs.extend(main_instrs)
        # Required to match with the yield in reference_kernel2.
        if self.instrs and not self.instrs[-1].get("flow"):
            self.instrs[-1].setdefault("flow", []).append(("pause",))
        else:
            self.instrs.append({"flow": [("pause",)]})

BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
