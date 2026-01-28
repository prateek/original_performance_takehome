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

        instrs: list[dict[str, list[tuple]]] = []
        cur: dict[str, list[tuple]] = {}
        written: set[int] = set()

        def flush():
            nonlocal cur, written
            if cur:
                instrs.append(cur)
                cur = {}
                written = set()

        for engine, slot in slots:
            engine = str(engine)
            if engine == "debug":
                continue
            reads, writes = self._slot_scratch_io(engine, slot)

            limit = SLOT_LIMITS.get(engine, 1)
            cur_engine_slots = cur.get(engine)
            cur_engine_len = len(cur_engine_slots) if cur_engine_slots is not None else 0
            if cur and (
                cur_engine_len >= limit or (reads & written) or (writes & written)
            ):
                flush()

            cur.setdefault(engine, []).append(slot)
            written.update(writes)

            is_barrier = engine == "flow" and slot and slot[0] in {
                "pause",
                "halt",
                "jump",
                "jump_indirect",
                "cond_jump",
                "cond_jump_rel",
            }
            if is_barrier:
                flush()

        flush()
        return instrs

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
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
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
            "rounds",
            "n_nodes",
            "batch_size",
            "forest_height",
            "forest_values_p",
            "inp_indices_p",
            "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        hdr_idxs = self.alloc_scratch("hdr_idxs", len(init_vars))
        prologue_slots: list[tuple[Engine, tuple]] = []
        for i in range(len(init_vars)):
            prologue_slots.append(("load", ("const", hdr_idxs + i, i)))
        for i, v in enumerate(init_vars):
            prologue_slots.append(("load", ("load", self.scratch[v], hdr_idxs + i)))
        self.instrs.extend(self.build(prologue_slots, vliw=True))

        # Pause instructions are matched up with yield statements in the reference
        # kernel to let you debug at intermediate steps. The testing harness in this
        # file requires these match up to the reference kernel's yields, but the
        # submission harness ignores them.
        self.add("flow", ("pause",))
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

        one_vec = self.alloc_scratch("one_vec", VLEN)
        zero_vec = self.alloc_scratch("zero_vec", VLEN)
        two_vec = self.alloc_scratch("two_vec", VLEN)
        n_nodes_vec = self.alloc_scratch("n_nodes_vec", VLEN)
        forest_values_p_vec = self.alloc_scratch("forest_values_p_vec", VLEN)
        # Hot node constants for fast-path rounds (root + depth-1).
        node0_vec = self.alloc_scratch("node0_vec", VLEN)
        node2_vec = self.alloc_scratch("node2_vec", VLEN)
        node1_minus_node2_vec = self.alloc_scratch("node1_minus_node2_vec", VLEN)

        vector_slots: list[tuple[Engine, tuple]] = [
            ("valu", ("vbroadcast", one_vec, one_const)),
            ("valu", ("vbroadcast", two_vec, two_const)),
            ("valu", ("vbroadcast", n_nodes_vec, self.scratch["n_nodes"])),
            ("valu", ("vbroadcast", forest_values_p_vec, self.scratch["forest_values_p"])),
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

        self.instrs.extend(self.build(const_slots, vliw=True))
        self.instrs.extend(self.build(vector_slots, vliw=True))

        # Precompute contiguous vload/vstore base addresses (reused across all rounds)
        n_groups = batch_size // VLEN
        idx_ptrs = self.alloc_scratch("idx_ptrs", n_groups)
        val_ptrs = self.alloc_scratch("val_ptrs", n_groups)
        idx_cache = self.alloc_scratch("idx_cache", batch_size)
        val_cache = self.alloc_scratch("val_cache", batch_size)
        node_cache = self.alloc_scratch("node_cache", batch_size)
        tmp_hash2_arr = self.alloc_scratch("tmp_hash2_arr", batch_size)
        addr_buf0 = self.alloc_scratch("addr_buf0", VLEN)
        addr_buf1 = self.alloc_scratch("addr_buf1", VLEN)
        # Load small, frequently-used forest nodes once and materialize vectors for
        # fast-path rounds where indices are known to be in {0,1,2}.
        body.append(("alu", ("+", tmp2, self.scratch["forest_values_p"], one_const)))
        body.append(("alu", ("+", tmp3, self.scratch["forest_values_p"], two_const)))
        body.append(("load", ("load", tmp1, self.scratch["forest_values_p"])))
        body.append(("load", ("load", tmp2, tmp2)))
        body.append(("load", ("load", tmp3, tmp3)))
        body.append(("valu", ("vbroadcast", node0_vec, tmp1)))
        body.append(("valu", ("vbroadcast", node2_vec, tmp3)))
        body.append(("valu", ("vbroadcast", node1_minus_node2_vec, tmp2)))
        body.append(("valu", ("-", node1_minus_node2_vec, node1_minus_node2_vec, node2_vec)))
        body.append(("alu", ("+", idx_ptrs + 0, self.scratch["inp_indices_p"], zero_const)))
        body.append(("alu", ("+", val_ptrs + 0, self.scratch["inp_values_p"], zero_const)))
        for g in range(1, n_groups):
            body.append(("alu", ("+", idx_ptrs + g, idx_ptrs + g - 1, stride_const)))
            body.append(("alu", ("+", val_ptrs + g, val_ptrs + g - 1, stride_const)))

        # Cache idx/val in scratch to avoid per-round vload/vstore traffic.
        for g in range(n_groups):
            body.append(("load", ("vload", idx_cache + g * VLEN, idx_ptrs + g)))
            body.append(("load", ("vload", val_cache + g * VLEN, val_ptrs + g)))

        # Pack the setup/caching phase with the generic VLIW packer.
        body_instrs = self.build(body, vliw=True)
        self.instrs.extend(body_instrs)

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
        MASK_LT = 11
        IDX_SELECT = 12
        STORE_OUT = 13
        DONE = 14

        period = forest_height + 1
        state = [ROOT_XOR for _ in range(n_groups)]
        g_round = [0 for _ in range(n_groups)]
        ready = [0 for _ in range(n_groups)]
        hash_stage = [0 for _ in range(n_groups)]

        from collections import deque

        load_queue: deque[int] = deque()
        store_queue: deque[int] = deque()

        load_group: int | None = None
        load_buf_idx: int | None = None
        load_off = 0

        prefetch_group: int | None = None
        prefetch_buf_idx: int | None = None
        prefetch_ready = 0
        buf_toggle = 0

        rr_ptr = 0
        cycle = 0
        main_instrs: list[dict[str, list[tuple]]] = []

        def group_iter(start: int):
            for di in range(n_groups):
                yield (start + di) % n_groups

        done_count = 0
        while done_count < n_groups:
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
                prefetch_group = None
                prefetch_buf_idx = None

            # Load engine: 2 scalar gathers per cycle for the active load_group.
            if load_group is not None and load_buf_idx is not None:
                g = load_group
                addr_buf = addr_bufs[load_buf_idx]
                node_base = node_cache + g * VLEN
                for _ in range(SLOT_LIMITS["load"]):
                    if load_off < VLEN:
                        instr_load.append(
                            ("load_offset", node_base, addr_buf, load_off)
                        )
                        load_off += 1
                if load_off >= VLEN:
                    # node_cache is written at end of this cycle; XOR can start next cycle.
                    state[g] = XOR
                    ready[g] = cycle + 1
                    load_group = None
                    load_buf_idx = None

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

                if g_pref is not None and valu_budget > 0:
                    if load_buf_idx is not None:
                        buf_idx = 1 - load_buf_idx
                    else:
                        buf_idx = buf_toggle
                        buf_toggle = 1 - buf_toggle
                    idx_base = idx_cache + g_pref * VLEN
                    instr_valu.append(
                        ("+", addr_bufs[buf_idx], forest_values_p_vec, idx_base)
                    )
                    prefetch_group = g_pref
                    prefetch_buf_idx = buf_idx
                    prefetch_ready = cycle + 1
                    state[g_pref] = PREFETCHED
                    valu_budget -= 1

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
                    if hash_stage[g] + 1 < len(HASH_STAGES):
                        hash_stage[g] += 1
                        state[g] = HASH_TMP
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
                    MASK_LT,
                    IDX_MAD,
                    STEP_AND,
                    XOR,
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

                    if st == XOR:
                        instr_valu.append(("^", val_base, val_base, node_base))
                        hash_stage[g] = 0
                        state[g] = HASH_TMP
                        ready[g] = cycle + 1
                    elif st == ROOT_XOR:
                        instr_valu.append(("^", val_base, val_base, node0_vec))
                        hash_stage[g] = 0
                        state[g] = HASH_TMP
                        ready[g] = cycle + 1
                    elif st == DEPTH1_MASK:
                        # idx is guaranteed in {1,2}; mask = idx&1.
                        instr_valu.append(("&", t2, idx_base, one_vec))
                        state[g] = DEPTH1_NODE
                        ready[g] = cycle + 1
                    elif st == DEPTH1_NODE:
                        instr_valu.append(("multiply_add", node_base, t2, node1_minus_node2_vec, node2_vec))
                        state[g] = XOR
                        ready[g] = cycle + 1
                    elif st == STEP_AND:
                        instr_valu.append(("&", t1, val_base, one_vec))
                        state[g] = STEP_SELECT
                        ready[g] = cycle + 1
                    elif st == IDX_MAD:
                        instr_valu.append(("multiply_add", idx_base, idx_base, two_vec, t1))
                        state[g] = MASK_LT
                        ready[g] = cycle + 1
                    elif st == MASK_LT:
                        instr_valu.append(("<", t2, idx_base, n_nodes_vec))
                        state[g] = IDX_SELECT
                        ready[g] = cycle + 1
                    else:
                        raise AssertionError("unreachable")

                    valu_budget -= 1
                    scheduled = True
                    break

                if scheduled:
                    continue
                break

            # flow engine: use vselect to offload simple vector ops from valu.
            flow_budget = SLOT_LIMITS["flow"]
            if flow_budget > 0:
                g_flow = find_ready(IDX_SELECT)
                if g_flow is not None:
                    idx_base = idx_cache + g_flow * VLEN
                    t2 = tmp_hash2_arr + g_flow * VLEN
                    instr_flow.append(("vselect", idx_base, t2, idx_base, zero_vec))
                    next_round = g_round[g_flow] + 1
                    g_round[g_flow] = next_round
                    if next_round < rounds:
                        depth = next_round % period
                        if depth == 0:
                            state[g_flow] = ROOT_XOR
                        elif depth == 1:
                            state[g_flow] = DEPTH1_MASK
                        else:
                            state[g_flow] = LOAD
                            load_queue.append(g_flow)
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

            # store engine: once a group completes all rounds, write it back (1 group/cycle).
            store_budget = SLOT_LIMITS["store"]
            if store_budget >= 2:
                g_store = None
                for _ in range(len(store_queue)):
                    cand = store_queue[0]
                    if state[cand] == STORE_OUT and ready[cand] <= cycle:
                        g_store = cand
                        store_queue.popleft()
                        break
                    store_queue.append(store_queue.popleft())

                if g_store is not None:
                    instr_store.append(
                        ("vstore", idx_ptrs + g_store, idx_cache + g_store * VLEN)
                    )
                    instr_store.append(
                        ("vstore", val_ptrs + g_store, val_cache + g_store * VLEN)
                    )
                    state[g_store] = DONE
                    done_count += 1

            instr: dict[str, list[tuple]] = {}
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
        # Required to match with the yield in reference_kernel2
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
