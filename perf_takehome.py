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
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        one_const = self.scratch_const(1)

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

        # Vector scratch registers
        tmp_idx = self.alloc_scratch("tmp_idx", VLEN)
        tmp_val = self.alloc_scratch("tmp_val", VLEN)
        tmp_node_val = self.alloc_scratch("tmp_node_val", VLEN)
        tmp_addr = self.alloc_scratch("tmp_addr", VLEN)
        tmp_idx_dbl = self.alloc_scratch("tmp_idx_dbl", VLEN)
        tmp_hash1 = self.alloc_scratch("tmp_hash1", VLEN)
        tmp_hash2 = self.alloc_scratch("tmp_hash2", VLEN)
        tmp_step = self.alloc_scratch("tmp_step", VLEN)
        tmp_mask = self.alloc_scratch("tmp_mask", VLEN)

        # Vector constants and scalars broadcasted to vectors
        one_vec = self.alloc_scratch("one_vec", VLEN)
        n_nodes_vec = self.alloc_scratch("n_nodes_vec", VLEN)
        forest_values_p_vec = self.alloc_scratch("forest_values_p_vec", VLEN)

        self.add("valu", ("vbroadcast", one_vec, one_const))
        self.add("valu", ("vbroadcast", n_nodes_vec, self.scratch["n_nodes"]))
        self.add(
            "valu",
            ("vbroadcast", forest_values_p_vec, self.scratch["forest_values_p"]),
        )

        # Vector constants used by the hash stages
        vec_const_map = {}
        for op1, val1, op2, op3, val3 in HASH_STAGES:
            for v in (val1, val3):
                if v in vec_const_map:
                    continue
                v_scalar = self.scratch_const(v)
                v_vec = self.alloc_scratch(f"const_{v:x}_vec", VLEN)
                self.add("valu", ("vbroadcast", v_vec, v_scalar))
                vec_const_map[v] = v_vec

        # Precompute contiguous vload/vstore base addresses (reused across all rounds)
        n_groups = batch_size // VLEN
        idx_ptrs = self.alloc_scratch("idx_ptrs", n_groups)
        val_ptrs = self.alloc_scratch("val_ptrs", n_groups)
        idx_cache = self.alloc_scratch("idx_cache", batch_size)
        val_cache = self.alloc_scratch("val_cache", batch_size)
        for g in range(n_groups):
            off_const = self.scratch_const(g * VLEN)
            body.append(
                ("alu", ("+", idx_ptrs + g, self.scratch["inp_indices_p"], off_const))
            )
            body.append(
                ("alu", ("+", val_ptrs + g, self.scratch["inp_values_p"], off_const))
            )

        # Cache idx/val in scratch to avoid per-round vload/vstore traffic.
        for g in range(n_groups):
            body.append(("load", ("vload", idx_cache + g * VLEN, idx_ptrs + g)))
            body.append(("load", ("vload", val_cache + g * VLEN, val_ptrs + g)))

        for round in range(rounds):
            for g in range(n_groups):
                idx_base = idx_cache + g * VLEN
                val_base = val_cache + g * VLEN

                # tmp_addr[v] = forest_values_p + idx[v]
                body.append(("valu", ("+", tmp_addr, forest_values_p_vec, idx_base)))
                # tmp_idx_dbl[v] = 2 * idx[v]
                body.append(("valu", ("+", tmp_idx_dbl, idx_base, idx_base)))

                # node_val[v] = mem[forest_values_p + idx[v]] (gather)
                for off in range(VLEN):
                    body.append(("load", ("load_offset", tmp_node_val, tmp_addr, off)))

                # val = myhash(val ^ node_val)
                body.append(("valu", ("^", val_base, val_base, tmp_node_val)))
                body.extend(
                    self.build_hash_vec(val_base, tmp_hash1, tmp_hash2, vec_const_map)
                )

                # idx = 2*idx + (1 + (val & 1))
                body.append(("valu", ("&", tmp_step, val_base, one_vec)))
                body.append(("valu", ("+", tmp_step, tmp_step, one_vec)))
                body.append(("valu", ("+", idx_base, tmp_idx_dbl, tmp_step)))

                # idx = idx if idx < n_nodes else 0  (masking avoids flow bottleneck)
                body.append(("valu", ("<", tmp_mask, idx_base, n_nodes_vec)))
                body.append(("valu", ("*", idx_base, idx_base, tmp_mask)))

        # Write back cached idx/val once at the end.
        for g in range(n_groups):
            body.append(("store", ("vstore", idx_ptrs + g, idx_cache + g * VLEN)))
            body.append(("store", ("vstore", val_ptrs + g, val_cache + g * VLEN)))

        body_instrs = self.build(body, vliw=True)
        self.instrs.extend(body_instrs)
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
