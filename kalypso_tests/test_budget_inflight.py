import asyncio
import importlib.util
import pathlib
import sys
import types


def load_budget():
    # budget.py imports torch/transformers at module level; stub them during load.
    saved = {k: sys.modules.get(k) for k in ("torch", "transformers")}
    torch = types.ModuleType("torch")
    torch.dtype = object
    torch.float16 = torch.bfloat16 = torch.float32 = "fp"
    tf = types.ModuleType("transformers")
    tf.AutoConfig = tf.AutoTokenizer = object
    sys.modules["torch"], sys.modules["transformers"] = torch, tf
    try:
        path = pathlib.Path(__file__).resolve().parents[1] / "vllm/kalypso/budget.py"
        spec = importlib.util.spec_from_file_location("kalypso_budget", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


budget = load_budget()


def make_manager(capacity=1000):
    m = budget.KVMemoryManager.__new__(budget.KVMemoryManager)  # skip __init__
    m._capacity = capacity
    m._global_used = 0
    m._stage_capacity, m._stage_min_capacity, m._stage_max_capacity = {}, {}, {}
    m._stage_used, m._stage_inflight = {}, {}
    m._cond = asyncio.Condition()
    return m


def test_total_inflight_tracks_admit_release_reset():
    async def go():
        m = make_manager()
        m.register_stage(1, 0.5)
        m.register_stage(2, 0.5)
        assert m.total_inflight() == 0
        assert await m.try_allocate_stage(1, 10)
        assert await m.try_allocate_stage(1, 10)
        assert await m.try_allocate_stage(2, 10)
        assert m.total_inflight() == 3
        await m.release_stage(1, 10)
        assert m.total_inflight() == 2
        m.reset_stages()
        assert m.total_inflight() == 0
    asyncio.run(go())


def test_refused_allocation_is_not_counted():
    async def go():
        m = make_manager()
        m.register_stage(1, 1.0)
        assert not await m.try_allocate_stage(1, 10_000)  # larger than capacity
        assert m.total_inflight() == 0
    asyncio.run(go())


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for name, fn in tests:
        fn()
        print("ok  ", name)
    print(f"\n{len(tests)} tests passed")