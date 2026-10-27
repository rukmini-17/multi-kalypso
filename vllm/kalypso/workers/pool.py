import json
from dataclasses import dataclass
from pathlib import Path
import asyncio



class Worker:
    def __init__(self, name, executor, manager, url=None, model = None, model_path=None):
        self.name = name
        self.executor = executor
        self.manager = manager
        self.url = url
        self.model = model
        self.model_path = model_path if model_path is not None else model

    def __repr__(self):
        return f"Worker({self.name})"


class WorkerPool:
    def __init__(self, workers):
        workers = list(workers)
        if not workers:
            raise ValueError("a pool needs at least one worker")
        names = [w.name for w in workers]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate worker names: {names}")
        
        urls = [w.url for w in workers if w.url is not None]
        if len(set(urls)) != len(urls):
            raise ValueError(f"duplicate worker urls: {urls}")

        if len({id(w.manager) for w in workers}) != len(workers):
            raise ValueError("every worker needs its own KVMemoryManager")
        self.workers = workers


    @classmethod
    def simulated(cls, n, executor, manager_factory):
        """n logical workers sharing one real engine, each with its own budget.

        manager_factory() must return a fresh KVMemoryManager each call.
        """
        return cls([Worker(f"w{i}", executor, manager_factory()) for i in range(n)])

    
    @classmethod
    async def discover(cls, specs, executor_factory, manager_factory, retries = 10, delay=2.0):
        import aiohttp
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            infos = await asyncio.gather(*(_fetch_with_retry(session, s.url, retries, delay) for s in specs))
        models = {i["model"] for i in infos}
        if len(models) != 1:
            raise RuntimeError(f"workers must serve the same model, got {sorted(models)}")
        model = models.pop()

        return cls([
            Worker(s.name, executor_factory(s.url, model, s.name),
                manager_factory(info["model_path"], info["budget_bytes"]),
                url=s.url, model=model, model_path=info["model_path"])
            for s, info in zip(specs, infos)
        ])

    def manager(self, w):
        return w.manager

    def managers(self):
        return [w.manager for w in self.workers]

    def outstanding(self, w):
        return w.manager.total_inflight()

    def total_capacity(self):
       return sum(m.capacity() for m in self.managers())


async def _fetch_with_retry(session, url, retries=10, delay=2.0):
    """Workers may still be loading the model; keep trying for a while."""
    for attempt in range(1, retries + 1):
        try:
            info = await _fetch_worker_info(session, url)
            print(f"[WorkerPool] {url} ready: model={info['model']} "
                  f"budget={info['budget_bytes'] / 1e9:.1f} GB")
            return info
        except Exception as e:
            if attempt == retries:
                print(f"[WorkerPool] {url} failed after {retries} attempts: {e!r}")
                raise
            print(f"[WorkerPool] {url} not ready (attempt {attempt}/{retries}): "
                  f"{type(e).__name__}; retrying in {delay:.0f}s")
            await asyncio.sleep(delay)

async def _fetch_worker_info(session, url: str) -> dict:
    async with session.get(f"{url}/v1/semantic/kv_budget") as resp:
        resp.raise_for_status()
        info = await resp.json()
    async with session.get(f"{url}/v1/models") as resp:
        resp.raise_for_status()
        models = (await resp.json())["data"]
    if not models or not models[0].get("id"):
        raise ValueError(f"worker {url} returned no served model id")
    info["model_path"] = info["model"]
    info["model"] = models[0]["id"]
    return info



@dataclass
class WorkerSpec:
    """A worker we intend to use, before we know whether it is up."""
    name: str
    url: str
 
 
@dataclass
class PoolConfig:
    workers: list[WorkerSpec]
    policy: str | None = None  # None = use --policy or the default
 
 
def _normalize_url(url):
    # "localhost:8001/" -> "http://localhost:8001", so duplicates are caught.
    url = str(url).strip().rstrip("/")
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    return url
 
 
def _check_specs(specs):
    if not specs:
        raise ValueError("no workers configured")
    names = [s.name for s in specs]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate worker names: {names}")
    urls = [s.url for s in specs]
    if len(set(urls)) != len(urls):
        raise ValueError(f"duplicate worker urls: {urls}")
    return specs
 
 
def parse_urls(urls):
    """'http://a:8001,http://b:8002' (the --workers flag) -> [WorkerSpec]."""
    if isinstance(urls, str):
        urls = urls.split(",")
    urls = [u.strip() for u in urls if u.strip()]
    return _check_specs(
        [WorkerSpec(f"w{i}", _normalize_url(u)) for i, u in enumerate(urls)]
    )
 
 
def parse_config(data):
    """A dict loaded from YAML/JSON -> PoolConfig, with every entry checked."""
    if not isinstance(data, dict) or not isinstance(data.get("workers"), list):
        raise ValueError("config must have a 'workers' list")
    specs = []
    for i, entry in enumerate(data["workers"]):
        if isinstance(entry, str):          # allow "- localhost:8001" shorthand
            entry = {"url": entry}
        if not isinstance(entry, dict) or "url" not in entry:
            raise ValueError(f"worker #{i} needs a 'url'")
        specs.append(WorkerSpec(
            name=str(entry.get("name", f"w{i}")),
            url=_normalize_url(entry["url"]),
        ))
    return PoolConfig(workers=_check_specs(specs), policy=data.get("policy"))
 
 
def load_config(path):
    """Read a .yaml/.yml or .json pool config file."""
    path = Path(path)
    text = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        import yaml  # imported here so the rest of pool.py works without pyyaml
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    return parse_config(data)
 
