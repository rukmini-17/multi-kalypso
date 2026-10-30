from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class WorkloadConfig:
    name: str
    rows: int
    data_root: str | None = None


@dataclass(frozen=True)
class Instrumentation:
    record_requests: bool = True
    poll_interval: float | None = None
    kv_cache_metrics_sample: float | None = None
    capture_kv_stats: bool = True


@dataclass(frozen=True)
class Vllm:
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.95
    extra_args: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Timeouts:
    worker_ready_s: float = 1800
    coordinator_ready_s: float = 120
    query_s: float = 3600


@dataclass(frozen=True)
class Hardware:
    gpu_type: str = "A10"


@dataclass(frozen=True)
class Experiment:
    model: str
    policies: list[str]
    workloads: list[WorkloadConfig]
    n_workers: list[int]
    reps: int = 3
    reset_prefix_cache: bool = True
    allow_fewer_rows: bool = False
    hardware: Hardware = field(default_factory=Hardware)
    vllm: Vllm = field(default_factory=Vllm)
    instrumentation: Instrumentation = field(default_factory=Instrumentation)
    timeouts: Timeouts = field(default_factory=Timeouts)


@dataclass(frozen=True)
class Run:
    n_workers: int
    policy: str
    workload: WorkloadConfig
    rep: int

    @property
    def name(self) -> str:
        return f"{self.policy}/{self.workload.name}/rep_{self.rep}"


def parse_config(text: str) -> Experiment:
    values = yaml.safe_load(text)
    if not isinstance(values, dict):
        raise ValueError("Experiment config must be a YAML mapping")
    values = dict(values)
    for key, cls in [
        ("hardware", Hardware),
        ("vllm", Vllm),
        ("instrumentation", Instrumentation),
        ("timeouts", Timeouts),
    ]:
        values[key] = cls(**values.get(key, {}))
    values["workloads"] = [WorkloadConfig(**w) for w in values.get("workloads", [])]
    config = Experiment(**values)
    if not config.model or type(config.reps) is not int or config.reps < 1:
        raise ValueError("model and positive integer reps are required")
    if not config.n_workers or any(
        type(n) is not int or n < 1 for n in config.n_workers
    ):
        raise ValueError("n_workers must contain positive integers")
    if not config.policies or not config.workloads:
        raise ValueError("policies and workloads must not be empty")
    for policy in config.policies:
        if (
            not isinstance(policy, str)
            or not policy
            or not all(c.isalnum() or c == "_" for c in policy)
        ):
            raise ValueError(f"Invalid policy name: {policy!r}")
    for w in config.workloads:
        if type(w.rows) is not int or w.rows < 1:
            raise ValueError(f"Invalid workload: {w}")
    for items in (
        config.n_workers,
        config.policies,
        [w.name for w in config.workloads],
    ):
        if len(items) != len(set(items)):
            raise ValueError("Duplicate matrix entries would overwrite run outputs")
    if any(
        t <= 0
        for t in (
            config.timeouts.worker_ready_s,
            config.timeouts.coordinator_ready_s,
            config.timeouts.query_s,
        )
    ):
        raise ValueError("Timeouts must be positive")
    if config.vllm.max_model_len < 1 or not 0 < config.vllm.gpu_memory_utilization <= 1:
        raise ValueError("Invalid vllm context length or memory utilization")
    sample = config.instrumentation.kv_cache_metrics_sample
    if sample is not None and not 0 < sample <= 1:
        raise ValueError("kv_cache_metrics_sample must be greater than 0 and at most 1")
    poll = config.instrumentation.poll_interval
    if poll is not None and poll <= 0:
        raise ValueError("poll_interval must be positive or null")
    if not isinstance(config.vllm.extra_args, list) or not all(
        isinstance(a, str) for a in config.vllm.extra_args
    ):
        raise ValueError("vllm.extra_args must be a list of strings")
    return config


def load_config(path: str | Path) -> Experiment:
    return parse_config(Path(path).read_text(encoding="utf-8"))


def expand(config: Experiment, n_workers: int | None = None) -> list[Run]:
    if n_workers is not None and n_workers not in config.n_workers:
        raise ValueError(f"N={n_workers} is not in configured n_workers")
    return [
        Run(n, policy, workload, rep)
        for n in config.n_workers
        if n_workers is None or n == n_workers
        for policy in config.policies
        for workload in config.workloads
        for rep in range(1, config.reps + 1)
    ]
