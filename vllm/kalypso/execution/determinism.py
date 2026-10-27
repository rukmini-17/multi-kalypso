# execution/determinism.py
"""Seeded reply replacement for the multi-worker correctness harness.

When KALYPSO_COIN_FLIP_SEED is set, each LLM reply is replaced after real
inference by a hash of the seed and prompt, shaped by the operator's strategy.
"""

import hashlib
import json
import os
from abc import ABC, abstractmethod

from .executor import CompletionResult


def determinism_seed() -> str | None:
    """Return the harness seed, or None when determinism is off."""
    return os.environ.get("KALYPSO_COIN_FLIP_SEED") or None


class Determinism(ABC):
    """How an operator's replies are replaced when determinism is on.

    Each operator declares one so the reply has the shape its parser expects.
    """

    @abstractmethod
    def reply(self, digest: bytes) -> str:
        pass


class Choice(Determinism):
    """Pick one option uniformly, for example true/false or a class name."""

    def __init__(self, options: list[str]):
        if not options:
            raise ValueError("Choice determinism needs at least one option")
        self.options = list(options)

    def reply(self, digest: bytes) -> str:
        return self.options[int.from_bytes(digest[:8], "big") % len(self.options)]


class Bucket(Determinism):
    """Hash each call into a fixed number of buckets.

    With the default of 10 buckets, each bucket holds about one tenth of the
    calls on average; skew is acceptable.
    """

    def __init__(self, num_buckets: int = 10):
        if num_buckets < 1:
            raise ValueError("Bucket determinism needs at least one bucket")
        self.num_buckets = num_buckets

    def reply(self, digest: bytes) -> str:
        return f"bucket {int.from_bytes(digest[:8], 'big') % self.num_buckets}"


class Text(Determinism):
    """Free text that is unique per prompt, for operators like sem_map."""

    def reply(self, digest: bytes) -> str:
        return digest.hex()[:8]


BOOLEAN = Choice(["true", "false"])


def apply_determinism(
    prompt: str | list[dict[str, object]],
    result: CompletionResult,
    determinism: Determinism | None,
) -> CompletionResult:
    """Make harness replies reproducible across worker counts after real inference."""
    seed = determinism_seed()
    if seed is None:
        return result
    if determinism is None:
        raise RuntimeError(
            "Determinism is on but this LLM call has no determinism strategy"
        )
    canonical_prompt = (
        prompt if isinstance(prompt, str) else json.dumps(prompt, sort_keys=True)
    )
    digest = hashlib.sha256(f"{seed}|{canonical_prompt}".encode()).digest()
    return CompletionResult(determinism.reply(digest), result.request_id, "stop")


class DeterminismUnsupportedError(ValueError):
    pass


def check_determinism(logical_ops, physical_ops) -> None:
    """Fail before execution if determinism is on and an operator lacks support."""
    if determinism_seed() is None:
        return
    unsupported = [
        f"{logical_ops[op.position]['op']} at position {op.position} "
        f"({op.__class__.__name__})"
        for op in physical_ops
        if op.CALLS_LLM and op.determinism is None
    ]
    if unsupported:
        raise DeterminismUnsupportedError(
            "Determinism is on (KALYPSO_COIN_FLIP_SEED is set), but these "
            "operators do not support it: " + "; ".join(unsupported)
        )
