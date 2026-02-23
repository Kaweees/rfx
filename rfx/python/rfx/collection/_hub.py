"""rfx.collection._hub — HuggingFace Hub push/pull operations."""

from __future__ import annotations

from pathlib import Path

from ._dataset import Dataset


def push(dataset: Dataset, repo_id: str | None = None) -> str:
    """Push a dataset to HuggingFace Hub. Returns URL."""
    effective_id = repo_id or dataset.repo_id
    dataset.push(repo_id)
    return f"https://huggingface.co/datasets/{effective_id}"


def pull(repo_id: str, *, root: str | Path = "datasets") -> Dataset:
    """Pull a dataset from HuggingFace Hub."""
    return Dataset.from_hub(repo_id, root=root)


def from_hub(repo_id: str, *, root: str | Path = "datasets") -> Dataset:
    """Alias for pull()."""
    return pull(repo_id, root=root)
