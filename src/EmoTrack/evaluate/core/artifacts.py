from __future__ import annotations

from enum import Enum
from pathlib import Path


class ArtifactNamespace(str, Enum):
    """Canonical detector artifact namespaces under a seeded result directory."""

    DETECTOR = ""
    TRANSFER = "transfer"


ArtifactNamespaceLike = ArtifactNamespace | str | None


def normalize_artifact_namespace(namespace: ArtifactNamespaceLike = None) -> ArtifactNamespace:
    if namespace is None or namespace == "":
        return ArtifactNamespace.DETECTOR
    if isinstance(namespace, ArtifactNamespace):
        return namespace
    raw = str(namespace).strip()
    for candidate in ArtifactNamespace:
        if raw == candidate.value or raw == candidate.name:
            return candidate
    raise ValueError(f"Unsupported detector artifact namespace: {namespace!r}")


def artifact_namespace_subdir(namespace: ArtifactNamespaceLike = None) -> str:
    return normalize_artifact_namespace(namespace).value


def detector_artifact_filename(*, method_name: str, transcript_view: str) -> str:
    safe_method = method_name.replace("-", "_")
    safe_view = transcript_view.replace("-", "_")
    if safe_view == "full":
        return f"{safe_method}.json"
    return f"{safe_method}_{safe_view}.json"


def detector_artifact_seed_dir(
    artifact_dir: str | Path,
    *,
    seed: int = 0,
    namespace: ArtifactNamespaceLike = None,
) -> Path:
    seed_dir = Path(artifact_dir) / f"seed{seed}"
    subdir = artifact_namespace_subdir(namespace)
    return seed_dir / subdir if subdir else seed_dir


def detector_output_path(
    artifact_dir: str | Path,
    *,
    method_name: str,
    transcript_view: str,
    seed: int = 0,
    namespace: ArtifactNamespaceLike = None,
) -> Path:
    return detector_artifact_seed_dir(
        artifact_dir,
        seed=seed,
        namespace=namespace,
    ) / detector_artifact_filename(method_name=method_name, transcript_view=transcript_view)
