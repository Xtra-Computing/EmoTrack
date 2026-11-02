from __future__ import annotations

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PACKAGE_ROOT.parent
REPO_ROOT = SRC_ROOT.parent


def repo_relative(path: Path) -> str:
    """Return *path* as a string relative to REPO_ROOT, preserving symlinks.

    ``Path.resolve()`` follows symlinks, so ``REPO_ROOT / "data" / "foo"``
    becomes e.g. ``/external-storage/.../data/foo`` which is outside REPO_ROOT.  This
    helper first tries non-resolving ``os.path.abspath``, then walks
    top-level repo symlinks to remap resolved paths back to their logical
    repo-relative form (e.g. ``data/conversation_profiles/...``).
    """
    # Fast path: logical (non-resolved) absolute path is under REPO_ROOT.
    try:
        return str(Path(os.path.abspath(path)).relative_to(Path(os.path.abspath(REPO_ROOT))))
    except ValueError:
        pass

    # The path was already resolved or constructed through a symlink target.
    resolved = path.resolve()
    for entry in sorted(REPO_ROOT.iterdir()):
        if not entry.is_symlink() or not entry.is_dir():
            continue
        target = entry.resolve()
        try:
            rel = resolved.relative_to(target)
            return str(Path(entry.name) / rel)
        except ValueError:
            continue

    # Last resort: relative to resolved REPO_ROOT.
    try:
        return str(resolved.relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def resolve_seed_cache_path(config_path: str, seed: int) -> str:
    """Return aida_cache_path with its seed directory set to inference_seed."""
    if not config_path:
        return config_path
    p = Path(config_path)
    parts = list(p.parts)
    for i, part in enumerate(parts):
        if part.startswith("seed") and part[4:].isdigit():
            parts[i] = f"seed{seed}"
            return str(Path(*parts))
    raise ValueError(
        f"aida_cache_path must include a seed<N> directory: {config_path}",
    )
