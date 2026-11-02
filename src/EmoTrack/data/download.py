"""Fetch the released core conversations and frozen splits, without research outputs."""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory

import zstandard
from huggingface_hub import hf_hub_download

from EmoTrack.data.bundles import CONVERSATION_FILES, require_complete_bundle

REPO_ID = "hiddensev/LongCounsel-8"
REVISION = "ad75f7e2e4248b055764487586a0d6590877fd5f"


def _checked_file(filename: str, checksum: str) -> Path:
    path = Path(
        hf_hub_download(REPO_ID, filename, repo_type="dataset", revision=REVISION)
    )
    _require_checksum(path, checksum)
    return path


def _require_checksum(path: Path, checksum: str) -> None:
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if digest != checksum:
        raise ValueError(f"Checksum mismatch for {path}")


def extract_archive(archive: Path, destination: Path, prefix: str) -> int:
    """Extract regular core files; reject traversal, links and unexpected assets."""
    count = 0
    with (
        archive.open("rb") as compressed,
        zstandard.ZstdDecompressor().stream_reader(compressed) as stream,
        tarfile.open(fileobj=stream, mode="r|") as bundle,
    ):
        for member in bundle:
            path = PurePosixPath(member.name)
            if (
                path.is_absolute()
                or ".." in path.parts
                or not path.parts
                or path.parts[0] != prefix
            ):
                raise ValueError(f"Unsafe archive path: {member.name}")
            relative = PurePosixPath(*path.parts[1:])
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError(
                    f"Links and special files are not allowed: {member.name}"
                )
            if (
                len(relative.parts) != 3
                or relative.parts[1] != "common"
                or relative.name not in CONVERSATION_FILES
            ):
                raise ValueError(f"Unexpected dataset asset: {member.name}")
            target = destination.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            source = bundle.extractfile(member)
            if source is None:
                raise ValueError(f"Cannot read {member.name}")
            with source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
            count += 1
    return count


def download_dataset(name: str, *, destination: Path) -> Path:
    manifest_path = hf_hub_download(
        REPO_ID, "manifest.json", repo_type="dataset", revision=REVISION
    )
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest.get("schema_version") != 3:
        raise ValueError("Expected dataset manifest version 3")
    entry = manifest["datasets"][name]
    receipt = {
        "repo_id": REPO_ID,
        "revision": REVISION,
        "archive_sha256": entry["archive_sha256"],
    }
    marker = destination / "download.json"
    if marker.is_file() and json.loads(marker.read_text()) == receipt:
        split = destination / "train_val_test_split.json"
        _require_checksum(split, entry["split_sha256"])
        require_complete_bundle(destination, split, CONVERSATION_FILES)
        print(f"Already downloaded: {destination}")
        return destination
    if destination.exists():
        raise FileExistsError(
            f"{destination} already exists; use it directly or move it before downloading."
        )
    archive = _checked_file(entry["archive"], entry["archive_sha256"])
    split = _checked_file(entry["split_file"], entry["split_sha256"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".download-", dir=destination.parent) as temporary:
        staging = Path(temporary) / name
        staging.mkdir()
        count = extract_archive(archive, staging, entry["archive_root"])
        if count != entry["core_files"]:
            raise ValueError(
                f"Expected {entry['core_files']} core files, found {count}"
            )
        shutil.copyfile(split, staging / "train_val_test_split.json")
        require_complete_bundle(
            staging, staging / "train_val_test_split.json", CONVERSATION_FILES
        )
        (staging / "download.json").write_text(json.dumps(receipt, indent=2) + "\n")
        staging.rename(destination)
    print(f"Downloaded {count} files to {destination}")
    return destination
