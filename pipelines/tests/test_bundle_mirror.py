"""Drift test: web/api/_crew/bundle must be byte-identical to models/crew-v1.

Only pipelines/export_model_bundle.py writes either copy. If this fails, someone regenerated or edited
one copy: run `python pipelines/export_model_bundle.py --mirror-only --force` (or re-export both).
"""
import hashlib
import pathlib

import pytest
from _crew.bundle import DEFAULT_BUNDLE_DIR, load_bundle

ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE = ROOT / "models" / "crew-v1"


def _digests(directory: pathlib.Path) -> dict[str, str]:
    return {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(directory.iterdir()) if f.is_file()}


def test_mirror_has_the_same_files_and_bytes_as_the_bundle():
    assert DEFAULT_BUNDLE_DIR.is_dir(), "mirror missing: run pipelines/export_model_bundle.py --mirror-only"
    assert _digests(DEFAULT_BUNDLE_DIR) == _digests(SOURCE)


@pytest.mark.parametrize("directory", [SOURCE, DEFAULT_BUNDLE_DIR], ids=["original", "mirror"])
def test_no_bundle_file_contains_a_carriage_return(directory):
    """Bundle files are LF only (.gitattributes -text): a CR changes the bytes git stores and bundle_sha256."""
    files = sorted(f for f in directory.iterdir() if f.is_file())
    assert len(files) == 5
    assert [f.name for f in files if b"\r" in f.read_bytes()] == []


def test_function_loads_the_mirror_by_default():
    assert load_bundle().sha256 == load_bundle(SOURCE).sha256


def test_mirror_refuses_to_overwrite_without_force(tmp_path):
    from export_model_bundle import mirror_bundle

    destination = tmp_path / "bundle"
    assert len(mirror_bundle(SOURCE, destination)) == 5
    with pytest.raises(FileExistsError):
        mirror_bundle(SOURCE, destination)
    assert _digests(destination) == _digests(SOURCE)
