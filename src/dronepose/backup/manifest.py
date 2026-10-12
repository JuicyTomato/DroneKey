"""SHA-256 manifest of a directory tree, to verify backups and detect corruption."""
import hashlib
from pathlib import Path

_JUNK = {".DS_Store"}


def _is_junk(path: Path) -> bool:
    return path.name.startswith("._") or path.name in _JUNK


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def _files(root: Path, exclude_suffixes: tuple[str, ...] = ()):
    for p in sorted(root.rglob("*")):
        if p.is_file() and not _is_junk(p) and not p.name.endswith(exclude_suffixes):
            yield p.relative_to(root).as_posix(), p


def build_manifest(root: Path, existing: dict[str, str] | None = None,
                   exclude_suffixes: tuple[str, ...] = ()) -> dict[str, str]:
    existing = existing or {}
    return {rel: existing.get(rel) or sha256_file(p) for rel, p in _files(root, exclude_suffixes)}


def write_manifest(manifest: dict[str, str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rel in sorted(manifest):
            f.write(f"{manifest[rel]}  {rel}\n")


def read_manifest(path: Path) -> dict[str, str]:
    out = {}
    with open(path) as f:
        for line in f:
            digest, rel = line.rstrip("\n").split("  ", 1)
            out[rel] = digest
    return out


def verify_manifest(root: Path, manifest: dict[str, str],
                    exclude_suffixes: tuple[str, ...] = ()) -> dict[str, list[str]]:
    on_disk = dict(_files(root, exclude_suffixes))
    missing = sorted(set(manifest) - set(on_disk))
    extra = sorted(set(on_disk) - set(manifest))
    changed = sorted(rel for rel in set(manifest) & set(on_disk) if sha256_file(on_disk[rel]) != manifest[rel])
    return {"missing": missing, "changed": changed, "extra": extra}
