from dronepose.backup.manifest import (build_manifest, read_manifest, sha256_file,
                                       verify_manifest, write_manifest)


def _make_tree(root):
    (root / "a").mkdir(parents=True)
    (root / "a" / "x.json").write_text("{}")
    (root / "a" / "y.png").write_bytes(b"\x89PNG")
    (root / "a" / "._y.png").write_bytes(b"junk")
    (root / ".DS_Store").write_bytes(b"junk")


def test_sha256_known_value(tmp_path):
    f = tmp_path / "f"
    f.write_bytes(b"abc")
    assert sha256_file(f) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_manifest_skips_macos_junk_and_roundtrips(tmp_path):
    root = tmp_path / "ds"
    _make_tree(root)
    m = build_manifest(root)
    assert sorted(m) == ["a/x.json", "a/y.png"]
    write_manifest(m, tmp_path / "M.sha256")
    assert read_manifest(tmp_path / "M.sha256") == m


def test_verify_detects_missing_changed_extra(tmp_path):
    root = tmp_path / "ds"
    _make_tree(root)
    m = build_manifest(root)
    (root / "a" / "x.json").write_text('{"changed": 1}')
    (root / "a" / "y.png").unlink()
    (root / "a" / "z.txt").write_text("new")
    report = verify_manifest(root, m)
    assert report == {"missing": ["a/y.png"], "changed": ["a/x.json"], "extra": ["a/z.txt"]}


def test_excluded_suffixes_are_not_read(tmp_path):
    root = tmp_path / "ds"
    _make_tree(root)
    (root / "a" / "big.zip").write_bytes(b"zip")
    assert "a/big.zip" not in build_manifest(root, exclude_suffixes=(".zip",))
    assert "a/big.zip" in verify_manifest(root, build_manifest(root, exclude_suffixes=(".zip",)))["extra"]
    assert verify_manifest(root, build_manifest(root, exclude_suffixes=(".zip",)),
                           exclude_suffixes=(".zip",)) == {"missing": [], "changed": [], "extra": []}


def test_build_manifest_resumes_from_existing(tmp_path):
    root = tmp_path / "ds"
    _make_tree(root)
    fake = {"a/x.json": "precomputed"}
    m = build_manifest(root, existing=fake)
    assert m["a/x.json"] == "precomputed"      # non ricalcolato
    assert m["a/y.png"] == sha256_file(root / "a" / "y.png")
