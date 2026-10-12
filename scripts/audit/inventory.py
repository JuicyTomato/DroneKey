"""Inventory of 6DroneSyn: files on disk vs annotation entries."""
import json, os, re, sys, hashlib
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/Volumes/MakeMePasta/6DroneSyn")


def real(p):  # skip macOS AppleDouble files
    return not p.name.startswith("._")


rows = []
problems = []
for scene_dir in sorted(ROOT.glob("scene_*")):
    inner = [p for p in scene_dir.iterdir() if p.is_dir() and real(p)]
    if len(inner) != 1 or inner[0].name != scene_dir.name:
        problems.append(f"{scene_dir.name}: inner folder(s) = {[p.name for p in inner]}")
    for inner_dir in inner:
        for sub in sorted(p for p in inner_dir.iterdir() if p.is_dir() and real(p)):
            annos = [p for p in sub.glob("*_anno.json") if real(p)]
            if len(annos) != 1:
                problems.append(f"{sub}: {len(annos)} anno files")
                continue
            anno = json.load(open(annos[0]))
            keys = [k for k in anno if k != "camera_parameters"]
            # images on disk
            disk = {}
            for bg in sorted(p for p in sub.iterdir() if p.is_dir() and real(p)):
                for img in bg.glob("*.png"):
                    if real(img):
                        disk[f"{bg.name}_{img.stem}"] = img
            ann_ids = set(keys)
            missing_img = ann_ids - set(disk)
            missing_ann = set(disk) - ann_ids
            # path field consistency
            path_scenes = Counter(anno[k]["path"].split("/")[1] for k in keys)
            bgs = Counter(anno[k]["background"] for k in keys)
            degs = Counter(anno[k]["degree"] for k in keys)
            dtypes = Counter(anno[k]["drone_type"] for k in keys)
            stypes = Counter(anno[k]["seq_type"] for k in keys)
            rows.append(dict(scene=scene_dir.name, inner=inner_dir.name, sub=sub.name,
                             n_ann=len(keys), n_img=len(disk),
                             miss_img=len(missing_img), miss_ann=len(missing_ann),
                             path_scene=dict(path_scenes), bg=dict(bgs), deg=dict(degs),
                             dtype=dict(dtypes), stype=dict(stypes),
                             K=anno["camera_parameters"]["intrinsic"]["camera_matrix_K"],
                             res=anno["camera_parameters"]["intrinsic"]["resolution"],
                             cam_loc=anno["camera_parameters"]["extrinsic"]["location"],
                             cam_rot=anno["camera_parameters"]["extrinsic"]["rotation_euler"]))

print(f"{'scene':9}{'inner':10}{'sub':7}{'ann':>5}{'img':>5}{'mImg':>5}{'mAnn':>5}  path_scene / bg / degree / drone_type / seq_type")
for r in rows:
    print(f"{r['scene']:9}{r['inner']:10}{r['sub']:7}{r['n_ann']:5}{r['n_img']:5}{r['miss_img']:5}{r['miss_ann']:5}  "
          f"{r['path_scene']} {r['bg']} {r['deg']} {r['dtype']} {r['stype']}")

print("\nTOTAL annotations:", sum(r["n_ann"] for r in rows), " images:", sum(r["n_img"] for r in rows))
print("Distinct K:", {json.dumps(r["K"]) for r in rows}.__len__(), " distinct res:", {tuple(r["res"]) for r in rows})
print("Distinct camera loc/rot:", Counter((tuple(r["cam_loc"]), tuple(r["cam_rot"])) for r in rows))
print("\nPROBLEMS:")
for p in problems:
    print(" -", p)
