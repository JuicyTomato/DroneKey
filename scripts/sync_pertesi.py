"""Copy the files of each work phase into PerTesi/<chapter>/ (local, git-ignored) to ease writing the thesis.

Only copies: never creates or overwrites README.md files (those are written by hand), never deletes anything.
Re-run it whenever the source files change.   Usage: python scripts/sync_pertesi.py [--out PerTesi]
"""
import argparse
import shutil
from pathlib import Path

SKIP = {"__pycache__", ".DS_Store"}
S = "src/dronepose"
# chapter -> sub-folder -> list of sources (file, directory or glob relative to the repo root)
# a source written as ("path", N) copies only the first N lines of a text file (sample of a big dataset)
PLAN = {
    "0_Contesto_e_piano": {
        "documenti": ["PROJECT_MEMORY.md", "docs/PIANO_DI_LAVORO.md", "README.md", "pyproject.toml", "requirements-lock.txt"],
    },
    "1_Audit_del_dataset": {
        "codice": ["scripts/audit"],
        "figure": ["docs/figures"],
        "documenti": ["docs/1mailDraft.txt", "docs/2response.txt", "docs/3response.txt"],
    },
    "2_Formato_canonico_e_infrastruttura": {
        "codice": [f"{S}/common", f"{S}/data", f"{S}/backup", "scripts/make_manifest.py", "scripts/verify_manifest.py",
                   "scripts/build_drone_models.py", "scripts/build_canonical.py", "scripts/build_yolo_crops.py",
                   "scripts/package_cache.py"],
        "test": ["tests/conftest.py", "tests/fixtures", "tests/test_geometry.py", "tests/test_sixdronesyn.py",
                 "tests/test_drone_models.py", "tests/test_canonical.py", "tests/test_crops_export.py",
                 "tests/test_manifest.py", "tests/test_real_dataset.py"],
        "dati": ["configs/drone_models.json", "data/manifest", ("data/processed/canonical_val.jsonl", 30)],
        "documenti": ["docs/superpowers/plans/2026-10-12-fase0-scheletro.md"],
    },
    "3_Scheletro_end_to_end": {
        "codice": [f"{S}/pose/pnp.py", f"{S}/eval/metrics.py", f"{S}/trajectory", f"{S}/keypoints/predictors.py",
                   f"{S}/pipeline/skeleton.py", f"{S}/viz", "scripts/run_skeleton.py", "scripts/kaggle"],
        "test": ["tests/test_pnp.py", "tests/test_baselines.py", "tests/test_pipeline.py", "tests/test_kaggle_paths.py"],
        "risultati": ["experiments/skeleton_gt", "experiments/skeleton_noise2", "experiments/skeleton_out"],
        "addestramento_yolo": ["experiments/skeleton_run"],
        "documenti": ["docs/kaggle/README.md"],
    },
    "4_Disambiguazione_IPPE_e_sensibilita": {
        "codice": [f"{S}/pose/disambiguation.py", f"{S}/pose/replay.py", f"{S}/pose/sensitivity.py",
                   f"{S}/keypoints/export.py", "scripts/replay_disambiguation.py", "scripts/sensitivity_pnp.py",
                   "scripts/diagnose_disambiguation.py", "scripts/kaggle/export_keypoints.py"],
        "test": ["tests/test_disambiguation.py", "tests/test_replay.py", "tests/test_sensitivity.py",
                 "tests/test_keypoint_export.py"],
        "risultati": ["experiments/replay_val", "experiments/replay_14_03_09", "experiments/replay_check",
                      "experiments/sensitivity", ("experiments/keypoints/keypoints_val.jsonl", 200)],
        "documenti": ["docs/superpowers/plans/2026-10-14-disambiguazione-sensibilita.md",
                      "docs/superpowers/handoff/2026-10-15-fix-temporal-reject.md"],
    },
    "5_Strategia_globale": {
        "codice": [f"{S}/pose/global_disambiguation.py", f"{S}/pose/tuning.py", "scripts/tune_global.py",
                   "scripts/compare_strategies.py", "scripts/plot_strategy_summary.py"],
        "test": ["tests/test_global_disambiguation.py", "tests/test_tuning.py"],
        "risultati": ["experiments/replay_val_global", "experiments/tune_global", "experiments/tune_global_ext",
                      "experiments/summary"],
        "video_e_fotogrammi": ["experiments/compare"],
        "documenti": ["docs/superpowers/specs/2026-10-15-global-disambiguation-design.md",
                      "docs/superpowers/plans/2026-10-15-global-disambiguation.md"],
    },
    "6_Drone_08_e_simmetria": {
        "codice": [f"{S}/eval/metrics.py", "scripts/audit/drone08_probe.py"],
        "test": ["tests/test_symmetry_metric.py"],
        "risultati": ["experiments/summary/drone08_probe_output.txt"],
    },
    "7_Fase3_dataset_e_baseline": {
        "codice": [f"{S}/forecast", "configs/forecast.json"],
        "script": ["scripts/forecast"],
        "test": ["tests/forecast"],
        "risultati": ["experiments/forecast/tuning", "experiments/forecast/baselines_val"],
        "video_e_fotogrammi": ["experiments/forecast/videos"],
        "documenti": ["docs/superpowers/specs/2026-10-16-fase3-dataset-baseline-design.md",
                      "docs/superpowers/plans/2026-10-16-fase3-dataset-baseline.md"],
    },
}


def copy_one(src: Path, dst_dir: Path) -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    if src.is_dir():
        for p in sorted(src.rglob("*")):
            if p.is_file() and not (set(p.parts) & SKIP) and not p.name.startswith("._"):
                t = dst_dir / src.name / p.relative_to(src)
                t.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, t)
                n += 1
    else:
        shutil.copy2(src, dst_dir / src.name)
        n = 1
    return n


ap = argparse.ArgumentParser()
ap.add_argument("--out", default="PerTesi")
args = ap.parse_args()
out, total, missing = Path(args.out), 0, []
for chapter, folders in PLAN.items():
    for folder, sources in folders.items():
        for item in sources:
            path, head = (item if isinstance(item, tuple) else (item, None))
            src = Path(path)
            if not src.exists():
                missing.append(path)
                continue
            if head:                                       # first N lines of a big text file, named *_prime_N_righe
                dst = out / chapter / folder
                dst.mkdir(parents=True, exist_ok=True)
                lines = src.read_text().splitlines()[:head]
                (dst / f"{src.stem}_prime_{head}_righe{src.suffix}").write_text("\n".join(lines) + "\n")
                total += 1
            else:
                total += copy_one(src, out / chapter / folder)
size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e6
print(f"copiati {total} file in {out}/ ({size:.0f} MB)")
if missing:
    print("sorgenti mancanti:", ", ".join(missing))
