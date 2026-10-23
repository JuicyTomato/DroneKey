"""Parameter counts of DroneKey (code vs paper wiring) against the paper's Tab. II / III -> experiments/dronekey/param_count.json."""
import json
from pathlib import Path

from dronepose.keypoints.dronekey.model import DroneKeyNet, count_params

OUT = Path("experiments/dronekey/param_count.json")
TAB2 = {2: 42.41, 3: 59.21, 6: 84.41, 8: 101.22, 10: 118.02, 12: 134.82}  # d = 1024; L=2 is a known exception
TAB3 = {128: 27.92, 256: 33.24, 512: 46.63, 768: 63.69, 1024: 84.42, 2048: 199.84}  # N = 6; d=2048 is a known exception
EXCEPTIONS = {("II", 2), ("III", 2048)}


def main() -> None:
    rows = []
    for table, cases in (("II", TAB2), ("III", TAB3)):
        for v, tab_m in cases.items():
            L, d = (v, 1024) if table == "II" else (6, v)
            row = {"table": table, "L": L, "d": d, "paper_table_M": tab_m, "known_exception": (table, v) in EXCEPTIONS}
            for wiring in ("code", "paper"):
                n = count_params(DroneKeyNet(d=d, n_layers=L, wiring=wiring, pretrained=False))
                row[f"{wiring}_wiring_M"] = round(n / 1e6, 3)
            row["code_gap_M"] = round(row["code_wiring_M"] - tab_m, 3)
            rows.append(row)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2))
    print(f"{'tab':>4} {'L':>3} {'d':>5} {'paper':>8} {'code':>8} {'gap':>7} {'paper-wiring':>13}")
    for r in rows:
        note = "  (known exception)" if r["known_exception"] else ""
        print(f"{r['table']:>4} {r['L']:>3} {r['d']:>5} {r['paper_table_M']:>8.2f} {r['code_wiring_M']:>8.2f} "
              f"{r['code_gap_M']:>+7.2f} {r['paper_wiring_M']:>13.2f}{note}")
    print(f"written {OUT}")


if __name__ == "__main__":
    main()
