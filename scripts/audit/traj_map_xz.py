import json
from pathlib import Path
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path("/Volumes/MakeMePasta/6DroneSyn")
scenes = {"03": "Mini3", "09": "Mini2", "06": "Mavic3", "07": "Air2"}
fig, axs = plt.subplots(4, 7, figsize=(21, 11))
for i, sc in enumerate(scenes):
    for j, p in enumerate(sorted(q for q in ROOT.glob(f"scene_{sc}/*/*/*_anno.json") if not q.name.startswith("._"))):
        a = json.load(open(p)); es = [e for k, e in a.items() if k != "camera_parameters"]
        bg = es[0]["background"]; es = sorted([e for e in es if e["background"] == bg], key=lambda e: e["filename"])
        T = np.array([e["drone"]["3Dtranslation(camera)"] for e in es])
        ax = axs[i, j]; ax.plot(T[:, 0], T[:, 2], "k-"); ax.plot(T[0, 0], T[0, 2], "go"); ax.plot(T[-1, 0], T[-1, 2], "rs")
        ax.set_title(f"scene {sc} drone {es[0]['drone_type']}", fontsize=10); ax.set_aspect("equal", "datalim"); ax.set_xticks([]); ax.set_yticks([])
    axs[i, 0].set_ylabel(f"paper: {scenes[sc]}", fontsize=12)
plt.tight_layout(); plt.savefig(S := str(Path(__file__).parent / "vis" / "traj_map_xz.png"), dpi=60); print(S)
