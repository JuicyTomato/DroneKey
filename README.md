# dronepose

Tesi magistrale: keypoint dei droni → posa 6-DoF → predizione della traiettoria.
Piano: `docs/PIANO_DI_LAVORO.md`. Decisioni e fatti sui dati: `PROJECT_MEMORY.md`.

## Installazione
    python3 -m venv .venv
    .venv/bin/python -m pip install -e ".[dev]"

## Test
    .venv/bin/python -m pytest                                   # senza dataset
    SIXDRONESYN_ROOT=/Volumes/MakeMePasta/6DroneSyn .venv/bin/python -m pytest   # anche sul dataset vero

## Rigenerare i dati (serve il disco esterno)
    export SIXDRONESYN_ROOT=/Volumes/MakeMePasta/6DroneSyn
    .venv/bin/python scripts/build_drone_models.py
    .venv/bin/python scripts/build_canonical.py
    .venv/bin/python scripts/build_yolo_crops.py
    .venv/bin/python scripts/package_cache.py

Il dataset 6DroneSyn non è nel repo (licenza: vietata la redistribuzione). Neanche i PDF dei paper (`docs/papers/`, copyright IEEE): stanno su OneDrive.
