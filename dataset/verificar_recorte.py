"""Comprueba que cada frame del dataset recortado es idéntico (imágenes y acciones) al frame
correspondiente del original. Uso: python verificar_recorte.py <original> <recortado>"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main(raw_root, trim_root):
    cuts = {int(k): v for k, v in json.load(open(Path(trim_root) / "meta" / "recortes.json")).items()}
    raw = LeRobotDataset("raw", root=raw_root)
    trim = LeRobotDataset("trim", root=trim_root)
    cams = raw.meta.camera_keys
    malos = 0
    for ep in sorted(cuts):
        start, end = cuts[ep]
        n = end - start
        assert trim.meta.episodes[ep]["length"] == n, (ep, trim.meta.episodes[ep]["length"], n)
        raw0 = raw.meta.episodes[ep]["dataset_from_index"]
        tr0 = trim.meta.episodes[ep]["dataset_from_index"]
        for k in (0, n // 2, n - 1):  # primer frame, mitad y último de cada episodio
            a, b = raw[raw0 + start + k], trim[tr0 + k]
            ok = int(a["episode_index"]) == int(b["episode_index"]) == ep
            ok &= torch.equal(a["action"], b["action"]) and torch.equal(a["observation.state"], b["observation.state"])
            for c in cams:
                ok &= torch.equal(a[c], b[c])
            if not ok:
                malos += 1
                print(f"DIFERENTE: episodio {ep}, frame {k}")
    print(f"Verificación: {len(cuts)} episodios x 3 frames, {malos} diferencias")
    sys.exit(1 if malos else 0)


if __name__ == "__main__":  # obligatorio en Windows
    main(sys.argv[1], sys.argv[2])
