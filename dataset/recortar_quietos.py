"""Crea una copia de un dataset LeRobot quitando los frames quietos del principio de cada episodio.

Inicio del movimiento = primer frame en el que algún motor de la acción se aleja más de --umbral
(grados, o % en la pinza) de la postura del primer frame. Se conservan --margen frames antes.
El dataset original no se modifica.

Uso:
  python recortar_quietos.py --src-repo FlipiFlopi007/so101_fase1_completo --src-root ctest/orig \
      --dst-repo FlipiFlopi007/so101_fase1_recortado --dst-root trimrun/viejo
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import VideoEncodingManager

SKIP_KEYS = {"timestamp", "frame_index", "episode_index", "index", "task_index"}


def onset_frames(root, umbral, margen, margen_final):
    """Devuelve {episodio: (primer_frame, fin_exclusivo)} leyendo solo los parquet de datos.

    Fin del movimiento = último frame que se aleja más de 'umbral' de la postura final; se conservan
    'margen_final' frames de sujeción quieta después."""
    files = sorted(Path(root, "data").rglob("*.parquet"))
    df = pd.concat(pd.read_parquet(f, columns=["episode_index", "frame_index", "action"]) for f in files)
    out = {}
    for ep, g in df.sort_values(["episode_index", "frame_index"]).groupby("episode_index"):
        act = np.stack(g["action"].values)
        moved = np.abs(act - act[0]).max(axis=1) > umbral
        onset = int(np.argmax(moved)) if moved.any() else 0
        moving_end = np.abs(act - act[-1]).max(axis=1) > umbral
        last_move = int(np.where(moving_end)[0].max()) if moving_end.any() else len(act) - 1
        out[int(ep)] = (max(0, onset - margen), min(len(act), last_move + 1 + margen_final))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-repo", required=True)
    ap.add_argument("--src-root", required=True)
    ap.add_argument("--dst-repo", required=True)
    ap.add_argument("--dst-root", required=True)
    ap.add_argument("--umbral", type=float, default=3.0, help="Grados (o %% pinza) que cuentan como movimiento")
    ap.add_argument("--margen", type=int, default=6, help="Frames que se conservan antes del movimiento (6 = 0,2 s)")
    ap.add_argument("--margen-final", type=int, default=30,
                    help="Frames de sujeción quieta que se conservan al final (30 = 1 s)")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--max-episodes", type=int, default=None, help="Solo para pruebas")
    ap.add_argument("--solo-informe", action="store_true", help="Calcula los recortes sin escribir nada")
    args = ap.parse_args()

    src = LeRobotDataset(args.src_repo, root=args.src_root)
    fps = src.meta.fps
    cuts = onset_frames(args.src_root, args.umbral, args.margen, args.margen_final)
    lengths = {int(e["episode_index"]): int(e["length"]) for e in src.meta.episodes}
    ini = np.array([c[0] for c in cuts.values()]) / fps
    fin = np.array([lengths[ep] - c[1] for ep, c in cuts.items()]) / fps
    queda = np.array([c[1] - c[0] for c in cuts.values()]) / fps
    print(f"{len(cuts)} episodios · se quitan de media {ini.mean():.1f} s del principio y {fin.mean():.1f} s del final "
          f"· quedan {queda.mean():.1f} s por episodio (mín {queda.min():.1f} s)", flush=True)
    for ep, (start, end) in cuts.items():
        if (end - start) / fps < 4:
            print(f"  aviso: el episodio {ep} queda en solo {(end - start) / fps:.1f} s", flush=True)
    if args.solo_informe:
        return

    features = {k: dict(v) for k, v in src.meta.features.items() if k not in SKIP_KEYS}
    for v in features.values():
        v.pop("info", None)
    dst_root = Path(args.dst_root)
    if dst_root.exists():
        shutil.rmtree(dst_root)
    dst = LeRobotDataset.create(
        args.dst_repo, fps=fps, root=dst_root, robot_type=src.meta.robot_type, features=features,
        use_videos=True, image_writer_threads=4 * len(src.meta.camera_keys),
        # La compresión por lotes falla en lerobot 0.6.1: se comprime en streaming
        streaming_encoding=True, encoder_threads=2,
    )
    camera_keys = src.meta.camera_keys
    episodes = src.meta.episodes
    n_eps = src.meta.total_episodes if args.max_episodes is None else min(args.max_episodes, src.meta.total_episodes)

    with VideoEncodingManager(dst):
        for ep in range(n_eps):
            first = episodes[ep]["dataset_from_index"]
            start = first + cuts[ep][0]
            last = first + cuts[ep][1]
            loader = DataLoader(torch.utils.data.Subset(src, range(start, last)), batch_size=None,
                                shuffle=False, num_workers=args.num_workers)
            for item in loader:
                frame = {"task": item["task"]}
                for key in features:
                    if key in camera_keys:
                        frame[key] = (item[key].permute(1, 2, 0) * 255).round().to(torch.uint8).numpy()
                    else:
                        frame[key] = item[key].numpy()
                dst.add_frame(frame)
            dst.save_episode()
            print(f"Episodio {ep}: {cuts[ep][0] / fps:.1f} s quitados al principio, "
                  f"{(episodes[ep]['dataset_to_index'] - last) / fps:.1f} s al final -> {last - start} frames", flush=True)

    dst.finalize()
    print(f"Hecho: {dst.meta.total_episodes} episodios recortados en {dst_root}")


if __name__ == "__main__":  # obligatorio en Windows: los workers del DataLoader reimportan este archivo
    main()
