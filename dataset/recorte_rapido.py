"""Recorta los frames quietos del principio/final de cada episodio SIN volver a codificar los vídeos.

Cada episodio guarda en meta/episodes en qué segundo de su archivo de vídeo empieza (from_timestamp) y
termina (to_timestamp). Se desplazan esos límites y se eliminan/renumeran las filas de data/*.parquet;
los mp4 no se tocan. Después hay que recalcular las estadísticas (lerobot-edit-dataset recompute_stats)
y verificar con verificar_recorte.py.

Uso: python recorte_rapido.py <carpeta_origen> <carpeta_destino>
"""

import json
import shutil
import sys
from pathlib import Path

import pandas as pd

from recortar_quietos import onset_frames

UMBRAL, MARGEN, MARGEN_FINAL = 3.0, 6, 30  # 3 grados, 0,2 s antes del movimiento, 1 s de sujeción final


def main(src, dst):
    src, dst = Path(src), Path(dst)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    info = json.load(open(dst / "meta" / "info.json"))
    fps = info["fps"]
    video_keys = [k for k, v in info["features"].items() if v["dtype"] == "video"]

    cuts = onset_frames(dst, UMBRAL, MARGEN, MARGEN_FINAL)  # {ep: (inicio, fin_exclusivo)} en frames

    # 1) Tablas de datos: quitar filas y renumerar frame_index / timestamp / index global
    data_files = sorted((dst / "data").rglob("*.parquet"))
    tables = {f: pd.read_parquet(f) for f in data_files}
    new_len, keep = {}, {}
    for f, df in tables.items():
        lo = df["episode_index"].map(lambda e: cuts[int(e)][0])
        hi = df["episode_index"].map(lambda e: cuts[int(e)][1])
        mask = (df["frame_index"] >= lo) & (df["frame_index"] < hi)
        df = df[mask].copy()
        df["frame_index"] = df["frame_index"] - df["episode_index"].map(lambda e: cuts[int(e)][0])
        df["timestamp"] = (df["frame_index"] / fps).astype("float32")
        keep[f] = df
    order = sorted(cuts)  # episodios en orden; el índice global sigue ese orden
    start_index, total = {}, 0
    for ep in order:
        start_index[ep] = total
        new_len[ep] = cuts[ep][1] - cuts[ep][0]
        total += new_len[ep]
    for f, df in keep.items():
        df["index"] = df["episode_index"].map(lambda e: start_index[int(e)]) + df["frame_index"]
        df.reset_index(drop=True).to_parquet(f)

    # 2) Metadata de episodios: longitud, índices globales y ventana de vídeo
    for f in sorted((dst / "meta" / "episodes").rglob("*.parquet")):
        df = pd.read_parquet(f)
        for i, row in df.iterrows():
            ep = int(row["episode_index"])
            start, end = cuts[ep]
            df.at[i, "length"] = new_len[ep]
            df.at[i, "dataset_from_index"] = start_index[ep]
            df.at[i, "dataset_to_index"] = start_index[ep] + new_len[ep]
            for k in video_keys:
                t0 = float(row[f"videos/{k}/from_timestamp"]) + start / fps
                df.at[i, f"videos/{k}/from_timestamp"] = t0
                df.at[i, f"videos/{k}/to_timestamp"] = t0 + new_len[ep] / fps
        df.to_parquet(f)

    info["total_frames"] = total
    json.dump(info, open(dst / "meta" / "info.json", "w"), indent=4)
    quitado = sum(c[0] + (sum(1 for _ in []) or 0) for c in cuts.values())
    print(f"Recortado: {len(cuts)} episodios, {total} frames (antes {sum(len(t) for t in tables.values())})")
    json.dump({str(k): v for k, v in cuts.items()}, open(dst / "meta" / "recortes.json", "w"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
