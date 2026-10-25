"""Ayuda a devolver la camara top a la posicion de las demos.

Muestra en directo la imagen de la C920 superpuesta (50/50) con un frame de las demos, y el desplazamiento
estimado en pixeles. Mueve la camara con la mano hasta que la imagen deje de verse doble y dx, dy ~ 0.
Uso: python alinear_camara.py [--camara 1] [--episodio 20]     (q o cerrar la ventana = salir)
"""
import argparse
import glob
import time
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd

from iluminar import Camara, Vista

# Dataset con el que se compara la cámara top (el de la linterna tumbada, descargado de Hugging Face con
# hf download FlipiFlopi007/so101_tumbada_full --repo-type dataset --local-dir dataset_tumbada)
DATASET = Path(r"C:\lerobot-work\tumbada\full")
if not DATASET.is_dir():
    DATASET = Path(__file__).resolve().parent.parent / "dataset_tumbada"


def frame_demo(ep):
    eps = pd.read_parquet(glob.glob(str(DATASET / "meta/episodes/*/*.parquet"))[0])
    r = eps[eps.episode_index == ep].iloc[0]
    k = "videos/observation.images.top/"
    p = DATASET / f"videos/observation.images.top/chunk-{int(r[k + 'chunk_index']):03d}/file-{int(r[k + 'file_index']):03d}.mp4"
    ts = float(r[k + "from_timestamp"])
    c = av.open(str(p))
    s = c.streams.video[0]
    c.seek(int(ts / s.time_base), stream=s)
    for fr in c.decode(s):
        if fr.time >= ts - 0.02:
            img = fr.to_ndarray(format="bgr24")
            break
    c.close()
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camara", type=int, default=1)
    ap.add_argument("--episodio", type=int, default=20)
    ap.add_argument("--referencia", default=str(Path(__file__).with_name("referencia_top_demo.png")),
                    help="imagen de referencia; se usa si no esta el dataset (p. ej. en otro PC)")
    a = ap.parse_args()

    ref = frame_demo(a.episodio) if DATASET.exists() else cv2.imread(a.referencia)
    gris = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)[:200]  # pared/mesa: parte estatica
    cam = Camara(a.camara, 640, 480, None)
    vista = Vista()
    try:
        while True:
            f = cam.leer(despues_de=time.perf_counter() - 0.05)
            (dx, dy), conf = cv2.phaseCorrelate(gris(ref), gris(f))
            mezcla = cv2.addWeighted(ref, 0.5, f, 0.5, 0)
            ok = abs(dx) < 3 and abs(dy) < 3
            txt = f"dx={dx:+.0f}px dy={dy:+.0f}px  " + ("ALINEADA" if ok else "mueve la camara hasta que no se vea doble")
            cv2.putText(mezcla, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if ok else (0, 0, 255), 2)
            if vista.mostrar(mezcla) == "q":
                break
    finally:
        cam.cerrar()
        vista.cerrar()


if __name__ == "__main__":
    main()
