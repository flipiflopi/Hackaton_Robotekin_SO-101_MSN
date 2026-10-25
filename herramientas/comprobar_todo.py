"""Chequeo antes de lanzar el robot: camaras (presentes, no congeladas, no intercambiadas) y motores.

Uso: python comprobar_todo.py [COM13]
Guarda chequeo_camaras.png con lo que ve cada indice.
"""
import subprocess
import sys
import time

import cv2
import numpy as np

ESPERADO = {0: "wrist (pinza)", 1: "top (C920)"}


def capturar(indice, segundos=2.0):
    cap = cv2.VideoCapture(indice)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    frames, t0 = [], time.time()
    while time.time() - t0 < segundos:
        ok, f = cap.read()
        if ok:
            frames.append(f)
    cap.release()
    return frames


def parece_top(frame):
    # La C920 ve la pared blanca arriba; la de la pinza ve sobre todo la esterilla oscura
    return frame[: frame.shape[0] // 3].mean() > frame[2 * frame.shape[0] // 3:].mean() + 25


def main():
    puerto = sys.argv[1] if len(sys.argv) > 1 else "COM13"
    ok_total, vistas = True, []
    for i, nombre in ESPERADO.items():
        frames = capturar(i)
        if len(frames) < 5:
            print(f"[FALLO] camara {i} ({nombre}): no da imagen. Revisa el USB (mejor directo al portatil, sin hub).")
            ok_total = False
            vistas.append(np.zeros((480, 640, 3), np.uint8))
            continue
        f = frames[-1]
        mov = float(np.abs(frames[-1].astype(int) - frames[len(frames) // 2].astype(int)).mean())
        es_top = parece_top(f)
        cambiada = (i == 1) != es_top
        estado = "OK"
        if cambiada:
            estado, ok_total = "PARECE INTERCAMBIADA (indices 0/1 cambiados)", False
        if f.mean() < 15:
            estado, ok_total = "IMAGEN NEGRA", False
        print(f"[{'OK' if estado == 'OK' else 'FALLO'}] camara {i} ({nombre}): {len(frames)} frames en 2 s, "
              f"brillo {f.mean():.0f}, ruido entre frames {mov:.1f} -> {estado}")
        vistas.append(cv2.resize(f, (640, 480)))
    cv2.imwrite("chequeo_camaras.png", np.hstack(vistas))
    print("Imagenes guardadas en chequeo_camaras.png (izquierda = indice 0, derecha = indice 1)\n")

    r = subprocess.run([sys.executable, "diagnostico_motores.py", puerto, "1"], capture_output=True, text=True)
    print(r.stdout.strip() or r.stderr.strip())
    if "FALLA" in r.stdout or r.returncode != 0:
        ok_total = False
    print("\n" + ("TODO OK: puedes lanzar el robot." if ok_total else "HAY PROBLEMAS: no lances el robot todavia."))


if __name__ == "__main__":
    main()
