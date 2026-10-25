"""Resume cada intento de un log de rollout: modelo, n_action_steps, Hz, recortes y si la pinza sujeta algo.

Uso: python resumen_intentos.py [log] [desde "AAAA-MM-DD HH:MM"]
"""
import re
import sys
from collections import Counter


def main():
    log = sys.argv[1] if len(sys.argv) > 1 else r"C:\lerobot-work\outputs\rollout_demo.log"
    desde = sys.argv[2] if len(sys.argv) > 2 else ""
    raw = open(log, "rb").read()
    t = raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8", "replace")
    # demo_fase1.ps1 escribe antes de cada intento: "=== fecha hora modelo=... pasos=N"
    cabeceras = [(m.start(), m.group(1), m.group(2)) for m in re.finditer(r"=== \S+ \S+ modelo=(\S+) pasos=(\d+)", t)]
    pos = 0
    for b in re.split(r"(?=INFO \S+ \S+ \S+ Building rollout context)", t):
        inicio, pos = pos, pos + len(b)
        hora = re.search(r"INFO (\S+ \S+) \S+ Building rollout context", b)
        if not hora or hora.group(1) < desde:
            continue
        previas = [c for c in cabeceras if c[0] < inicio]
        if previas:
            ruta, pasos = previas[-1][1], previas[-1][2]
            partes = ruta.replace("\\pretrained_model", "").split("\\")
            modelo = f"{partes[-1]} p{pasos}"
        else:
            modelo = "?"
        hz = sorted(float(x) for x in re.findall(r"running slower \(([\d.]+) Hz", b))
        cl = re.findall(r"'(\w+)': \{\s*'original goal_pos': ([-\d.e]+),\s*'safe goal_pos': ([-\d.e]+)", b)
        c = Counter(j for j, _, _ in cl)
        # pinza pidiendo cerrar (<5) sin poder: esta sujetando la linterna
        sujeta = sum(1 for j, o, s in cl if j == "gripper" and float(o) < 5 and float(s) > float(o) + 1)
        estado = "ERROR" if "Traceback" in b else ("ok" if "Rollout finished" in b else "cortado")
        print(f"{hora.group(1)[11:]}  {modelo:>28s}  avisos_lentos={len(hz):2d} "
              f"mediana={hz[len(hz) // 2] if hz else 0:5.1f}Hz  recortes={dict(c)}  "
              f"pinza_sujetando={sujeta:3d}  {estado}")


if __name__ == "__main__":
    main()
