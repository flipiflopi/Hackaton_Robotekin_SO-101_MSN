"""Resume un log de lerobot-rollout: Hz del bucle y articulaciones recortadas por max_relative_target."""
import re
import sys
from collections import Counter
from datetime import datetime


def main():
    raw = open(sys.argv[1], "rb").read()
    texto = raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8", "replace")
    eventos = []  # (hora, tipo, dato)
    for m in re.finditer(r"WARNING (\S+ \S+) .*?running slower \(([\d.]+) Hz\)", texto):
        eventos.append((datetime.fromisoformat(m.group(1)), "hz", float(m.group(2))))
    for m in re.finditer(r"WARNING (\S+ \S+) \S+ Relative goal position .*?clamped to be safe\.\s*(\{.*?\}\})", texto, re.S):
        juntas = re.findall(r"'(\w+)': \{\s*'original goal_pos': ([-\d.e]+),\s*'safe goal_pos': ([-\d.e]+)", m.group(2))
        eventos.append((datetime.fromisoformat(m.group(1)), "clamp", [(j, float(o) - float(s)) for j, o, s in juntas]))
    eventos.sort(key=lambda e: e[0])

    # Separar ejecuciones: hueco > 8 s entre avisos
    runs, actual = [], []
    for e in eventos:
        if actual and (e[0] - actual[-1][0]).total_seconds() > 8:
            runs.append(actual)
            actual = []
        actual.append(e)
    runs.append(actual)

    for i, r in enumerate(runs, 1):
        hz = [d for _, t, d in r if t == "hz"]
        cl = [d for _, t, d in r if t == "clamp"]
        dur = (r[-1][0] - r[0][0]).total_seconds() + 1
        juntas, recorte = Counter(), Counter()
        for c in cl:
            for j, d in c:
                juntas[j] += 1
                recorte[j] += abs(d)
        hz_s = sorted(hz)
        print(f"Ejecucion {i}: {r[0][0]:%H:%M:%S}  ~{dur:.0f}s  avisos lentos={len(hz)}  "
              f"Hz min/mediana={hz_s[0] if hz else '-'}/{hz_s[len(hz_s)//2] if hz else '-'}  "
              f"<15Hz={sum(h < 15 for h in hz)}  clamps={len(cl)}")
        print("   recortes por articulacion (veces, grados medios recortados):",
              {j: (n, round(recorte[j] / n, 1)) for j, n in juntas.most_common()})


if __name__ == "__main__":
    main()
