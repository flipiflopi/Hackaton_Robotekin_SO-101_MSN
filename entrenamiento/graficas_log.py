"""Graficas de entrenamiento a partir del log de lerobot-train (sin GPU, sin dataset).

Uso: python graficas_log.py <log> <carpeta_salida> [--ref <log_referencia>]
"""
import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TQDM_RE = re.compile(r"(\d+)/(\d+) \[")
TRAIN_RE = re.compile(  # \s+ entre campos: la consola parte la linea en sitios distintos segun el ancho
    r"step:(\S+)\s+smpl:\S+\s+ep:\S+\s+epch:([\d.]+)\s+loss:([\d.]+)\s+grdn:([\d.]+)\s+"
    r"lr:([\d.e+-]+)\s+updt_s:([\d.]+)\s+data_s:([\d.]+)\s+smp/s:([\d.]+)\s+mem_gb:([\d.]+)\s+"
    r"l1_loss:([\d.]+)\s+kld_loss:([\d.]+)"
)
EVAL_RE = re.compile(r"step (\d+): eval_loss=([\d.]+)")
CAMPOS = ["epoch", "loss", "grdn", "lr", "updt_s", "data_s", "smp_s", "mem_gb", "l1", "kld"]


def leer(path):
    raw = Path(path).read_bytes()
    texto = raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8", "replace")
    # El log redondea el paso ("step:12K"), asi que se saca de la barra de tqdm
    # anterior: N/T con T = pasos restantes al (re)arrancar -> paso global = N + total - T
    barras = [(m.start(), int(m.group(1)), int(m.group(2))) for m in TQDM_RE.finditer(texto)]
    total = max(b[2] for b in barras)
    train, j = {}, 0
    for m in TRAIN_RE.finditer(texto):
        while j + 1 < len(barras) and barras[j + 1][0] < m.start():
            j += 1
        _, n, t = barras[j]
        paso = round((n + total - t) / 100) * 100
        # Al reanudar se repiten pasos: nos quedamos con la ultima aparicion
        train[paso] = dict(zip(CAMPOS, map(float, m.groups()[1:])))
    evals = {int(m.group(1)): float(m.group(2)) for m in EVAL_RE.finditer(texto)}
    return dict(sorted(train.items())), dict(sorted(evals.items()))


def suavizar(y, k=5):
    return [sum(y[max(0, i - k + 1): i + 1]) / len(y[max(0, i - k + 1): i + 1]) for i in range(len(y))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("out")
    ap.add_argument("--ref", help="log de otra red para comparar (p. ej. act_fase1)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    train, evals = leer(a.log)
    ref_train, ref_evals = leer(a.ref) if a.ref else ({}, {})
    if ref_train:
        tope = max(train) + 4000
        ref_train = {k: v for k, v in ref_train.items() if k <= tope}
        ref_evals = {k: v for k, v in ref_evals.items() if k <= tope}
    pasos = list(train)
    nom = Path(a.log).stem
    nom_ref = Path(a.ref).stem if a.ref else ""
    col = lambda c, d=train: [d[s][c] for s in d]

    fig, ax = plt.subplots(3, 3, figsize=(18, 13))
    fig.suptitle(f"Entrenamiento {Path(a.log).stem}  (hasta paso {pasos[-1]})", fontsize=15)

    # 1. Overfitting: L1 de entrenamiento vs validacion
    g = ax[0, 0]
    g.plot(pasos, col("l1"), alpha=0.25, color="C0")
    g.plot(pasos, suavizar(col("l1")), color="C0", label="L1 entrenamiento (suavizada)")
    g.plot(list(evals), list(evals.values()), "o-", color="C3", label="eval_loss (validación)")
    if ref_evals:
        g.plot(list(ref_evals), list(ref_evals.values()), "s--", color="gray", alpha=0.7, label=f"eval_loss {nom_ref}")
    g.axhline(0.221, color="gray", ls=":", label="0,221 = mejor de act_fase1")
    g.set_title("Entrenamiento vs validación (overfitting)")
    g.set_ylim(0, 0.5)
    g.legend(fontsize=8)

    # 2. Hueco validacion - entrenamiento
    g = ax[0, 1]
    l1s = dict(zip(pasos, suavizar(col("l1"))))
    hueco = [(s, v - l1s[min(l1s, key=lambda p: abs(p - s))]) for s, v in evals.items()]
    g.bar([s for s, _ in hueco], [h for _, h in hueco], width=900, color="C3")
    g.set_title("Hueco eval_loss − L1 entrenamiento (crece = memoriza)")

    # 3. eval_loss ampliada
    g = ax[0, 2]
    g.plot(list(evals), list(evals.values()), "o-", color="C3", label=nom)
    if ref_evals:
        g.plot(list(ref_evals), list(ref_evals.values()), "s--", color="gray", label=nom_ref)
    for s, v in evals.items():
        g.annotate(f"{v:.3f}", (s, v), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
    best = min(evals, key=evals.get)
    g.axvline(best, color="green", ls=":", label=f"mejor: paso {best}")
    g.axhline(0.221, color="gray", ls=":")
    g.set_title("eval_loss (validación)")
    g.legend(fontsize=8)

    # 4. Loss total
    g = ax[1, 0]
    g.plot(pasos, col("loss"), alpha=0.3, color="C1")
    g.plot(pasos, suavizar(col("loss")), color="C1", label=nom)
    if ref_train:
        g.plot(list(ref_train), suavizar(col("loss", ref_train)), color="gray", alpha=0.7, label=nom_ref)
    g.set_yscale("log")
    g.set_title("Loss total = L1 + 10·KLD (log)")
    g.legend(fontsize=8)

    # 5. KLD
    g = ax[1, 1]
    g.plot(pasos, col("kld"), color="C4")
    g.set_yscale("log")
    g.set_title("KLD (parte VAE de ACT)")

    # 6. Norma del gradiente
    g = ax[1, 2]
    g.plot(pasos, col("grdn"), alpha=0.3, color="C2")
    g.plot(pasos, suavizar(col("grdn")), color="C2")
    g.set_yscale("log")
    g.set_title("Norma del gradiente")

    # 7. Learning rate
    g = ax[2, 0]
    g.plot(pasos, col("lr"), color="C5")
    g.set_title("Learning rate")

    # 8. Tiempos
    g = ax[2, 1]
    g.plot(pasos, col("updt_s"), label="update (GPU) s/paso")
    g.plot(pasos, col("data_s"), label="carga datos s/paso")
    g.set_title("Tiempos por paso")
    g.legend(fontsize=8)

    # 9. Memoria GPU y velocidad
    g = ax[2, 2]
    g.plot(pasos, col("mem_gb"), color="C6", label="memoria GPU (GB)")
    g2 = g.twinx()
    g2.plot(pasos, col("smp_s"), color="C7", alpha=0.6, label="muestras/s")
    g.set_title("Memoria GPU y velocidad")
    g.legend(loc="upper left", fontsize=8)
    g2.legend(loc="upper right", fontsize=8)

    for g in ax.flat:
        g.set_xlabel("paso")
        g.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "metricas_entreno.png", dpi=110)

    with open(out / "resumen.txt", "w", encoding="utf-8") as f:
        f.write("paso\teval_loss\tL1_train(suav)\thueco\n")
        for s, h in hueco:
            f.write(f"{s}\t{evals[s]:.4f}\t{evals[s] - h:.4f}\t{h:.4f}\n")
        if ref_evals:
            f.write(f"\n{nom_ref} (referencia):\n")
            for s, v in ref_evals.items():
                f.write(f"{s}\t{v:.4f}\n")
    print((out / "resumen.txt").read_text(encoding="utf-8"))
    print("Ultimo paso:", pasos[-1], train[pasos[-1]])


if __name__ == "__main__":
    main()
