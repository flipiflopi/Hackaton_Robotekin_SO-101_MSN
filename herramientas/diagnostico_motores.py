"""Diagnostico del follower (solo lectura): ping, voltaje, temperatura, estado y par de cada motor.

Uso: python diagnostico_motores.py [COM13] [rondas]
"""
import sys
import time

from scservo_sdk import PacketHandler, PortHandler

NOMBRES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
# Bits del registro de estado (65) de los STS3215
BITS = {0: "voltaje", 1: "sensor", 2: "temperatura", 3: "corriente", 5: "sobrecarga"}


def leer(h, x, i, addr):
    v, com, err = h.read1ByteTxRx(x, i, addr)
    return v if com == 0 else None, err


def main():
    puerto = sys.argv[1] if len(sys.argv) > 1 else "COM13"
    rondas = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    h, x = PacketHandler(0), PortHandler(puerto)
    if not (x.openPort() and x.setBaudRate(1_000_000)):
        raise SystemExit(f"No se pudo abrir {puerto} (¿otro programa lo esta usando?)")
    fallos = {n: 0 for n in NOMBRES}
    try:
        for r in range(rondas):
            for i, n in enumerate(NOMBRES, 1):
                ping = h.ping(x, i)[1] == 0
                v, err = leer(h, x, i, 62)
                t, _ = leer(h, x, i, 63)
                st, _ = leer(h, x, i, 65)
                tq, _ = leer(h, x, i, 40)
                if not ping or None in (v, t, st, tq):
                    fallos[n] += 1
                alarmas = [b for k, b in BITS.items() if st is not None and st >> k & 1]
                print(f"r{r} {i} {n:13s} ping={'ok' if ping else 'FALLA'}  "
                      f"V={'?' if v is None else v / 10:>4}  T={'?' if t is None else t:>3}C  "
                      f"par={'?' if tq is None else tq}  alarmas={alarmas or '-'}  err_byte={err}")
            time.sleep(0.3)
    finally:
        x.closePort()
    print("\nLecturas fallidas por motor:", {n: f for n, f in fallos.items() if f} or "ninguna")


if __name__ == "__main__":
    main()
