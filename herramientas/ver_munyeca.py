"""Muestra en directo el giro de muñeca (wrist_roll) del leader y del follower, para alinearlos antes de grabar.

Solo lee: no mueve nada ni toca el par. Ctrl+C para salir.
Uso: python ver_munyeca.py [COM_follower] [COM_leader]
"""
import sys
import time

from scservo_sdk import PacketHandler, PortHandler

DEMOS = -167  # valor tipico de wrist_roll en las demos (grados)


def grados(raw):
    return (raw - 2047.5) * 360 / 4095


def main():
    pf = sys.argv[1] if len(sys.argv) > 1 else "COM13"
    pl = sys.argv[2] if len(sys.argv) > 2 else "COM14"
    h = PacketHandler(0)
    puertos = {}
    for n, p in (("follower", pf), ("leader", pl)):
        x = PortHandler(p)
        x.openPort()
        x.setBaudRate(1_000_000)
        puertos[n] = x
    print(f"Objetivo: los dos cerca de {DEMOS} grados (como en las demos). OJO: el borde es -180/+180;")
    print("si el leader pasa de -180 salta a +180 y el follower daria casi una vuelta entera.\n")
    try:
        while True:
            v = {n: grados(h.read2ByteTxRx(x, 5, 56)[0]) for n, x in puertos.items()}
            dif = v["leader"] - v["follower"]
            estado = "OK" if abs(dif) < 10 and abs(v["leader"] - DEMOS) < 20 else "gira el leader"
            print(f"\rfollower {v['follower']:+7.1f}   leader {v['leader']:+7.1f}   diferencia {dif:+7.1f}   {estado:15s}", end="")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print()
    finally:
        for x in puertos.values():
            x.closePort()


if __name__ == "__main__":
    main()
