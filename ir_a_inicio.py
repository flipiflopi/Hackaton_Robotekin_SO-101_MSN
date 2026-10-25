"""Lleva el follower, despacio, a la postura de inicio de las demos y lo deja ahi sujeto.

La red solo ha visto episodios que empiezan en esta postura (mediana de las 97 demos, pinza CERRADA).
Si arranca desde otra (pinza abierta, brazo a medio camino porque se corto un intento con Ctrl+C...),
la red recibe algo que no conoce y da ordenes raras: sube y baja, trompicones.

Uso: python ir_a_inicio.py [--puerto COM13] [--segundos 3] [--postura tumbada]
"""
import argparse
import time

import numpy as np

JUNTAS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
POSTURAS = {
    # Mediana del primer frame de los 97 episodios de trimrun/full (linterna de pie)
    "de_pie": [0.0, -101.5, 96.5, 74.0, -167.0, 1.0],
    # Mediana del primer frame de los 54 episodios de tumbada/full (muneca girada ~160 grados respecto a de_pie)
    "tumbada": [0.4, -103.1, 96.2, 53.4, -7.5, 0.6],
}


def main():
    from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig

    ap = argparse.ArgumentParser()
    ap.add_argument("--puerto", default="COM13")
    ap.add_argument("--segundos", type=float, default=3.0)
    ap.add_argument("--postura", choices=POSTURAS, default="de_pie")
    a = ap.parse_args()

    # A veces el motor 5 responde mal al reactivar el par tras configurar ("Incorrect status packet"):
    # es puntual, se reintenta
    for intento in range(1, 4):
        robot = SOFollower(SOFollowerRobotConfig(port=a.puerto, id="mi_follower", disable_torque_on_disconnect=False))
        try:
            robot.connect(calibrate=False)
            break
        except ConnectionError as e:
            print(f"Fallo de comunicacion al conectar (intento {intento}/3): {e}")
            try:
                robot.bus.disconnect(False)
            except Exception:
                pass
            if intento == 3:
                raise
            time.sleep(1)
    try:
        obs = robot.get_observation()
        q0 = np.array([obs[f"{j}.pos"] for j in JUNTAS])
        q1 = np.array(POSTURAS[a.postura])
        print("Ahora:  ", dict(zip(JUNTAS, q0.round(1))))
        print("Destino:", dict(zip(JUNTAS, q1.round(1))))
        pasos = max(1, int(a.segundos * 30))
        for k in range(1, pasos + 1):
            s = k / pasos
            s = s * s * (3 - 2 * s)  # arranque y parada suaves
            robot.send_action({f"{j}.pos": float(v) for j, v in zip(JUNTAS, q0 + s * (q1 - q0))})
            time.sleep(1 / 30)
        time.sleep(0.7)
        obs = robot.get_observation()
        q = np.array([obs[f"{j}.pos"] for j in JUNTAS])
        err = np.abs(q - q1)
        print("Llegado:", dict(zip(JUNTAS, q.round(1))))
        if err[:5].max() > 6:
            print(f"AVISO: {JUNTAS[int(err[:5].argmax())]} se ha quedado a {err[:5].max():.0f} grados (algo lo bloquea?)")
        else:
            print("OK: en postura de inicio.")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
