"""Fase 1 para vision_ibvs_esfera.py: lleva el brazo a la postura de inicio y ejecuta el lerobot-rollout oficial.

Lo llama ejecutar_fase1() con:
  --puerto --modelo --pasos --segundos --wrist --top --device [--encadenar] [--salida carpeta] [--inicio "q1,...,q6"]
Guarda toda la salida del rollout en <salida>/consola.log. Codigo de salida 0 si el rollout termino bien.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

JUNTAS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
# Mediana del primer frame de los 54 episodios de la linterna tumbada (tumbada/full)
INICIO_TUMBADA = "0.4,-103.1,96.2,53.4,-7.5,0.6"
# Antes de ir a la postura de inicio se ABRE la pinza (por si aún sujeta la linterna de la ejecución anterior:
# si no, el brazo la arrastra hasta el inicio y la pinza se queda bloqueada contra ella). Se cierra al llegar,
# porque en las demos la pinza empieza cerrada.
PINZA_ABIERTA = 35.0
PINZA_SOLTAR_S = 1.5          # espera con la pinza abierta antes de mover el brazo (que caiga la linterna)


def _mover(robot, q0, q1, segundos):
    pasos = max(1, int(segundos * 30))
    for k in range(1, pasos + 1):
        s = k / pasos
        s = s * s * (3 - 2 * s)  # arranque y parada suaves
        robot.send_action({f"{j}.pos": float(v) for j, v in zip(JUNTAS, q0 + s * (q1 - q0))})
        time.sleep(1 / 30)


def ir_a_inicio(puerto, q1, segundos=3.0):
    from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig

    # El motor 5 a veces responde mal al reactivar el par ("Incorrect status packet"): se reintenta
    for intento in range(1, 4):
        robot = SOFollower(SOFollowerRobotConfig(port=puerto, id="mi_follower", disable_torque_on_disconnect=False))
        try:
            robot.connect(calibrate=False)
            break
        except ConnectionError as e:
            print(f"Fallo de comunicacion al conectar (intento {intento}/3): {e}", flush=True)
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
        print("Ahora:  ", dict(zip(JUNTAS, q0.round(1))), flush=True)
        print("Destino:", dict(zip(JUNTAS, q1.round(1))), flush=True)
        # 1) abrir la pinza donde está (suelta la linterna si la tenía cogida)
        abierta = np.array(q0)
        abierta[5] = max(q0[5], PINZA_ABIERTA)
        print(f"Abro la pinza ({q0[5]:.1f} -> {abierta[5]:.1f}) por si sujeta la linterna", flush=True)
        _mover(robot, q0, abierta, 0.8)
        time.sleep(PINZA_SOLTAR_S)
        # 2) brazo a la postura de inicio con la pinza abierta
        destino = np.array(q1)
        destino[5] = abierta[5]
        _mover(robot, abierta, destino, segundos)
        # 3) cerrar la pinza (las demos empiezan con la pinza cerrada)
        _mover(robot, destino, q1, 0.8)
        time.sleep(0.7)
        obs = robot.get_observation()
        err = np.abs(np.array([obs[f"{j}.pos"] for j in JUNTAS]) - q1)
        if err[:5].max() > 6:
            print(f"AVISO: {JUNTAS[int(err[:5].argmax())]} se ha quedado a {err[:5].max():.0f} grados", flush=True)
        else:
            print("OK: en postura de inicio.", flush=True)
    finally:
        robot.disconnect()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--puerto", default="COM13")
    ap.add_argument("--modelo", required=True)
    ap.add_argument("--pasos", type=int, default=60)
    ap.add_argument("--segundos", type=float, default=20)
    ap.add_argument("--wrist", type=int, default=0)
    ap.add_argument("--top", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--encadenar", action="store_true", help="al acabar no vuelve a reposo y sigue sujetando")
    ap.add_argument("--salida", default=os.path.join("logs", "fase1"))
    ap.add_argument("--inicio", default=INICIO_TUMBADA, help='postura de inicio, "q1,...,q6" en grados')
    # Con la linterna cogida, la red sigue pidiendo cerrar (~0) y el limite deja la orden 'limite' por debajo
    # de donde esta la pinza: con 10 el motor 6 aprieta tanto que salta su proteccion de sobrecarga y deja de
    # responder. Con 3 aprieta menos y sigue cerrando igual de rapido (3 x 30 Hz = 90 %/s)
    ap.add_argument("--limite-pinza", type=float, default=3)
    a = ap.parse_args()

    salida = Path(a.salida)
    salida.mkdir(parents=True, exist_ok=True)
    ir_a_inicio(a.puerto, np.array([float(x) for x in a.inicio.split(",")]))

    exe = Path(sys.executable).with_name("lerobot-rollout.exe")
    cmd = [str(exe if exe.exists() else "lerobot-rollout"), "--strategy.type=base",
           f"--policy.path={a.modelo}", f"--policy.n_action_steps={a.pasos}", f"--policy.device={a.device}",
           "--robot.max_relative_target={shoulder_pan: 10, shoulder_lift: 10, elbow_flex: 10, wrist_flex: 10, "
           f"wrist_roll: 10, gripper: {a.limite_pinza}}}",
           "--robot.type=so101_follower", f"--robot.port={a.puerto}", "--robot.id=mi_follower",
           f"--robot.cameras={{ wrist: {{type: opencv, index_or_path: {a.wrist}, width: 640, height: 480, fps: 30}}, "
           f"top: {{type: opencv, index_or_path: {a.top}, width: 640, height: 480, fps: 30}} }}",
           "--task=Coger la linterna tumbada", f"--duration={a.segundos}"]
    if a.encadenar:
        cmd += ["--return_to_initial_position=false", "--robot.disable_torque_on_disconnect=false"]
    env = dict(os.environ, HF_HUB_OFFLINE="1", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")

    # Mismo reintento que demo_fase1.ps1: si el motor 5 falla al conectar, el rollout aborta sin moverse
    for intento in range(1, 4):
        print("$ " + " ".join(cmd), flush=True)
        texto = []
        with open(salida / "consola.log", "a", encoding="utf-8") as log:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                                 text=True, encoding="utf-8", errors="replace", bufsize=1)
            for linea in p.stdout:
                print(linea.rstrip(), flush=True)
                log.write(linea)
                texto.append(linea)
            p.wait()
        t = "".join(texto)
        if "Incorrect status packet" in t and "control loop started" not in t:
            print(f"Fallo puntual de comunicacion con un motor al conectar; reintentando ({intento}/3)...", flush=True)
            time.sleep(1)
            continue
        if a.encadenar and p.returncode == 0:
            aflojar_pinza(a.puerto)
        sys.exit(p.returncode)
    sys.exit(1)


def aflojar_pinza(puerto, apriete=1.5):
    """Tras el rollout encadenado la pinza se queda con la ultima orden (cerrar del todo) mientras sujeta la
    linterna: deja la orden justo por debajo de donde esta para que sujete sin forzar el motor 6 hasta el IBVS.
    Sin configure(): eso quitaria el par a todo el brazo y la linterna se caeria."""
    from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig

    robot = SOFollower(SOFollowerRobotConfig(port=puerto, id="mi_follower", disable_torque_on_disconnect=False))
    try:
        robot.bus.connect(handshake=False)
        ahora = float(robot.bus.read("Present_Position", "gripper", num_retry=3))
        robot.bus.write("Goal_Position", "gripper", max(0.0, ahora - apriete), num_retry=3)
        print(f"Pinza: sujeta en {ahora:.1f} (orden {max(0.0, ahora - apriete):.1f}, sin forzar el motor)", flush=True)
    except Exception as e:
        print(f"AVISO: no se pudo aflojar la pinza ({e})", flush=True)
    finally:
        try:
            robot.bus.disconnect(False)  # False: mantener el par
        except Exception:
            pass


if __name__ == "__main__":
    main()
