"""Quita el par (torque) de todos los motores del follower para moverlo con la mano.

Uso: python liberar_robot.py [COM13]
Ojo: el brazo cae por su peso al soltarlo; sujetalo antes de ejecutar.
"""
import sys
import time

from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig

puerto = sys.argv[1] if len(sys.argv) > 1 else "COM13"
for intento in range(1, 4):
    robot = SOFollower(SOFollowerRobotConfig(port=puerto, id="mi_follower", disable_torque_on_disconnect=True))
    try:
        robot.connect(calibrate=False)  # configure() ya deja el par desactivado
        robot.bus.disable_torque()
        robot.disconnect()
        print("Robot liberado: ya puedes moverlo con la mano")
        break
    except ConnectionError as e:
        print(f"Fallo de comunicacion ({intento}/3): {e}")
        try:
            robot.bus.disconnect(False)
        except Exception:
            pass
        time.sleep(1)
else:
    sys.exit("No se pudo liberar el robot; desenchufa y vuelve a enchufar la alimentacion")
