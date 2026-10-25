"""Fase 2: apuntar la linterna (ya sujeta por el robot) a un punto de la PCB con control visual.

Sin deep learning: se detecta el centro del haz en la camara top, se compara con el punto objetivo
(en pixeles) y se corrigen las articulaciones en tiempo real. La relacion "grados de cada
articulacion -> pixeles que se mueve el haz" (jacobiano) se mide al empezar con pequenos movimientos
de prueba y se va actualizando sola, asi que da igual como haya quedado la linterna en la pinza.

Modos:
  referencia Guarda la imagen sin haz que se usa para detectarlo (hacer al montar la escena).
  foto       Guarda una imagen de depuracion (haz detectado y mascara). No usa el robot.
  pose       Sin par motor: mueves el follower a mano y muestra la --pose-inicial a copiar.
  ver        Ventana en directo para ajustar umbral/exposicion. No usa el robot. Tecla q = salir.
  controlar  Conecta el follower y apunta el haz al objetivo hasta Ctrl+C (o --duracion).

Ejemplos:
  python iluminar.py referencia
  python iluminar.py ver --objetivo 320,240
  python iluminar.py controlar --objetivo punto.json --puerto COM13 --sin-configurar
"""
import argparse
import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np

JUNTAS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


# --------------------------------------------------------------------------- camara

class Camara:
    """Lee la camara en un hilo y entrega siempre el ultimo frame (sin frames atrasados)."""

    def __init__(self, indice, ancho, alto, exposicion):
        self.cap = cv2.VideoCapture(indice, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, ancho)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, alto)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if exposicion is not None:
            # Exposicion manual: con la automatica el brillo cambia al entrar el haz y la deteccion baila
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            self.cap.set(cv2.CAP_PROP_EXPOSURE, exposicion)
        if not self.cap.isOpened():
            raise RuntimeError(f"No se pudo abrir la camara {indice}")
        self.frame, self.t = None, 0.0
        self.vivo = True
        self.hilo = threading.Thread(target=self._bucle, daemon=True)
        self.hilo.start()
        while self.frame is None:
            time.sleep(0.01)

    def _bucle(self):
        while self.vivo:
            ok, f = self.cap.read()
            if ok:
                self.frame, self.t = f, time.perf_counter()

    def leer(self, despues_de=0.0):
        """Ultimo frame; si se pasa despues_de, espera a uno capturado despues de ese instante."""
        while self.t <= despues_de:
            time.sleep(0.002)
        return self.frame.copy()

    def cerrar(self):
        self.vivo = False
        self.hilo.join(timeout=1)
        self.cap.release()


# --------------------------------------------------------------------------- deteccion del haz

def blancura(frame):
    """min(R, G, B) suavizado: alto solo si los tres canales son altos (luz blanca)."""
    return cv2.GaussianBlur(frame, (9, 9), 0).min(axis=2).astype(np.int16)


def detectar_haz(frame, objetivo, args, ref=None):
    """Devuelve ((x, y), mascara) del centro del haz o (None, mascara).

    Luz blanca sobre PCB verde: en el haz los tres canales suben; en la PCB el rojo y el azul son
    bajos y el brazo naranja no tiene azul. Con imagen de referencia (sin haz) se mide cuanto ha
    subido la blancura, lo que ignora la etiqueta blanca de la PCB y la pared. Se coge la mancha
    mas grande, primero cerca del objetivo y, si no hay, en toda la imagen.
    """
    b = blancura(frame)
    if ref is not None:
        b = np.clip(b - ref, 0, 255)
    umbral = args.umbral_dif if ref is not None else args.umbral
    mascara = (b >= umbral).astype(np.uint8)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    zonas = [None]
    if objetivo is not None and args.radio_roi > 0:
        roi = np.zeros_like(mascara)
        cv2.circle(roi, (int(objetivo[0]), int(objetivo[1])), args.radio_roi, 1, -1)
        zonas = [roi, None]
    for roi in zonas:
        m = mascara if roi is None else mascara * roi
        n, etiquetas, stats, _ = cv2.connectedComponentsWithStats(m)
        if n <= 1:
            continue
        i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        if stats[i, cv2.CC_STAT_AREA] < args.area_min:
            continue
        ys, xs = np.nonzero(etiquetas == i)
        w = b[ys, xs].astype(np.float64) - umbral + 1  # centroide ponderado por brillo
        return (float((xs * w).sum() / w.sum()), float((ys * w).sum() / w.sum())), mascara
    return None, mascara


def dibujar(frame, mascara, objetivo, haz, texto=""):
    vista = frame.copy()
    vista[mascara > 0] = (0.5 * vista[mascara > 0] + (0, 0, 127)).astype(np.uint8)
    if objetivo is not None:
        cv2.drawMarker(vista, (int(objetivo[0]), int(objetivo[1])), (0, 255, 0), cv2.MARKER_CROSS, 30, 2)
    if haz is not None:
        cv2.circle(vista, (int(haz[0]), int(haz[1])), 8, (0, 0, 255), 2)
        if objetivo is not None:
            cv2.line(vista, (int(haz[0]), int(haz[1])), (int(objetivo[0]), int(objetivo[1])), (255, 255, 0), 1)
    cv2.putText(vista, texto, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return vista


class Vista:
    """Ventana en directo con matplotlib (el OpenCV de LeRobot no trae ventanas)."""

    def __init__(self):
        import matplotlib

        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt

        self.plt = plt
        plt.ion()
        self.fig, ax = plt.subplots(figsize=(9, 7))
        ax.axis("off")
        self.img = None
        self.ax = ax
        self.tecla = None
        self.fig.canvas.mpl_connect("key_press_event", lambda ev: setattr(self, "tecla", ev.key))
        self.fig.canvas.mpl_connect("close_event", lambda ev: setattr(self, "tecla", "q"))

    def mostrar(self, bgr):
        rgb = bgr[:, :, ::-1]
        if self.img is None:
            self.img = self.ax.imshow(rgb)
        else:
            self.img.set_data(rgb)
        self.fig.canvas.draw_idle()
        self.plt.pause(0.001)
        t, self.tecla = self.tecla, None
        return t

    def cerrar(self):
        self.plt.close(self.fig)


def leer_objetivo(valor):
    if valor is None:
        return None
    if Path(valor).is_file():
        d = json.loads(Path(valor).read_text(encoding="utf-8"))
        return float(d["u"]), float(d["v"])
    u, v = valor.split(",")
    return float(u), float(v)


# --------------------------------------------------------------------------- robot

def cargar_referencia(args, cam):
    """Blancura de la imagen sin haz: del archivo --referencia o, si no existe, del frame actual."""
    if args.sin_referencia:
        return None
    if Path(args.referencia).is_file():
        return blancura(cv2.imread(args.referencia))
    print(f"AVISO: no existe {args.referencia}; tomo la referencia ahora (el haz NO debe tocar la PCB)")
    return blancura(cam.leer(despues_de=time.perf_counter()))


def conectar_robot(args):
    from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig

    cfg = SOFollowerRobotConfig(
        port=args.puerto,
        id="mi_follower",
        max_relative_target=args.paso_max * 2,
        disable_torque_on_disconnect=False,  # al salir sigue sujetando la linterna
    )
    robot = SOFollower(cfg)
    if args.sin_configurar:
        # connect() normal quita el par motor para configurar los motores -> el brazo caeria con la
        # linterna. Si la fase 1 acaba de dejarlo sujetando, los motores ya estan configurados.
        robot.configure = lambda: None
    robot.connect(calibrate=False)
    return robot


def leer_pose(args):
    """Quita el par y muestra las articulaciones mientras se mueve el follower a mano."""
    from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig

    robot = SOFollower(SOFollowerRobotConfig(port=args.puerto, id="mi_follower"))
    robot.connect(calibrate=False)
    robot.bus.disable_torque()
    print("Par quitado: mueve el follower con la mano (sujeta la linterna). Ctrl+C para terminar.")
    try:
        while True:
            q = posicion(robot)
            print("\r--pose-inicial " + ",".join(f"{v:.1f}" for v in q[:5]) + "    ", end="")
            time.sleep(0.2)
    except KeyboardInterrupt:
        print()
    finally:
        robot.disconnect()


def posicion(robot):
    obs = robot.get_observation()
    return np.array([obs[f"{j}.pos"] for j in JUNTAS], dtype=np.float64)


def mandar(robot, q):
    robot.send_action({f"{j}.pos": float(v) for j, v in zip(JUNTAS, q)})


def medir_haz(cam, objetivo, args, ref, n=3):
    """Mediana de n detecciones con frames nuevos (filtra ruido)."""
    puntos = []
    for _ in range(n):
        p, _ = detectar_haz(cam.leer(despues_de=time.perf_counter()), objetivo, args, ref)
        if p is not None:
            puntos.append(p)
    return np.median(np.array(puntos), axis=0) if len(puntos) >= (n + 1) // 2 else None


def estimar_jacobiano(robot, cam, objetivo, q_cmd, idx, args, ref):
    """Mueve cada articulacion +-delta y mide cuantos px se desplaza el haz por grado."""
    J = np.zeros((2, len(idx)))
    for c, k in enumerate(idx):
        medidas = []
        for signo in (+1, -1):
            q = q_cmd.copy()
            q[k] += signo * args.delta_prueba
            mandar(robot, q)
            time.sleep(args.espera_prueba)
            medidas.append(medir_haz(cam, objetivo, args, ref))
        mandar(robot, q_cmd)
        time.sleep(args.espera_prueba)
        if medidas[0] is None or medidas[1] is None:
            raise RuntimeError(f"El haz se sale de la imagen al mover {JUNTAS[k]}; reduce --delta-prueba "
                               "o acerca la pose inicial al objetivo")
        J[:, c] = (medidas[0] - medidas[1]) / (2 * args.delta_prueba)
        print(f"  {JUNTAS[k]:13s}: {J[0, c]:+6.1f} px/grado en x, {J[1, c]:+6.1f} px/grado en y")
    return J


def controlar(args, objetivo, cam):
    if objetivo is None:
        raise SystemExit("Falta --objetivo")
    idx = [JUNTAS.index(j) for j in args.juntas.split(",")]
    robot = conectar_robot(args)
    vista = Vista() if args.ventana else None
    try:
        # Referencia sin haz: en este momento la linterna NO debe estar iluminando la PCB
        ref = cargar_referencia(args, cam)
        q_cmd = posicion(robot)
        q_cmd[5] = max(0.0, q_cmd[5] - args.apriete)  # pinza: un poco mas cerrada para no soltarla
        if args.pose_inicial:
            destino = np.array([float(x) for x in args.pose_inicial.split(",")] + [q_cmd[5]])
            print("Yendo a la pose inicial...")
            for a in np.linspace(0, 1, 60):  # ~2 s, suave
                mandar(robot, q_cmd + a * (destino - q_cmd))
                time.sleep(1 / 30)
            q_cmd = destino
            time.sleep(0.5)
        mandar(robot, q_cmd)

        if medir_haz(cam, objetivo, args, ref) is None:
            raise RuntimeError("No veo el haz. Usa el modo 'ver' para ajustar --umbral/--exposicion, "
                               "o pasa una --pose-inicial que apunte hacia la PCB")
        print("Midiendo jacobiano (pequenos movimientos de prueba)...")
        J = estimar_jacobiano(robot, cam, objetivo, q_cmd, idx, args, ref)
        if np.linalg.svd(J, compute_uv=False)[-1] < 0.5:
            print("AVISO: con estas articulaciones el haz casi no se mueve en alguna direccion. "
                  "Prueba a anadir wrist_roll o shoulder_lift en --juntas.")

        lim = {k: (q_cmd[k] - args.rango, q_cmd[k] + args.rango) for k in idx}
        t0, perdido, centrado = time.perf_counter(), 0, 0
        p_prev, q_prev = None, None
        print("Controlando. Ctrl+C para parar (el brazo sigue sujetando la linterna).")
        while args.duracion <= 0 or time.perf_counter() - t0 < args.duracion:
            t_ciclo = time.perf_counter()
            frame = cam.leer(despues_de=t_ciclo - 0.02)
            haz, mascara = detectar_haz(frame, objetivo, args, ref)
            if haz is None:
                perdido += 1
                if perdido == int(args.hz):  # 1 s sin haz: volver a la ultima posicion buena
                    print("Haz perdido; vuelvo a la ultima posicion buena")
                    if q_prev is not None:
                        q_cmd = q_prev.copy()
                        mandar(robot, q_cmd)
                texto = "HAZ NO DETECTADO"
            else:
                perdido = 0
                haz = np.array(haz)
                # Actualizacion de Broyden: corrige J con lo que el haz se ha movido de verdad
                if p_prev is not None:
                    dq = (posicion(robot) - q_prev)[idx]
                    if np.linalg.norm(dq) > 0.3:
                        dp = haz - p_prev
                        J += args.broyden * np.outer(dp - J @ dq, dq) / (dq @ dq)
                e = np.array(objetivo) - haz
                err = float(np.linalg.norm(e))
                centrado = centrado + 1 if err < args.tolerancia else 0
                q_prev, p_prev = posicion(robot), haz
                if err > args.tolerancia / 2:
                    # Minimos cuadrados amortiguados: dq = K * J^T (J J^T + l I)^-1 e
                    dq = args.ganancia * J.T @ np.linalg.solve(J @ J.T + args.amortiguacion * np.eye(2), e)
                    dq = np.clip(dq, -args.paso_max, args.paso_max)
                    for c, k in enumerate(idx):
                        q_cmd[k] = np.clip(q_cmd[k] + dq[c], *lim[k])
                    mandar(robot, q_cmd)
                texto = f"error {err:5.1f} px" + ("  ILUMINADO" if centrado >= args.hz / 2 else "")
            if vista is not None and vista.mostrar(dibujar(frame, mascara, objetivo, haz, texto)) == "q":
                break
            if int((time.perf_counter() - t0) * 2) % 2 == 0 and haz is not None:
                print(f"\r{texto:40s}", end="")
            time.sleep(max(0.0, 1 / args.hz - (time.perf_counter() - t_ciclo)))
    except KeyboardInterrupt:
        pass
    finally:
        print("\nParado. El brazo se queda sujetando la linterna.")
        robot.disconnect()
        if vista is not None:
            vista.cerrar()


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("modo", choices=["referencia", "foto", "ver", "pose", "controlar"])
    ap.add_argument("--objetivo", help="'u,v' en pixeles o JSON con claves u, v")
    ap.add_argument("--camara", type=int, default=1, help="indice OpenCV de la camara top")
    ap.add_argument("--ancho", type=int, default=640)
    ap.add_argument("--alto", type=int, default=480)
    ap.add_argument("--exposicion", type=float, default=None,
                    help="exposicion manual (DirectShow, p. ej. -6); sin valor = automatica")
    ap.add_argument("--umbral", type=int, default=200, help="blancura minima (0-255) sin referencia")
    ap.add_argument("--umbral-dif", type=int, default=35, help="subida minima de blancura respecto a la referencia")
    ap.add_argument("--sin-referencia", action="store_true", help="detectar por blancura absoluta")
    ap.add_argument("--referencia", default="referencia_sin_haz.png",
                    help="imagen sin haz (modo referencia la crea); si no existe se toma al arrancar")
    ap.add_argument("--area-min", type=int, default=150, help="area minima de la mancha en px")
    ap.add_argument("--radio-roi", type=int, default=130, help="buscar primero a esta distancia del objetivo")
    ap.add_argument("--salida", default="iluminar_foto.png")
    # Control
    ap.add_argument("--puerto", default="COM13")
    ap.add_argument("--sin-configurar", action="store_true",
                    help="no quitar el par al conectar (usar justo despues de la fase 1)")
    ap.add_argument("--juntas", default="shoulder_pan,wrist_flex", help="articulaciones que se mueven")
    ap.add_argument("--pose-inicial", help="5 valores pan,lift,elbow,w_flex,w_roll para apuntar a la PCB")
    ap.add_argument("--apriete", type=float, default=2.0, help="grados extra de cierre de la pinza")
    ap.add_argument("--hz", type=float, default=15)
    ap.add_argument("--ganancia", type=float, default=0.4)
    ap.add_argument("--amortiguacion", type=float, default=5.0)
    ap.add_argument("--paso-max", type=float, default=1.5, help="grados maximos por ciclo y articulacion")
    ap.add_argument("--rango", type=float, default=40, help="grados maximos que se aleja cada articulacion")
    ap.add_argument("--tolerancia", type=float, default=10, help="px de error para considerar iluminado")
    ap.add_argument("--broyden", type=float, default=0.2)
    ap.add_argument("--delta-prueba", type=float, default=3.0)
    ap.add_argument("--espera-prueba", type=float, default=0.5)
    ap.add_argument("--duracion", type=float, default=0, help="segundos (0 = hasta Ctrl+C)")
    ap.add_argument("--ventana", action="store_true", help="mostrar la camara durante el control")
    args = ap.parse_args()

    if args.modo == "pose":
        leer_pose(args)
        return
    objetivo = leer_objetivo(args.objetivo)
    cam = Camara(args.camara, args.ancho, args.alto, args.exposicion)
    try:
        time.sleep(1.0)  # dejar que se estabilice la exposicion
        if args.modo == "controlar":
            controlar(args, objetivo, cam)
            return
        if args.modo == "referencia":
            input("Linterna apagada o apuntando fuera de la PCB, nadie delante. ENTER para guardar...")
            cv2.imwrite(args.referencia, cam.leer(despues_de=time.perf_counter()))
            print("Referencia guardada en", args.referencia)
            return
        ref = cargar_referencia(args, cam)
        if args.modo == "foto":
            frame = cam.leer(despues_de=time.perf_counter())
            haz, mascara = detectar_haz(frame, objetivo, args, ref)
            cv2.imwrite(args.salida, dibujar(frame, mascara, objetivo, haz, f"haz={haz}"))
            print("Haz:", haz, "| guardado en", args.salida)
            return
        print("En la ventana: q = salir, +/- = umbral, r = nueva referencia")
        vista = Vista()
        while True:
            frame = cam.leer(despues_de=time.perf_counter() - 0.02)
            haz, mascara = detectar_haz(frame, objetivo, args, ref)
            u = args.umbral_dif if ref is not None else args.umbral
            txt = "-" if haz is None else f"{haz[0]:.0f},{haz[1]:.0f}"
            tecla = vista.mostrar(dibujar(frame, mascara, objetivo, haz, f"umbral {u}  haz {txt}"))
            if tecla == "q":
                break
            if tecla in ("+", "-"):
                campo = "umbral_dif" if ref is not None else "umbral"
                setattr(args, campo, int(np.clip(getattr(args, campo) + (5 if tecla == "+" else -5), 0, 255)))
            if tecla == "r" and ref is not None:
                ref = blancura(frame)
        vista.cerrar()
    finally:
        cam.cerrar()


if __name__ == "__main__":
    main()
