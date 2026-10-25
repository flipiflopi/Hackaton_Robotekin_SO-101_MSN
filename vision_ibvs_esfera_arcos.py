#!/usr/bin/env python3
"""
Segmentación desde cámara + marcado por voz del punto al que apunta un boli azul.

Pipeline:
  1) Alfombrilla gris oscura (HSV + apertura + envolvente convexa).
  2) PCB verde rectangular dentro de la alfombrilla -> geometría del plano de la PCB
     (homografía imagen -> mm, y pose con solvePnP).
  3) Boli azul: se segmenta DENTRO del recorte enderezado de la PCB (no en la imagen completa),
     con su eje principal (fitLine) y su punta.
  4) Orden "ILUMINA AQUI" (voz) o tecla 'k' -> se guarda el punto 3D de la punta
     (en puntos_ilumina.csv) y se dibuja sobre la imagen y sobre el zoom de la PCB.
  5) El punto guardado (en coordenadas de la CÁMARA) se transforma a la BASE del robot.
     La cadena se construye partiendo de la base: traslación -> giro Z -> giro X (ver constantes
     TRASLACION_MM, ROT_Z_DEG, ROT_X_DEG). Se dibuja paso a paso en la ventana "Ejes 3D" y los
     ejes de la base se proyectan sobre la imagen para comprobar la transformación a ojo.
  6) Cinemática directa del SO-101 (URDF oficial, el mismo que usa LeRobot): con las articulaciones
     q (leídas con LeRobot o fijadas a mano) se obtiene la pose del efector en la BASE, el punto
     visto desde el efector y el objetivo con un offset de seguridad sobre el eje Z del efector.

Sistema de coordenadas 3D:
  - PCB (mm): origen en la esquina superior izquierda de la PCB, X a la derecha, Y hacia abajo,
    Z = 0 (la punta se asume apoyada en el plano de la PCB).
  - Cámara (mm): sistema de la cámara (X derecha, Y abajo, Z hacia delante), con intrínsecos
    aproximados (ver CAM_FOCAL_FACTOR).

Controles:
  k      : guardar el punto apuntado (igual que la orden de voz)
  i      : ILUMINAR el último punto guardado con IBVS (cámara externa, ver bloque IBVS)
  l      : FASE 1 otra vez: coger la linterna con la red (ACT). Al arrancar se hace sola (FASE1_AL_INICIO)
  x      : borrar de pantalla los puntos dibujados (el CSV no se toca)
  e      : mostrar/ocultar los ejes de la BASE (y de la PCB) proyectados sobre la imagen
  g      : mostrar/ocultar la ventana "Ejes 3D" (cadena de giros; arrastra con el ratón)
  v      : en "Ejes 3D", incluir (o no) el punto apuntado en la escala
  r      : mostrar/ocultar el esqueleto del SO-101 (cinemática directa) sobre la imagen
  f      : mostrar/ocultar la ventana "Robot FK" (esqueleto, efector, punto y objetivo)
  + / -  : zoom digital de la cámara
  m      : mostrar/ocultar máscaras
  t      : mostrar/ocultar ventana de umbrales
  s      : guardar capturas
  q/ESC  : salir

Dependencias:  pip install opencv-python numpy SpeechRecognition pyaudio
Opcional (leer las articulaciones del robot real):  pip install lerobot[feetech]
"""

import csv
import json
import os
import sys
import threading
import time
import traceback
import unicodedata
from collections import deque

import cv2
import numpy as np

# ----------------------------------------------------------------------------- Configuración
CAMERA_INDEX = 1
CAP_W, CAP_H = 640, 480           # sube a 1280x720 para más precisión en la punta
SCALE = 0.5                       # factor de reducción SOLO para procesar
ZOOM_CAMARA = 1.0                 # zoom digital inicial (1.0 = sin zoom)
ZOOM_STEP = 0.1
ZOOM_MAX = 4.0
PCB_ZOOM_LARGO = 600              # px del lado largo de la ventana "PCB zoom"
PCB_MARGEN = 1.08                 # margen alrededor de la PCB (1.0 = justo)

# Geometría real  >>> PON AQUÍ LAS MEDIDAS REALES DE TU PCB (mm) <<<
PCB_MM = (100.0, 60.0)            # (lado largo, lado corto)
CAM_FOCAL_FACTOR = 0.9            # focal en px ~ factor * ancho de imagen (aprox. webcam ~65° FOV)

# Transformación CÁMARA -> base del robot (cámara externa).
# Se construye PARTIENDO DE LA BASE, paso a paso (cada giro es sobre los ejes ya girados):
#   0) base   1) traslación   2) giro sobre Z   3) giro sobre X   ->  ejes de la cámara
# Giros en grados, regla de la mano derecha (positivo = antihorario mirando hacia el eje).
TRASLACION_MM = (22.0, 70.0, 38.0)
TRASLACION_ESCALA = 10.0          # mm por unidad del vector: 10 si [22, 70, 38] está en cm, 1 si ya está en mm
ROT_Z_DEG = 180.0 - 17.447
ROT_X_DEG = -26.87                # negativo = la cámara cabecea hacia ABAJO (mira a la base desde arriba)
# Convención de ejes de la cámara a la que se refieren esos giros:
#   "opencv": X derecha, Y abajo, Z adelante (lo que devuelve solvePnP; sin conversión)
#   "cuerpo": X derecha, Y adelante, Z arriba (se convierte antes de aplicar la cadena)
CAM_CONVENCION = "cuerpo"

# Robot SO-101 (cinemática directa)
ROBOT_PUERTO = None               # p. ej. "COM5": lee las articulaciones con LeRobot (solo lectura).
                                  # None = no conecta y usa Q_MANUAL_DEG.
ROBOT_ID = "my_awesome_follower_arm"   # id con el que se calibró el brazo en LeRobot
# Articulaciones (grados, como LeRobot con use_degrees=True) si no hay robot conectado.
# Pose de reposo aproximada; ajústala a la pose real para comprobar el esqueleto sobre la imagen.
Q_MANUAL_DEG = {"shoulder_pan": 0.0, "shoulder_lift": -100.0, "elbow_flex": 97.0,
                "wrist_flex": 60.0, "wrist_roll": 0.0}
# El origen del URDF (base_link, a ras de mesa bajo el brazo) expresado en TU sistema BASE, y su giro
# en Z. Déjalo en 0 si tu BASE coincide con la del URDF; si el esqueleto sale desplazado o girado
# respecto al robot real, ajústalo aquí.
BASE_URDF_EN_BASE_MM = (0.0, 0.0, 0.0)
BASE_URDF_YAW_DEG = 0.0
# Offset de seguridad (mm) sobre el eje Z del EFECTOR (el que sale de la pinza hacia delante).
# Positivo = el objetivo se queda OFFSET mm ANTES del punto (la linterna no llega a tocar).
OFFSET_Z_MM = 10.0

# Boli
MIN_AREA_BOLI = 150               # área mínima (px del recorte de la PCB)
BOLI_ELONGACION = 2.0             # largo/ancho mínimo para considerarlo un boli

# Voz
IDIOMA_VOZ = "es-ES"
HIST_SEG = 3.0                    # segundos de historial de la punta
VOZ_RETRO_S = 2.5                 # al oír la orden se busca el punto más estable en los últimos s
TECLA_RETRO_S = 0.6
PUNTOS_CSV = "puntos_ilumina.csv"

MIN_AREA_MAT = int(8000 * SCALE * SCALE)
MIN_AREA_PCB = int(500 * SCALE * SCALE)

K_OPEN_MAT = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
K_CLOSE_MAT = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
K_PCB = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
K_CLOSE_BOLI = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

FONT = cv2.FONT_HERSHEY_SIMPLEX

# Umbrales (se editan con la ventana "Ajustes" si se activa con 't')
PARAMS = {
    "Mat S max": 70, "Mat V min": 25, "Mat V max": 105,
    "PCB H min": 35, "PCB H max": 90, "PCB S min": 60, "PCB V min": 30,
    "Boli H min": 100, "Boli H max": 130, "Boli S min": 90, "Boli V min": 40,
}
RANGOS = {
    "Mat S max": 255, "Mat V min": 255, "Mat V max": 255,
    "PCB H min": 179, "PCB H max": 179, "PCB S min": 255, "PCB V min": 255,
    "Boli H min": 179, "Boli H max": 179, "Boli S min": 255, "Boli V min": 255,
}


# ----------------------------------------------------------------------------- Utilidades
def nothing(_):
    pass


def abrir_camara(idx):
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    cap = cv2.VideoCapture(idx, backend)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def abrir_ajustes():
    cv2.namedWindow("Ajustes", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Ajustes", 450, 420)
    for n, v in PARAMS.items():
        cv2.createTrackbar(n, "Ajustes", v, RANGOS[n], nothing)


def sincronizar_ajustes():
    for n in PARAMS:
        PARAMS[n] = cv2.getTrackbarPos(n, "Ajustes")


def zoom_digital(frame, z):
    if z <= 1.0:
        return frame
    h, w = frame.shape[:2]
    cw, ch = int(w / z), int(h / z)
    x0, y0 = (w - cw) // 2, (h - ch) // 2
    return cv2.resize(frame[y0:y0 + ch, x0:x0 + cw], (w, h), interpolation=cv2.INTER_LINEAR)


def mayor_contorno(mask, min_area):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    return c if cv2.contourArea(c) >= min_area else None


def up(pts):
    return (np.asarray(pts, np.float32) / SCALE).astype(np.int32)


def marcar(img, p, texto, color, r=6):
    """Círculo en p con texto a su lado (admite varias líneas separadas por \n)."""
    c = (int(p[0]), int(p[1]))
    cv2.circle(img, c, r, color, -1)
    cv2.circle(img, c, r + 3, (255, 255, 255), 1)
    for i, linea in enumerate(texto.split("\n") if texto else []):
        pos = (c[0] + 10, c[1] - 8 + 16 * i)
        cv2.putText(img, linea, pos, FONT, 0.5, (0, 0, 0), 3)
        cv2.putText(img, linea, pos, FONT, 0.5, color, 1)


def placeholder(texto):
    img = np.full((300, PCB_ZOOM_LARGO, 3), 40, np.uint8)
    cv2.putText(img, texto, (20, 155), FONT, 0.8, (200, 200, 200), 2)
    return img


# ----------------------------------------------------------------------------- Segmentación
def segmentar_alfombrilla(hsv):
    lower = np.array([0, 0, PARAMS["Mat V min"]], np.uint8)
    upper = np.array([179, PARAMS["Mat S max"], PARAMS["Mat V max"]], np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, K_OPEN_MAT)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, K_CLOSE_MAT, iterations=2)

    cnt = mayor_contorno(mask, MIN_AREA_MAT)
    blob = np.zeros_like(mask)
    if cnt is None:
        return blob, None, None
    hull = cv2.convexHull(cnt)
    cv2.drawContours(blob, [hull], -1, 255, cv2.FILLED)
    quad = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True)
    return blob, hull, quad


def segmentar_pcb(hsv, blob_mat):
    lower = np.array([PARAMS["PCB H min"], PARAMS["PCB S min"], PARAMS["PCB V min"]], np.uint8)
    upper = np.array([PARAMS["PCB H max"], 255, 255], np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.bitwise_and(mask, blob_mat)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, K_PCB)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, K_PCB, iterations=2)
    cnt = mayor_contorno(mask, MIN_AREA_PCB)
    rect = cv2.minAreaRect(cnt) if cnt is not None else None
    return mask, rect


def segmentar_boli(pcb_img):
    """
    Boli azul DENTRO del recorte enderezado de la PCB: máscara HSV, blob alargado mayor y eje
    con fitLine. La PUNTA es el extremo del eje más alejado del borde del recorte (el boli
    entra desde fuera, sujeto por la mano, así que su otro extremo toca el borde).
    Devuelve (máscara, dict|None) con coordenadas en píxeles del recorte.
    """
    hsv = cv2.cvtColor(cv2.GaussianBlur(pcb_img, (5, 5), 0), cv2.COLOR_BGR2HSV)
    lower = np.array([PARAMS["Boli H min"], PARAMS["Boli S min"], PARAMS["Boli V min"]], np.uint8)
    upper = np.array([PARAMS["Boli H max"], 255, 255], np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, K_PCB)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, K_CLOSE_BOLI)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mejor, mejor_area = None, 0
    for c in cnts:
        a = cv2.contourArea(c)
        if a < MIN_AREA_BOLI:
            continue
        _, (rw, rh), _ = cv2.minAreaRect(c)
        largo, ancho = max(rw, rh), max(min(rw, rh), 1.0)
        if largo / ancho < BOLI_ELONGACION:
            continue
        if a > mejor_area:
            mejor, mejor_area = c, a
    if mejor is None:
        return mask, None

    pts = mejor.reshape(-1, 2).astype(np.float32)
    vx, vy, x0, y0 = cv2.fitLine(mejor, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    proj = (pts - np.array([x0, y0], np.float32)) @ np.array([vx, vy], np.float32)
    pa, pb = pts[np.argmin(proj)], pts[np.argmax(proj)]

    h, w = pcb_img.shape[:2]

    def dist_borde(p):
        return min(p[0], p[1], w - 1 - p[0], h - 1 - p[1])

    punta, cola = (pa, pb) if dist_borde(pa) >= dist_borde(pb) else (pb, pa)
    return mask, {"punta": punta, "cola": cola, "cnt": mejor}


# ----------------------------------------------------------------------------- Geometría 3D
def ordenar_esquinas(pts):
    """Orden canónico: arriba-izq, arriba-der, abajo-der, abajo-izq (según la imagen)."""
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()          # y - x
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], np.float32)


def geometria_pcb(rect):
    """Homografía imagen(px, frame completo) -> plano de la PCB (mm)."""
    c = ordenar_esquinas(cv2.boxPoints(rect).astype(np.float32) / SCALE)
    w = float(np.linalg.norm(c[1] - c[0]))
    h = float(np.linalg.norm(c[3] - c[0]))
    if w < 5 or h < 5:
        return None
    largo, corto = max(PCB_MM), min(PCB_MM)
    W, Hh = (largo, corto) if w >= h else (corto, largo)
    obj = np.array([[0, 0], [W, 0], [W, Hh], [0, Hh]], np.float32)
    H = cv2.getPerspectiveTransform(c, obj)
    return {"c": c, "obj": obj, "H": H, "Hinv": np.linalg.inv(H), "W": W, "Hh": Hh}


def matriz_K(shape, zoom):
    """Intrínsecos aproximados de la cámara (focal estimada, centro óptico en el centro)."""
    h, w = shape[:2]
    f = CAM_FOCAL_FACTOR * w * zoom
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], np.float64)


def pose_pcb(geo, shape, zoom):
    """Pose de la PCB respecto a la cámara (intrínsecos aproximados). Devuelve (R, t) o None."""
    K = matriz_K(shape, zoom)
    obj3 = np.hstack([geo["obj"], np.zeros((4, 1), np.float32)])
    ok, rvec, tvec = cv2.solvePnP(obj3, geo["c"], K, None, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.reshape(3)


def vista_pcb(frame, geo):
    """PCB enderezada y ampliada. Devuelve (imagen, T) con T: mm -> px de esa imagen."""
    W, Hh = geo["W"], geo["Hh"]
    s = PCB_ZOOM_LARGO / (max(W, Hh) * PCB_MARGEN)
    off = (PCB_MARGEN - 1) / 2
    Wv, Hv = max(int(W * PCB_MARGEN * s), 1), max(int(Hh * PCB_MARGEN * s), 1)
    T = np.array([[s, 0, off * W * s], [0, s, off * Hh * s], [0, 0, 1]], np.float64)
    img = cv2.warpPerspective(frame, T @ geo["H"], (Wv, Hv), flags=cv2.INTER_CUBIC)
    return img, T


def px_a_mm(geo, p):
    return cv2.perspectiveTransform(np.array([[p]], np.float32), geo["H"])[0, 0]


def mm_a_px(geo, xy):
    return cv2.perspectiveTransform(np.array([[xy]], np.float32), geo["Hinv"])[0, 0]


def analizar(frame, zoom):
    """Procesa un frame completo y devuelve todo lo detectado."""
    small = cv2.resize(frame, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (5, 5), 0)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

    r = {"blob_mat": None, "hull": None, "quad": None, "mask_pcb": None, "mask_boli": None,
         "geo": None, "pose": None, "pcb_img": None, "T": None, "boli": None}
    r["blob_mat"], r["hull"], r["quad"] = segmentar_alfombrilla(hsv)
    r["mask_pcb"] = np.zeros_like(r["blob_mat"])
    r["mask_boli"] = np.zeros_like(r["blob_mat"])
    if r["hull"] is None:
        return r

    r["mask_pcb"], rect = segmentar_pcb(hsv, r["blob_mat"])
    if rect is not None:
        r["geo"] = geometria_pcb(rect)
    if r["geo"] is not None:
        r["pose"] = pose_pcb(r["geo"], frame.shape, zoom)
        r["pcb_img"], r["T"] = vista_pcb(frame, r["geo"])
        # El boli se busca SOLO dentro del recorte enderezado de la PCB
        r["mask_boli"], r["boli"] = segmentar_boli(r["pcb_img"])
    return r


# ----------------------------------------------------------------------------- Transformación a la base
def rot_x(deg):
    c, s_ = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[1, 0, 0], [0, c, -s_], [0, s_, c]])


def rot_z(deg):
    c, s_ = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])


def hom(R=None, t=None):
    """Matriz homogénea 4x4 a partir de rotación y/o traslación."""
    M = np.eye(4)
    if R is not None:
        M[:3, :3] = R
    if t is not None:
        M[:3, 3] = t
    return M


# (x, y, z) OpenCV -> (x, z, -y) cuerpo   [X derecha, Y adelante, Z arriba]
C_OPENCV_A_CUERPO = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)


def cadena_frames():
    """Pose (4x4, expresada en la BASE) de cada paso de la construcción. El último es la cámara.
    T_k = T_(k-1) @ giro_k, es decir, cada giro se hace sobre los ejes ya girados."""
    T1 = hom(t=np.asarray(TRASLACION_MM, float) * TRASLACION_ESCALA)
    T2 = T1 @ hom(rot_z(ROT_Z_DEG))
    T3 = T2 @ hom(rot_x(ROT_X_DEG))
    fin = " = CAMARA" if CAM_CONVENCION == "opencv" else ""
    pasos = [("0 Base", np.eye(4)),
             ("1 + traslacion", T1),
             (f"2 + Rz {ROT_Z_DEG:.2f}", T2),
             (f"3 + Rx {ROT_X_DEG:.2f}{fin}", T3)]
    if CAM_CONVENCION == "cuerpo":
        pasos.append(("4 ejes OpenCV = CAMARA", T3 @ hom(C_OPENCV_A_CUERPO)))
    return pasos


def matriz_cam_a_base():
    """Pose de la cámara en la base: p_base = M @ p_cam (p_cam en ejes OpenCV)."""
    return cadena_frames()[-1][1]


def tf_a_base(p):
    """Punto 3D en la cámara (mm) -> coordenadas en la base del robot (mm)."""
    return (matriz_cam_a_base() @ np.array([p[0], p[1], p[2], 1.0]))[:3]


def punto_a_base(cam):
    return None if cam is None else tf_a_base(cam)


# ----------------------------------------------------------------------------- Ejes sobre la imagen
def px_de_cam(K, Pc):
    """Puntos 3D en ejes de cámara (Nx3) -> píxeles; None si están detrás de la cámara."""
    out = []
    for p in np.asarray(Pc, float):
        if p[2] <= 1e-6:
            out.append(None)
            continue
        u = K[0, 0] * p[0] / p[2] + K[0, 2]
        v = K[1, 1] * p[1] / p[2] + K[1, 2]
        out.append((int(np.clip(u, -30000, 30000)), int(np.clip(v, -30000, 30000))))
    return out


COLORES_EJES = [(0, 0, 255), (0, 200, 0), (255, 80, 0)]          # X rojo, Y verde, Z azul (BGR)


def dibujar_ejes(img, K, T_cam, largo, etiqueta):
    """Dibuja X(rojo) Y(verde) Z(azul) de un marco cuya pose en la CÁMARA es T_cam (4x4).
    Devuelve False si el origen queda detrás de la cámara."""
    P = np.array([[0, 0, 0], [largo, 0, 0], [0, largo, 0], [0, 0, largo]], float)
    Pc = (T_cam[:3, :3] @ P.T).T + T_cam[:3, 3]
    px = px_de_cam(K, Pc)
    if px[0] is None:
        return False
    for i, (c, n) in enumerate(zip(COLORES_EJES, "XYZ"), start=1):
        if px[i] is not None:
            cv2.arrowedLine(img, px[0], px[i], c, 2, tipLength=0.15)
            cv2.putText(img, n, (px[i][0] + 4, px[i][1] - 4), FONT, 0.5, c, 2)
    cv2.circle(img, px[0], 4, (255, 255, 255), -1)
    cv2.putText(img, etiqueta, (px[0][0] + 8, px[0][1] + 16), FONT, 0.5, (0, 0, 0), 3)
    cv2.putText(img, etiqueta, (px[0][0] + 8, px[0][1] + 16), FONT, 0.5, (255, 255, 255), 1)
    return True


# ----------------------------------------------------------------------------- Ventana "Ejes 3D"
VISTA3D = {"azim": -35.0, "elev": 25.0, "drag": None, "con_punto": False}
CELDA_3D = 340


def _mouse3d(evt, x, y, flags, _):
    v = VISTA3D
    if evt == cv2.EVENT_LBUTTONDOWN:
        v["drag"] = (x, y)
    elif evt == cv2.EVENT_LBUTTONUP:
        v["drag"] = None
    elif evt == cv2.EVENT_MOUSEMOVE and v["drag"] is not None:
        v["azim"] += (x - v["drag"][0]) * 0.5
        v["elev"] = float(np.clip(v["elev"] - (y - v["drag"][1]) * 0.5, -89, 89))
        v["drag"] = (x, y)


def _proj3d(P, azim, elev):
    a, e = np.radians(azim), np.radians(elev)
    P = np.atleast_2d(np.asarray(P, float))
    x = P[:, 0] * np.cos(a) - P[:, 1] * np.sin(a)
    y = P[:, 0] * np.sin(a) + P[:, 1] * np.cos(a)
    vert = y * np.sin(e) + P[:, 2] * np.cos(e)
    return np.stack([x, vert], axis=1)


def render_ejes3d(punto_base=None):
    """Una celda por paso de la cadena: ejes de la base en gris y el marco de ese paso en color."""
    pasos = cadena_frames()
    n = len(pasos)
    cols = 2 if n <= 4 else 3
    filas = (n + cols - 1) // cols
    cel = CELDA_3D
    img = np.full((filas * cel, cols * cel, 3), 30, np.uint8)

    orig = [T[:3, 3] for _, T in pasos]
    incluir = punto_base is not None and VISTA3D["con_punto"]
    pts = np.array(orig + ([punto_base] if incluir else []), float)
    L = max(0.4 * float(np.ptp(pts, axis=0).max()), 25.0)
    todos = [np.zeros(3)] + [v * L for v in np.eye(3)]
    for _, T in pasos:
        todos += [T[:3, 3]] + [T[:3, 3] + T[:3, :3] @ v * L for v in np.eye(3)]
    todos = np.array(todos + list(pts))
    centro = (todos.max(0) + todos.min(0)) / 2
    radio = max(float(np.linalg.norm(todos.max(0) - todos.min(0))) / 2, 1.0)
    esc = 0.85 * (cel / 2) / radio
    az, el = VISTA3D["azim"], VISTA3D["elev"]

    for k, (nombre, T) in enumerate(pasos):
        f, c = divmod(k, cols)
        celda = img[f * cel:(f + 1) * cel, c * cel:(c + 1) * cel]
        cv2.rectangle(celda, (0, 0), (cel - 1, cel - 1), (80, 80, 80), 1)

        def P(p):
            q = _proj3d(np.asarray(p, float) - centro, az, el)[0]
            return (int(cel / 2 + esc * q[0]), int(cel / 2 - esc * q[1]))

        o = P([0, 0, 0])                                   # base en gris
        for v, nm in zip(np.eye(3) * L, ("Xb", "Yb", "Zb")):
            cv2.arrowedLine(celda, o, P(v), (150, 150, 150), 1, tipLength=0.12)
            cv2.putText(celda, nm, (P(v)[0] + 3, P(v)[1] - 3), FONT, 0.4, (150, 150, 150), 1)

        R, t = T[:3, :3], T[:3, 3]
        ot = P(t)
        if k > 0:
            cv2.line(celda, o, ot, (120, 120, 120), 1)
        for i, (col, nm) in enumerate(zip(COLORES_EJES, "XYZ")):
            tip = P(t + R[:, i] * L)
            cv2.arrowedLine(celda, ot, tip, col, 3, tipLength=0.15)
            cv2.putText(celda, nm, (tip[0] + 4, tip[1] - 4), FONT, 0.5, col, 2)
            v = R[:, i]
            cv2.putText(celda, f"{nm}: ({v[0]:+.2f},{v[1]:+.2f},{v[2]:+.2f})",
                        (8, cel - 8 - 14 * (2 - i)), FONT, 0.4, col, 1)
        if punto_base is not None:
            cv2.circle(celda, P(punto_base), 5, (255, 0, 255), -1)
        cv2.putText(celda, nombre, (8, 18), FONT, 0.5, (255, 255, 255), 1)

    cv2.putText(img, "arrastra con el raton para girar | v: punto en escala",
                (img.shape[1] - 330, 14), FONT, 0.35, (200, 200, 200), 1)
    return img


# ----------------------------------------------------------------------------- Robot SO-101: cinemática directa
# Datos del URDF oficial so101_new_calib.urdf (TheRobotStudio/SO-ARM100), el que usa LeRobot con
# RobotKinematics(target_frame_name="gripper_frame_link"). Metros en el URDF -> mm aquí.
# Todas las articulaciones giran sobre su Z local; los ángulos son los de LeRobot (grados).
ARTICULACIONES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
SO101_CADENA = [   # (articulación, eslabón hijo, xyz [m], rpy [rad])
    ("shoulder_pan", "shoulder_link", (0.0388353, -8.97657e-09, 0.0624), (3.14159, 4.18253e-17, -3.14159)),
    ("shoulder_lift", "upper_arm_link", (-0.0303992, -0.0182778, -0.0542), (-1.5708, -1.5708, 0.0)),
    ("elbow_flex", "lower_arm_link", (-0.11257, -0.028, 1.73763e-16), (-3.63608e-16, 8.74301e-16, 1.5708)),
    ("wrist_flex", "wrist_link", (-0.1349, 0.0052, 3.62355e-17), (4.02456e-15, 8.67362e-16, -1.5708)),
    ("wrist_roll", "gripper_link", (5.55112e-17, -0.0611, 0.0181), (1.5708, 0.0486795, 3.14159)),
]
SO101_EFECTOR = ("gripper_frame_link", (-0.0079, -0.000218121, -0.0981274), (0.0, 3.14159, 0.0))   # fija


def _rot_rpy(r, p, y):
    """Convención URDF: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def cinematica_directa(q_deg):
    """Pose (4x4, en tu BASE) de cada eslabón del SO-101 para las articulaciones q_deg (dict, grados).
    Devuelve [(nombre, T), ...]; el último es gripper_frame_link (el efector)."""
    T = hom(rot_z(BASE_URDF_YAW_DEG), np.asarray(BASE_URDF_EN_BASE_MM, float))
    frames = [("base_link", T.copy())]
    for art, link, xyz, rpy in SO101_CADENA:
        T = T @ hom(_rot_rpy(*rpy), np.asarray(xyz) * 1000.0) @ hom(rot_z(q_deg[art]))
        frames.append((link, T.copy()))
    nombre, xyz, rpy = SO101_EFECTOR
    T = T @ hom(_rot_rpy(*rpy), np.asarray(xyz) * 1000.0)
    frames.append((nombre, T.copy()))
    return frames


def objetivo_robot(p_base, q_deg):
    """Aplica la cinemática directa al punto (en la BASE):
      p_ef      : el punto visto desde el efector (T_ef^-1 @ p)
      p_ef_obj  : p_ef con el offset de seguridad sobre Z del efector (se queda OFFSET_Z_MM antes)
      p_obj     : ese objetivo expresado de nuevo en la BASE (lo que necesitaría una cinemática inversa)."""
    frames = cinematica_directa(q_deg)
    T_ef = frames[-1][1]
    p_ef = (np.linalg.inv(T_ef) @ np.array([p_base[0], p_base[1], p_base[2], 1.0]))[:3]
    p_ef_obj = p_ef - np.array([0.0, 0.0, OFFSET_Z_MM])
    p_obj = (T_ef @ np.array([p_ef_obj[0], p_ef_obj[1], p_ef_obj[2], 1.0]))[:3]
    return {"frames": frames, "T_ef": T_ef, "p_ef": p_ef, "p_ef_obj": p_ef_obj, "p_obj": p_obj}


class LectorRobot:
    """Lee las articulaciones del SO-101 con LeRobot en un hilo (solo lectura). Si ROBOT_PUERTO es
    None, o no se puede conectar (puerto ocupado por otro programa, sin lerobot...), se usa Q_MANUAL_DEG."""

    def __init__(self):
        self.q = dict(Q_MANUAL_DEG)
        self.estado = "q manual"
        self._stop = threading.Event()
        self.robot = None                    # [IBVS] conexión abierta (la usa el IBVS si existe)
        self.pausa = threading.Event()       # [IBVS] pausa la lectura mientras el IBVS mueve el robot

    def iniciar(self):
        if not ROBOT_PUERTO:
            return
        try:
            try:
                from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
            except ImportError:                                  # versiones anteriores de LeRobot
                from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
            robot = SO101Follower(SO101FollowerConfig(port=ROBOT_PUERTO, id=ROBOT_ID, use_degrees=True))
            robot.connect(calibrate=False)
        except Exception as e:
            self.estado = f"sin robot ({type(e).__name__}): q manual"
            return
        self.estado = "robot (LeRobot)"
        self.robot = robot                                       # [IBVS]
        while not self._stop.is_set():
            if self.pausa.is_set():                              # [IBVS] el IBVS usa el bus
                time.sleep(0.05)
                continue
            try:
                obs = robot.get_observation()
                self.q = {n: float(obs[f"{n}.pos"]) for n in ARTICULACIONES}
            except Exception as e:
                self.estado = f"error lectura ({type(e).__name__}): ultima q"
            time.sleep(0.05)
        try:
            robot.disconnect()
        except Exception:
            pass

    def parar(self):
        self._stop.set()


def render_fk3d(frames, p_base, p_obj, q_deg, tam=520):
    """Esqueleto del SO-101 (cinemática directa), ejes del efector, punto apuntado y objetivo."""
    img = np.full((tam, tam, 3), 30, np.uint8)
    pts = [T[:3, 3] for _, T in frames]
    extra = [p for p in (p_base, p_obj) if p is not None]
    todos = np.array(pts + [np.zeros(3)] + [v * 100.0 for v in np.eye(3)] + extra)
    centro = (todos.max(0) + todos.min(0)) / 2
    radio = max(float(np.linalg.norm(todos.max(0) - todos.min(0))) / 2, 1.0)
    esc = 0.8 * (tam / 2) / radio
    az, el = VISTA3D["azim"], VISTA3D["elev"]

    def P(p):
        q = _proj3d(np.asarray(p, float) - centro, az, el)[0]
        return (int(tam / 2 + esc * q[0]), int(tam / 2 - esc * q[1]))

    o = P([0, 0, 0])                                              # BASE (gris)
    for v, nm in zip(np.eye(3) * 100.0, ("Xb", "Yb", "Zb")):
        cv2.arrowedLine(img, o, P(v), (150, 150, 150), 1, tipLength=0.12)
        cv2.putText(img, nm, (P(v)[0] + 3, P(v)[1] - 3), FONT, 0.4, (150, 150, 150), 1)
    for a, b in zip(pts[:-1], pts[1:]):                           # eslabones
        cv2.line(img, P(a), P(b), (0, 200, 255), 3)
    for pp in pts:
        cv2.circle(img, P(pp), 4, (0, 200, 255), -1)
    T_ef = frames[-1][1]                                          # efector
    for i, (col, nm) in enumerate(zip(COLORES_EJES, "XYZ")):
        tip = P(T_ef[:3, 3] + T_ef[:3, i] * 50.0)
        cv2.arrowedLine(img, P(T_ef[:3, 3]), tip, col, 3, tipLength=0.15)
        cv2.putText(img, nm, (tip[0] + 4, tip[1] - 4), FONT, 0.5, col, 2)
    if p_base is not None:
        cv2.circle(img, P(p_base), 6, (255, 0, 255), -1)
        cv2.putText(img, "punto", (P(p_base)[0] + 8, P(p_base)[1] - 6), FONT, 0.45, (255, 0, 255), 1)
    if p_obj is not None:
        cv2.circle(img, P(p_obj), 6, (0, 255, 255), -1)
        cv2.putText(img, "objetivo", (P(p_obj)[0] + 8, P(p_obj)[1] + 14), FONT, 0.45, (0, 255, 255), 1)
        if p_base is not None:
            cv2.line(img, P(p_obj), P(p_base), (0, 255, 255), 1)
    corto = {"shoulder_pan": "pan", "shoulder_lift": "lift", "elbow_flex": "elbow",
             "wrist_flex": "wflex", "wrist_roll": "wroll"}
    qtxt = " ".join(f"{corto[k]}={q_deg[k]:.0f}" for k in ARTICULACIONES)
    cv2.putText(img, "SO-101 FK  q(deg): " + qtxt, (6, 14), FONT, 0.38, (255, 255, 255), 1)
    pe = T_ef[:3, 3]
    cv2.putText(img, f"EF en BASE (mm): ({pe[0]:.0f}, {pe[1]:.0f}, {pe[2]:.0f})", (6, tam - 8),
                FONT, 0.45, (255, 255, 255), 1)
    return img


# ----------------------------------------------------------------------------- Punto apuntado
def muestra_punta(geo, pose, T, punta_vista):
    """Una muestra de la punta: (t, xy_mm, xyz_cam | None, px_frame). Entrada: px del recorte."""
    v = np.linalg.solve(T, np.array([punta_vista[0], punta_vista[1], 1.0]))
    xy = v[:2] / v[2]                                   # recorte (px) -> plano PCB (mm)
    px = mm_a_px(geo, xy)                               # y a píxeles del frame original
    cam = None
    if pose is not None:
        R, t = pose
        cam = R @ np.array([xy[0], xy[1], 0.0]) + t
    return (time.time(), xy, cam, np.asarray(px, np.float32))


def punto_estable(hist, retro, ventana=0.5, min_muestras=4):
    """Busca en los últimos `retro` s la ventana más quieta de la punta y devuelve su mediana."""
    ahora = time.time()
    m = [h for h in hist if ahora - h[0] <= retro]
    if len(m) < min_muestras:
        return None
    mejor, mejor_sp = None, 1e18
    for i in range(len(m)):
        g = [h for h in m[i:] if h[0] - m[i][0] <= ventana]
        if len(g) < min_muestras:
            continue
        sp = float(np.array([h[1] for h in g]).std(axis=0).sum())
        if sp < mejor_sp:
            mejor, mejor_sp = g, sp
    if mejor is None:
        return None
    xy = np.median(np.array([h[1] for h in mejor]), axis=0)
    px = np.median(np.array([h[3] for h in mejor]), axis=0)
    cams = [h[2] for h in mejor if h[2] is not None]
    cam = np.median(np.array(cams), axis=0) if cams else None
    return xy, px, cam


CSV_HEADER = ["n", "fecha", "origen", "x_pcb_mm", "y_pcb_mm", "z_pcb_mm",
              "x_cam_mm", "y_cam_mm", "z_cam_mm", "x_base_mm", "y_base_mm", "z_base_mm",
              "x_ef_mm", "y_ef_mm", "z_ef_mm", "x_obj_mm", "y_obj_mm", "z_obj_mm",
              "q_shoulder_pan", "q_shoulder_lift", "q_elbow_flex", "q_wrist_flex", "q_wrist_roll",
              "px", "py", "dentro_pcb"]


def _ruta_csv():
    """Ruta del CSV a usar. Si existe uno con otro formato (versión anterior) lo aparta.
    La cabecera se lee con el archivo CERRADO antes de renombrar (Windows no deja renombrar
    un archivo abierto)."""
    if not os.path.exists(PUNTOS_CSV):
        return PUNTOS_CSV
    try:
        with open(PUNTOS_CSV, "r", encoding="utf-8") as f:
            cabecera = f.readline().strip()
    except OSError:
        cabecera = ""
    if cabecera == ",".join(CSV_HEADER):
        return PUNTOS_CSV
    try:
        os.replace(PUNTOS_CSV, PUNTOS_CSV.replace(".csv", f"_antiguo_{int(time.time())}.csv"))
        return PUNTOS_CSV
    except OSError:                                  # p. ej. abierto en Excel
        alt = PUNTOS_CSV.replace(".csv", f"_{int(time.time())}.csv")
        print(f"[aviso] No se pudo apartar {PUNTOS_CSV} (¿abierto en Excel?). Guardo en {alt}")
        return alt


def guardar_punto(n, origen, xy, cam, base, px, geo, rob=None):
    ruta = _ruta_csv()
    dentro = int(0 <= xy[0] <= geo["W"] and 0 <= xy[1] <= geo["Hh"])
    c = ["", "", ""] if cam is None else [f"{v:.2f}" for v in cam]
    b = ["", "", ""] if base is None else [f"{v:.2f}" for v in base]
    ef = ["", "", ""] if rob is None else [f"{v:.2f}" for v in rob["p_ef_obj"]]
    ob = ["", "", ""] if rob is None else [f"{v:.2f}" for v in rob["p_obj"]]
    qs = [""] * 5 if rob is None else [f"{rob['q'][k]:.2f}" for k in ARTICULACIONES]
    fila = [n, time.strftime("%Y-%m-%d %H:%M:%S"), origen,
            f"{xy[0]:.2f}", f"{xy[1]:.2f}", "0.00", *c, *b, *ef, *ob, *qs,
            f"{px[0]:.1f}", f"{px[1]:.1f}", dentro]
    for intento in range(2):
        try:
            nuevo = not os.path.exists(ruta)
            with open(ruta, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if nuevo:
                    w.writerow(CSV_HEADER)
                w.writerow(fila)
            break
        except PermissionError:                      # CSV bloqueado: no tumbar la visión
            ruta = ruta.replace(".csv", f"_{int(time.time())}.csv")
            print(f"[aviso] CSV bloqueado (¿abierto en Excel?). Guardo en {ruta}")
    return dentro


# ----------------------------------------------------------------------------- Voz
def sin_acentos(t):
    return "".join(c for c in unicodedata.normalize("NFD", t.lower())
                   if unicodedata.category(c) != "Mn")


def es_orden(texto):
    """'ilumina aquí', 'iluminar', 'marca aquí', 'guarda aquí', 'apunta ahí'..."""
    t = sin_acentos(texto)
    if "lumin" in t:
        return True
    verbos = ("marca", "guarda", "apunta", "senala", "registra", "anota")
    return ("aqui" in t or "ahi" in t) and any(v in t for v in verbos)


class EscuchaVoz:
    def __init__(self):
        self.evento = threading.Event()
        self.estado = "iniciando micro..."
        self.ultima = ""
        self._parar = None

    def iniciar(self):
        try:
            import speech_recognition as sr
        except ImportError:
            self.estado = "sin voz (pip install SpeechRecognition pyaudio) - usa 'k'"
            return
        try:
            self.sr = sr
            rec = sr.Recognizer()
            rec.dynamic_energy_threshold = True
            mic = sr.Microphone()
            with mic as fuente:
                rec.adjust_for_ambient_noise(fuente, duration=1.0)
            self._parar = rec.listen_in_background(mic, self._callback, phrase_time_limit=4)
            self.estado = "escuchando"
        except Exception as e:                       # sin micro, sin pyaudio, etc.
            self.estado = f"micro no disponible ({type(e).__name__}) - usa 'k'"

    def _callback(self, rec, audio):
        try:
            texto = rec.recognize_google(audio, language=IDIOMA_VOZ)
        except self.sr.UnknownValueError:
            return
        except Exception as e:                       # sin internet, etc.
            self.estado = f"error voz ({type(e).__name__})"
            return
        self.estado = "escuchando"
        self.ultima = texto
        if es_orden(texto):
            self.evento.set()

    def parar(self):
        if self._parar:
            self._parar(wait_for_stop=False)



# ============================================================================= ILUMINACIÓN IBVS
# VERSIÓN "SEMIESFERA" (vision_ibvs_esfera.py). Igual que vision_ibvs.py, pero al llegar a la consigna NO
# termina: vigila en tiempo real si algo se interpone entre la linterna y la PCB y, si es así, busca otra
# posición de la linterna sobre la SEMIESFERA de radio D (distancia de iluminación) centrada en el punto.
#
#  1) REFERENCIA al converger: % del PUNTO (círculo de R_PUNTO_PX) iluminado (cian o núcleo blanco), radio del
#     halo, % de PCB iluminada en un anillo y exceso de azul (estos dos últimos, informativos).
#  2) OCLUSIÓN (OCL_N_FRAMES seguidos): el punto pierde la luz (< OCL_Q_MIN*ref), o el halo ENCOGE
#     (< OCL_R_MIN*ref: el haz choca antes con algo, p. ej. una mano), o sombra negra, o no se ve la luz.
#     Medido en los logs: normal radio 48 px; mano encima 31-35 px. Mientras hay oclusión NO se persigue el
#     centro del halo (está deformado por el obstáculo).
#  3) SEMIESFERA: P (el punto) se calcula con la cinemática (el haz corta la PCB a Z_PCB_MM); D = distancia
#     actual linterna-P (o DIST_ILUM_MM). Se trazan N_CIRCUNFERENCIAS circunferencias máximas de la semiesfera
#     que pasan por la vertical de P, con orientaciones repartidas (cada 180/N grados), y en cada una
#     varios puntos (ESFERA_ELEVS, a ambos lados de la vertical). Para cada punto L se calcula la cinemática
#     inversa (linterna en L apuntando a P).
#  4) ALCANZABILIDAD: con la linterna PERPENDICULAR al eje Z de la pinza (EJE_HAZ_PINZA), wrist_roll cambia
#     la dirección del haz: hay 5 grados de libertad útiles (3 posición + 2 dirección) y la linterna puede
#     apuntar a P desde casi toda la semiesfera. Para cada L se resuelve la cinemática inversa (5
#     articulaciones) y se comprueba con la cinemática directa el DESVÍO del haz respecto a P; se descartan
#     los L fuera de límites o con desvío > ESFERA_DESVIO_MAX_MM (el IBVS corrige lo que quede).
#  5) ORDEN DE PRUEBA = RECORRER CIRCUNFERENCIAS: si el haz está tapado solo en parte, el halo que queda se
#     desplaza alejándose del obstáculo (pista). Con pista: primero la circunferencia más alineada con
#     "alejarse", por ese lado, bajando desde la vertical (82, 74, 66...). Sin pista (tapado entero): todas las
#     circunferencias nivel a nivel (todas a 82, luego todas a 74...), para probar pronto todas las direcciones.
#     Si un intento ya consigue algo de luz (calidad >= 0.3), se sigue bajando por ESA circunferencia y lado. Se prueban de uno en uno (ir, re-apuntar con IBVS si el halo es normal, medir) y se queda
#     con el PRIMERO que ilumina el punto (calidad >= OCL_OK). Si ninguno, vuelve al mejor.
#  6) Si la luz se aparta de la consigna SIN oclusión (> REAJUSTE_PX), se recentra con IBVS.
# Teclas extra en la ventana IBVS: 'r' = tomar referencia ahora, 'o' = buscar en la semiesfera ya.
# ESPACIO para, ESC vuelve.
# IBVS con la CÁMARA EXTERNA (la misma de arriba, cámara fija: "eye-to-hand"). Tras guardar el punto
# con 'k' (o voz), la tecla 'i' lanza el control:
#
#  1. La PCB ya está segmentada (geo del momento en que se guardó el punto; la cámara y la PCB no se
#     mueven, así que aunque el brazo tape la PCB la consigna no se pierde). Consigna en la imagen:
#     s* = mm_a_px(geo, punto).
#  2. Se segmenta el CÍRCULO DE LUZ dentro de la PCB (brillo alto, poca saturación) y su centroide
#     ponderado por el brillo: s = (u, v).
#  3. Error IBVS: e = s - s*  (px).  Paso del efector  dx = -GANANCIA * J^-1 * e  en el plano X-Y de la
#     base, con la ALTURA del efector y la orientación de la pinza FIJAS (la altura es la que tenga el
#     brazo al pulsar 'i'; se ajusta con w/s). J = de/dx (px/mm) se mide con 4 movimientos de prueba
#     de +-PASO_ID_MM y se corrige con Broyden (no hace falta calibración cámara-base).
#  4. Convergido -> ¿el punto se ve VERDE desde la cámara? (con la luz encima, mirando el halo, y
#     apartando la luz un instante). Si no (el brazo, la linterna u otro objeto lo tapa): búsqueda
#     izquierda, derecha, diagonales, arriba, abajo (en la imagen de la cámara externa), re-apuntando
#     la luz a la consigna con shoulder_pan + wrist_flex. Sigue en la dirección que mejora; si ninguna
#     mejora reduce el paso (o lo amplía si todo está tapado). Al final vuelve a la mejor.
#
# Antes de todo, el robot va DESPACIO a la POSE DE SEGURIDAD (pinza hacia abajo, ver POSE_ARCHIVO).
# Ventana "IBVS":  ESPACIO parar | i repetir | p ir a la pose de seguridad | g guardar la pose actual
#   como pose de seguridad | b solo búsqueda | c re-identificar J | w / s subir/bajar 5 mm |
#   t umbrales de la luz | m máscara de la luz | ENTER / q / ESC volver
IBVS_AUTO = False             # True: lanzar el IBVS nada más guardar el punto (si no, tecla 'i')

# ----------------------------------------------------------------------------- FASE 1: coger la linterna (red ACT)
# Lo PRIMERO que hace el programa (antes de abrir la cámara): ejecuta la red entrenada que coge la linterna,
# con rollout_registro.py (misma carpeta): comprueba la muñeca, lleva el brazo a la postura de inicio de las
# demos, ejecuta el lerobot-rollout OFICIAL y GRABA TODO (logs/<ejecución>/fase1_*/: pasos.csv, chunks.npz,
# vídeos de las dos cámaras, resumen.txt). Al terminar NO vuelve a reposo y sigue sujetando la linterna
# (--encadenar). Después sigue el programa normal: 'k' para marcar el punto, 'i' para iluminar.
FASE1_AL_INICIO = True        # False: empezar directamente con la visión (la linterna ya está cogida)
FASE1_PREGUNTAR = True        # pedir ENTER antes de mover el robot (escribir 'n' = saltar la fase 1)
# Red de la linterna TUMBADA (act_tumbada, 54 episodios, checkpoint 18k)
# Se usa el del PC donde se entrenó si existe; si no, el descargado de Hugging Face en modelo/act_tumbada
# (hf download FlipiFlopi007/act_tumbada --local-dir modelo/act_tumbada)
FASE1_MODELO = r"C:\lerobot-work\outputs\train\act_tumbada\checkpoints\018000\pretrained_model"
if not os.path.isdir(FASE1_MODELO):
    FASE1_MODELO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modelo", "act_tumbada")
FASE1_SEGUNDOS = 15           # duración de la red (las demos de la linterna tumbada duran 12,8 s de media, máx. 15,8;
                              # con más tiempo se queda arriba temblando: ya no sabe qué hacer)
FASE1_PASOS = 60              # n_action_steps (cada cuántas acciones recalcula; 60 fue lo que mejor fue)
FASE1_CAM_WRIST = 0           # índice de la cámara de la muñeca (como en el entrenamiento)
FASE1_CAM_TOP = 1             # índice de la cámara de arriba (= CAMERA_INDEX)
import shutil as _shutil
FASE1_DEVICE = "cuda" if _shutil.which("nvidia-smi") else "cpu"   # GPU NVIDIA si la hay
FASE1_SCRIPT = "rollout_registro.py"
# Postura a la que va el brazo ANTES de la red (la del inicio de las demos del modelo): mediana del primer
# frame de los 54 episodios de la linterna tumbada. None = la que trae rollout_registro.py (la misma).
FASE1_INICIO = "0.4,-103.1,96.2,53.4,-7.5,0.6"


def ejecutar_fase1(preguntar=None):
    """Fase 1: la red coge la linterna (proceso aparte: necesita las dos cámaras y el robot). True si terminó."""
    import subprocess
    preguntar = FASE1_PREGUNTAR if preguntar is None else preguntar
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), FASE1_SCRIPT)
    if not os.path.exists(script):
        print(f"[FASE1] No encuentro {script}: copia rollout_registro.py junto a este script")
        return False
    print("\n" + "=" * 70 + "\n[FASE1] COGER LA LINTERNA CON LA RED (ACT)\n"
          f"        modelo: {FASE1_MODELO}\n        {FASE1_SEGUNDOS:.0f} s, n_action_steps={FASE1_PASOS}, "
          f"puerto {IBVS_PUERTO}, cámaras muñeca={FASE1_CAM_WRIST} arriba={FASE1_CAM_TOP}\n" + "=" * 70)
    if preguntar:
        r = input("[FASE1] Linterna en la mesa y brazo libre. ENTER = empezar | 'n' + ENTER = saltar: ").strip().lower()
        if r == "n":
            print("[FASE1] Saltada")
            return False
    salida = REG.sub("fase1_" + time.strftime("%H%M%S")) or os.path.join("logs", "fase1")
    cmd = [sys.executable, script, "--puerto", IBVS_PUERTO, "--modelo", FASE1_MODELO,
           "--pasos", str(FASE1_PASOS), "--segundos", str(FASE1_SEGUNDOS), "--wrist", str(FASE1_CAM_WRIST),
           "--top", str(FASE1_CAM_TOP), "--device", FASE1_DEVICE, "--encadenar", "--salida", salida]
    if FASE1_INICIO:
        cmd += ["--inicio", str(FASE1_INICIO)]
    print("[FASE1] " + " ".join(cmd))
    env = dict(os.environ, HF_HUB_OFFLINE="1", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    t0 = time.time()
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                                text=True, encoding="utf-8", errors="replace", bufsize=1)
        for linea in proc.stdout:                      # se ve en la consola y queda en consola.log
            print("    " + linea.rstrip())
        proc.wait()
    except KeyboardInterrupt:
        print("[FASE1] Interrumpida con Ctrl+C")
        try:
            proc.terminate()
        except Exception:
            pass
        return False
    print(f"[FASE1] Terminada en {time.time() - t0:.0f} s (código {proc.returncode}). Datos en {salida}")
    return proc.returncode == 0
IBVS_PUERTO = "COM13"         # follower (si ROBOT_PUERTO no es None se usa la conexión de LectorRobot)
IBVS_ROBOT_ID = "mi_follower"  # id de la calibración del follower (NO recalibrar)
IBVS_SIN_ROBOT = False        # True: solo visión (ver si detecta la luz), sin mover el robot
MAX_RELATIVE_TARGET = 8.0     # grados máx. por comando respecto a la posición actual (seguridad)

# Pose de SEGURIDAD (pinza apuntando abajo, "L invertida", como el final de las demos de la fase 1).
# Antes del IBVS el robot va a ella despacio. Se guarda leyendo el brazo real (tecla 'g' en la ventana
# IBVS, o el comando de una línea) en pose_seguridad.json. Si el archivo no existe, se usa POSE_SEGURIDAD_DEG.
IR_A_POSE_SEGURIDAD = True
POSE_ARCHIVO = "pose_seguridad.json"
POSE_SEGURIDAD_DEG = None     # p. ej. {"shoulder_pan": 0, "shoulder_lift": -20, "elbow_flex": 30, "wrist_flex": 75, "wrist_roll": 0}
VEL_POSE_DEG_S = 45.0         # velocidad para ir a la pose de seguridad (wrist_roll gira ~155 grados: ~3,5 s)

GANANCIA = 0.4               # Kp (proporcional)
KI = 0.10                     # Ki (integral, por iteración). 0 = solo P
I_ZONA_PX = 12.0              # solo se integra cuando el error es menor que esto (evita windup lejos)
I_MAX_PX = 15.0               # límite del término integral (anti-windup)
N_FILTRO = 3                  # medidas (mediana) por iteración del control
PASO_MAX_MM = 5.0             # desplazamiento máx. del efector por iteración
PASO_ID_MM = 10.0             # movimientos de prueba para medir J
BROYDEN_BETA = 0.3
TOL_PX = 3.0                  # error aceptado (px de la cámara externa)
N_OK = 6                      # iteraciones seguidas dentro de tolerancia
TIMEOUT_SERVO_S = 90.0
DESPL_MAX_MM = 150.0          # no alejarse más de esto de donde estaba al pulsar 'i'
VEL_DEG_S = 30.0              # velocidad máx. de articulación
ASENTAR_S = 0.25              # espera tras cada paso antes de medir
JOG_MM = 5.0

APUNTAR_ART = ("wrist_flex", "wrist_roll")   # re-apuntar girando la linterna SIN desplazar el brazo
PASO_ID_DEG = 2.0
PASO_MAX_DEG = 3.0
TIMEOUT_APUNTAR_S = 20.0

R_VERDE_PX = 8                # radio (px) alrededor de la consigna para ver si es verde
VERDE_MIN = 0.70
VIGILAR = True                # al llegar, seguir vigilando (oclusión / desvío) en vez de terminar
OCL_N_FRAMES = 12             # frames seguidos con oclusión (~0.4 s) para reaccionar
OCL_Q_MIN = 0.75              # anillo de PCB iluminada por debajo de esto * referencia -> oclusión
OCL_AZUL_MIN = 0.60           # exceso de azul en el anillo por debajo de esto * referencia -> oclusión
OCL_R_MIN = 0.80              # radio del halo por debajo de esto * referencia -> algo corta el haz
OCL_R_OK = 0.60               # para dar por recuperado basta con esto (una sombra parcial también lo encoge)
OCL_OK = 0.85                 # calidad para dar por recuperada la iluminación
ANILLO_R0, ANILLO_R1 = 13, 28 # anillo (px) alrededor de la consigna (solo informativo)
R_PUNTO_PX = 4                # círculo del "punto" que tiene que estar iluminado (~3 mm)
Z_PCB_MM = 5.0                # altura de la superficie de la PCB en la base del robot (mesa = 0)
ORBITA_PASO_DEG = 8.0
ORBITA_MAX_DEG = 40.0
ORBITA_ELEV_MIN, ORBITA_ELEV_MAX = 35.0, 145.0   # límites del ángulo de elevación (90 = vertical)
# --- Semiesfera
DIST_ILUM_MM = None           # radio D de la semiesfera (mm). None = la distancia actual linterna-punto
N_CIRCUNFERENCIAS = 6         # circunferencias máximas por la vertical del punto, cada 180/N grados (6 -> 12
                              # direcciones, una cada 30 grados alrededor del punto)
# Elevaciones (grados sobre la horizontal) a probar en cada dirección: arcos PRONUNCIADOS desde el primer intento
# (22, 35 y 48 grados de inclinación respecto a la vertical). Antes 82/74/66...: con 8 grados la mano casi
# siempre seguía tapando el haz y hacían falta 10-13 intentos.
# VERSIÓN "ARCOS": en cada dirección se hace UN arco largo y abombado hasta la inclinación MÁXIMA alcanzable
# (se apura la dirección), mirando la luz por el camino: si a mitad de arco el punto ya se ilumina, se para
# ahí. ESFERA_ELEVS solo sirve para buscar hasta dónde se puede bajar en cada dirección.
ESFERA_ELEVS = (75, 65, 55, 45, 35, 25, 15)   # hasta 15 grados: linterna casi paralela al suelo
ESFERA_ALTURA_MIN_MM = 40.0   # la linterna no baja de esta altura sobre la mesa (no rozar la mesa / PCB)
# Ir de una posición a otra por un ARCO ABOMBADO (alejándose del punto a mitad de camino) en vez de en línea
# recta: el obstáculo suele estar entre la linterna y el punto, y la línea recta lo atravesaba.
ABOMBADO = 0.35               # a mitad de camino, la linterna está a D x (1 + ABOMBADO) del punto
ARCO_PUNTOS = 12              # puntos intermedios del arco (más puntos = se mira la luz más a menudo)
# Tras el primer IBVS, acercar la linterna a la PCB a lo largo del haz (foco menos difuminado). 0 = no acercar.
ACERCAR_MM = 30.0
# Si ya se esquivó un obstáculo y vuelve a haber otro, volver primero a la pose de iluminación inicial (la del
# primer IBVS); si allí también está tapado, buscar en la semiesfera.
VOLVER_A_INICIAL = True
# Mismo arco en TODAS las direcciones: si a la distancia D no se alcanza (el lado lejano del robot), se prueba
# la misma inclinación más cerca del punto (D x 0,85, D x 0,7) en vez de descartar esa dirección.
ESFERA_D_FACTORES = (1.0, 0.85, 0.7)
ESFERA_DESVIO_MAX_MM = 12.0   # desvío máx. del haz respecto a P (kinemática) para aceptar un candidato
ESFERA_MAX_INTENTOS = 36      # candidatos (posiciones) que se prueban como mucho
ESFERA_IK_RESTO_MM = 4.0      # resto máx. de la cinemática inversa en posición
# --- ESQUIVE CON PARÁBOLAS (sustituye a la búsqueda por toda la semiesfera)
# Desde la pose de iluminación inicial se traza una PARÁBOLA ABOMBADA hacia cada dirección de la base del robot,
# en este orden: la linterna AVANZA en horizontal desde donde está (lo más posible, hasta el ancho máximo de esa
# dirección) y la pinza gira por el camino para seguir apuntando al punto. Si no ilumina, se vuelve por el mismo
# camino a la pose inicial y se prueba la siguiente.
# (nombre, dirección, orientación, ancho máx. mm): "apuntar" = la pinza gira para seguir apuntando al punto;
# "fija" = la pinza NO se reorienta (mantiene la orientación de la pose inicial)
PARABOLA_DIRS = (("+x", (1.0, 0.0, 0.0), "apuntar", 140.0),
                 ("+y", (0.0, 1.0, 0.0), "apuntar", 100.0),
                 ("+x+y", (0.7071, 0.7071, 0.0), "apuntar", 120.0))     # diagonal entre +x y +y
PARABOLA_PUNTOS = 14          # puntos de la parábola (tras cada uno se mira si el punto ya se ilumina)
PARABOLA_ABOMBADO = 0.5       # altura extra a mitad de camino = PARABOLA_ABOMBADO x D (sube por encima del obstáculo)
                              # si no llega tan alto, se prueba con el 60 % de esto
# ANCHURA: se prueba desde el ancho máximo de la dirección hacia abajo, de PARABOLA_ANCHO_PASO_MM en
# PARABOLA_ANCHO_PASO_MM hasta PARABOLA_ANCHO_MIN_MM, y se queda la primera (la más ancha) que se puede hacer.
PARABOLA_ANCHO_PASO_MM = 20.0
PARABOLA_ANCHO_MIN_MM = 40.0
PARABOLA_BAJAR_MM = (-30.0, 0.0, -60.0)   # el final queda esto más abajo que el inicio (orden de preferencia)
# PROLONGAR: al acabar la parábola la trayectoria sigue y se curva suavemente hacia el punto, BAJANDO hacia la
# PCB y apuntando siempre a él, hasta PARABOLA_PROLONGAR_MM de recorrido, PARABOLA_Z_FINAL_MM sobre la mesa,
# PARABOLA_DIST_MIN_MM del punto o hasta donde se pueda.
PARABOLA_PROLONGAR_MM = 100.0
PARABOLA_PROLONGAR_PASO_MM = 10.0
PARABOLA_Z_FINAL_MM = 50.0
PARABOLA_DIST_MIN_MM = 70.0
# Después, acercamiento extra A LO LARGO DEL HAZ (la luz no se mueve de sitio). 0 = no (ya lo hace la prolongación)
PARABOLA_ACERCAR_MM = 0.0
PARABOLA_ACERCAR_PASO_MM = 10.0
# Pasar de "esquivando" a "iluminando" NO es instantáneo: el punto tiene que verse iluminado de forma continua
# durante esto (brazo quieto). Se toleran ILUMINADO_FALLOS_MAX frames malos seguidos (ruido de la cámara).
ILUMINADO_CONFIRMAR_S = 2.0
ILUMINADO_FALLOS_MAX = 2
PARABOLA_MIN_PUNTOS = 3      # si la parábola no se puede hacer entera, se hace el trozo alcanzable si tiene al menos
                              # estos puntos (como -x y -y, que chocan con el límite de wrist_roll)
PARABOLA_DESVIO_TRAMO_MM = 25.0   # desvío máx. del haz respecto al punto en los puntos intermedios
PARABOLA_WRIST_ROLL_MAX = 70.0    # tope físico de wrist_roll hacia +79 con la linterna tumbada: no acercarse
PARABOLA_SALTO_ROLL_DEG = 40.0    # giro máx. de wrist_roll entre dos puntos seguidos (evita vueltas raras)
SOMBRA_N_FRAMES = 15          # frames seguidos con sombra (~0.5 s) para reaccionar
REAJUSTE_PX = 6.0             # si la luz se aparta más que esto de la consigna (REAJUSTE_N frames) -> IBVS
REAJUSTE_N = 5
SOMBRA_ESPERA_S = 5.0         # tras un reajuste por sombra, esperar esto antes de otro
BUSQ_PASO_MM = 20.0
BUSQ_PASO_MIN_MM = 5.0
BUSQ_RADIO_MAX_MM = 80.0
BUSQ_MARGEN = 0.05
DIRECCIONES = [               # (nombre, dirección en la imagen de la cámara externa)
    ("izquierda", (-1, 0)), ("derecha", (1, 0)),
    ("diag arriba-izq", (-1, -1)), ("diag arriba-der", (1, -1)),
    ("diag abajo-izq", (-1, 1)), ("diag abajo-der", (1, 1)),
    ("arriba", (0, -1)), ("abajo", (0, 1)),
]

# --- EJE DEL HAZ de la linterna en el sistema de la PINZA (gripper_frame_link de la cinemática).
# La linterna va PERPENDICULAR al eje Z de la pinza (girada 90 grados): el haz sale por el eje X o el Y de
# la pinza, así que wrist_roll SÍ cambia hacia dónde apunta (5 grados de libertad útiles: 3 de posición y 2
# de dirección -> la linterna puede apuntar al punto desde casi toda la semiesfera).
# "auto": en la pose de seguridad (linterna apuntando a la PCB) se elige, de +X, -X, +Y, -Y, el eje que
# apunta más hacia abajo. También se puede fijar: "x", "-x", "y", "-y" (o "z" = agarre antiguo).
EJE_HAZ_PINZA = "auto"
EJE_HAZ_VEC = np.array([1.0, 0.0, 0.0])      # se ajusta con elegir_eje_haz()
EJE_HAZ_ELEGIDO = False
IK_ART = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
# Articulaciones a las que el IBVS manda órdenes (todas menos la pinza). wrist_roll: con la calibración de
# este brazo su rango es 0-4095 y -180/+180 grados son el MISMO sitio físico. Se limita a +-175 (no se
# acerca al salto) y nunca se manda una orden que lo cruce (el motor daría casi una vuelta entera).
ART_MOVER = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


def cruza_salto(actual, objetivo):
    """True si ir de `actual` a `objetivo` (wrist_roll) cruza el salto +-180 (vuelta larga del motor)."""
    directo = abs(objetivo - actual)
    return directo > 180.0 and (360.0 - directo) < directo


def elegir_eje_haz(q, forzar=False):
    """Fija EJE_HAZ_VEC. En "auto", el eje (+-X, +-Y de la pinza) que apunta más hacia abajo en la pose q."""
    global EJE_HAZ_VEC, EJE_HAZ_ELEGIDO
    if EJE_HAZ_ELEGIDO and not forzar:
        return
    ejes = {"x": [1, 0, 0], "-x": [-1, 0, 0], "y": [0, 1, 0], "-y": [0, -1, 0], "z": [0, 0, 1], "-z": [0, 0, -1]}
    if EJE_HAZ_PINZA != "auto":
        EJE_HAZ_VEC = np.array(ejes[EJE_HAZ_PINZA], float)
        print(f"[IBVS] Eje del haz fijado: {EJE_HAZ_PINZA} de la pinza")
    else:
        R = cinematica_directa(q)[-1][1][:3, :3]
        dz = {k: float((R @ np.array(v, float))[2]) for k, v in ejes.items() if "z" not in k}
        mejor = min(dz, key=dz.get)
        EJE_HAZ_VEC = np.array(ejes[mejor], float)
        print(f"[IBVS] Eje del haz (auto): {mejor} de la pinza | componente vertical de cada eje: "
              + ", ".join(f"{k}={v:+.2f}" for k, v in dz.items()))
        if dz[mejor] > -0.7:
            print("[IBVS] AVISO: ningún eje apunta claramente hacia abajo en esta pose. Revisa la pose de "
                  "seguridad (la linterna debe apuntar a la PCB) o fija EJE_HAZ_PINZA a mano")
    EJE_HAZ_ELEGIDO = True
LIMITES_DEG = {"shoulder_pan": (-120, 120), "shoulder_lift": (-110, 110), "elbow_flex": (-110, 110),
               "wrist_flex": (-110, 110), "wrist_roll": (-180, 180)}

# La linterna da luz AZUL saturada (medido en los logs: H 92-103, S 140-207, V 200-252, B-(G+R)/2 = +100..120),
# no blanca. Se detecta por "exceso de azul" B - (G+R)/2, que separa la luz de los componentes blancos (+37),
# la alfombrilla (+5) y el cuerpo plateado de la linterna (+3).
PARAMS_LUZ = {"Luz azul min": 40, "Luz V min": 100, "Luz area min": 15, "Luz margen PCB": 150,
              "Sombra V max": 60, "Sombra % max": 30, "Sombra radio": 10}
ESTATICO_MM = 12.0            # si el efector se mueve esto y el "brillo" no se mueve -> no es la luz
ESTATICO_PX = 2.5
J_MAX_DIF_DEG = 10.0          # si la pose actual difiere más de esto de la de J guardada -> medir J otra vez
MEDIR_J_SIEMPRE = False       # medir J al empezar cada IBVS (5 s). La luz, la PCB o la cámara cambian entre pruebas
J_MIN_PX_MM = 0.15            # si mover la pinza 1 mm mueve la "luz" menos que esto, no es la luz (o J está mal)
DIVERGE_FACTOR = 1.3          # si el error crece así 3 pasos seguidos -> se vuelve a medir J (una vez)
RANGOS_LUZ = {"Luz azul min": 255, "Luz V min": 255, "Luz area min": 2000, "Luz margen PCB": 300,
              "Sombra V max": 255, "Sombra % max": 100, "Sombra radio": 40}
J_ARCHIVO = "ibvs_jacobiano.json"
LOGS_DIR = "logs"             # cada ejecución: logs/AAAAMMDD_HHMMSS/ (consola, puntos, IBVS con imágenes)
LOG_IMG_CADA_S = 0.5          # foto anotada cada X s mientras el IBVS mueve el robot (y en cada evento)
LOG_IMG_MAX = 2000            # máximo de imágenes por sesión IBVS


class _Tee:
    """Copia todo lo que sale por consola (print, errores) a un archivo, sin romper la consola."""

    def __init__(self, original, archivo):
        self.original, self.archivo = original, archivo

    def write(self, txt):
        self.original.write(txt)
        try:
            self.archivo.write(txt)
            self.archivo.flush()
        except (OSError, ValueError):
            pass

    def flush(self):
        self.original.flush()

    def __getattr__(self, nombre):
        return getattr(self.original, nombre)


class Registro:
    """Carpeta de logs de la ejecución: consola.log (todo lo impreso) + subcarpetas por punto e IBVS."""

    def __init__(self):
        self.dir = None

    def iniciar(self):
        self.dir = os.path.abspath(os.path.join(LOGS_DIR, time.strftime("%Y%m%d_%H%M%S")))
        try:
            os.makedirs(self.dir, exist_ok=True)
            f = open(os.path.join(self.dir, "consola.log"), "a", encoding="utf-8", errors="replace")
            sys.stdout, sys.stderr = _Tee(sys.stdout, f), _Tee(sys.stderr, f)
            print(f"[LOG] Registro de esta ejecución en: {self.dir}")
            print(f"[LOG] Config: CAMERA_INDEX={CAMERA_INDEX} ROBOT_PUERTO={ROBOT_PUERTO} IBVS_PUERTO={IBVS_PUERTO} "
                  f"IBVS_SIN_ROBOT={IBVS_SIN_ROBOT} GANANCIA={GANANCIA} PASO_MAX_MM={PASO_MAX_MM} "
                  f"PASO_ID_MM={PASO_ID_MM} VEL_DEG_S={VEL_DEG_S} ASENTAR_S={ASENTAR_S} TOL_PX={TOL_PX} "
                  f"VERDE_MIN={VERDE_MIN} PARAMS={PARAMS} PARAMS_LUZ={PARAMS_LUZ}")
            print(f"[LOG] Pose de seguridad: {cargar_pose_seguridad()}")
        except OSError as e:
            print(f"[LOG] No se pudo crear la carpeta de logs ({e}): sigo sin logs")
            self.dir = None

    def sub(self, nombre):
        """Subcarpeta nueva dentro de la ejecución (o None si no hay logs)."""
        base = self.dir or os.path.abspath(os.path.join(LOGS_DIR, time.strftime("%Y%m%d_%H%M%S")))
        d = os.path.join(base, nombre)
        try:
            os.makedirs(d, exist_ok=True)
            return d
        except OSError:
            return None

    def imagen(self, ruta, img):
        try:
            if ruta and img is not None:
                cv2.imwrite(ruta, img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        except (OSError, cv2.error):
            pass


REG = Registro()


# ----------------------------------------------------------------------------- IBVS: robot
class RobotIBVS:
    """SO-101 follower con LeRobot. Se conecta SIN reconfigurar los motores si ya tienen par
    (LeRobot quita el par un instante al configurar y la linterna se caería). La pinza no se toca."""

    def __init__(self, puerto, simulado, lector=None):
        self.puerto, self.simulado, self.lector = puerto, simulado, lector
        self.q = dict(Q_MANUAL_DEG)
        self.estado = "SIN ROBOT (IBVS_SIN_ROBOT)" if simulado else "desconectado"
        self.r, self.prestado = None, False

    def conectar(self):
        if self.simulado:
            return
        if self.lector is not None and self.lector.robot is not None:
            self.lector.pausa.set()                       # LectorRobot ya tiene el puerto abierto
            time.sleep(0.15)
            self.r, self.prestado = self.lector.robot, True
            self.r.config.max_relative_target = MAX_RELATIVE_TARGET
            self.estado = "robot (via LectorRobot)"
            self.leer()
            limites_de_calibracion(self.r)
            return
        try:
            from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        except ImportError:
            try:
                from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
            except ImportError:
                raise RuntimeError("No encuentro LeRobot: ejecuta con C:\\lerobot-env\\Scripts\\python.exe")
        cfg = SO101FollowerConfig(port=self.puerto, id=IBVS_ROBOT_ID, use_degrees=True,
                                  max_relative_target=MAX_RELATIVE_TARGET,
                                  disable_torque_on_disconnect=False)
        self.r = SO101Follower(cfg)
        if not self.r.calibration:
            raise RuntimeError(f"No encuentro la calibración '{IBVS_ROBOT_ID}'. NO recalibres: copia el json")
        try:
            self.r.bus.connect()
        except RuntimeError as e:
            if "Missing motor IDs" in str(e) and "- 6 " in str(e):
                raise RuntimeError("La pinza (motor 6) no responde: ha saltado su protección de sobrecarga al "
                                   "apretar la linterna. Sujeta el brazo, apaga y enciende la fuente del robot "
                                   "y repite la fase 1 (tecla 'l').") from e
            raise
        par = self.r.bus.sync_read("Torque_Enable", normalize=False, num_retry=3)
        if not all(int(v) == 1 for v in par.values()):
            print("[IBVS] Motores sin par -> configuro (LeRobot quita/pone el par un instante)")
            self.r.configure()
        self.estado = f"robot {self.puerto}"
        self.leer()
        limites_de_calibracion(self.r)



    def leer(self):
        if not self.simulado:
            pos = self.r.bus.sync_read("Present_Position", num_retry=3)
            self.q = {a: float(pos[a]) for a in ARTICULACIONES}
        return dict(self.q)

    def enviar(self, q):
        q = limitar(q)
        if self.simulado:
            self.q.update({a: q[a] for a in ARTICULACIONES})
            return
        if cruza_salto(self.q.get("wrist_roll", q["wrist_roll"]), q["wrist_roll"]):
            q = dict(q)                                                  # nunca cruzar el salto +-180
            q["wrist_roll"] = self.q["wrist_roll"]
        accion = {f"{a}.pos": float(q[a]) for a in ART_MOVER}          # sin la pinza (no se toca)
        for intento in range(3):
            try:
                self.r.send_action(accion)
                return
            except Exception as e:                                       # paquete perdido en el bus
                if intento == 2:
                    raise RuntimeError(f"Fallo enviando al robot: {e}")
                time.sleep(0.02)

    def desconectar(self):
        if self.prestado:
            self.lector.pausa.clear()
            return
        if self.r is not None:
            try:
                self.r.bus.disconnect(False)                             # False: mantener el par
            except Exception:
                pass


def limites_de_calibracion(robot, margen=2.0):
    """LIMITES_DEG reales a partir de la calibración (grados de LeRobot con use_degrees=True)."""
    try:
        for a in IK_ART:
            c = robot.calibration[a]
            mid = (c.range_min + c.range_max) / 2
            lo, hi = (c.range_min - mid) * 360 / 4095, (c.range_max - mid) * 360 / 4095
            m_ = 5.0 if a == "wrist_roll" else margen       # wrist_roll: lejos del salto +-180
            LIMITES_DEG[a] = (lo + m_, hi - m_)
        print("[IBVS] límites (calibración): " + ", ".join(f"{a}=[{LIMITES_DEG[a][0]:.0f}, {LIMITES_DEG[a][1]:.0f}]"
                                                          for a in IK_ART))
    except Exception as e:
        print(f"[IBVS] aviso: no pude leer los límites de la calibración ({e}); uso los genéricos")


def limitar(q):
    return {a: float(np.clip(v, *LIMITES_DEG.get(a, (-180, 180)))) for a, v in q.items()}


def fk(q):
    """Posición (mm) de la pinza y DIRECCIÓN DEL HAZ de la linterna (eje EJE_HAZ_VEC de la pinza) en la base."""
    T = cinematica_directa(q)[-1][1]
    return T[:3, 3].copy(), T[:3, :3] @ EJE_HAZ_VEC


def _pinv_amort(A, lam):
    return A.T @ np.linalg.inv(A @ A.T + lam ** 2 * np.eye(A.shape[0]))


def ik(q, p_obj, z_obj, iters=25, lam=0.5):
    """Cinemática inversa numérica con prioridades: 1) posición p_obj, 2) eje de la pinza z_obj
    (lo que permita la articulación sobrante: 4 articulaciones no dan para todo)."""
    qk = dict(q)
    n = len(IK_ART)
    for _ in range(iters):
        p, z = fk(qk)
        e1, e2 = p_obj - p, z_obj - z
        if np.linalg.norm(e1) < 0.1 and np.linalg.norm(e2) < 0.005:
            break
        J1, J2 = np.zeros((3, n)), np.zeros((3, n))
        for j, a in enumerate(IK_ART):
            q2 = dict(qk)
            q2[a] += 0.5
            p2, z2 = fk(q2)
            J1[:, j] = (p2 - p) / 0.5
            J2[:, j] = (z2 - z) / 0.5
        # Articulaciones en su tope que el paso empujaría más allá: se bloquean (columna a cero) y el
        # resto compensa la posición. Así una muñeca en su límite no impide moverse.
        bloq = set()
        for _ in range(n):
            J1m, J2m = J1.copy(), J2.copy()
            for j in bloq:
                J1m[:, j] = 0.0
                J2m[:, j] = 0.0
            J1p = _pinv_amort(J1m, lam)
            dq1 = J1p @ e1
            N = np.eye(n) - J1p @ J1m
            dq = dq1 + N @ (_pinv_amort(J2m @ N, 0.02) @ (e2 - J2m @ dq1))
            nuevos = set()
            for j, a in enumerate(IK_ART):
                lo, hi = LIMITES_DEG.get(a, (-180, 180))
                if j not in bloq and ((dq[j] > 0 and qk[a] >= hi - 0.5) or (dq[j] < 0 and qk[a] <= lo + 0.5)):
                    nuevos.add(j)
            if not nuevos:
                break
            bloq |= nuevos
        dq = np.clip(dq, -5.0, 5.0)
        for j, a in enumerate(IK_ART):
            qk[a] += dq[j]
        qk = limitar(qk)
    p, _ = fk(qk)
    return qk, float(np.linalg.norm(p_obj - p))


# ----------------------------------------------------------------------------- IBVS: visión (cámara externa)
def zona_pcb(geo, shape, margen):
    z = np.zeros(shape[:2], np.uint8)
    cv2.fillConvexPoly(z, geo["c"].astype(np.int32), 255)
    if margen > 0:
        z = cv2.dilate(z, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margen + 1, 2 * margen + 1)))
    return z


def segmentar_luz(bgr, hsv, zona, previo, excluir=None):
    """Halo de la linterna: EXCESO DE AZUL B-(G+R)/2 >= "Luz azul min" y brillo V >= "Luz V min"
    (+ el núcleo quemado blanco pegado a esa zona azul), fuera de `excluir` (brillos fijos).
    Centroide ponderado por el exceso de azul. (máscara, centroide | None, radio, candidatos)."""
    P = PARAMS_LUZ
    b, g, r = [c.astype(np.int16) for c in cv2.split(bgr)]
    exceso = b - (g + r) // 2
    azul = ((exceso >= P["Luz azul min"]) & (hsv[:, :, 2] >= P["Luz V min"])).astype(np.uint8) * 255
    quemado = ((hsv[:, :, 2] >= 250) & (hsv[:, :, 1] <= 60)).astype(np.uint8) * 255
    m = cv2.bitwise_or(azul, cv2.bitwise_and(quemado, cv2.dilate(azul, np.ones((15, 15), np.uint8))))
    m = cv2.bitwise_and(m, zona)
    if excluir is not None:
        m = cv2.bitwise_and(m, cv2.bitwise_not(excluir))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mejor, mejor_p, cands = None, -1.0, []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < P["Luz area min"]:
            continue
        mo = cv2.moments(c)
        if mo["m00"] <= 0:
            continue
        x, y = mo["m10"] / mo["m00"], mo["m01"] / mo["m00"]
        cands.append((x, y, a))
        punt = a                                                 # el halo es el blob azul más grande
        if previo is not None:                                   # seguimiento: preferir el cercano
            punt /= 1.0 + np.hypot(x - previo[0], y - previo[1]) / 60.0
        if punt > mejor_p:
            mejor, mejor_p = c, punt
    if mejor is None:
        return m, None, 0.0, cands
    blob = np.zeros_like(m)
    cv2.drawContours(blob, [mejor], -1, 255, cv2.FILLED)
    w = np.clip(exceso, 0, None).astype(np.float32) * (blob > 0)
    M = cv2.moments(w)                                           # centroide ponderado por el azul
    if M["m00"] <= 0:
        return m, None, 0.0, cands
    c = np.array([M["m10"] / M["m00"], M["m01"] / M["m00"]], np.float32)
    return m, c, float(np.sqrt(cv2.contourArea(mejor) / np.pi)), cands


def fraccion_verde(hsv, centro, r):
    """Fracción del círculo (radio r FIJO) en la consigna que es "PCB visible": color de PCB, verde sin
    luz o cian/azul/violeta con la luz azul encima (tono PCB H min..130, S >= la mitad de PCB S min),
    o el núcleo blanco quemado de la luz. Lo demás (brazo naranja, cinta negra, cuerpo plateado de la
    linterna, alfombrilla gris) cuenta como obstáculo. None si el círculo se sale de la imagen."""
    h, w = hsv.shape[:2]
    x, y = int(round(centro[0])), int(round(centro[1]))
    if not (r <= x < w - r and r <= y < h - r):
        return None
    circ = np.zeros((2 * r + 1, 2 * r + 1), np.uint8)
    cv2.circle(circ, (r, r), r, 1, -1)
    circ = circ > 0
    sub = hsv[y - r:y + r + 1, x - r:x + r + 1].astype(np.int16)
    H, S, V = sub[:, :, 0], sub[:, :, 1], sub[:, :, 2]
    pcb = (H >= PARAMS["PCB H min"]) & (H <= 130) & (S >= 0.5 * PARAMS["PCB S min"]) & (V >= PARAMS["PCB V min"])
    nucleo = (V >= 230) & (S <= 80)
    return float(((pcb | nucleo) & circ).sum()) / float(circ.sum())


def cargar_pose_seguridad():
    try:
        with open(POSE_ARCHIVO, encoding="utf-8") as f:
            d = json.load(f)
        return {a: float(d[a]) for a in ARTICULACIONES if a in d}
    except (OSError, ValueError, KeyError):
        return dict(POSE_SEGURIDAD_DEG) if POSE_SEGURIDAD_DEG else None


def guardar_pose_seguridad(q):
    d = {a: round(float(q[a]), 1) for a in ARTICULACIONES}
    with open(POSE_ARCHIVO, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)
    return d


def fraccion_oscura(hsv, centro, r):
    """Fracción de píxeles OSCUROS (sombra: V < "Sombra V max") en un círculo de radio r alrededor de la
    consigna. None si el círculo se sale de la imagen."""
    h, w = hsv.shape[:2]
    x, y = int(round(centro[0])), int(round(centro[1]))
    r = max(int(r), 2)
    if not (r <= x < w - r and r <= y < h - r):
        return None
    circ = np.zeros((2 * r + 1, 2 * r + 1), np.uint8)
    cv2.circle(circ, (r, r), r, 1, -1)
    circ = circ > 0
    v = hsv[y - r:y + r + 1, x - r:x + r + 1, 2]
    return float(((v < PARAMS_LUZ["Sombra V max"]) & circ).sum()) / float(circ.sum())


def calidad_anillo(hsv, exceso, centro, r0=None, r1=None):
    """(% de PCB iluminada (cian) en el anillo r0..r1 alrededor de la consigna, exceso de azul medio)."""
    r0 = ANILLO_R0 if r0 is None else r0
    r1 = ANILLO_R1 if r1 is None else r1
    h, w = hsv.shape[:2]
    x, y = int(round(centro[0])), int(round(centro[1]))
    if not (r1 <= x < w - r1 and r1 <= y < h - r1):
        return None, None
    yy, xx = np.mgrid[-r1:r1 + 1, -r1:r1 + 1]
    d = np.hypot(xx, yy)
    ani = (d >= r0) & (d <= r1)
    sub = hsv[y - r1:y + r1 + 1, x - r1:x + r1 + 1].astype(np.int16)
    H, S, V = sub[:, :, 0], sub[:, :, 1], sub[:, :, 2]
    luz = (H >= 80) & (H <= 130) & (S >= 80) & (V >= 120)
    ex = exceso[y - r1:y + r1 + 1, x - r1:x + r1 + 1]
    return float(luz[ani].mean()), float(ex[ani].mean())


def calidad_punto(hsv, centro, r=None):
    """Fracción del círculo PEQUEÑO (R_PUNTO_PX) sobre la consigna que está ILUMINADA: PCB cian bajo la
    luz azul o núcleo blanco quemado del halo. Es lo que importa: que el PUNTO tenga luz."""
    r = R_PUNTO_PX if r is None else r
    h, w = hsv.shape[:2]
    x, y = int(round(centro[0])), int(round(centro[1]))
    if not (r <= x < w - r and r <= y < h - r):
        return None
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    disco = np.hypot(xx, yy) <= r
    sub = hsv[y - r:y + r + 1, x - r:x + r + 1].astype(np.int16)
    H, S, V = sub[:, :, 0], sub[:, :, 1], sub[:, :, 2]
    lit = ((H >= 80) & (H <= 130) & (S >= 80) & (V >= 120)) | ((V >= 235) & (S <= 70))
    return float(lit[disco].mean())


def anomalia_halo(hsv, centro, radio):
    """Centro (relativo a la consigna, px) de los píxeles dentro del halo que NO son PCB iluminada ni el
    núcleo quemado: dónde está lo que tapa. None si casi no hay."""
    h, w = hsv.shape[:2]
    r = int(max(radio, ANILLO_R1))
    x, y = int(round(centro[0])), int(round(centro[1]))
    x0, x1, y0, y1 = max(0, x - r), min(w, x + r + 1), max(0, y - r), min(h, y + r + 1)
    sub = hsv[y0:y1, x0:x1].astype(np.int16)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dentro = np.hypot(xx - centro[0], yy - centro[1]) <= r
    H, S, V = sub[:, :, 0], sub[:, :, 1], sub[:, :, 2]
    ok = ((H >= 80) & (H <= 130) & (S >= 80) & (V >= 120)) | ((V >= 240) & (S <= 70) &
                                                            (np.hypot(xx - centro[0], yy - centro[1]) < ANILLO_R0))
    raro = dentro & ~ok
    if raro.sum() < 30:
        return None
    return np.array([xx[raro].mean() - centro[0], yy[raro].mean() - centro[1]])


def hay_sombra(sombra):
    return sombra is not None and 100.0 * sombra > PARAMS_LUZ["Sombra % max"]


# ----------------------------------------------------------------------------- IBVS: control
class Controlador:
    """Secuencias escritas como generadores: cada `yield` = esperar al siguiente frame (la ventana
    sigue viva mientras el robot se mueve)."""

    def __init__(self, s):
        self.s = s                 # SesionIBVS
        self.J = None              # 2x2  de/dx  (px por mm del efector en X, Y de la base)
        self.paso_parcial = False  # el último paso no mantuvo la orientación (articulación en su tope)
        self.q_ref = None          # pose en la que se midió J
        self.j_sesion = False      # J medida en ESTA sesión (no la del archivo)
        self.j_ap_sesion = False   # J_ap medida en ESTA sesión
        self.J_ap = None           # 2x2  de/dq  (px por grado de APUNTAR_ART)
        try:
            with open(J_ARCHIVO, encoding="utf-8") as f:
                d = json.load(f)
            self.J = np.array(d["J"]) if d.get("J") else None
            self.J_ap = np.array(d["J_ap"]) if d.get("J_ap") else None
            self.q_ref = d.get("q_ref")
        except (OSError, ValueError, KeyError):
            pass
        self.J_archivo = None if self.J is None else self.J.copy()   # para volver a ella sin movimientos de prueba

    def restaurar_j(self):
        """En vez de re-medir J con los 4 movimientos de prueba, vuelve a la J del archivo (Broyden la afina)."""
        if self.J_archivo is not None:
            self.J = self.J_archivo.copy()
        self.j_sesion = False

    def guardar(self):
        d = {"J": None if self.J is None else self.J.tolist(),
             "J_ap": None if self.J_ap is None else self.J_ap.tolist(), "fecha": time.strftime("%F %T"),
             "q_ref": self.q_ref}
        try:
            with open(J_ARCHIVO, "w", encoding="utf-8") as f:
                json.dump(d, f, indent=2)
        except OSError as e:
            print(f"[IBVS] aviso: no pude guardar {J_ARCHIVO} ({e})")
        self.s.evento(f"J = {d['J']}  J_ap = {d['J_ap']}")

    def estado(self, txt):
        self.s.estado = txt
        print(f"[IBVS] {txt}")
        self.s.evento(txt)

    def esperar(self, t):
        t0 = time.time()
        while time.time() - t0 < t:
            yield

    def medir(self, n=4, timeout=3.0):
        """Mediana de n errores e = luz - consigna (px) en frames NUEVOS. None si no se ve la luz."""
        datos, ult, t0 = [], -1, time.time()
        while len(datos) < n:
            if time.time() - t0 > timeout:
                return None
            m = self.s.med
            if m["id"] != ult and m["e"] is not None:
                datos.append(m["e"])
                ult = m["id"]
            yield
        return np.median(np.array(datos, float), axis=0)

    def ir_a(self, q_obj, asentar=ASENTAR_S, vel=None):
        """Lleva las articulaciones a q_obj interpolando a VEL_DEG_S (o a `vel`)."""
        vel = VEL_DEG_S if vel is None else vel
        q_obj = limitar(q_obj)              # nunca pedir más allá de los límites de la calibración
        s = self.s
        t = time.time()
        while True:
            dq = {a: q_obj[a] - s.q_cmd[a] for a in q_obj}
            mx = max(abs(v) for v in dq.values())
            if mx < 1e-3:
                break
            ahora = time.time()
            f = min(1.0, vel * max(ahora - t, 1 / 60) / mx)
            t = ahora
            for a in dq:
                s.q_cmd[a] += dq[a] * f
            s.robot.enviar(s.q_cmd)
            yield
        yield from self.esperar(asentar)

    def mover_cart(self, d, z_eje):
        """Desplaza el efector d (mm, base) con la orientación fija. Devuelve el desplazamiento real."""
        s = self.s
        p, _ = fk(s.q_cmd)
        if s.p_inicio is not None and np.linalg.norm(p + d - s.p_inicio) > DESPL_MAX_MM:
            raise RuntimeError(f"Me alejaría más de {DESPL_MAX_MM:.0f} mm del inicio: paro")
        q1, res = ik(s.q_cmd, p + np.asarray(d, float), z_eje)
        if res > 10.0:
            raise RuntimeError(f"Cinemática inversa no alcanza el punto (resto {res:.1f} mm)")
        self.paso_parcial = res > 1.0 or abs(1.0 - float(fk(q1)[1] @ z_eje)) > 1e-3   # tope / orientación cambia
        if res > 1.0:                       # parcial: se ejecuta lo alcanzable, la imagen corrige el resto
            topes = [a for a in IK_ART if min(abs(q1[a] - LIMITES_DEG[a][0]), abs(q1[a] - LIMITES_DEG[a][1])) < 1.0]
            print(f"[IBVS] aviso: paso parcial (faltan {res:.1f} mm); articulaciones en su tope: {topes}")
        yield from self.ir_a(q1)
        return fk(s.q_cmd)[0] - p

    def marcar_estatico(self, pt, motivo, area=None):
        """Se excluye SOLO el área del brillo fijo (+3 px), no más: la luz puede pasar al lado."""
        if area is None:
            area = np.pi * self.s.med.get("r_luz", 4.0) ** 2
        r = float(np.sqrt(area / np.pi)) + 3.0
        self.s.estaticos.append((float(pt[0]), float(pt[1]), r))
        self.s.luz_prev = None
        self.estado(f"Descarto brillo FIJO en ({pt[0]:.0f},{pt[1]:.0f}) px: {motivo}")

    def aprender_estaticos(self, cands_antes, p_antes):
        """Brillos que siguen en el mismo sitio tras mover mucho el efector no son la luz."""
        p_ahora = fk(self.s.q_cmd)[0]
        if np.linalg.norm(p_ahora - p_antes) < 20.0:
            return
        for (u, v, area) in self.s.med.get("cands", []):
            if any(np.hypot(u - a, v - b) < 4.0 for (a, b, _) in cands_antes):
                self.marcar_estatico((u, v), f"no se movió aunque la pinza se movió {np.linalg.norm(p_ahora - p_antes):.0f} mm",
                                     area)

    def comprobar_luz(self):
        if self.s.med["luz"] is None:
            raise RuntimeError("No veo el circulo de luz sobre la PCB (ajusta umbrales con 't', mira 'm')")

    # ---------------------------------------------------------------- J por movimientos de prueba
    def identificar(self):
        self.comprobar_luz()
        s = self.s
        q0 = dict(s.q_cmd)
        _, z0 = fk(q0)
        J = np.zeros((2, 2))
        for i in range(2):
            meds = []
            for signo in (+1, -1):
                self.estado(f"Midiendo J: {'XY'[i]} {'+' if signo > 0 else '-'}{PASO_ID_MM:.0f} mm")
                d = np.zeros(3)
                d[i] = signo * PASO_ID_MM
                real = yield from self.mover_cart(d, z0)
                m = yield from self.medir(6)
                yield from self.ir_a(q0)
                if m is None:
                    raise RuntimeError("Perdí la luz al medir J (¿sale de la PCB? baja PASO_ID_MM)")
                meds.append((m, real[i]))
            (mp, dp), (mn, dn) = meds
            J[:, i] = (mp - mn) / (dp - dn)
        cols = np.linalg.norm(J, axis=0)
        if cols.min() < J_MIN_PX_MM:
            raise RuntimeError(f"La luz detectada apenas se mueve al mover la pinza ({cols.round(2).tolist()} px/mm): "
                               f"no es la luz o está tapada. Revisa 'm' y 't'")
        if abs(np.linalg.det(J)) < 0.25 * cols.prod():
            raise RuntimeError(f"J casi singular (X e Y mueven la luz en la misma dirección): {J.round(2).tolist()}")
        self.J = J
        self.j_sesion = True
        self.q_ref = {a: round(float(q0[a]), 1) for a in IK_ART}
        # no se guarda aquí: solo al CONVERGER (guardar() en servo), para no dejar en el archivo una J mala
        self.estado(f"J medida (px/mm): {J.round(2).tolist()}")

    # ---------------------------------------------------------------- servo IBVS
    def servo(self, timeout=None):
        """IBVS con PI: u = Kp*e + Ki*sum(e) (integral solo cerca, limitada y que se descarga al cambiar
        de signo), zona muerta (dentro de TOL_PX no se mueve) y mediana de N_FILTRO medidas."""
        self.comprobar_luz()
        s = self.s
        # J = px que se mueve la luz por mm que se mueve el efector CON LA ORIENTACIÓN FIJA: la luz se desplaza
        # sobre la mesa lo mismo que el efector, así que J depende de la cámara y de dónde está la luz en la
        # imagen, no de la pose del brazo. Se reutiliza la guardada (la de la última vez que CONVERGIÓ, ya
        # afinada con Broyden) sin hacer los 4 movimientos de prueba; solo se mide si no hay ninguna guardada
        # o si el servo ve que no funciona.
        if self.J is not None and not self.j_sesion and MEDIR_J_SIEMPRE:
            self.estado("Mido J al empezar (MEDIR_J_SIEMPRE)")
            self.J = None
        if self.J is None:
            yield from self.identificar()
        _, z_eje = fk(s.q_cmd)
        ok_seg, t0, previo, atasco = 0, time.time(), None, 0
        integ = np.zeros(2)
        hist = deque(maxlen=4)                 # (desplazamiento real del efector, posición luz) -> brillos fijos
        normas, remedida, contrario = deque(maxlen=3), False, 0
        q_ok = dict(s.q_cmd)                   # última pose en la que se vio la luz
        dx_ultimo = np.zeros(2)
        self.estado("Servo IBVS (PI): llevando la luz a la consigna")
        while True:
            if time.time() - t0 > (TIMEOUT_SERVO_S if timeout is None else timeout):
                self.estado("Servo: tiempo agotado")
                return False
            e = yield from self.medir(N_FILTRO)
            if e is None:
                if remedida:
                    self.estado("Servo: perdí la luz (ajusta 't' / mira 'm')")
                    return False
                self.estado("Perdí la luz: vuelvo a donde la veía (sin movimientos de prueba)")
                remedida = True
                yield from self.ir_a(q_ok, asentar=0.5)
                self.restaurar_j()
                normas.clear()
                hist.clear()
                previo, integ, dx_ultimo = None, np.zeros(2), np.zeros(2)
                continue
            q_ok = dict(s.q_cmd)
            luz = e + s.consigna_px
            hist.append((dx_ultimo.copy(), luz))
            if len(hist) == hist.maxlen and self.j_sesion:
                # px que la luz DEBERÍA haberse movido según J (medida en esta sesión) vs lo que se movió
                previsto = sum(np.linalg.norm(self.J @ h[0]) for h in list(hist)[1:])
                camino = sum(np.linalg.norm(h[0]) for h in list(hist)[1:])
                mov_luz = max(np.linalg.norm(h[1] - hist[0][1]) for h in hist)
                if previsto > 4 * ESTATICO_PX and camino > ESTATICO_MM and mov_luz < ESTATICO_PX:
                    self.marcar_estatico(luz, f"la pinza recorrió {camino:.0f} mm (debía mover la luz "
                                              f"{previsto:.0f} px) y no se movió")
                    hist.clear()
                    previo, integ = None, np.zeros(2)
                    yield from self.esperar(0.2)
                    continue
            if previo is not None and BROYDEN_BETA > 0 and not self.paso_parcial:   # corrección de Broyden
                e_ant, dxr = previo
                if np.linalg.norm(dxr) > 0.5:
                    self.J += BROYDEN_BETA * np.outer((e - e_ant) - self.J @ dxr, dxr) / (dxr @ dxr)
            norma = float(np.hypot(*e))
            s.log(["servo", *np.round(e, 2).tolist(), "I", *np.round(integ, 2).tolist()])
            normas.append(norma)
            # ¿la luz se movió hacia donde J predecía? (coseno entre cambio real y previsto)
            # (con una articulación en su tope la orientación cambia y J predice peor: se exige más)
            if previo is not None and np.linalg.norm(previo[1]) > 0.5:
                previsto = self.J @ previo[1]
                real_de = e - previo[0]
                if np.linalg.norm(previsto) > 2.0 and np.linalg.norm(real_de) > 2.0:
                    cos = float(previsto @ real_de / (np.linalg.norm(previsto) * np.linalg.norm(real_de)))
                    aleja = norma > float(np.hypot(*previo[0]))
                    umbral = -0.5 if self.paso_parcial else 0.0
                    contrario = contrario + 1 if (cos < umbral and aleja) else 0
            crece = len(normas) == normas.maxlen and all(normas[k + 1] > normas[k] for k in range(len(normas) - 1)) \
                and normas[-1] > DIVERGE_FACTOR * normas[0] and normas[-1] > 3 * TOL_PX
            if contrario >= 2 or crece:
                contrario = 0
                if remedida:
                    self.estado(f"Servo: la luz no va hacia la consigna ({normas[0]:.0f} -> {normas[-1]:.0f} px) "
                                f"incluso con J nueva. Paro")
                    return False
                self.estado(f"La luz no se mueve como predice J ({normas[0]:.0f} -> {normas[-1]:.0f} px): "
                            f"vuelvo a la J guardada (sin movimientos de prueba)")
                remedida = True
                yield from self.ir_a(q_ok, asentar=0.3)
                self.restaurar_j()
                normas.clear()
                hist.clear()
                previo, integ, dx_ultimo = None, np.zeros(2), np.zeros(2)
                continue
            if norma < TOL_PX:                 # zona muerta: no se mueve (no "tiembla")
                ok_seg += 1
                previo = None
                dx_ultimo = np.zeros(2)
                if ok_seg >= N_OK:
                    self.guardar()
                    self.estado("Servo: CONVERGIDO")
                    return True
                continue
            ok_seg = 0
            # integral con anti-windup
            if norma < I_ZONA_PX:
                if previo is not None:
                    cambio = np.sign(e) != np.sign(previo[0])
                    integ[cambio] *= 0.5        # se ha pasado de largo: descargar ese eje
                integ = np.clip(integ + e, -I_MAX_PX, I_MAX_PX)
            else:
                integ[:] = 0.0
            u = GANANCIA * e + KI * integ
            dx = -np.linalg.solve(self.J, u)
            nrm = np.linalg.norm(dx)
            if nrm > PASO_MAX_MM:
                dx *= PASO_MAX_MM / nrm
            real = yield from self.mover_cart([dx[0], dx[1], 0.0], z_eje)
            dx_ultimo = np.asarray(real[:2], float)
            atasco = atasco + 1 if np.linalg.norm(real[:2]) < 0.3 else 0
            if atasco >= 5:
                self.estado("Servo: el brazo no puede moverse más en esa dirección (límites). Paro")
                return False
            previo = (e, real[:2])

    # ---------------------------------------------------------------- re-apuntar (pan + wrist_flex)
    def identificar_apuntar(self):
        s = self.s
        q0 = dict(s.q_cmd)
        J = np.zeros((2, len(APUNTAR_ART)))
        for j, a in enumerate(APUNTAR_ART):
            meds = []
            for signo in (+1, -1):
                q1 = dict(q0)
                q1[a] += signo * PASO_ID_DEG
                yield from self.ir_a(q1)
                m = yield from self.medir(5)
                if m is None:
                    yield from self.ir_a(q0)
                    raise RuntimeError("Perdí la luz midiendo el re-apuntado")
                meds.append(m)
            J[:, j] = (meds[0] - meds[1]) / (2 * PASO_ID_DEG)
            yield from self.ir_a(q0)
        self.J_ap = J
        self.j_ap_sesion = True
        self.guardar()

    def apuntar(self):
        """IBVS en articulaciones: lleva la luz a la consigna girando pan + wrist_flex."""
        s = self.s
        if self.J_ap is None or not self.j_ap_sesion:
            self.estado("Midiendo re-apuntado")
            yield from self.identificar_apuntar()
        t0, ok_seg, previo = time.time(), 0, None
        while time.time() - t0 < TIMEOUT_APUNTAR_S:
            e = yield from self.medir(2)
            if e is None:
                return False
            if previo is not None:
                e_ant, dq = previo
                if np.linalg.norm(dq) > 0.1:
                    self.J_ap += BROYDEN_BETA * np.outer((e - e_ant) - self.J_ap @ dq, dq) / (dq @ dq)
            ok_seg = ok_seg + 1 if np.hypot(*e) < TOL_PX * 1.5 else 0
            if ok_seg >= 4:
                return True
            dq = np.clip(-GANANCIA * np.linalg.lstsq(self.J_ap, e, rcond=None)[0], -PASO_MAX_DEG, PASO_MAX_DEG)
            q1 = dict(s.q_cmd)
            for j, a in enumerate(APUNTAR_ART):
                q1[a] += dq[j]
            yield from self.ir_a(q1)
            previo = (e, dq)
        return False

    # ---------------------------------------------------------------- ¿verde? (obstáculos)
    def _muestras(self, clave, n=8, timeout=3.0):
        vals, ult, t0 = [], -1, time.time()
        while len(vals) < n and time.time() - t0 < timeout:
            m = self.s.med
            if m["id"] != ult:
                ult = m["id"]
                if m.get(clave) is not None:
                    vals.append(m[clave])
            yield
        return float(np.median(vals)) if vals else None

    def evaluar_verde(self):
        """¿Se ve la PCB en la consigna (sin nada delante)? Mediana de 10 frames, SIN mover el robot
        (antes se apartaba la luz con pan/wrist_flex y eso hacía oscilar el brazo)."""
        v = yield from self._muestras("verde", n=10)
        v = 0.0 if v is None else v
        sombra = yield from self._muestras("sombra", n=10)
        print(f"        PCB visible en la consigna: {100 * v:.0f}% | sombra: "
              f"{'--' if sombra is None else f'{100 * sombra:.0f}%'}")
        if hay_sombra(sombra):
            # con sombra nunca se da por bueno (< VERDE_MIN), pero menos sombra = mejor puntuación
            v = min(v, 0.5 * (1.0 - sombra))
        return v

    # ---------------------------------------------------------------- búsqueda heurística
    def dir_base(self, d_img):
        """Dirección en la imagen de la cámara externa -> dirección del efector en la base (X-Y)."""
        v = np.linalg.solve(self.J, np.asarray(d_img, float))
        return np.array([v[0], v[1], 0.0]) / (np.linalg.norm(v) + 1e-9)

    def probar(self, q0, offset, nombre):
        s = self.s
        p0, z0 = fk(q0)
        self.estado(f"Busqueda: {nombre}  offset ({offset[0]:.0f},{offset[1]:.0f}) mm")
        q1, res = ik(q0, p0 + offset, z0)
        if res > 3.0:
            return False, 0.0, None
        # pre-apuntado: lo que se movería la luz con la traslación, compensado con pan/wrist_flex
        dq = -np.linalg.lstsq(self.J_ap, self.J @ offset[:2], rcond=None)[0]
        for j, a in enumerate(APUNTAR_ART):
            q1[a] += float(np.clip(dq[j], -15, 15))
        yield from self.ir_a(limitar(q1), asentar=0.3)
        ok = yield from self.apuntar()
        if not ok:
            return False, 0.0, dict(s.q_cmd)
        v = yield from self.evaluar_verde()
        s.log(["busqueda", nombre, offset[0], offset[1], v])
        return True, v, dict(s.q_cmd)

    def busqueda(self, q0=None, v0=None):
        s = self.s
        if self.J is None:
            yield from self.identificar()
        q0 = dict(s.q_cmd) if q0 is None else q0
        if v0 is None:
            v0 = yield from self.evaluar_verde()
        if self.J_ap is None or not self.j_ap_sesion:
            self.estado("Midiendo re-apuntado")
            yield from self.identificar_apuntar()
        mejor_v, mejor_off, mejor_q = v0, np.zeros(3), dict(q0)
        centro, paso, ampliado = np.zeros(3), BUSQ_PASO_MM, False
        while paso >= BUSQ_PASO_MIN_MM:
            mejoro = False
            for nombre, d in DIRECCIONES:
                u = self.dir_base(d)
                k = 1
                while True:
                    off = centro + u * paso * k
                    if np.linalg.norm(off) > BUSQ_RADIO_MAX_MM:
                        break
                    ok, v, q = yield from self.probar(q0, off, nombre)
                    print(f"        -> {nombre}: verde {v:.2f}" if ok else f"        -> {nombre}: no pude re-apuntar")
                    if ok and v >= VERDE_MIN:
                        self.estado(f"VISTA LIBRE ({nombre}): verde {v:.2f}")
                        return True
                    if ok and v > mejor_v + BUSQ_MARGEN:
                        mejor_v, mejor_off, mejor_q = v, off, q
                        mejoro = True
                        k += 1                                     # mejora: sigo en esa dirección
                        continue
                    break
                if mejoro:
                    centro = mejor_off.copy()
                    break
            if not mejoro:
                if mejor_v < 0.05 and paso * 2 <= BUSQ_RADIO_MAX_MM and not ampliado:
                    paso, ampliado = paso * 2, True                # todo tapado: miro más lejos
                else:
                    paso /= 2                                      # refino alrededor de lo mejor
        self.estado(f"Sin vista libre. Vuelvo a la mejor (verde {mejor_v:.2f})")
        yield from self.ir_a(mejor_q)
        return False

    # ---------------------------------------------------------------- pose de seguridad
    def ir_pose_seguridad(self):
        pose = cargar_pose_seguridad()
        if pose is None:
            raise RuntimeError(f"No hay pose de seguridad: pon el brazo en L invertida y pulsa 'g' ({POSE_ARCHIVO})")
        self.s.q_cmd = self.s.robot.leer()
        q = dict(self.s.q_cmd)
        for a, v in pose.items():
            if a not in ART_MOVER:
                continue
            if a == "wrist_roll" and cruza_salto(q[a], v):
                print(f"[IBVS] AVISO: wrist_roll {q[a]:.1f} -> {v:.1f} cruza el salto +-180 (vuelta larga). La dejo "
                      f"donde está: gira la muñeca a mano o vuelve a grabar la pose ('g')")
                continue
            # El resto tiene rango limitado por la calibración: se mueve aunque el giro sea grande (despacio).
            if abs(v - q[a]) > 60:
                print(f"[IBVS] {a}: giro grande {q[a]:.1f} -> {v:.1f} ({v - q[a]:+.0f} grados), voy despacio")
            q[a] = v
        self.estado(f"Yendo a la pose de seguridad (ESPACIO para parar): {({k: round(v, 1) for k, v in q.items()})}")
        yield from self.ir_a(q, asentar=1.0, vel=VEL_POSE_DEG_S)
        # Si una articulacion se ha quedado atascada (p. ej. wrist_roll contra un tope), dejar de empujar
        # (orden = donde esta) y parar: si no, el motor sigue forzando hasta saltar su proteccion de sobrecarga
        real = self.s.robot.leer()
        malas = {a: round(real[a], 1) for a in ART_MOVER if abs(real[a] - q[a]) > 15}
        if malas:
            self.s.robot.enviar(real)
            self.s.q_cmd = dict(real)
            raise RuntimeError(f"No llega a la pose de seguridad (atascado en {malas}; objetivo "
                               f"{ {a: q[a] for a in malas} }). Revisa {POSE_ARCHIVO}")
        self.estado("En la pose de seguridad")

    # ---------------------------------------------------------------- secuencia completa
    def secuencia(self):
        s = self.s
        if IR_A_POSE_SEGURIDAD:
            cands0 = list(s.cands_inicio or s.med.get("cands", []))
            p0 = fk(s.q_cmd)[0]
            yield from self.ir_pose_seguridad()
            yield from self.esperar(0.3)
            elegir_eje_haz(s.q_cmd, forzar=True)          # la linterna apunta a la PCB en la pose de seguridad
            self.aprender_estaticos(cands0, p0)
            yield from self.esperar(0.2)
        elegir_eje_haz(s.q_cmd)
        s.p_inicio = fk(s.q_cmd)[0]
        ok = yield from self.servo()
        if not ok:
            return
        if ACERCAR_MM > 0:
            yield from self.acercar(ACERCAR_MM)
        v = yield from self.evaluar_verde()
        if v >= VERDE_MIN:
            self.estado(f"HECHO: punto iluminado y se ve verde ({v:.2f})")
        else:
            self.estado(f"Obstaculo o sombra: puntuacion {v:.2f} < {VERDE_MIN}. Busco otro punto de vista")
            yield from self.busqueda(dict(s.q_cmd), v)
        if VIGILAR:
            yield from self.vigilar()

    def acercar(self, mm):
        """Acerca la linterna al punto `mm` A LO LARGO DEL HAZ (la luz no se mueve de sitio, solo se concentra:
        foco menos difuminado) y re-centra con el IBVS. Nunca por debajo de ESFERA_ALTURA_MIN_MM."""
        s = self.s
        p, z = fk(s.q_cmd)
        if z[2] > -0.2:
            return
        mm = min(mm, max(0.0, (p[2] - ESFERA_ALTURA_MIN_MM) / -z[2]))
        if mm < 5:
            return
        self.estado(f"Acerco la linterna {mm:.0f} mm a la PCB (foco mas concentrado)")
        try:
            yield from self.mover_cart(z * mm, z)
        except RuntimeError as e:
            self.estado(f"No puedo acercarme ({e}); sigo a esta distancia")
            return
        s.p_inicio = fk(s.q_cmd)[0]
        yield from self.servo(timeout=10.0)

    # ---------------------------------------------------------------- vigilancia + órbita
    def tomar_referencia(self, n=15):
        s = self.s
        q = yield from self._muestras("q_anillo", n=n)
        a = yield from self._muestras("azul_anillo", n=n)
        r = yield from self._muestras("r_luz", n=n)
        p = yield from self._muestras("q_punto", n=n)
        s.ref_punto = 0.0 if p is None else float(p)       # % del punto iluminado (sin el mínimo de 0,5)
        if q is None or a is None or r is None or r <= 0 or p is None:
            self.estado("No pude tomar la referencia de buena iluminacion")
            return
        if p < 0.5:
            self.estado(f"AVISO: al tomar la referencia el punto solo esta iluminado al {100 * p:.0f}%")
        s.ref = (q, a, r, max(p, 0.5))
        self.estado(f"Referencia: punto {100 * p:.0f}% | anillo {100 * q:.0f}% | azul {a:.0f} | halo {r:.0f} px")

    def ocluido(self, m):
        ref = self.s.ref
        if m["luz"] is None:
            return True, "no se ve la luz"
        if hay_sombra(m.get("sombra")):
            return True, f"sombra negra {100 * m['sombra']:.0f}%"
        if ref is None:
            return False, ""
        motivos = []
        if m.get("q_punto") is not None and m["q_punto"] < OCL_Q_MIN * ref[3]:
            motivos.append(f"punto iluminado {100 * m['q_punto']:.0f}% (ref {100 * ref[3]:.0f}%)")
        if m.get("r_luz") and m["r_luz"] < OCL_R_MIN * ref[2]:
            motivos.append(f"halo {m['r_luz']:.0f}px (ref {ref[2]:.0f})")
        return bool(motivos), ", ".join(motivos)

    def calidad(self, n=8):
        """Calidad relativa a la referencia (1 = como al converger)."""
        ref = self.s.ref
        p = yield from self._muestras("q_punto", n=n, timeout=2.0)
        r = yield from self._muestras("r_luz", n=n, timeout=2.0)
        if p is None or ref is None:
            return 0.0
        cal = min(p / max(ref[3], 1e-3), 1.0)
        if r is None or r < OCL_R_OK * ref[2]:
            cal *= 0.5                                       # halo muy pequeño: el haz choca con algo antes
        return float(cal)

    def vigilar(self):
        """Vigilancia en tiempo real: si algo corta el haz u oscurece el punto -> órbita guiada; si la luz se
        aparta de la consigna sin oclusión -> IBVS."""
        s = self.s
        yield from self.tomar_referencia()
        # Pose de iluminación INICIAL (la del primer IBVS): tras esquivar un obstáculo, en la siguiente
        # oclusión se vuelve aquí antes de buscar otra vez en la semiesfera
        q_inicial, ref_inicial = dict(s.q_cmd), s.ref
        en_inicial = True                 # ¿el brazo está AHORA en la pose inicial? (se pone a False al salir)
        self.estado("Vigilando oclusion / desvio (ESPACIO para parar)")
        n_ocl = n_desvio = 0
        t_ultimo, ult = 0.0, -1

        def volver_a_inicial(motivo):
            """Vuelve a la pose de iluminación inicial (arco abombado) y devuelve la calidad allí."""
            nonlocal en_inicial
            self.estado(f"{motivo} -> vuelvo a la pose de iluminacion inicial")
            geo = self.punto_mesa(s.q_cmd)
            if geo is not None:
                yield from self.ir_por_arco(q_inicial, geo[0], vel=VEL_POSE_DEG_S)
            else:
                yield from self.ir_a(q_inicial, asentar=0.3, vel=VEL_POSE_DEG_S)
            s.ref = ref_inicial
            en_inicial = True
            yield from self.esperar(0.3)
            cal = yield from self.calidad()
            print(f"        pose inicial -> calidad {cal:.2f}")
            if cal >= OCL_OK and not (yield from self.confirmar_iluminado("pose inicial")):
                cal = 0.0                                    # no aguanta iluminado: se busca con las parábolas
            if cal >= OCL_OK:
                self.estado(f"ILUMINADO de nuevo desde la pose inicial (calidad {cal:.2f})")
            return cal

        while True:
            m = s.med
            if m["id"] == ult:
                yield
                continue
            ult = m["id"]
            ocl, motivo = self.ocluido(m)
            s.ocl = ocl
            n_ocl = n_ocl + 1 if ocl else 0
            e = m["e"]
            n_desvio = n_desvio + 1 if (not ocl and e is not None and np.hypot(*e) > REAJUSTE_PX) else 0
            if n_desvio >= REAJUSTE_N:
                if not en_inicial:
                    # en una pose de esquive que ilumina: SE QUEDA aquí (solo vuelve a la inicial si hay un
                    # obstáculo nuevo). Se recentra la luz con el IBVS sin moverse de esta zona.
                    self.estado(f"La luz se ha apartado {np.hypot(*e):.0f} px en la pose de esquive: reajusto aqui "
                                f"con IBVS (no vuelvo a la inicial)")
                    s.p_inicio = fk(s.q_cmd)[0]
                    yield from self.servo(timeout=10.0)
                    yield from self.tomar_referencia()
                else:
                    self.estado(f"La luz se ha apartado {np.hypot(*e):.0f} px (sin oclusion): reajusto con IBVS")
                    s.p_inicio = fk(s.q_cmd)[0]
                    yield from self.servo()
                    yield from self.tomar_referencia()
                    if s.ref_punto >= 0.5:                   # la inicial queda recentrada (solo si ilumina bien)
                        q_inicial, ref_inicial = dict(s.q_cmd), s.ref
                    else:                                    # reajuste malo: volver a la inicial buena
                        yield from volver_a_inicial(f"Tras reajustar el punto solo esta iluminado al "
                                                    f"{100 * s.ref_punto:.0f}%")
                n_desvio = n_ocl = 0
                self.estado("Vigilando oclusion / desvio (ESPACIO para parar)")
            elif n_ocl >= OCL_N_FRAMES and time.time() - t_ultimo > SOMBRA_ESPERA_S:
                cal = 0.0
                if not en_inicial and VOLVER_A_INICIAL:
                    # obstáculo (o luz perdida) estando en una pose de esquive: primero, volver a la inicial
                    cal = yield from volver_a_inicial(f"OCLUSION: {motivo}")
                    motivo = f"la pose inicial tambien esta tapada (calidad {cal:.2f})"
                if cal < OCL_OK:
                    self.estado(f"OCLUSION: {motivo} -> busco en la semiesfera alrededor del punto")
                    en_inicial = False                       # esfera() mueve el brazo
                    if (yield from self.esfera(q_inicial)):
                        # nueva posición (otra distancia/ángulo: otro tamaño de halo): nueva referencia, o la
                        # vigilancia vería "halo encogido" y volvería a buscar sin motivo
                        yield from self.tomar_referencia()
                    elif VOLVER_A_INICIAL:
                        # ninguna dirección ilumina: esperar en la pose inicial, no en la "mejor" (que no ilumina)
                        yield from volver_a_inicial("Semiesfera sin exito")
                t_ultimo = time.time()
                n_desvio = n_ocl = 0
                self.estado("Vigilando oclusion / desvio (ESPACIO para parar)")
            yield

    def punto_mesa(self, q):
        """P (base) donde el haz (EJE_HAZ_VEC de la pinza) corta la PCB, y datos de la órbita actual."""
        p, z = fk(q)
        if z[2] > -0.2:
            return None
        t = (Z_PCB_MM - p[2]) / z[2]
        P = p + t * z
        rho = np.array([P[0], P[1], 0.0])
        rho /= (np.linalg.norm(rho) + 1e-9)
        w = p - P
        R = float(np.linalg.norm(w))
        theta = float(np.degrees(np.arctan2(w[2], w @ rho)))
        return P, rho, R, theta

    def pista_obstaculo(self, n=10):
        """Dirección (base, plano XY) en la que conviene ALEJAR la linterna: si el haz está tapado solo en
        parte, el halo que queda se desplaza alejándose del obstáculo. None si no hay pista (tapado entero)."""
        s = self.s
        desp = []
        for _ in range(n):
            m = s.med
            if m["e"] is not None and s.ref is not None and m["r_luz"] >= OCL_R_OK * s.ref[2]:
                desp.append(np.asarray(m["e"], float))
            yield
        if len(desp) < n // 2:
            return None
        d_img = np.median(np.array(desp), axis=0)
        if np.linalg.norm(d_img) < 3.0:
            return None
        d_base = np.linalg.solve(self.J, d_img)              # px -> mm del efector (X, Y de la base)
        return d_base / (np.linalg.norm(d_base) + 1e-9)

    def candidatos_esfera(self, q0, P, D, alejar):
        """Puntos de la semiesfera (radio D, centro P) alcanzables apuntando a P (desvío del haz pequeño)."""
        u0 = (fk(q0)[0] - P) / D
        rad = np.arctan2(P[1], P[0])                         # azimut del plano del brazo
        ez = np.array([0.0, 0.0, 1.0])
        cands, por_circ = [], {}
        for k in range(N_CIRCUNFERENCIAS):
            yield                                            # repartir el cálculo entre frames
            phi = rad + k * np.pi / N_CIRCUNFERENCIAS
            h = np.array([np.cos(phi), np.sin(phi), 0.0])
            por_circ[k] = 0
            for el in ESFERA_ELEVS:
                for sg in (+1, -1):
                    u = np.cos(np.radians(el)) * sg * h + np.sin(np.radians(el)) * ez
                    ang = float(np.degrees(np.arccos(np.clip(u @ u0, -1, 1))))
                    if ang < 4.0:
                        continue
                    for f in ESFERA_D_FACTORES:                   # misma inclinación, más cerca si no llega
                        if P[2] + f * D * u[2] < ESFERA_ALTURA_MIN_MM:
                            continue
                        q1, res = ik(q0, P + f * D * u, -u)
                        if res > ESFERA_IK_RESTO_MM:
                            continue
                        p1, z1 = fk(q1)
                        w = P - p1
                        desvio = float(np.linalg.norm(w - (w @ z1) * z1))     # distancia de P al haz
                        if z1[2] <= -0.2 and desvio <= ESFERA_DESVIO_MAX_MM:
                            break
                    else:
                        continue
                    # ORDEN = recorrer circunferencias completas:
                    #  con pista: primero la circunferencia más alineada con "alejarse del obstáculo", por el lado
                    #  que se aleja, bajando desde la vertical (82, 74, 66...); el lado que se acerca, al final.
                    #  sin pista: TODAS las circunferencias nivel a nivel (todas a 82, luego todas a 74...).
                    if alejar is not None:
                        alin = float(h[:2] @ alejar)
                        bueno = sg * alin > 0.2
                        neutro = abs(alin) <= 0.2
                        orden = (0 if bueno else 1 if neutro else 2, -round(abs(alin), 2), -el, sg)
                    else:                            # sin pista: todas las direcciones, nivel a nivel
                        orden = (-el, min(k, N_CIRCUNFERENCIAS - k), k, sg)
                    cands.append({"orden": orden, "circ": k, "phi": float(np.degrees(phi - rad)), "el": el,
                                  "lado": sg, "q": q1, "desvio": desvio, "ang": ang, "escala": f})
                    por_circ[k] += 1
        cands.sort(key=lambda c: c["orden"])
        return cands, por_circ

    def confirmar_iluminado(self, donde, escala=1.0, segundos=None):
        """Antes de pasar de "esquivando" a "iluminando": el punto tiene que verse iluminado de forma CONTINUA
        durante ILUMINADO_CONFIRMAR_S (el brazo quieto). Se toleran ILUMINADO_FALLOS_MAX frames malos seguidos
        (ruido de la cámara); uno más y no se confirma."""
        s = self.s
        segundos = ILUMINADO_CONFIRMAR_S if segundos is None else segundos
        self.estado(f"Luz recuperada ({donde}): confirmando {segundos:.0f} s...")
        t0, ult, malos = time.time(), -1, 0
        while time.time() - t0 < segundos:
            m = s.med
            if m["id"] != ult:
                ult = m["id"]
                ref = s.ref
                bien = (m["luz"] is not None and ref is not None and m.get("q_punto") is not None
                        and m["q_punto"] >= OCL_OK * ref[3] and m["r_luz"] >= OCL_R_OK * ref[2] * escala)
                malos = 0 if bien else malos + 1
                if malos > ILUMINADO_FALLOS_MAX:
                    self.estado(f"Luz recuperada ({donde}) pero no aguanta ({time.time() - t0:.1f} s < {segundos:.0f} s)")
                    return False
            yield
        return True

    def luz_recuperada(self, escala=0.7):
        """¿El punto vuelve a estar iluminado en el frame actual? (comprobación rápida, sin esperar)."""
        m, ref = self.s.med, self.s.ref
        return (m["luz"] is not None and ref is not None and m.get("q_punto") is not None
                and m["q_punto"] >= OCL_OK * ref[3] and m["r_luz"] >= OCL_R_MIN * ref[2] * escala)

    def ir_por_arco(self, q1, P, vel=None, mirar=False):
        """Va a q1 por un arco ABOMBADO alrededor de P: la dirección se interpola por la esfera (slerp) y la
        distancia a P crece hacia la mitad del camino (x (1 + ABOMBADO)), apuntando siempre a P. Así la
        linterna rodea el obstáculo (que suele estar entre ella y P) en vez de atravesarlo en línea recta.
        Si un punto intermedio no se alcanza, se prueba con menos abombado; si no, se salta.
        mirar=True: tras cada tramo se mira si el punto ya se ilumina; si es así se PARA ahí y devuelve True."""
        s = self.s
        p0 = fk(s.q_cmd)[0]
        p1 = fk(q1)[0]
        r0, r1 = float(np.linalg.norm(p0 - P)), float(np.linalg.norm(p1 - P))
        u0, u1 = (p0 - P) / r0, (p1 - P) / r1
        om = float(np.arccos(np.clip(u0 @ u1, -1.0, 1.0)))
        if om < np.radians(5):                               # casi el mismo sitio: directo
            yield from self.ir_a(q1, asentar=0.3, vel=vel)
            return False
        q = dict(s.q_cmd)
        for i in range(1, ARCO_PUNTOS):
            t = i / ARCO_PUNTOS
            u = (np.sin((1 - t) * om) * u0 + np.sin(t * om) * u1) / np.sin(om)
            for b in (ABOMBADO, ABOMBADO / 2, 0.0):
                L = P + ((1 - t) * r0 + t * r1) * (1 + b * np.sin(np.pi * t)) * u
                if L[2] < ESFERA_ALTURA_MIN_MM:
                    continue
                qi, res = ik(q, L, -u)
                if res <= ESFERA_IK_RESTO_MM and not cruza_salto(q["wrist_roll"], qi["wrist_roll"]):
                    q = qi
                    yield from self.ir_a(qi, asentar=0.0, vel=vel)
                    break
            if mirar and self.luz_recuperada():
                yield from self.esperar(0.15)                # confirmar con otro frame (ya parado)
                if self.luz_recuperada():
                    return True
        yield from self.ir_a(q1, asentar=0.3, vel=vel)
        return False

    def _punto_ok(self, q, qi, res, P, final, apuntar=True):
        """¿El punto de la parábola es válido? (alcanzable, haz hacia abajo y apuntando a P, muñeca sin saltos).
        apuntar=False: no se exige que el haz pase por P (solo la FORMA de la parábola)."""
        p1, z1 = fk(qi)
        w = P - p1
        desvio = float(np.linalg.norm(w - (w @ z1) * z1))
        tope = np.inf if not apuntar else ESFERA_DESVIO_MAX_MM if final else PARABOLA_DESVIO_TRAMO_MM
        return not (res > ESFERA_IK_RESTO_MM or z1[2] > -0.2 or desvio > tope
                    or qi["wrist_roll"] > PARABOLA_WRIST_ROLL_MAX
                    or (q is not None and (cruza_salto(q["wrist_roll"], qi["wrist_roll"])
                                           or abs(qi["wrist_roll"] - q["wrist_roll"]) > PARABOLA_SALTO_ROLL_DEG)))

    def _trazar(self, q0, L0, E, P, h, apuntar=True, z_fija=None):
        """Articulaciones de la parábola L0 -> E (abombado h) hasta el primer punto no válido. Si se completa,
        se PROLONGA hacia la PCB: la trayectoria sigue saliendo de la parábola y se va curvando suavemente hacia el
        punto (bajando), apuntando siempre a él, mientras sea válida.
        z_fija: dirección del haz que se mantiene en todo el camino (la pinza no se reorienta)."""
        ez = np.array([0.0, 0.0, 1.0])
        ruta, q = [], dict(q0)
        for i in range(1, PARABOLA_PUNTOS + 1):
            t = i / PARABOLA_PUNTOS
            L = (1 - t) * L0 + t * E + 4.0 * h * t * (1 - t) * ez     # parábola: sube a mitad de camino
            if L[2] < ESFERA_ALTURA_MIN_MM:
                return ruta
            z_obj = z_fija if z_fija is not None else (P - L) / np.linalg.norm(P - L)  # gira para apuntar a P
            qi, res = ik(q, L, z_obj, iters=150 if i == PARABOLA_PUNTOS else 40)
            if not self._punto_ok(q, qi, res, P, i == PARABOLA_PUNTOS, apuntar and z_fija is None):
                return ruta
            ruta.append(qi)
            q = qi
        # --- prolongación hacia la PCB
        t1 = 1 - 1 / PARABOLA_PUNTOS
        L1 = (1 - t1) * L0 + t1 * E + 4.0 * h * t1 * (1 - t1) * ez
        tang = (E - L1) / (np.linalg.norm(E - L1) + 1e-9)                   # dirección al final de la parábola
        L = E.copy()
        n = int(PARABOLA_PROLONGAR_MM // PARABOLA_PROLONGAR_PASO_MM)
        for k in range(1, n + 1):
            a = min(1.0, k / max(1, n // 2))                                # se curva hacia el punto
            hacia_p = (P - L) / np.linalg.norm(P - L)
            v = (1 - a) * tang + a * hacia_p
            L = L + PARABOLA_PROLONGAR_PASO_MM * v / (np.linalg.norm(v) + 1e-9)
            if L[2] < PARABOLA_Z_FINAL_MM or np.linalg.norm(P - L) < PARABOLA_DIST_MIN_MM:
                break
            z_obj = z_fija if z_fija is not None else (P - L) / np.linalg.norm(P - L)
            qi, res = ik(q, L, z_obj, iters=150)
            if not self._punto_ok(q, qi, res, P, True, apuntar and z_fija is None):
                break
            ruta.append(qi)
            q = qi
        return ruta

    def _acercar_final(self, ruta):
        """Añade a la ruta pasos A LO LARGO DEL HAZ hacia la PCB desde el final (misma orientación, luz en el
        mismo sitio), hasta PARABOLA_ACERCAR_MM o hasta donde se pueda."""
        if not ruta or PARABOLA_ACERCAR_MM <= 0:
            return ruta
        q = ruta[-1]
        p, z = fk(q)
        if z[2] > -0.2:
            return ruta
        ruta = list(ruta)
        n = int(PARABOLA_ACERCAR_MM // PARABOLA_ACERCAR_PASO_MM)
        for k in range(1, n + 1):
            L = p + z * (k * PARABOLA_ACERCAR_PASO_MM)
            if L[2] < ESFERA_ALTURA_MIN_MM:
                break
            qi, res = ik(q, L, z, iters=40)
            if (res > ESFERA_IK_RESTO_MM or float(fk(qi)[1] @ z) < 0.995 or cruza_salto(q["wrist_roll"], qi["wrist_roll"])
                    or qi["wrist_roll"] > PARABOLA_WRIST_ROLL_MAX):
                break
            ruta.append(qi)
            q = qi
        return ruta

    def plan_parabola(self, q0, P, D, d, orientacion="apuntar", ancho_max=120.0):
        plan = yield from self._plan_parabola(q0, P, D, d, orientacion, ancho_max)
        if plan is None:
            return None
        ruta, el, f, modo = plan
        return self._acercar_final(ruta), el, f, modo

    def _plan_parabola(self, q0, P, D, d, orientacion="apuntar", ancho_max=120.0):
        """Articulaciones de la PARÁBOLA ABOMBADA desde la pose q0: la linterna AVANZA en horizontal en la dirección
        d (base) desde donde está (lo más posible, hasta ancho_max) y baja PARABOLA_BAJAR_MM, apuntando siempre al
        punto P (orientacion="apuntar") o sin reorientar la pinza ("fija"). Si no hay ninguna entera, el trozo
        alcanzable más largo (>= PARABOLA_MIN_PUNTOS).
        Devuelve (ruta, elevación del final sobre P, distancia final / D, modo: "entera" | "fija" | "trozo") o None."""
        L0, z0 = fk(q0)
        d = np.array([d[0], d[1], 0.0], float)
        d /= np.linalg.norm(d)
        ez = np.array([0.0, 0.0, 1.0])
        z_fija = z0 if orientacion == "fija" else None
        finales = []                                         # primero el que más avanza
        s = float(ancho_max)
        while s >= PARABOLA_ANCHO_MIN_MM - 1e-6:
            for dz in PARABOLA_BAJAR_MM:
                finales.append((s, L0 + s * d + dz * ez))
            s -= PARABOLA_ANCHO_PASO_MM

        def info(ruta, modo):
            w = fk(ruta[-1])[0] - P
            return ruta, float(np.degrees(np.arctan2(w[2], np.linalg.norm(w[:2])))), float(np.linalg.norm(w) / D), modo

        mejor = None
        for s, E in finales:
            yield                                            # repartir el cálculo entre frames
            if E[2] < ESFERA_ALTURA_MIN_MM:
                continue
            if z_fija is None:                               # el final tiene que poder apuntar al punto: si no, ni se traza
                haz_e = (P - E) / np.linalg.norm(P - E)
                qe, res = ik(q0, E, haz_e, iters=40)
                if not self._punto_ok(None, qe, res, P, True) and res <= ESFERA_IK_RESTO_MM:
                    pe, ze = fk(qe)
                    w = P - pe
                    if float(np.linalg.norm(w - (w @ ze) * ze)) < 3 * ESFERA_DESVIO_MAX_MM:
                        qe, res = ik(qe, E, haz_e, iters=150)  # casi: afinar
                if not self._punto_ok(None, qe, res, P, True):
                    continue
            for b in (PARABOLA_ABOMBADO, 0.6 * PARABOLA_ABOMBADO):
                yield
                ruta = self._trazar(q0, L0, E, P, b * D, z_fija=z_fija)
                if len(ruta) >= PARABOLA_PUNTOS:
                    return info(ruta, "fija" if z_fija is not None else "entera")
                if ruta and (mejor is None or len(ruta) > len(mejor)):
                    mejor = ruta
        if (mejor is None or len(mejor) < PARABOLA_MIN_PUNTOS) and z_fija is None:
            # ningún final alcanzable apuntando al punto: el trozo que se pueda (el más ancho y el más corto)
            for s, E in finales:
                if s not in (finales[0][0], finales[-1][0]):
                    continue
                yield
                ruta = self._trazar(q0, L0, E, P, PARABOLA_ABOMBADO * D)
                if ruta and (mejor is None or len(ruta) > len(mejor)):
                    mejor = ruta
        if mejor is None or len(mejor) < PARABOLA_MIN_PUNTOS:
            return None
        return info(mejor, "trozo")

    def volver_por_ruta(self, hecho, q_inicio):
        """Vuelve a la pose de iluminación inicial deshaciendo el mismo camino (no atraviesa el obstáculo)."""
        for q in reversed(hecho):
            yield from self.ir_a(q, asentar=0.0, vel=VEL_POSE_DEG_S)
        yield from self.ir_a(q_inicio, asentar=0.3, vel=VEL_POSE_DEG_S)

    def esfera(self, q_inicio=None):
        """Esquive con PARÁBOLAS ABOMBADAS (PARABOLA_DIRS: +x, +y, -x, -y de la base) desde la pose de iluminación inicial. Si una
        no ilumina el punto (o no se puede hacer), vuelve a la pose inicial y prueba la siguiente."""
        s = self.s
        if self.J is None:
            self.estado("Parabolas: falta J (pulsa 'c')")
            return False
        q_inicio = dict(s.q_cmd) if q_inicio is None else dict(q_inicio)
        geo = self.punto_mesa(q_inicio)
        if geo is None:
            self.estado("Parabolas: la linterna no apunta hacia abajo en la pose inicial")
            return False
        P, _, D_act, _ = geo
        D = float(DIST_ILUM_MM) if DIST_ILUM_MM else D_act
        if max(abs(q_inicio[a] - s.q_cmd[a]) for a in ART_MOVER) > 2.0:
            yield from self.ir_a(q_inicio, asentar=0.3, vel=VEL_POSE_DEG_S)
        self.estado(f"Parabolas: punto inicial guardado {({a: round(q_inicio[a], 1) for a in ART_MOVER})} | "
                    f"P=({P[0]:.0f},{P[1]:.0f},{P[2]:.0f}) mm, D={D:.0f} mm")
        fallos, N = [], len(PARABOLA_DIRS)
        for n, (nombre, d, orientacion, ancho) in enumerate(PARABOLA_DIRS, start=1):
            plan = yield from self.plan_parabola(q_inicio, P, D, d, orientacion, ancho)
            if plan is None:
                fallos.append(f"{nombre} no alcanzable")
                s.avisar(f"Parabola {n}/{N} ({nombre}): NO SE PUEDE hacer (fuera de alcance / muneca en su limite)")
                s.log(["parabola", nombre, "no_alcanzable"])
                continue
            ruta, el, f, modo = plan
            entera = modo != "trozo"
            if modo == "entera":
                self.estado(f"Parabola {n}/{N} ({nombre}): hasta {90 - el:.0f} grados de la vertical, D x {f:.2f}")
            elif modo == "fija":
                self.estado(f"Parabola {n}/{N} ({nombre}): sin reorientar la pinza (luz hacia abajo)")
            elif modo == "forma":
                s.avisar(f"Parabola {n}/{N} ({nombre}): entera, pero la muneca no deja apuntar al punto todo el "
                         f"camino (luz desviada al final)")
            else:
                s.avisar(f"Parabola {n}/{N} ({nombre}): solo se puede hacer un trozo ({len(ruta)}/{PARABOLA_PUNTOS} puntos, muneca en su limite). Hago lo que se pueda")
            hecho, a_mitad = [], False
            for qi in ruta:
                yield from self.ir_a(qi, asentar=0.0, vel=VEL_POSE_DEG_S)
                hecho.append(qi)
                if qi is not ruta[-1] and self.luz_recuperada(escala=f):
                    yield from self.esperar(0.15)            # otro frame (ya parado) antes de pararse aquí
                    if self.luz_recuperada(escala=f):
                        if modo not in ("forma", "fija"):
                            s.p_inicio = fk(s.q_cmd)[0]      # afinar el apuntado antes de confirmar
                            yield from self.servo(timeout=6.0)
                        if (yield from self.confirmar_iluminado(f"parabola {nombre}, a mitad", escala=f)):
                            a_mitad = True
                            break
                        self.estado(f"Parabola {n}/{N} ({nombre}): la luz no aguanta {ILUMINADO_CONFIRMAR_S:.0f} s: "
                                    f"sigo la parabola")
            if a_mitad:
                s.log(["parabola", nombre, el, f, 1, 1.0])
                self.estado(f"ILUMINADO de nuevo: parabola {nombre}, a mitad de camino (intento {n})")
                return True
            yield from self.esperar(0.2)
            # ¿se ha quedado una articulación atascada (tope físico)? Dejar de empujar, avisar y volver
            real = s.robot.leer()
            malas = {a: round(real[a], 1) for a in ART_MOVER if abs(real[a] - s.q_cmd[a]) > 15}
            if malas:
                s.robot.enviar(real)
                s.q_cmd = dict(real)
                fallos.append(f"{nombre} atascada")
                s.avisar(f"Parabola {n}/{N} ({nombre}): NO PUEDE seguir (atascado en {malas}). Vuelvo al inicio")
                yield from self.volver_por_ruta(hecho[:-1], q_inicio)
                continue
            m = s.med
            # en modo "forma" la luz acaba lejos del punto: el IBVS solo deshacería la parábola (si se iluminó
            # por el camino, a_mitad, sí se afina)
            if (modo not in ("forma", "fija") and m["luz"] is not None and s.ref is not None
                    and m["r_luz"] >= OCL_R_MIN * s.ref[2] * f):
                s.p_inicio = fk(s.q_cmd)[0]                  # el haz llega entero: afinar el apuntado
                yield from self.servo(timeout=6.0)
            cal = yield from self.calidad()
            donde = f"al final ({90 - el:.0f} grados)" if entera else "al final del trozo alcanzable"
            print(f"        parabola {n}/{N} ({nombre}): {donde} -> calidad {cal:.2f}")
            if cal >= OCL_OK and not (yield from self.confirmar_iluminado(f"parabola {nombre}, {donde}", escala=f)):
                cal = 0.0                                    # no ha aguantado iluminado ILUMINADO_CONFIRMAR_S
            s.log(["parabola", nombre, el, f, 0, round(cal, 3)])
            if cal >= OCL_OK:
                self.estado(f"ILUMINADO de nuevo: parabola {nombre}, {donde} (calidad {cal:.2f})")
                return True
            fallos.append(f"{nombre} tapada ({cal:.2f})")
            s.avisar(f"Parabola {n}/{N} ({nombre}): no ilumina el punto (calidad {cal:.2f}). Vuelvo al inicio")
            yield from self.volver_por_ruta(hecho, q_inicio)
            if self.luz_recuperada() and (yield from self.confirmar_iluminado("pose inicial")):
                self.estado("ILUMINADO de nuevo desde la pose inicial")   # el obstáculo se ha quitado
                return True
        s.avisar("NINGUNA de las " + str(N) + " parabolas ilumina el punto: " + ", ".join(fallos), t=8.0)
        self.estado("Parabolas sin exito: " + ", ".join(fallos) + ". Espero en la pose inicial")
        return False

    def orbitar(self):
        """Órbita guiada en el plano del brazo, alejándose del lado del obstáculo."""
        s = self.s
        if self.J is None:
            self.estado("Orbita: falta J (pulsa 'c')")
            return False
        q0 = dict(s.q_cmd)
        geo = self.punto_mesa(q0)
        if geo is None:
            self.estado("Orbita: la linterna no apunta hacia abajo")
            return False
        P, rho, R, th0 = geo
        # ¿De qué lado está el obstáculo? Si tapa SOLO PARTE del haz, el trozo de halo que queda se desplaza
        # ALEJÁNDOSE del obstáculo (pista fiable). Si lo tapa entero (halo encogido sobre el obstáculo) no hay
        # forma de saberlo desde la cámara: se prueba primero hacia la vertical (90 grados).
        g = self.J @ rho[:2]                                  # dirección en la imagen de "alejarse de la base"
        g = g / (np.linalg.norm(g) + 1e-9)
        desp = []
        for _ in range(10):
            m = s.med
            if m["e"] is not None and s.ref is not None and m["r_luz"] >= OCL_R_MIN * s.ref[2]:
                desp.append(np.asarray(m["e"], float))
            yield
        lado = None
        if len(desp) >= 5 and np.linalg.norm(np.median(desp, axis=0)) > 3.0:
            d_ = np.median(desp, axis=0)
            lado = -float(d_ @ g) / (np.linalg.norm(d_) + 1e-9)   # >0: obstáculo hacia fuera
        if lado is not None and abs(lado) > 0.3:
            signos = [+1, -1] if lado > 0 else [-1, +1]      # alejarse del obstáculo
            pista = f"halo desplazado: obstaculo {'hacia fuera' if lado > 0 else 'hacia la base'} ({lado:+.2f})"
        else:
            signos = [+1, -1] if th0 <= 90 else [-1, +1]      # sin pista: primero hacia la vertical
            pista = "haz tapado entero (sin pista de lado): primero hacia la vertical"
        self.estado(f"Orbita: P=({P[0]:.0f},{P[1]:.0f},{P[2]:.0f}) mm, R={R:.0f} mm, elevacion {th0:.0f} grados; "
                    f"{pista} -> primero {'hacia la base' if signos[0] > 0 else 'hacia fuera'}")
        q_ini = dict(q0)
        mejor = (-1.0, dict(q0), th0)
        for sg in signos:
            k = 1
            while k * ORBITA_PASO_DEG <= ORBITA_MAX_DEG + 1e-6:
                th = th0 + sg * k * ORBITA_PASO_DEG
                if not (ORBITA_ELEV_MIN <= th <= ORBITA_ELEV_MAX):
                    break
                t_ = np.radians(th)
                L = P + R * (np.cos(t_) * rho + np.sin(t_) * np.array([0.0, 0.0, 1.0]))
                zdir = -(L - P) / R
                q1, res = ik(s.q_cmd, L, zdir)
                if res > 4.0:
                    self.estado(f"Orbita: elevacion {th:.0f} grados no alcanzable (resto {res:.0f} mm)")
                    break
                self.estado(f"Orbita: elevacion {th:.0f} grados ({'hacia la base' if sg > 0 else 'hacia fuera'})")
                yield from self.ir_a(q1, asentar=0.3, vel=VEL_POSE_DEG_S)
                yield from self.esperar(0.2)
                m = s.med
                halo_ok = m["luz"] is not None and s.ref is not None and m["r_luz"] >= OCL_R_MIN * s.ref[2]
                if halo_ok:                                    # el haz llega entero: afinar el apuntado
                    s.p_inicio = fk(s.q_cmd)[0]
                    yield from self.servo(timeout=8.0)
                cal = yield from self.calidad()
                ocl, motivo = self.ocluido(s.med)
                s.log(["orbita", round(th, 1), round(cal, 3), int(ocl), motivo])
                print(f"        elevacion {th:.0f}: calidad {cal:.2f} {'(' + motivo + ')' if ocl else ''}")
                if cal > mejor[0]:
                    mejor = (cal, dict(s.q_cmd), th)
                if cal >= OCL_OK:
                    self.estado(f"ILUMINADO de nuevo desde elevacion {th:.0f} grados (calidad {cal:.2f})")
                    return True
                k += 1
            yield from self.ir_a(q_ini, asentar=0.3, vel=VEL_POSE_DEG_S)
        self.estado(f"Orbita: sin vista libre en el plano del brazo; me quedo en la mejor "
                    f"(elevacion {mejor[2]:.0f}, calidad {mejor[0]:.2f})")
        yield from self.ir_a(mejor[1], asentar=0.3, vel=VEL_POSE_DEG_S)
        return False

    def solo(self, gen):
        elegir_eje_haz(self.s.q_cmd)
        self.s.p_inicio = fk(self.s.q_cmd)[0]
        yield from gen


# ----------------------------------------------------------------------------- IBVS: sesión
class SesionIBVS:
    """IBVS sobre la cámara externa para UN punto consigna (mm de la PCB)."""

    def __init__(self, consigna_mm, geo, lector):
        self.consigna_mm = np.asarray(consigna_mm, float)
        self.geo = geo                                         # fija: la del momento de guardar el punto
        self.consigna_px = np.asarray(mm_a_px(geo, self.consigna_mm), np.float32)
        self.estado, self.aviso, self.aviso_hasta = "conectando robot", "", 0.0
        self.med = {"id": 0, "e": None, "luz": None, "r_luz": 0.0, "verde": None, "verde_sin_luz": None,
                    "sombra": None, "q_anillo": None, "azul_anillo": None, "anomalia": None, "q_punto": None}
        self.ref = None            # referencia de buena iluminación (q_anillo, azul_anillo, r_luz)
        self.ocl = False           # oclusión detectada ahora
        self.p_inicio, self.tarea, self.luz_prev, self.zona = None, None, None, None
        self.estaticos = []        # brillos fijos descartados: [(u, v, radio)]
        self.cands_inicio = None   # brillos vistos en el primer frame (antes de mover el robot)
        self.ver_mascara = self.ver_ajustes = False
        # --- registro de la sesión: logs/<ejecución>/ibvs_HHMMSS/{eventos.txt, datos.csv, img/}
        self.dir_log = REG.sub("ibvs_" + time.strftime("%H%M%S"))
        self.dir_img = os.path.join(self.dir_log, "img") if self.dir_log else None
        self._log = self._logw = self._eventos = None
        self.n_img, self.t_img, self.n_frame, self._ult_estado = 0, 0.0, 0, None
        self._img_actual = None
        if self.dir_log:
            try:
                os.makedirs(self.dir_img, exist_ok=True)
                self._eventos = open(os.path.join(self.dir_log, "eventos.txt"), "a", encoding="utf-8")
                self._log = open(os.path.join(self.dir_log, "datos.csv"), "a", newline="", encoding="utf-8")
                self._logw = csv.writer(self._log)
                self._logw.writerow(["t", "frame", "estado", "tarea", "luz_u", "luz_v", "r_luz", "cons_u", "cons_v",
                                     "e_u", "e_v", "e_norm", "verde_con_luz", "verde_sin_luz", "sombra", "q_anillo", "azul_anillo",
                                     "q_punto", "oclusion",
                                     *[f"q_{a}" for a in ARTICULACIONES], "ef_x", "ef_y", "ef_z"])
            except OSError as e:
                print(f"[IBVS] aviso: sin logs de la sesión ({e})")
        print(f"[IBVS] logs de esta sesión: {self.dir_log}")
        self.evento(f"Consigna PCB {self.consigna_mm.round(2).tolist()} mm -> px {self.consigna_px.round(1).tolist()} | "
                    f"geo esquinas {np.asarray(geo['c']).round(1).tolist()} | J guardada: "
                    f"{'si' if os.path.exists(J_ARCHIVO) else 'no'}")
        self.robot = RobotIBVS(IBVS_PUERTO, IBVS_SIN_ROBOT, lector)
        self.robot.conectar()
        self.q_cmd = self.robot.leer()
        self.ctrl = Controlador(self)
        print(f"[IBVS] {self.robot.estado} | q = " + ", ".join(f"{k}={v:.1f}" for k, v in self.q_cmd.items()))

    def log(self, fila):
        self.evento("DATO " + " ".join(str(x) for x in fila), foto=False)

    def evento(self, txt, foto=True):
        """Línea con hora en eventos.txt y foto anotada del momento."""
        try:
            if self._eventos:
                self._eventos.write(f"{time.strftime('%H:%M:%S')}.{int(time.time() * 1000) % 1000:03d} "
                                    f"[frame {getattr(self, 'n_frame', 0)}] {txt}\n")
                self._eventos.flush()
        except (OSError, ValueError):
            pass
        if foto and getattr(self, "_img_actual", None) is not None:
            self.foto("evento")

    def foto(self, motivo):
        if not getattr(self, "dir_img", None) or self.n_img >= LOG_IMG_MAX:
            return
        self.n_img += 1
        REG.imagen(os.path.join(self.dir_img, f"{self.n_img:05d}_f{self.n_frame}_{motivo}.jpg"), self._img_actual)

    def registrar_frame(self, img):
        """Fila de datos por frame + fotos periódicas mientras hay tarea."""
        self._img_actual = img
        m = self.med
        if self._logw is not None:
            try:
                ef = fk(self.q_cmd)[0]
                luz = m["luz"] if m["luz"] is not None else (None, None)
                e = m["e"] if m["e"] is not None else (None, None)
                r = lambda x: "" if x is None else f"{x:.2f}"  # noqa: E731
                self._logw.writerow([f"{time.time():.3f}", self.n_frame, self.estado, int(self.tarea is not None),
                                     r(luz[0]), r(luz[1]), r(m["r_luz"]), r(self.consigna_px[0]), r(self.consigna_px[1]),
                                     r(e[0]), r(e[1]), r(None if m["e"] is None else float(np.hypot(*m["e"]))),
                                     r(m["verde"]), r(m["verde_sin_luz"]), r(m.get("sombra")),
                                     r(m.get("q_anillo")), r(m.get("azul_anillo")), r(m.get("q_punto")),
                                     int(self.ocl),
                                     *[r(self.q_cmd[a]) for a in ARTICULACIONES], *[r(v) for v in ef]])
                if self.n_frame % 30 == 0:
                    self._log.flush()
            except (OSError, ValueError):
                pass
        ahora = time.time()
        cada = LOG_IMG_CADA_S if self.tarea is not None else 3.0
        if ahora - self.t_img >= cada:
            self.t_img = ahora
            self.foto("mov" if self.tarea is not None else "quieto")
        if self.n_frame == 1 and self.dir_log:
            REG.imagen(os.path.join(self.dir_log, "mascara_luz_inicio.png"), m.get("mask"))

    def avisar(self, txt, t=4.0):
        self.aviso, self.aviso_hasta = txt, time.time() + t
        print(f"[IBVS] {txt}")
        self.evento("AVISO " + txt)

    def procesar(self, frame):
        if self.zona is None or self.zona.shape != frame.shape[:2] or self.ver_ajustes:
            self.zona = zona_pcb(self.geo, frame.shape, PARAMS_LUZ["Luz margen PCB"])
        blur = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        excl = None
        if self.estaticos:
            excl = np.zeros(frame.shape[:2], np.uint8)
            for (u, v, r) in self.estaticos:
                cv2.circle(excl, (int(u), int(v)), int(r), 255, -1)
        mask, luz, r_luz, cands = segmentar_luz(blur, hsv, self.zona, self.luz_prev, excl)
        if luz is None and self.luz_prev is not None:
            mask, luz, r_luz, cands = segmentar_luz(blur, hsv, self.zona, None, excl)
        if luz is None and self.estaticos:
            # la luz puede estar ENCIMA de un brillo fijo: se acepta el blob unido si es claramente mayor
            m2, l2, r2, c2 = segmentar_luz(blur, hsv, self.zona, None, None)
            if l2 is not None:
                for (u, v, r) in self.estaticos:
                    if np.hypot(l2[0] - u, l2[1] - v) < r + r2 and r2 > 1.25 * max(r - 3.0, 1.0):
                        mask, luz, r_luz, cands = m2, l2, r2, c2
                        break
        if self.cands_inicio is None:
            self.cands_inicio = list(cands)
        self.luz_prev = luz
        c = self.consigna_px
        e = None if luz is None else np.array(luz - c, float)
        r_v = R_VERDE_PX                       # círculo FIJO (no crece con el halo: si no, mide la alfombrilla)
        v = fraccion_verde(hsv, c, r_v)
        sombra = fraccion_oscura(hsv, c, PARAMS_LUZ["Sombra radio"])
        bb, gg, rr = [x_.astype(np.int16) for x_ in cv2.split(blur)]
        q_ani, azul_ani = calidad_anillo(hsv, bb - (gg + rr) // 2, c)
        q_pto = calidad_punto(hsv, c)
        anom = anomalia_halo(hsv, c, r_luz if luz is not None else ANILLO_R1) if (self.med["id"] % 3 == 0) else \
            self.med.get("anomalia")
        self.med = {"id": self.med["id"] + 1, "e": e, "luz": luz, "r_luz": r_luz, "r_verde": r_v, "cands": cands,
                    "verde": v, "verde_sin_luz": None, "mask": mask,
                    "sombra": sombra, "q_anillo": q_ani, "azul_anillo": azul_ani, "anomalia": anom,
                    "q_punto": q_pto}

    def dibujar(self, frame):
        m, out = self.med, frame.copy()
        cv2.polylines(out, [self.geo["c"].astype(np.int32)], True, (0, 0, 255), 2)
        c = tuple(int(v) for v in self.consigna_px)
        v = m["verde"] if m["verde"] is not None else m["verde_sin_luz"]
        cv2.circle(out, c, m["r_verde"], (0, 255, 0) if (v is not None and v >= VERDE_MIN) else (0, 0, 255), 1)
        cv2.drawMarker(out, c, (255, 0, 255), cv2.MARKER_CROSS, 18, 2)
        som = m.get("sombra")
        cv2.circle(out, c, int(PARAMS_LUZ["Sombra radio"]), (0, 0, 255) if hay_sombra(som) else (255, 255, 0), 1)
        col_ani = (0, 0, 255) if self.ocl else (0, 200, 0)
        cv2.circle(out, c, ANILLO_R0, col_ani, 1)
        cv2.circle(out, c, ANILLO_R1, col_ani, 1)
        if m.get("anomalia") is not None and self.ocl:
            a_ = m["anomalia"]
            cv2.arrowedLine(out, c, (int(c[0] + a_[0]), int(c[1] + a_[1])), (0, 0, 255), 2, tipLength=0.3)
        for (eu, ev, er) in self.estaticos:                    # brillos fijos descartados
            cv2.circle(out, (int(eu), int(ev)), int(er), (0, 0, 255), 1)
            cv2.line(out, (int(eu - er), int(ev - er)), (int(eu + er), int(ev + er)), (0, 0, 255), 1)
        if m["luz"] is not None:
            cl = tuple(int(x) for x in m["luz"])
            cv2.circle(out, cl, max(int(m["r_luz"]), 3), (0, 255, 255), 1)
            cv2.circle(out, cl, 2, (0, 255, 255), -1)
            cv2.arrowedLine(out, cl, c, (0, 200, 255), 1, tipLength=0.2)
        vtxt = "--" if v is None else f"{100 * v:.0f}%"
        lin = [f"IBVS: {self.estado[:70]}",
               "Luz: NO DETECTADA (t: umbrales, m: mascara)" if m["e"] is None else
               f"e = ({m['e'][0]:+.1f}, {m['e'][1]:+.1f}) px   |e| = {np.hypot(*m['e']):.1f}",
               f"Verde en la consigna: {vtxt} (min {100 * VERDE_MIN:.0f}%) | J: {'si' if self.ctrl.J is not None else 'no'}",
               f"Sombra en la consigna: {'--' if m.get('sombra') is None else f'{100 * m['sombra']:.0f}%'} "
               f"(max {PARAMS_LUZ['Sombra % max']}%, V<{PARAMS_LUZ['Sombra V max']})" + ("  << SOMBRA" if hay_sombra(m.get('sombra')) else ""),
               (f"Punto iluminado {100 * (m.get('q_punto') or 0):.0f}% | anillo {100 * (m.get('q_anillo') or 0):.0f}% "
                f"| halo {m.get('r_luz', 0):.0f}px" + ("" if self.ref is None else
                f" | ref {100 * self.ref[3]:.0f}% / {100 * self.ref[0]:.0f}% / {self.ref[2]:.0f}px") +
                ("  << OCLUSION" if self.ocl else "")),
               f"{self.robot.estado} | ESPACIO parar  i iniciar  p pose  g guardar pose  b busq  c J  w/s  ESC"]
        for i, t in enumerate(lin):
            cv2.putText(out, t, (8, 20 + 18 * i), FONT, 0.5, (0, 0, 0), 3)
            cv2.putText(out, t, (8, 20 + 18 * i), FONT, 0.5, (255, 255, 255), 1)
        if time.time() < self.aviso_hasta:
            cv2.putText(out, self.aviso[:70], (8, out.shape[0] - 14), FONT, 0.55, (0, 0, 0), 4)
            cv2.putText(out, self.aviso[:70], (8, out.shape[0] - 14), FONT, 0.55, (0, 255, 255), 2)
        return out

    def lanzar(self, gen, nombre):
        self.q_cmd = self.robot.leer()
        self.tarea, self.estado = gen, nombre

    def paso_tarea(self):
        if self.tarea is None:
            return
        try:
            next(self.tarea)
        except StopIteration:
            self.tarea = None
        except Exception as ex:
            self.tarea = None
            self.estado = f"PARADO: {ex}"
            self.avisar(f"PARADO: {ex}", 8.0)
            traceback.print_exc()
            self.evento("TRACEBACK\n" + traceback.format_exc())

    def jog_z(self, dz):
        if self.tarea is None:
            self.q_cmd = self.robot.leer()
            _, z = fk(self.q_cmd)
            self.tarea = self.ctrl.solo(self.ctrl.mover_cart([0.0, 0.0, dz], z))

    def ejecutar(self, cap, zoom):
        cv2.namedWindow("IBVS")
        self.lanzar(self.ctrl.secuencia(), "arrancando")
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = zoom_digital(frame, zoom)
                if self.ver_ajustes:
                    try:
                        for k_ in PARAMS_LUZ:
                            PARAMS_LUZ[k_] = cv2.getTrackbarPos(k_, "Ajustes luz")
                    except cv2.error:                  # la ventana se cerró con la X
                        self.ver_ajustes = False
                        print(f"[IBVS] umbrales de la luz: {PARAMS_LUZ}")
                self.n_frame += 1
                if self.n_frame == 1 and self.dir_log:
                    REG.imagen(os.path.join(self.dir_log, "frame_inicio.jpg"), frame)
                self.procesar(frame)
                self.paso_tarea()
                img = self.dibujar(frame)
                self.registrar_frame(img)
                cv2.imshow("IBVS", img)
                if self.ver_mascara:
                    cv2.imshow("Mascara luz", self.med["mask"])
                key = cv2.waitKey(1) & 0xFF
                if key in (27, 13, ord("q")):
                    break
                elif key == ord(" "):
                    self.tarea, self.estado = None, "PARADO (espacio)"
                    self.q_cmd = self.robot.leer()
                    self.evento("Usuario: ESPACIO (parar)")
                elif key == ord("i"):
                    self.lanzar(self.ctrl.secuencia(), "arrancando")
                elif key == ord("b"):
                    self.lanzar(self.ctrl.solo(self.ctrl.busqueda()), "busqueda")
                elif key == ord("c"):
                    self.lanzar(self.ctrl.solo(self.ctrl.identificar()), "midiendo J")
                elif key == ord("r"):
                    if self.tarea is None:
                        self.lanzar(self.ctrl.solo(self.ctrl.tomar_referencia()), "referencia")
                elif key == ord("o"):
                    self.lanzar(self.ctrl.solo(self.ctrl.esfera()), "semiesfera")
                elif key == ord("p"):
                    self.lanzar(self.ctrl.solo(self.ctrl.ir_pose_seguridad()), "pose de seguridad")
                elif key == ord("g"):
                    if self.tarea is None:
                        self.q_cmd = self.robot.leer()
                        self.avisar(f"Pose de seguridad guardada: {guardar_pose_seguridad(self.q_cmd)}")
                elif key == ord("w"):
                    self.jog_z(+JOG_MM)
                elif key == ord("s"):
                    self.jog_z(-JOG_MM)
                elif key == ord("m"):
                    self.ver_mascara = not self.ver_mascara
                    if not self.ver_mascara:
                        try:
                            cv2.destroyWindow("Mascara luz")
                        except cv2.error:
                            pass
                elif key == ord("t"):
                    self.ver_ajustes = not self.ver_ajustes
                    if self.ver_ajustes:
                        cv2.namedWindow("Ajustes luz", cv2.WINDOW_NORMAL)
                        cv2.resizeWindow("Ajustes luz", 450, 200)
                        for k_, v_ in PARAMS_LUZ.items():
                            cv2.createTrackbar(k_, "Ajustes luz", v_, RANGOS_LUZ[k_], nothing)
                    else:
                        try:
                            cv2.destroyWindow("Ajustes luz")
                        except cv2.error:
                            pass
        finally:
            self.robot.desconectar()            # el robot se queda donde está, con par
            self.evento(f"FIN: {self.estado}")
            for f_ in (self._log, self._eventos):
                try:
                    if f_:
                        f_.close()
                except OSError:
                    pass
            for w in ("IBVS", "Mascara luz", "Ajustes luz"):
                try:
                    cv2.destroyWindow(w)
                except cv2.error:
                    pass
        print(f"[IBVS] fin: {self.estado}")
        return self.estado


def ejecutar_ibvs(xy, geo, lector, cap, zoom):
    """Ilumina el punto xy (mm de la PCB) con IBVS sobre la cámara externa. Vuelve al terminar."""
    print(f"[IBVS] consigna PCB ({xy[0]:.1f}, {xy[1]:.1f}) mm")
    try:
        return SesionIBVS(xy, geo, lector).ejecutar(cap, zoom)
    except Exception as e:
        traceback.print_exc()
        print(f"[IBVS] no se pudo ejecutar: {e}")
        return f"error: {e}"


# ----------------------------------------------------------------------------- Main
def main():
    global ZOOM_CAMARA
    REG.iniciar()                                                # [IBVS] logs de toda la ejecución
    if FASE1_AL_INICIO and not ROBOT_PUERTO:                     # [FASE1] lo primero: coger la linterna
        ejecutar_fase1()
    cap = abrir_camara(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la cámara {CAMERA_INDEX}")

    voz = EscuchaVoz()
    threading.Thread(target=voz.iniciar, daemon=True).start()

    ver_mascaras = False
    ver_ajustes = False
    ver_ejes = True
    ver_3d = True
    ver_robot = True
    ver_fk = True
    cv2.namedWindow("Ejes 3D")
    cv2.setMouseCallback("Ejes 3D", _mouse3d)
    cv2.namedWindow("Robot FK")
    cv2.setMouseCallback("Robot FK", _mouse3d)
    lector = LectorRobot()
    threading.Thread(target=lector.iniciar, daemon=True).start()
    hist = deque()
    guardados = []                 # [{"n": int, "xy": np.array([x_mm, y_mm])}]
    ultimo_ibvs = None             # [IBVS] (xy, geo) del último punto guardado
    aviso, aviso_hasta = "", 0.0
    n_cap = 0
    t_prev = time.time()
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = zoom_digital(frame, ZOOM_CAMARA)
        out = frame.copy()
        if ver_ajustes:
            sincronizar_ajustes()

        q_actual = dict(lector.q)
        r = analizar(frame, ZOOM_CAMARA)
        geo, pose, boli = r["geo"], r["pose"], r["boli"]
        pcb_img, T = r["pcb_img"], r["T"]

        if r["hull"] is not None:
            cv2.polylines(out, [up(r["quad"])], True, (255, 200, 0), 2)
            x, y = up(r["quad"])[:, 0, :].min(axis=0)
            cv2.putText(out, "Alfombrilla", (int(x), max(int(y) - 8, 15)), FONT, 0.6, (255, 200, 0), 2)
        else:
            cv2.putText(out, "Alfombrilla no detectada", (10, 25), FONT, 0.7, (0, 0, 255), 2)

        if geo is not None:
            cv2.polylines(out, [geo["c"].astype(np.int32)], True, (0, 0, 255), 2)

        # --- Boli y punta (detectados en el recorte de la PCB)
        if boli is not None and geo is not None:
            M = geo["Hinv"] @ np.linalg.inv(T)            # px del recorte -> px del frame

            def a_frame(p):
                return cv2.perspectiveTransform(np.array([[p]], np.float32), M)[0, 0]

            punta_f, cola_f = a_frame(boli["punta"]), a_frame(boli["cola"])
            cnt_f = cv2.perspectiveTransform(boli["cnt"].astype(np.float32), M).astype(np.int32)
            cv2.drawContours(out, [cnt_f], -1, (255, 255, 0), 1)
            cv2.line(out, tuple(int(v) for v in cola_f), tuple(int(v) for v in punta_f), (255, 0, 0), 2)
            cv2.drawContours(pcb_img, [boli["cnt"]], -1, (255, 255, 0), 1)
            cv2.line(pcb_img, tuple(int(v) for v in boli["cola"]),
                     tuple(int(v) for v in boli["punta"]), (255, 0, 0), 2)

            hist.append(muestra_punta(geo, pose, T, boli["punta"]))
            xy = hist[-1][1]
            marcar(out, punta_f, f"({xy[0]:.0f},{xy[1]:.0f})mm", (255, 0, 255), 5)
            marcar(pcb_img, boli["punta"], "", (255, 0, 255), 5)
        now = time.time()
        while hist and now - hist[0][0] > HIST_SEG:
            hist.popleft()

        # --- Puntos guardados (se reproyectan en cada frame sobre la PCB)
        if geo is not None:
            for g in guardados:
                p = mm_a_px(geo, g["xy"])
                txt = f"#{g['n']} PCB({g['xy'][0]:.1f},{g['xy'][1]:.1f},0)"
                if g["base"] is not None:
                    txt += f"\nBASE({g['base'][0]:.1f},{g['base'][1]:.1f},{g['base'][2]:.1f})"
                if g["rob"] is not None:
                    e = g["rob"]["p_ef_obj"]
                    txt += f"\nEF({e[0]:.1f},{e[1]:.1f},{e[2]:.1f})"
                marcar(out, p, txt, (0, 255, 0), 7)
                if T is not None:
                    v = T @ np.array([g["xy"][0], g["xy"][1], 1.0])
                    marcar(pcb_img, v[:2], f"#{g['n']}", (0, 255, 0), 7)

        # --- Ejes de la BASE y de la PCB proyectados sobre la imagen (comprobación visual)
        K = matriz_K(frame.shape, ZOOM_CAMARA)
        if ver_ejes:
            if pose is not None:
                dibujar_ejes(out, K, hom(*pose), 40.0, "PCB")
            if not dibujar_ejes(out, K, np.linalg.inv(matriz_cam_a_base()), 100.0, "BASE"):
                cv2.putText(out, "BASE detras de la camara (z<=0): revisa la transformacion",
                            (10, 75), FONT, 0.55, (0, 0, 255), 2)

        # --- Esqueleto del SO-101 (cinemática directa) proyectado sobre la imagen
        if ver_robot:
            Minv = np.linalg.inv(matriz_cam_a_base())
            frames_q = cinematica_directa(q_actual)
            Pb = np.array([T_[:3, 3] for _, T_ in frames_q])
            px_r = px_de_cam(K, (Minv[:3, :3] @ Pb.T).T + Minv[:3, 3])
            for a_, b_ in zip(px_r[:-1], px_r[1:]):
                if a_ is not None and b_ is not None:
                    cv2.line(out, a_, b_, (0, 200, 255), 3)
            for pp in px_r:
                if pp is not None:
                    cv2.circle(out, pp, 5, (0, 200, 255), -1)
            dibujar_ejes(out, K, Minv @ frames_q[-1][1], 40.0, "EF")
            for g in guardados[-1:]:                         # objetivo (con offset) del último punto
                if g["rob"] is not None:
                    pc = Minv[:3, :3] @ g["rob"]["p_obj"] + Minv[:3, 3]
                    pp = px_de_cam(K, [pc])[0]
                    if pp is not None:
                        marcar(out, pp, "OBJ", (0, 255, 255), 5)

        # --- Estado en pantalla
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))
        t_prev = now
        h_img = out.shape[0]
        cv2.putText(out, f"{fps:.0f} FPS  zoom x{ZOOM_CAMARA:.1f}", (10, h_img - 10), FONT, 0.6, (0, 255, 0), 2)
        cv2.putText(out, f"Voz: {voz.estado} | '{voz.ultima[:40]}'", (10, h_img - 32), FONT, 0.5, (255, 255, 255), 1)
        cv2.putText(out, f"Robot: {lector.estado}", (10, h_img - 54), FONT, 0.5, (255, 255, 255), 1)
        if now < aviso_hasta:
            cv2.putText(out, aviso, (10, 50), FONT, 0.8, (0, 0, 0), 4)
            cv2.putText(out, aviso, (10, 50), FONT, 0.8, (0, 255, 255), 2)

        cv2.imshow("Resultado", out)
        punto3d = None
        if hist and hist[-1][2] is not None:
            punto3d = punto_a_base(hist[-1][2])
        elif guardados and guardados[-1]["base"] is not None:
            punto3d = guardados[-1]["base"]
        if ver_3d:
            cv2.imshow("Ejes 3D", render_ejes3d(punto3d))
        if ver_fk:
            rob_v = objetivo_robot(punto3d, q_actual) if punto3d is not None else None
            cv2.imshow("Robot FK", render_fk3d(cinematica_directa(q_actual), punto3d,
                                               None if rob_v is None else rob_v["p_obj"], q_actual))
        cv2.imshow("PCB zoom", pcb_img if pcb_img is not None else placeholder("PCB no detectada"))
        if ver_mascaras:
            cv2.imshow("Mascara alfombrilla", r["blob_mat"])
            cv2.imshow("Mascara PCB", r["mask_pcb"])
            cv2.imshow("Mascara boli", r["mask_boli"])

        key = cv2.waitKey(1) & 0xFF
        origen = None
        if key == ord("k"):
            origen = "tecla"
        if voz.evento.is_set():
            voz.evento.clear()
            origen = origen or "voz"

        if origen is not None:
            retro = VOZ_RETRO_S if origen == "voz" else TECLA_RETRO_S
            p = punto_estable(hist, retro)
            if p is None or geo is None:
                aviso, aviso_hasta = "NO VEO LA PUNTA DEL BOLI EN LA PCB", now + 2.0
            else:
                xy, px, cam = p
                n = len(guardados) + 1
                p_base = punto_a_base(cam)               # CÁMARA -> base del robot (traslación, Rz, Rx)
                rob = None
                if p_base is not None:                   # cinemática directa del SO-101
                    rob = objetivo_robot(p_base, q_actual)
                    rob["q"] = dict(q_actual)
                dentro = guardar_punto(n, origen, xy, cam, p_base, px, geo, rob)
                guardados.append({"n": n, "xy": xy, "base": p_base, "rob": rob})
                aviso = f"PUNTO #{n} GUARDADO ({origen})" + ("" if dentro else " - FUERA DE PCB")
                aviso_hasta = now + 2.0
                print(f"[{aviso}]")
                print(f"    PCB  (mm): ({xy[0]:8.2f}, {xy[1]:8.2f}, {0.0:8.2f})")
                if cam is not None:
                    print(f"    CAM  (mm): ({cam[0]:8.2f}, {cam[1]:8.2f}, {cam[2]:8.2f})")
                if p_base is not None:
                    print(f"    BASE (mm): ({p_base[0]:8.2f}, {p_base[1]:8.2f}, {p_base[2]:8.2f})")
                if rob is not None:
                    e, o = rob["p_ef_obj"], rob["p_obj"]
                    pe = rob["T_ef"][:3, 3]
                    print(f"    EF   (mm): ({e[0]:8.2f}, {e[1]:8.2f}, {e[2]:8.2f})   <- punto visto desde el efector, "
                          f"con offset Z = {OFFSET_Z_MM:.1f} mm")
                    print(f"    OBJ  (mm): ({o[0]:8.2f}, {o[1]:8.2f}, {o[2]:8.2f})   <- objetivo del efector en la BASE")
                    print(f"    efector actual en BASE (mm): ({pe[0]:.1f}, {pe[1]:.1f}, {pe[2]:.1f}) | "
                          f"q = " + ", ".join(f"{k}={q_actual[k]:.1f}" for k in ARTICULACIONES))
                ultimo_ibvs = (xy, geo)                                   # [IBVS]
                d_p = REG.sub(f"punto_{n}_" + time.strftime("%H%M%S"))  # [IBVS] fotos del punto guardado
                if d_p:
                    REG.imagen(os.path.join(d_p, "resultado.jpg"), out)
                    REG.imagen(os.path.join(d_p, "frame.jpg"), frame)
                    if pcb_img is not None:
                        REG.imagen(os.path.join(d_p, "pcb_zoom.jpg"), pcb_img)
                if IBVS_AUTO:
                    aviso, aviso_hasta = "IBVS: " + ejecutar_ibvs(xy, geo, lector, cap, ZOOM_CAMARA), time.time() + 4.0

        if key in (ord("q"), 27):
            break
        elif key == ord("l"):                                    # [FASE1] coger la linterna otra vez
            if ROBOT_PUERTO:
                aviso, aviso_hasta = "FASE1: no con ROBOT_PUERTO (LectorRobot usa el puerto)", time.time() + 3.0
            else:
                cap.release()                                    # la red necesita las dos cámaras
                ok_f1 = ejecutar_fase1(preguntar=False)
                cap = abrir_camara(CAMERA_INDEX)
                aviso, aviso_hasta = f"FASE1 {'terminada' if ok_f1 else 'fallida/saltada'}", time.time() + 3.0
        elif key == ord("i"):                                    # [IBVS] relanzar con el último punto
            if ultimo_ibvs is None:
                aviso, aviso_hasta = "IBVS: primero guarda un punto ('k' o voz)", time.time() + 2.0
            else:
                aviso, aviso_hasta = "IBVS: " + ejecutar_ibvs(*ultimo_ibvs, lector, cap, ZOOM_CAMARA), time.time() + 4.0
        elif key in (ord("+"), ord("="), 171):
            ZOOM_CAMARA = min(ZOOM_MAX, ZOOM_CAMARA + ZOOM_STEP)
        elif key in (ord("-"), ord("_"), 173):
            ZOOM_CAMARA = max(1.0, ZOOM_CAMARA - ZOOM_STEP)
        elif key == ord("x"):
            guardados.clear()
        elif key == ord("e"):
            ver_ejes = not ver_ejes
        elif key == ord("g"):
            ver_3d = not ver_3d
            if not ver_3d:
                cv2.destroyWindow("Ejes 3D")
            else:
                cv2.namedWindow("Ejes 3D")
                cv2.setMouseCallback("Ejes 3D", _mouse3d)
        elif key == ord("v"):
            VISTA3D["con_punto"] = not VISTA3D["con_punto"]
        elif key == ord("r"):
            ver_robot = not ver_robot
        elif key == ord("f"):
            ver_fk = not ver_fk
            if not ver_fk:
                cv2.destroyWindow("Robot FK")
            else:
                cv2.namedWindow("Robot FK")
                cv2.setMouseCallback("Robot FK", _mouse3d)
        elif key == ord("m"):
            ver_mascaras = not ver_mascaras
            if not ver_mascaras:
                for w in ("Mascara alfombrilla", "Mascara PCB", "Mascara boli"):
                    cv2.destroyWindow(w)
        elif key == ord("t"):
            ver_ajustes = not ver_ajustes
            if ver_ajustes:
                abrir_ajustes()
            else:
                cv2.destroyWindow("Ajustes")
        elif key == ord("s"):
            cv2.imwrite(f"captura_{n_cap:03d}.png", out)
            if pcb_img is not None:
                cv2.imwrite(f"pcb_{n_cap:03d}.png", pcb_img)
            n_cap += 1

    voz.parar()
    lector.parar()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()