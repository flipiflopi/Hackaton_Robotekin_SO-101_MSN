"""Comandos de voz para vision_voz.py: diccionario de frases, intérprete y escucha en segundo plano.

Uso suelto (probar el diccionario sin robot):
    python comandos_voz.py                      -> prueba frases de ejemplo
    python comandos_voz.py "muevete a la derecha" -> interpreta esa frase
    python comandos_voz.py --micros             -> lista los micrófonos
    python comandos_voz_rev.py --micro [n]      -> mantén M pulsada, habla y suelta: imprime lo que entiende

Cada comando tiene una lista de frases (sin acentos, en minúsculas). Una frase puede llevar ".*" para dejar
palabras en medio ("muev\\w*.*arriba" vale para "muévete un poco hacia arriba"). Se elige la frase que
coincide con el trozo MÁS LARGO del texto; si empatan, gana el comando que va antes en PRIORIDAD.
"""
import collections
import queue
import re
import sys
import threading
import time
import unicodedata

# ----------------------------------------------------------------------------- Diccionario
COMANDOS = {
    # Fase 1: coger la linterna con la red (= tecla 'l'). Solo vale al principio (lo controla vision_voz.py)
    "coger": [
        "coge la linterna", "coge linterna", "cogela", "coge", "coger la linterna", "coger", "coja la linterna",
        "coge la lampara", "coge el foco", "coge la luz", "agarra la linterna", "agarra", "agarrala",
        "agarra la lampara", "pilla la linterna", "pillala", "toma la linterna", "tomala", "recoge la linterna",
        "recoge", "recogela", "levanta la linterna", "busca la linterna", "ve a por la linterna",
        "empieza", "empezar", "empecemos", "comienza", "comenzar", "inicia", "iniciar", "arranca", "arrancar",
        "a trabajar",
    ],
    # Guardar el punto que señala el boli (= tecla 'k')
    "guardar": [
        "guarda punto", "guarda el punto", "guarda este punto", "guarda ese punto", "guarda aqui", "guarda ahi",
        "guarda", "guardar", "guardar punto", "guardalo", "guardame", "guardame el punto", "guarda la posicion",
        "guarda esto", "guardame esto", "marca", "marca el punto", "marca aqui", "marca este punto", "marcalo",
        "marcar", "registra", "registra el punto", "registralo", "anota", "anota el punto", "anotalo",
        "apunta el punto", "memoriza", "memoriza el punto", "recuerda este punto", "fija el punto", "fijalo",
        "graba el punto", "salva el punto", "este punto", "ese punto", "aqui el punto", "toma el punto",
    ],
    # Iluminar el punto guardado (= tecla 'i')
    "iluminar": [
        "iluminame aqui", "ilumina aqui", "iluminame", "ilumina", "iluminar", "ilumina el punto",
        "ilumina ahi", "ilumina esto", "iluminalo", "iluminalo aqui", "a iluminar", "alumbra", "alumbra aqui",
        "alumbrame", "alumbralo", "alumbrar", "alumbra el punto", "enfoca", "enfoca aqui", "enfocalo",
        "enfoca el punto", "dale luz", "dale luz aqui", "pon la luz", "pon la luz aqui", "pon luz", "luz aqui",
        "luz al punto", "apunta la luz", "apunta aqui", "apunta ahi", "proyecta la luz", "senala con la luz",
    ],
    # Mover el punto iluminado en la base del robot (+x / -x / +y / -y)
    "derecha": [
        "muevete a la derecha", "mueve a la derecha", "muev\\w*.*derecha", "desplaz\\w*.*derecha",
        "ve a la derecha", "vete a la derecha", "ve hacia la derecha", "hacia la derecha", "a la derecha",
        "derecha", "a tu derecha", "mas a la derecha", "un poco a la derecha", "corre.*derecha", "gira.*derecha",
        "pasa a la derecha", "derechita",
    ],
    "izquierda": [
        "muevete a la izquierda", "mueve a la izquierda", "muev\\w*.*izquierda", "desplaz\\w*.*izquierda",
        "ve a la izquierda", "vete a la izquierda", "ve hacia la izquierda", "hacia la izquierda",
        "a la izquierda", "izquierda", "a tu izquierda", "mas a la izquierda", "un poco a la izquierda",
        "corre.*izquierda", "gira.*izquierda", "pasa a la izquierda",
    ],
    "adelante": [
        "mueve arriba", "muevete arriba", "mueve hacia arriba", "muevete hacia arriba", "muev\\w*.*arriba",
        "desplaz\\w*.*arriba", "ve arriba", "ve hacia arriba", "adelante", "hacia adelante", "delante",
        "hacia delante", "para delante", "para adelante", "al frente", "hacia el frente", "avanza", "avanzar",
        "mas adelante", "alante", "palante", "al fondo", "hacia el fondo", "muev\\w*.*adelante",
        "muev\\w*.*delante",
    ],
    "atras": [
        "mueve abajo", "muevete abajo", "mueve hacia abajo", "muevete hacia abajo", "muev\\w*.*abajo",
        "desplaz\\w*.*abajo", "ve abajo", "ve hacia abajo", "atras", "hacia atras", "detras", "retrocede",
        "retroceder", "mas atras", "para atras", "patras", "vuelve atras", "hacia mi", "muev\\w*.*atras",
    ],
    # Bajar la linterna en VERTICAL (perpendicular al suelo). ("arriba" / subir se ha QUITADO: no iba bien)
    "acercar": [
        "abajo", "baja", "bajate", "bajar", "baja un poco", "baja mas", "mas abajo", "para abajo", "hacia abajo",
        "mas bajo", "desciende", "descender", "acercate", "acerca", "acercar", "acercate a la placa",
        "mas cerca", "pegate", "arrimate",
    ],
    # Control
    "parar": ["para", "parate", "para ya", "stop", "quieto", "quieta", "detente", "detenlo", "alto", "espera",
              "frena", "basta", "pausa"],
    "seguir": ["sigue", "seguir", "continua", "continuar", "vigila", "sigue vigilando", "vigila otra vez",
               "reanuda", "reanudar"],
    "salir": ["sal", "salir", "termina", "terminar", "terminado", "fin", "acaba", "acabamos", "vuelve",
              "cierra", "hemos terminado"],
}
# Empates (mismo largo de coincidencia): gana el que va antes
PRIORIDAD = ["parar", "salir", "iluminar", "guardar", "coger", "derecha", "izquierda", "adelante", "atras",
             "acercar", "seguir"]
# Modificadores del tamaño del paso
FACTORES = {"un poquito": 0.5, "poquito": 0.5, "un poco": 0.5, "poco": 0.5, "ligeramente": 0.5,
            "mucho": 2.0, "bastante": 2.0, "mas lejos": 1.0}
MOVIMIENTOS = ("derecha", "izquierda", "adelante", "atras", "acercar")

# --- Filtro de CONVERSACIÓN: una orden es una frase corta con UN solo comando
# Palabras de relleno que no cuentan (artículos, "por favor", "un poco", "hacia"...)
RELLENO = set("""a al el la lo los las un una unos unas de del y e o en con por favor porfa hacia para mas muy
poco poquito mucho bastante ya ahora otra vez venga vale ok okay oye eh bueno pues tu te me se aqui ahi alli este
esta ese esa esto robot que he has ha hemos le les mi""".split())
MAX_PALABRAS_UTILES = 4      # más palabras con significado que esto -> es una conversación, no una orden
MAX_PALABRAS = 10            # más palabras en total que esto -> conversación
CONVERSACION_FRASES = 3      # si se oyen tantas frases (de lo que sea) en CONVERSACION_S segundos -> conversación:
CONVERSACION_S = 6.0         #   no se hace nada
# Órdenes que SOLO valen dichas sueltas (todas las palabras con significado son de esa orden): "para" es casi
# siempre una preposición ("para que está aquí..."), "coge" sale en conversaciones ("voy a coger la linterna...")
SOLO_SUELTAS = ("parar", "seguir", "salir", "coger")
# PULSAR PARA HABLAR
PTT_HZ = 16000               # audio del micro (mono, 16 bits)
PTT_TROZO = 1024             # muestras por lectura (~64 ms)
PTT_PREVIO = 4               # trozos de antes de pulsar que se añaden (~0,25 s: no cortar el principio)
PTT_MIN_S = 0.3              # menos que esto con la tecla pulsada: se ignora
PTT_MAX_S = 8.0              # como mucho esto de grabación por pulsación
ESPERA_S = 1.2               # una orden se ejecuta si en este tiempo NO se oye otra frase (si no: conversación);
INMEDIATOS = ("parar",)      #   salvo estas, que se ejecutan al momento (seguridad)


def normalizar(texto):
    t = "".join(c for c in unicodedata.normalize("NFD", texto.lower()) if unicodedata.category(c) != "Mn")
    t = re.sub(r"[^a-z0-9ñ ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


_PATRONES = [(cmd, re.compile(r"\b" + f.replace(".*", r"\b.*\b") + r"\b"))
             for cmd, frases in COMANDOS.items() for f in frases]


# palabras de cada orden (para comprobar las órdenes que solo valen sueltas)
VOCABULARIO = {c: {w for f in frases for w in re.sub(r"[^a-z ]", " ", f).split()} for c, frases in COMANDOS.items()}


def interpretar_detalle(texto):
    """Texto reconocido -> (comando, factor, motivo). comando None si no es una orden (motivo dice por qué)."""
    t = normalizar(texto)
    if not t:
        return None, 1.0, "vacio"
    palabras = t.split()
    utiles = [w for w in palabras if w not in RELLENO]
    if len(palabras) > MAX_PALABRAS or len(utiles) > MAX_PALABRAS_UTILES:
        return None, 1.0, f"frase larga ({len(utiles)} palabras utiles): conversacion"
    mejores = {}                                     # comando -> (largo, inicio, fin)
    for cmd, pat in _PATRONES:
        m = pat.search(t)
        if m and (cmd not in mejores or m.end() - m.start() > mejores[cmd][0]):
            mejores[cmd] = (m.end() - m.start(), m.start(), m.end())
    if not mejores:
        return None, 1.0, "no es un comando"
    cmd = max(mejores, key=lambda c: (mejores[c][0], -PRIORIDAD.index(c)))
    _, i0, i1 = mejores[cmd]
    # ¿otro comando en OTRA parte de la frase? ("guarda y luego ilumina") -> ambiguo: no se hace nada
    otros = [c for c, (_, a, b) in mejores.items() if c != cmd and (b <= i0 or a >= i1)]
    if otros:
        return None, 1.0, f"varias ordenes a la vez ({cmd} + {', '.join(otros)})"
    if cmd in SOLO_SUELTAS:
        extra = [w for w in utiles if w not in VOCABULARIO[cmd]]
        if extra:
            return None, 1.0, f"'{cmd}' solo vale como orden suelta (sobra: {' '.join(extra)})"
    factor = 1.0
    if cmd in MOVIMIENTOS:
        for f, v in sorted(FACTORES.items(), key=lambda kv: -len(kv[0])):
            if re.search(r"\b" + f + r"\b", t):
                factor = v
                break
    return cmd, factor, ""


def interpretar(texto):
    """Texto reconocido -> (comando, factor) o (None, 1.0). factor: 0.5 'un poco', 2 'mucho'."""
    cmd, factor, _ = interpretar_detalle(texto)
    return cmd, factor


# ----------------------------------------------------------------------------- Escucha
class EscuchaVoz:
    """Escucha el micro en segundo plano y deja los comandos en self.cola: (comando, factor, texto).
    Motor: "google" (SpeechRecognition + internet) o "vosk" (sin internet; carpeta del modelo en es)."""

    def __init__(self, idioma="es-ES", motor="auto", modelo_vosk="modelo_vosk", microfono=None, ptt=True, tecla="M"):
        self.idioma, self.motor, self.modelo_vosk = idioma, motor, modelo_vosk
        self.ptt, self.tecla = ptt, tecla.upper()   # PULSAR PARA HABLAR: solo se procesa lo grabado con la tecla pulsada
        self.grabando = False
        self.microfono = microfono                   # índice del micro (None = el predeterminado de Windows)
        self.cola = queue.Queue()
        self.estado = "iniciando micro..."
        self.ultima = ""
        self.ultimo_cmd = ""
        self._parar = None
        self._fin = threading.Event()
        self._tiempos = collections.deque()   # cuándo se oyó cada frase (para detectar conversaciones)
        self._pend = None                     # (orden, hora): orden esperando ESPERA_S sin más frases
        self._lock = threading.Lock()

    def iniciar(self):
        motor = self.motor
        if motor == "auto":
            import os
            motor = "vosk" if os.path.isdir(self.modelo_vosk) else "google"
        try:
            if self.ptt:
                self._iniciar_ptt(motor)
            else:
                self._iniciar_vosk() if motor == "vosk" else self._iniciar_google()
        except ImportError as e:
            self.estado = f"sin voz ({e.name} no instalado) - usa las teclas"
        except Exception as e:                       # sin micro, etc.
            self.estado = f"micro no disponible ({type(e).__name__}: {str(e)[:40]}) - usa las teclas"

    # --- Google (SpeechRecognition)
    def _iniciar_google(self):
        import speech_recognition as sr
        self.sr = sr
        rec = sr.Recognizer()
        rec.dynamic_energy_threshold = True
        rec.pause_threshold = 0.5                    # órdenes cortas: cortar antes
        mic = sr.Microphone(device_index=self.microfono)
        with mic as fuente:
            rec.adjust_for_ambient_noise(fuente, duration=1.0)
        self._parar = rec.listen_in_background(mic, self._callback_google, phrase_time_limit=4)
        self.estado = "escuchando (google)"

    def _callback_google(self, rec, audio):
        try:
            texto = rec.recognize_google(audio, language=self.idioma)
        except self.sr.UnknownValueError:
            return
        except Exception as e:                       # sin internet, etc.
            self.estado = f"error voz ({type(e).__name__})"
            return
        self.estado = "escuchando (google)"
        self._texto(texto)

    # --- Vosk (sin internet)
    def _iniciar_vosk(self):
        import json
        import pyaudio
        import vosk
        vosk.SetLogLevel(-1)
        modelo = vosk.Model(self.modelo_vosk)
        rec = vosk.KaldiRecognizer(modelo, 16000)
        pa = pyaudio.PyAudio()
        st = pa.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=4000,
                     input_device_index=self.microfono)
        self.estado = "escuchando (vosk, sin internet)"

        def bucle():
            try:
                while not self._fin.is_set():
                    datos = st.read(4000, exception_on_overflow=False)
                    if rec.AcceptWaveform(datos):
                        texto = json.loads(rec.Result()).get("text", "")
                        if texto:
                            self._texto(texto)
            finally:
                st.stop_stream()
                st.close()
                pa.terminate()

        threading.Thread(target=bucle, daemon=True).start()
        self._parar = lambda wait_for_stop=False: self._fin.set()

    # --- PULSAR PARA HABLAR (tecla mantenida)
    def tecla_pulsada(self):
        """¿La tecla está pulsada AHORA? (Windows: estado real del teclado, en cualquier ventana)."""
        try:
            import ctypes
            return bool(ctypes.windll.user32.GetAsyncKeyState(ord(self.tecla)) & 0x8000)
        except (AttributeError, OSError):
            return False

    def _iniciar_ptt(self, motor):
        import pyaudio
        if motor == "vosk":
            import vosk
            vosk.SetLogLevel(-1)
            self._modelo_vosk = vosk.Model(self.modelo_vosk)
        else:
            import speech_recognition as sr
            self.sr, self._rec = sr, sr.Recognizer()
        pa = pyaudio.PyAudio()
        st = pa.open(format=pyaudio.paInt16, channels=1, rate=PTT_HZ, input=True, frames_per_buffer=PTT_TROZO,
                     input_device_index=self.microfono)
        listo = f"mantén {self.tecla} pulsada para hablar ({'vosk' if motor == 'vosk' else 'google'})"
        self.estado = listo

        def bucle():
            previo = collections.deque(maxlen=PTT_PREVIO)   # audio de justo antes de pulsar (no cortar la 1.ª sílaba)
            grab = []
            try:
                while not self._fin.is_set():
                    datos = st.read(PTT_TROZO, exception_on_overflow=False)
                    if self.tecla_pulsada():
                        if not grab:
                            grab = list(previo)
                            self.grabando, self.estado = True, f"ESCUCHANDO... (suelta {self.tecla} al acabar)"
                        grab.append(datos)
                        if len(grab) * PTT_TROZO / PTT_HZ > PTT_MAX_S:      # tecla atascada: cortar
                            grab = grab[-int(PTT_MAX_S * PTT_HZ / PTT_TROZO):]
                    else:
                        previo.append(datos)
                        if grab:
                            audio, grab = b"".join(grab), []
                            self.grabando = False
                            dur = len(audio) / 2 / PTT_HZ
                            if dur - PTT_PREVIO * PTT_TROZO / PTT_HZ < PTT_MIN_S:
                                self.estado = f"muy corto: mantén {self.tecla} mientras hablas"
                            else:
                                self.estado = "entendiendo..."
                                threading.Thread(target=self._reconocer, args=(audio, motor, listo),
                                                 daemon=True).start()
            finally:
                st.stop_stream()
                st.close()
                pa.terminate()

        threading.Thread(target=bucle, daemon=True).start()
        self._parar = lambda wait_for_stop=False: self._fin.set()

    def _reconocer(self, audio, motor, listo):
        try:
            if motor == "vosk":
                import json
                import vosk
                rec = vosk.KaldiRecognizer(self._modelo_vosk, PTT_HZ)
                rec.AcceptWaveform(audio)
                texto = json.loads(rec.FinalResult()).get("text", "")
            else:
                texto = self._rec.recognize_google(self.sr.AudioData(audio, PTT_HZ, 2), language=self.idioma)
        except Exception as e:                       # no se entiende nada, sin internet...
            no_entiende = motor != "vosk" and isinstance(e, self.sr.UnknownValueError)
            self.estado = "no te he entendido, repite" if no_entiende else f"error voz ({type(e).__name__})"
            print(f"[VOZ] {self.estado}")
            return
        self.estado = listo
        if texto:
            self._texto(texto)
        else:
            self.estado = "no te he entendido, repite"

    def _texto(self, texto):
        self.ultima = texto
        if self.ptt:                                 # lo has dicho con la tecla pulsada: es para el robot
            cmd, factor, motivo = interpretar_detalle(texto)
            if cmd is None:
                self.ultimo_cmd = f"({motivo})"
                print(f"[VOZ] '{texto}' -> ignorado: {motivo}")
                return
            self.ultimo_cmd = cmd if factor == 1.0 else f"{cmd} x{factor:g}"
            print(f"[VOZ] '{texto}' -> {self.ultimo_cmd}")
            self.cola.put((cmd, factor, texto))
            return
        ahora = time.time()
        with self._lock:
            if self._pend is not None:              # otra frase justo después de una orden: conversación
                print(f"[VOZ] '{self._pend[0][2]}' -> anulado: le ha seguido otra frase ('{texto}')")
                self._pend = None
                self.ultimo_cmd = "(frases seguidas: ignoro)"
        self._tiempos.append(ahora)
        while self._tiempos and ahora - self._tiempos[0] > CONVERSACION_S:
            self._tiempos.popleft()
        if len(self._tiempos) >= CONVERSACION_FRASES:
            self.ultimo_cmd = "(conversacion: ignoro)"
            print(f"[VOZ] '{texto}' -> ignorado: {len(self._tiempos)} frases en {CONVERSACION_S:.0f} s (conversacion)")
            return
        cmd, factor, motivo = interpretar_detalle(texto)
        if cmd is None:
            self.ultimo_cmd = f"({motivo})"
            print(f"[VOZ] '{texto}' -> ignorado: {motivo}")
            return
        self.ultimo_cmd = cmd if factor == 1.0 else f"{cmd} x{factor:g}"
        print(f"[VOZ] '{texto}' -> {self.ultimo_cmd}" + ("" if cmd in INMEDIATOS else f" (en {ESPERA_S:g} s si no se oye nada mas)"))
        with self._lock:
            if cmd in INMEDIATOS:
                self.cola.put((cmd, factor, texto))
            else:
                self._pend = ((cmd, factor, texto), ahora)

    def siguiente(self):
        """Siguiente comando pendiente (comando, factor, texto) o None."""
        with self._lock:
            if self._pend is not None and time.time() - self._pend[1] >= ESPERA_S:
                self.cola.put(self._pend[0])
                self._pend = None
        try:
            return self.cola.get_nowait()
        except queue.Empty:
            return None

    def devolver(self, item):
        """Deja un comando para el siguiente que lea la cola (p. ej. 'coger' oído dentro del IBVS)."""
        self.cola.put(item)

    def parar(self):
        if self._parar:
            self._parar(wait_for_stop=False)


# ----------------------------------------------------------------------------- Prueba
EJEMPLOS = [
    # coger
    ("Coge la linterna", "coger"), ("coge", "coger"), ("agarra la linterna", "coger"), ("empieza", "coger"),
    ("cógela", "coger"), ("coge la linterna por favor", "coger"), ("recoge la lámpara", "coger"),
    # guardar
    ("Guarda punto", "guardar"), ("guarda el punto", "guardar"), ("guarda", "guardar"), ("guardar", "guardar"),
    ("guárdalo", "guardar"), ("marca aquí", "guardar"), ("este punto", "guardar"), ("memoriza el punto", "guardar"),
    ("vale guarda", "guardar"), ("ok guarda este punto", "guardar"),
    # iluminar
    ("Ilumíname aquí", "iluminar"), ("ilumina aquí", "iluminar"), ("ilumina", "iluminar"),
    ("iluminar", "iluminar"), ("alumbra", "iluminar"), ("enfoca aquí", "iluminar"), ("dale luz", "iluminar"),
    ("ilumina el punto que he guardado", "iluminar"), ("pon la luz aquí", "iluminar"),
    # mover el punto
    ("muévete a la derecha", "derecha"), ("a la derecha", "derecha"), ("derecha", "derecha"),
    ("muévete un poco a la derecha", "derecha"), ("izquierda", "izquierda"), ("mueve a la izquierda", "izquierda"),
    ("desplázate a la izquierda", "izquierda"), ("mueve arriba", "adelante"), ("muévete hacia arriba", "adelante"),
    ("muévete un poco hacia arriba", "adelante"), ("adelante", "adelante"), ("hacia delante", "adelante"),
    ("mueve abajo", "atras"), ("atrás", "atras"), ("muévete hacia abajo", "atras"), ("hacia mí", "atras"),
    # subir / bajar en vertical
    ("arriba", None), ("sube", None), ("aléjate", None),            # "arriba" QUITADO
    ("abajo", "acercar"), ("baja", "acercar"), ("acércate", "acercar"), ("baja un poco", "acercar"),
    ("para abajo", "acercar"), ("desciende", "acercar"), ("baja mucho", "acercar"),
    # control
    ("para", "parar"), ("quieto", "parar"), ("stop", "parar"), ("para ya", "parar"), ("sigue", "seguir"),
    ("sal", "salir"), ("hemos terminado", "salir"),
    # NO son órdenes: conversación, varias órdenes a la vez, nada
    ("hola qué tal", None), ("", None),
    ("oye pues yo creo que deberíamos mover esto un poco a la derecha", None),
    ("la linterna está encendida", None), ("qué luz más bonita", None), ("ese es el punto bueno", None),
    ("guarda y luego ilumina", None), ("coge la linterna e ilumina", None), ("muévete a la derecha y para", None),
    ("vamos a ver si funciona el robot ahora", None),
    ("para que está aquí en línea recta", None), ("como cuando voy a coger la linterna", None),
    ("espera que lo miro", None), ("sigue tú", "seguir"), ("bueno si acaba de cogerlo", None),
]


def _probar():
    malos = 0
    for frase, esperado in EJEMPLOS:
        cmd, f = interpretar(frase)
        ok = cmd == esperado
        malos += not ok
        print(f"  {'OK ' if ok else 'MAL'}  {frase!r:34} -> {cmd}{'' if f == 1.0 else f' x{f:g}'}"
              + ("" if ok else f"   (esperado {esperado})"))
    print(f"\n{len(EJEMPLOS) - malos}/{len(EJEMPLOS)} bien")
    return malos


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--micros":
        import pyaudio
        pa = pyaudio.PyAudio()
        for i in range(pa.get_device_count()):
            d = pa.get_device_info_by_index(i)
            if d["maxInputChannels"] > 0:
                print(i, d["name"])
        pa.terminate()
    elif len(sys.argv) > 1 and sys.argv[1] == "--micro":
        import time
        v = EscuchaVoz(microfono=int(sys.argv[2]) if len(sys.argv) > 2 else None)
        v.iniciar()
        print(v.estado, "- (Ctrl+C para salir)")
        ult = ""
        try:
            while True:
                time.sleep(0.1)
                if v.estado != ult:
                    ult = v.estado
                    print("  [estado]", ult)
        except KeyboardInterrupt:
            v.parar()
    elif len(sys.argv) > 1:
        print(interpretar(" ".join(sys.argv[1:])))
    else:
        sys.exit(1 if _probar() else 0)
