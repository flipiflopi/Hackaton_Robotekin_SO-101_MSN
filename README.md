# SO-101: coger una linterna e iluminar un punto de una PCB

Proyecto de hackathon con un brazo **SO-101** (LeRobot). El robot hace dos cosas seguidas:

1. **Fase 1 – coger la linterna** (aprendizaje por imitación). Una política **ACT** entrenada con 54
   demostraciones por teleoperación coge la linterna, que está tumbada en la mesa, y la levanta.
2. **Fase 2 – iluminar el punto** (visión clásica, sin deep learning). El usuario toca la PCB con un boli azul
   y pulsa `k`. El robot apunta el haz a ese punto con **control visual (IBVS)** desde la cámara de arriba. Si
   alguien tapa el haz, **esquiva el obstáculo** con trayectorias parabólicas y sigue iluminando.

| | |
|---|---|
| 🤗 Dataset | [FlipiFlopi007/so101_tumbada_full](https://huggingface.co/datasets/FlipiFlopi007/so101_tumbada_full) (54 episodios, 20.770 frames, 2 cámaras) |
| 🤗 Modelo | [FlipiFlopi007/act_tumbada](https://huggingface.co/FlipiFlopi007/act_tumbada) (ACT, checkpoint 18.000) |
| 🎥 Vídeo de la demo | _(enlace)_ |

<!-- Foto del montaje: guardarla como media/montaje.jpg y quitar este comentario
![Montaje](media/montaje.jpg)
-->

## Hardware

| Elemento | Detalle |
|---|---|
| Brazos | 2 × SO-101: **leader** (se mueve a mano, fuente 5 V) y **follower** (fuente 12 V, motores Feetech STS3215) |
| Cámara wrist | En la pinza, 640×480 @ 30 fps (índice OpenCV 0) |
| Cámara top | Logitech C920 **fija**, mirando la mesa en diagonal (índice OpenCV 1). La usan la política y el IBVS |
| PC | Windows 11, RTX 3050 Laptop 4 GB |
| Escena | Alfombrilla gris, PCB verde, linterna de luz azul, boli azul |

> ⚠️ La política es **muy sensible a la posición de la cámara top**. Tiene que estar donde estaba al grabar
> (`herramientas/alinear_camara.py` lo comprueba).

## Instalación

Python 3.12 en Windows (nativo, sin WSL).

```bash
pip install -r requirements.txt
```

PyTorch con CUDA va aparte (se usó 2.11.0+cu128); ver https://pytorch.org. Hace falta **`opencv-python` con
ventanas**: con la versión `headless`, `cv2.imshow` no funciona.

**Calibración:** `calibracion/` tiene la de nuestros brazos (`mi_follower`, `mi_leader`). Para usarla, cópiala a
`%USERPROFILE%\.cache\huggingface\lerobot\calibration\robots\so_follower\` y `...\teleoperators\so_leader\`.
Con otros brazos, calíbralos con `lerobot-calibrate`.

**Modelo:**
```bash
hf download FlipiFlopi007/act_tumbada --local-dir modelo/act_tumbada
```
`vision_ibvs_esfera_arcos.py` lo busca ahí automáticamente (`FASE1_MODELO`). Ajusta también el puerto del
follower (`IBVS_PUERTO`, `COM13`) y los índices de las cámaras si en tu PC son otros.

Para comprobar con `herramientas/alinear_camara.py` que la cámara top está donde al grabar, descarga también el
dataset:
```bash
hf download FlipiFlopi007/so101_tumbada_full --repo-type dataset --local-dir dataset_tumbada
```

## Uso

### Demo completa (fase 1 + fase 2)

```bash
python vision_ibvs_esfera_arcos.py
```

1. **Fase 1** (ENTER para empezar, `n` para saltarla): el brazo abre la pinza, va a la postura de inicio de las
   demos, la red coge la linterna (15 s, `n_action_steps=60`) y afloja la pinza para que no se sobrecargue.
2. **Visión**: detecta la PCB y la punta del boli. **`k`** guarda el punto; **`i`** lanza el IBVS; **`l`** repite
   la fase 1; **`q`** sale.
3. **IBVS**: va a la pose de seguridad (`pose_seguridad.json`) y lleva la luz al punto (control PI con la
   jacobiana imagen↔efector guardada en `ibvs_jacobiano.json`, corregida en línea con Broyden).
4. **Vigilancia y esquive**: si algo tapa el haz durante 12 frames, guarda la pose de iluminación inicial y
   prueba **3 parábolas** (+x, +y y la diagonal +x+y de la base). En cada una, la linterna sube por encima del
   obstáculo, se desplaza, baja hacia la PCB y la pinza gira por el camino para seguir apuntando al punto.
   Solo se da por iluminado si el punto se ve iluminado **2 s seguidos**; entonces se queda ahí. Si una
   parábola no ilumina, vuelve a la pose inicial por el mismo camino y prueba la siguiente. Vuelve a la
   inicial cuando aparece un obstáculo nuevo.

Cada ejecución guarda su log en `logs/<fecha_hora>/`. En [`logs_demo/20261025_091529`](logs_demo/20261025_091529)
hay una ejecución completa: 4 oclusiones, todas resueltas.

### Solo la fase 1

```bash
.\demo_fase1.ps1 -Postura tumbada -Pasos 60 -Modelo modelo\act_tumbada
```

### Grabar y entrenar

```bash
lerobot-record --robot.type=so101_follower --robot.port=COM13 --robot.id=mi_follower --robot.cameras="{ wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, top: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30} }" --teleop.type=so101_leader --teleop.port=COM14 --teleop.id=mi_leader --dataset.repo_id=FlipiFlopi007/so101_tumbada --dataset.single_task="Coger la linterna tumbada" --dataset.num_episodes=20 --dataset.episode_time_s=20 --dataset.reset_time_s=15 --dataset.push_to_hub=false
```

```bash
lerobot-train --dataset.repo_id=FlipiFlopi007/so101_tumbada_full --dataset.image_transforms.enable=true --dataset.image_transforms.max_num_transforms=4 --dataset.eval_split=0.08 --policy.type=act --policy.device=cuda --batch_size=4 --steps=20000 --save_freq=2000 --output_dir=outputs/train/act_tumbada --job_name=act_tumbada
```

Entre grabar y entrenar se recortaron los tramos quietos de las demos (`dataset/recorte_rapido.py` +
`dataset/verificar_recorte.py`). Las curvas y el log del entrenamiento están en [`entrenamiento/`](entrenamiento/).

## Estructura

| Ruta | Para qué |
|---|---|
| `vision_ibvs_esfera_arcos.py` | Programa de la demo: fase 1 + visión + IBVS + vigilancia/esquive |
| `rollout_registro.py` | Fase 1: abre la pinza, va a la postura de inicio, ejecuta `lerobot-rollout` y afloja la pinza |
| `ir_a_inicio.py`, `demo_fase1.ps1` | Postura de inicio de las demos y prueba de la fase 1 sola |
| `pose_seguridad.json`, `ibvs_jacobiano.json` | Pose desde la que empieza el IBVS y jacobiana guardada |
| `calibracion/` | Calibración LeRobot de nuestros brazos |
| `herramientas/` | Chequeos antes de usar el robot: motores, puertos, cámara top, muñeca, quitar el par |
| `dataset/` | Recorte de los tramos quietos y verificación |
| `entrenamiento/` | Gráfica de pérdidas, resumen y log del entrenamiento |
| `analisis/` | Resumen de los rollouts de la fase 1 |
| `logs_demo/` | Log completo de una ejecución buena (consola, CSV, eventos, imágenes) |

## Lecciones y limitaciones

- **Cámara top y postura de inicio:** la política solo funciona con la cámara top donde estaba al grabar y
  arrancando en la postura de inicio de las demos.
- **Ajustes de la fase 1:** `n_action_steps=60` sin temporal ensembling, 15 s de rollout y la pinza limitada a
  3 por orden. Con 10, el motor 6 entra en protección por sobrecarga.
- **Límite de la muñeca:** con la linterna tumbada, `wrist_roll` trabaja a −165…−175°, junto a su límite y al
  salto ±180°. Por eso el haz solo puede inclinarse hacia un lado: el esquive funciona hacia +x / +y, no hacia
  −x / −y.
- **Validación engañosa:** en el entrenamiento la pérdida de validación se estanca desde el paso 6.000 y no
  predice el comportamiento en el robot. El checkpoint se eligió probándolo en el robot.
- **Si un motor se bloquea** ("Overload error" / "Missing motor IDs"): sujeta el brazo y apaga y enciende la
  fuente de 12 V.

## Licencia

Apache-2.0, como LeRobot.
