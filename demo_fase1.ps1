# Fase 1 (coger la linterna) con el modelo congelado para la demo.
# Uso:  cd C:\lerobot-work; .\demo_fase1.ps1            (follower en COM13)
#       .\demo_fase1.ps1 -Puerto COM10                   (si ha cambiado el puerto)
#       .\demo_fase1.ps1 -Encadenar                      (al acabar NO vuelve a reposo: sigue sujetando
#                                                         la linterna para la fase 2, iluminar.py)
#       .\demo_fase1.ps1 -Pasos 30                       (recalcular cada 1 s; 60 = cada 2 s, por defecto)
#       .\demo_fase1.ps1 -Modelo C:\lerobot-work\outputs\train\act_dia2\checkpoints\012500\pretrained_model
#       .\demo_fase1.ps1 -LimitePinza 4                  (la pinza aprieta menos: evita "Overload error" del motor 6)
#       .\demo_fase1.ps1 -Postura tumbada -Modelo ...     (modelos de la linterna tumbada: otra postura de inicio)
param([string]$Puerto = "COM13", [int]$Segundos = 20, [int]$Pasos = 60, [switch]$Encadenar, [int]$LimitePinza = 10,
      [string]$Modelo = "C:\lerobot-work\modelos_demo\fase1_act_dia2_015000",
      [ValidateSet("de_pie", "tumbada")][string]$Postura = "de_pie")

$extra = @()
if ($Encadenar) { $extra = @("--return_to_initial_position=false", "--robot.disable_torque_on_disconnect=false") }

C:\lerobot-env\Scripts\Activate.ps1
$env:HF_HUB_OFFLINE = "1"
# Siempre desde la misma postura que las demos (pinza cerrada); si no, la red da ordenes raras
python C:\lerobot-work\ir_a_inicio.py --puerto $Puerto --postura $Postura
if ($LASTEXITCODE -ne 0) { Write-Host "No se pudo llevar el brazo a la postura de inicio"; exit 1 }
# Si el motor 5 responde mal al conectar ("Incorrect status packet"), el rollout aborta antes de
# moverse: se reintenta hasta 3 veces
for ($intento = 1; $intento -le 3; $intento++) {
  "=== $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') modelo=$Modelo pasos=$Pasos postura=$Postura" | Out-File -Append -Encoding unicode C:\lerobot-work\outputs\rollout_demo.log
  lerobot-rollout --strategy.type=base `
    --policy.path=$Modelo `
    --policy.n_action_steps=$Pasos `
    "--robot.max_relative_target={shoulder_pan: 10, shoulder_lift: 10, elbow_flex: 10, wrist_flex: 10, wrist_roll: 10, gripper: $LimitePinza}" `
    --robot.type=so101_follower --robot.port=$Puerto --robot.id=mi_follower `
    --robot.cameras="{ wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, top: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30} }" `
    --task="Coger la linterna" --duration=$Segundos @extra 2>&1 | Tee-Object -Append -FilePath C:\lerobot-work\outputs\rollout_demo.log | Tee-Object -Variable salida
  $texto = $salida | Out-String
  if ($texto -match "Incorrect status packet" -and $texto -notmatch "control loop started") {
    Write-Host "`nFallo puntual de comunicacion con un motor al conectar; reintentando ($intento/3)..." -ForegroundColor Yellow
    Start-Sleep -Seconds 1
  } else { break }
}
