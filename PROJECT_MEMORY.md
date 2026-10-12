# Project memory — Drone pose estimation & trajectory prediction (tesi magistrale)

Memoria viva del progetto: decisioni prese, fatti verificati sui dati, problemi aperti, idee future.
Aggiornare a ogni decisione. Non cancellare voci: marcarle come superate (~~barrato~~ + data + motivo).

Ultimo aggiornamento: 2026-10-12

---

## 1. Obiettivo

Pipeline in 3 fasi su video/immagini di droni in volo:

1. **Keypoint 2D** — rilevare le 4 eliche del drone nell'immagine.
2. **Posa 6-DoF** — PnP (keypoint 2D + modello 3D del drone + intrinseci `K`) → posizione XYZ + orientamento.
3. **Predizione traiettoria** — sequenza di pose passate → pose future (RNN/GRU/LSTM vs Kalman / modello fisico).

Primo esperimento: confronto **YOLO-Pose vs DroneKey** sui keypoint.

Risorse calcolo: GPU del laboratorio universitario + Kaggle.

---

## 2. Decisioni prese

| Data | Decisione | Motivo |
|---|---|---|
| 2026-09-16 | **Strategia 1: sintetico-first, reale come verifica.** Niente capitolo opzionale con sintetico proprio (Unity/Blender). | GT completa solo sul sintetico; reale senza GT di posa. |
| 2026-09-16 | Keypoint = **4 eliche** (schema del dataset), non bracci/centro/motori. | Coordinate 3D disponibili solo per quelle. |
| 2026-09-16 | Confronto fase 1 su **crop identico** per entrambi i modelli. Tabella con 3 setting: (A) crop da bbox GT, (B) crop da bbox del detector YOLO, (C) YOLO-Pose a frame intero (riga extra). | Isola qualità keypoint dalla detection; B = pipeline realistica. |
| 2026-09-16 | Interfaccia unica `KeypointDetector.predict(image, bbox) -> (4,3)` in **coordinate del frame originale**. | `K` è definita sul frame intero; coordinate crop = posa sbagliata in silenzio. |
| 2026-09-16 | Fase 2 non tocca reti: legge JSON di output della fase 1. | Fasi testabili separatamente. |
| 2026-09-16 | Rotazioni per fase 3 in quaternione o 6D-continuous, **mai Euler**. Input anche velocità. | Discontinuità ±180° rovina la loss. |
| 2026-09-16 | Split **per sequenza/traiettoria**, mai per frame. | Frame consecutivi quasi identici → leak. |
| 2026-09-16 | Formato canonico unico (JSON stile COCO-keypoints + `R`, `t`, `K`, `kpts_3d`, `drone_class`, `seq_id`, `frame_idx`) + converter verso YOLO-pose e DroneKey, con test round-trip. | Una sola fonte di verità. |
| 2026-10-07 | Dataset principale: **6DroneSyn** (DroneKey++), in `/Volumes/MakeMePasta/6DroneSyn`. | Ha tutto: 2D/3D keypoint, posa, K, sequenze. |
| 2026-10-07 | **Ordine keypoint canonico = circolare** (come droni 02, 03). Droni 01, 04, 05, 06, 08 → scambiare key3 ↔ key4 (anche nel modello 3D). `flip_idx = [1, 0, 3, 2]`. | Un solo modello multi-drone con identità coerenti; PnP con corrispondenze giuste. |
| 2026-10-07 | **Fronte = lato 1-2 (gimbal) per tutti i 7 droni**, confermato visivamente dall'utente. Semantica (punto di vista del drone): **key1 = anteriore destra, key2 = anteriore sinistra, key3 = posteriore sinistra, key4 = posteriore destra**. Coerente col modello dai dati: coppia 1-2 sempre allo stesso lato del frame body in tutti i droni. | Definizione unica di "davanti" per annotare il reale e dare senso agli angoli. Figure: `docs/figures/front_*.jpg`, script `scripts/audit/front_pilot.py`. |
| 2026-10-07 | Rotazione GT = **Kabsch** (modello 3D canonico ↔ `3DKey(camera)` riordinati), campi Euler del JSON ignorati (solo cross-check). Eulero estratto da `R` per report. | Campi Euler incoerenti per 6 droni; Kabsch dà `R` affidabile, Eulero è solo una rappresentazione. |
| 2026-10-07 | **Frame body FLU**: x = davanti (lato 1-2), y = sinistra, z = su. Eulero **`ZYX`** intrinseco: yaw (z), pitch (y), roll (x). | Convenzione aeronautica/ROS REP-103: nomi roll/pitch/yaw coerenti con la letteratura. |
| 2026-10-07 | **Split (opzione A)**: cartelle 11→10, 12→11, 13→12, 14→13 del paper. Quindi per cartella: **train** 01, 02, 04, 05, 08, 11, 12, 13 · **val** 03, 09, 14 · **test** 06, 07. Confermare con gli autori quando rispondono. | Conteggi tornano esatti (52.920); test 06–07 non ambiguo. |
| 2026-10-07 | **Origine del drone = centroide delle 4 eliche**: GT `t` = media di `3DKey(camera)`; modello 3D FLU ricentrato sul centroide. Per confronto con DroneKey++ si riporta anche `t_blender = t_centroide + R · offset` (offset 1.4–5.9 cm). | Misurabile anche sui droni reali; stessa scelta di DroneKey. |
| 2026-10-09 | Fase 3 approvata (dettagli §6) + **esperimento di predizione 2D su video reali** (opzione A). | Unica valutazione quantitativa reale possibile per la fase 3: non servono `K` né pose. |
| 2026-10-09 | **DroneKey si reimplementa e si riaddestra da noi** (dal paper + `train.py` parziale: ResNet + 6 layer transformer d=1024, 8 head, gated sum, loss Mahalanobis pose-adaptive α=5 D=10, input 224×224, Adam 1e-5, batch 8, 100 epoche). | Pesi persi (risposta autori), repo incompleto (mancano dataloader e loss). Va dichiarato in tesi: confronto con una nostra reimplementazione. |
| 2026-10-09 | Baseline PnP aggiuntiva **`SOLVEPNP_ITERATIVE` (default OpenCV)**, come usato dagli autori. | Riproduce il loro setup e mostra l'effetto dell'ambiguità non gestita. |
| 2026-10-10 | **Priorità all'MVP (traguardi T0–T4 di `docs/PIANO_DI_LAVORO.md`)**, estensioni dopo. Nell'MVP di sicuro: controllo di fedeltà di DroneKey, git, backup, cache, audit automatici, 3 semi, intervalli di confidenza, classe del drone, Fase 2 e Fase 3 in parallelo. | Tempo limitato: prima la base, poi si espande. |
| 2026-10-10 | **Fedeltà di DroneKey = scala di ablation "nero → bianco"** (passo 1.3b): dalla versione minima (backbone + regressione) alla completa, un componente alla volta, confrontando l'andamento con le ablation del paper; più 2DroneKey se scaricabile. | Idea dell'utente; controllo possibile senza i pesi originali. |
| 2026-10-10 | **Fase 3: generatore procedurale di traiettorie a keyframe** (passo 3.1b), tarato solo sul train, con manovre brusche; augmentation sull'osservato (frame persi, jitter). Solo traiettorie, non immagini (niente mesh/scene Blender). | Idea dell'utente: più animazioni e movimenti improvvisi; perturbare solo l'osservato non insegna manovre nel futuro. |
| 2026-10-10 | **Fase 1: copy-paste del drone su sfondi nuovi** con maschere SAM (passo 1.8), dopo la prima baseline E1. | Idea dell'utente sugli sfondi; per un modello a frame singolo conta la varietà, non il movimento dello sfondo. |
| 2026-10-10 | **Backup massimi e verificati** (regola 3-2-1, manifest SHA-256, test di ripristino, passo 0.9) e **training sempre riprendibile** da checkpoint (passo 0.10). | Richiesta utente: backup funzionanti su quasi tutto; disco esterno inaffidabile; sessioni Kaggle a tempo. |
| 2026-10-10 | **Calcolo:** lab 2× RTX 3090 (preferibilmente da remoto); Kaggle con dataset **privato ≤ 20 GB** → cache compatta (crop 256 px ~2 GB + frame a 640 px in JPEG ~3 GB) e notebook in catena con checkpoint. | Vincoli dell'utente. |
| 2026-10-12 | **Calcolo solo su Kaggle** (niente accesso remoto al lab); backup codice su **GitHub privato**, copia cloud su **OneDrive** fatta a mano dall'utente. | Il budget di calcolo diventa il vincolo principale: ~30 h GPU/settimana, sessioni a tempo, dataset privato ≤ 20 GB. |
| 2026-10-12 | **Piano v3** (`docs/PIANO_DI_LAVORO.md`) dopo la revisione esterna: **scheletro end-to-end** (YOLO-Pose veloce → IPPE → CV su una scena di val) entro 2–3 settimane, dopo una Fase 0 minima; il resto dell'infrastruttura cresce attorno. | Problemi d'integrazione (convenzioni, ordine, unità) emergono presto; evitare mesi di sola infrastruttura. |
| 2026-10-12 | Fase 3: generatore di traiettorie con **differential flatness** (spinta ∥ a + g → roll e pitch determinati, yaw libero) + **baseline fisica** (`a_h ≈ g·tan(tilt)`). L'ablation sulla rotazione si misura sulle traiettorie generate con fisica, **non** sul test di 6DroneSyn (lì non deve aiutare: risultato sul dataset). | Rende verificabile la parte "modelli basati sulla fisica" della traccia nonostante 6DroneSyn sia a keyframe. |
| 2026-10-12 | **M1 dopo l'MVP** (confermato); candidato: **EuRoC MAV, 6 sequenze Vicon** (posa 6D in CSV, velocità simili a 6DroneSyn; il CSV è dentro zip da 1–2 GB con le immagini). Blackbird scartato come primo candidato (troppo veloce, percorsi ripetuti). | Tempo limitato: prima la base. |
| 2026-10-12 | **M11 minimo nell'MVP** (passo R.6): calibrazione con scacchiera (anche telefono) + 2–3 min di video del drone disponibile → E7, posa reale valutata solo con proxy (flip, scatti temporali, distanza coerente). Errore di riproiezione = proxy debole (il raffinamento lo azzera con 4 punti). | L'utente ha accesso a drone e camera; chiude il buco di `K` e modello 3D sul reale (non quello della GT). |
| 2026-10-12 | E5: classificazione camera fissa/mobile + **compensazione del moto della camera con omografia** (ORB + RANSAC sullo sfondo, bbox esclusa); risultati separati per gruppo. | Per un brandeggio (rotazione pura) l'omografia è il modello esatto. |
| 2026-10-12 | Rumore per l'augmentation della Fase 3 da **archivio di errori reali** `stima − GT` sulle scene di **validazione** (non train: la rete ha visto il train); modello parametrico anisotropo, dipendente dalla distanza, con flip, come riserva. | L'errore del PnP non è i.i.d. (profondità, correlazione temporale, flip). |
| 2026-10-12 | **Budget:** 3 semi solo per i confronti finali di E1 ed E4; scala di ablation a 1 seme e schedule ridotto; setting A e B = stesso modello. Stima ~50–70 h GPU Kaggle. Resume: curva di loss entro tolleranza su GPU, identità esatta solo su CPU. | Solo Kaggle (~30 h/settimana). |
| 2026-10-12 | **Statistica:** bootstrap **appaiato** sulla differenza tra modelli, per traiettoria; 1–2 fold leave-scenes-out solo con YOLO-Pose. **Metrica principale di E1 = errore di posa dopo il PnP** (+ px, PCK stretto, identità); OKS-AP solo per confronto col paper; Pareto accuratezza/FPS con YOLO-Pose n/s/m. | Con 2 scene di test gli intervalli separati si sovrappongono; l'OKS-AP satura. |
| 2026-10-12 | **Fedeltà di DroneKey su tre basi:** test unitari contro le equazioni del paper + overfit su sottoinsieme + tendenze della scala di ablation. | La scala da sola prova tendenze, non correttezza. |
| 2026-10-12 | **Propagazione dell'incertezza dopo l'MVP** (versione leggera: σ px da val → covarianza della posa dallo Jacobiano del PnP → R del Kalman e input GRU → uscita gaussiana NLL). Agganci predisposti subito: campo covarianza nel JSON della Fase 2, Kalman con R per frame, GRU con testa di uscita separata. | Non necessaria per l'MVP; con gli agganci costa pochi giorni dopo. |
| 2026-10-07 | Rappresentazioni: report in tesi = **Eulero** (roll/pitch/yaw); metrica errore = **angolo geodetico** tra `R_stima` e `R_GT`; loss/input rete = quaternione o 6D. | Eulero leggibile ma discontinuo (±180°, gimbal lock). |
| 2026-10-07 | Unità: usare **sempre il frame camera** (`3DKey(camera)`, `3Dtranslation(camera)`); errori riportati in queste unità (≈ metri). | World = camera × 1.718, non metrico. |
| 2026-10-07 | Background 14 di `02_04` mancante: **ignorato**. | 120 frame su 52.920, irrilevante. |
| 2026-10-08 | **Conversione in metri veri rimandata alla fase 3.** Fasi 1–2 restano in unità del dataset (confrontabili con DroneKey++). Solo in fase 3 (e sul reale) si moltiplica l'intera `t` (x, y, z) per `S_reale/S_dataset` della classe; la rotazione non cambia. **PROMEMORIA: ricordarlo all'utente quando si inizia la fase 3** (vedi 7bis). | Scala sbagliata colpisce solo la profondità; conversione = una funzione per classe, facile da correggere quando arrivano le dimensioni giuste. In fase 2 sul sintetico il PnP usa il modello ricavato dai dati, quindi nessun errore di scala (vedi 7bis). |
| 2026-10-10 | **Report delle traslazioni in tre tabelle.** (1) **Unità camera** (calcolo interno e confronto principale); (2) **unità world** = camera × 1.718, per confrontarsi con DroneKey++ in entrambi i casi (il paper dice camera, gli autori dicono che la loro pipeline usa il world); (3) **metri veri stimati** = `t · s_c`, con `s_c = S_reale/S_dataset = 1/rapporto_camera`: 01 0.75, 02 0.81, 03 0.86, 04 0.93, 05 0.93, 06 0.86; **08 (Tello) escluso** finché non c'è una dimensione verificata. Tabella 3 dichiarata come **stima** (incertezza di `s_c` ~5–10%: specifiche DJI = diagonale rotori, noi misuriamo i mozzi, §7bis punto 2). Cambiare unità **non riduce l'errore**: cambia solo il numero scritto. Runtime: sul **reale** il PnP usa il modello 3D con dimensioni vere (specifiche DJI) e restituisce già metri; sul **sintetico** usa il modello dai dati e si moltiplica per `s_c` (classe oracle o predetta). | L'incertezza sull'unità dei 0.135 m di DroneKey++ rende il confronto valido solo riportando entrambe le unità; i metri veri servono per la fase 3 e per il reale (§7bis). |
| 2026-10-09 | **Fase 1: due esperimenti di confronto YOLO-Pose vs DroneKey.** **(1) Sintetico → sintetico:** entrambi addestrati su 6DroneSyn (stesso split, stesso crop, stesso ordine keypoint) e testati sul test sintetico (06–07): confronto di qualità alla pari. **(2) Sintetico → reale ("a nudo"):** gli stessi modelli, senza ulteriore addestramento, sul mini test-set reale annotato a mano (300–500 frame, solo quadricotteri): misura il gap sim-to-real e quale modello regge meglio. Niente confronto zero-shot "senza dataset" sui keypoint. | YOLO-Pose pre-addestrato conosce i punti del corpo umano, non le eliche; DroneKey nel repo pubblico non ha pesi né codice completo (vedi §4, stato del repo). Un confronto è alla pari solo con gli stessi dati di training. Se gli autori inviano i pesi: riga extra "pesi ufficiali" solo come riferimento, non alla pari (ordine keypoint e dataset diversi). |

---

## 3. Fatti verificati sui dati

### 3.1 6DroneSyn — struttura
- Path: `/Volumes/MakeMePasta/6DroneSyn/scene_XX/scene_XX/XX_DD/XX_DD_BB_GGG/NNNN.png` + `XX_DD_anno.json` per (scena, drone).
  - `XX` scena, `DD` drone_type, `BB` background (01–22), `GGG` = campo `degree` (000/090/180/270).
- Cartelle scene: `01–09, 11–14` (manca 10). `scene_13/` contiene cartella mal nominata `scene_12/12_xx`, ma il contenuto è scena 13 (ID e `path` = 13, zero overlap con scena 12).
- Droni: `01, 02, 03, 04, 05, 06, 08` (7 tipi, manca 07). `08` ≈ DJI Tello (dimensioni). Mapping completo drone_type → modello DJI **non presente** nei dati.
- Frame per sequenza: 120 (scene 01–09), 360 (scene 11–14). Totale annotazioni **52.920** = paper. Immagini su disco 52.800 (manca `02_04` bg 14, assente anche nello zip originale — zip integro).
- File `._*` = AppleDouble macOS da exFAT → ignorare sempre.
- Zip già estratti ancora sul disco (~140 GB): eliminabili, decisione dell'utente.
- Immagini 1920×1080. PNG con chunk eXIf non valido → warning libpng innocui.

### 3.2 6DroneSyn — campi JSON
- `camera_parameters`: `K` unica per tutto il dataset (f = 2666.67 px, c = (960, 540)), camera in origine, 4 yaw (0, 90, -180, -90 → `degree` 000/090/180/270). `extrinsic.rotation_matrix` ha scala 1.7184 (camera scalata in Blender).
- Per frame: `bbox`, `2DKey` (4), `3DKey(world)`, `3DKey(camera)`, `drone.{2Dlocation, 3Dtranslation/rotation (world|camera)}`. Euler in **radianti**, convenzione scipy `"xyz"`.

### 3.3 Verifiche superate
- `K · 3DKey(camera)` → `2DKey` con errore < 1e-12 px su tutti i frame. Idem centro drone.
- Nessun keypoint fuori immagine/bbox, nessun NaN, bbox tutte valide.
- `3DKey(camera) = 0.582 · R · 3DKey(world)` esatto (similarità, residuo 1e-13).
- Corpo rigido: distanze tra keypoint costanti (std ≤ 4.5e-5) → nessuno scambio di identità nel tempo.
- 4 keypoint **complanari** (sv3/sv1 ≈ 0, drone 06: 0.002) → PnP con `SOLVEPNP_IPPE` (2 soluzioni). Su GT senza rumore la prima soluzione è sempre giusta; con rumore l'ambiguità diventa reale.
- Temporale: 273 sequenze senza buchi, salto max 0.095 unità/frame, nessun teletrasporto.
- Controllo visivo su 28 frame casuali: keypoint sui mozzi delle eliche.

### 3.4 Problemi trovati e trattamento
1. **Campo rotazione sfasato per 6 droni su 7** (01, 03, 04, 05, 06, 08). Vale per `3Drotation(world)` e `3Drotation(camera)`. Al proiettare il modello 3D con la rotazione annotata: errore medio 0 px a `degree=000`, 60–325 px a 090/180/270. Drone 02 corretto a ogni `degree`.
   - Correzione verificata (residuo max 0.04°): `R = euler_xyz(rot) @ Rz(degree)` per drone ≠ 02; drone 02 invariato.
   - Causa probabile: export Blender (yaw del `degree` applicato all'oggetto ma non scritto nel campo, tranne che per il drone 02).
   - **Approccio adottato (confermato 2026-10-07):** GT di rotazione ricavata con **Kabsch** tra modello 3D canonico e `3DKey(camera)` → indipendente da convenzioni Euler. Campi Euler solo come cross-check.
   - Figure: `docs/figures/rotbug_1.jpg`, `docs/figures/rotbug_2.jpg`; campioni keypoint: `docs/figures/sheet_a.jpg`, `sheet_b.jpg`.
     **Attenzione nel leggerle:** riferimento = degree 000 (scelta arbitraria). Freccia verde = asse y del modello, NON il "davanti" (non ancora definito).
   - 2026-10-07, obiezione utente: a degree 180 la freccia verde annotata sembra corretta (camera ruotata 180° in Blender). Verifica: con modello costruito dai frame a 180, il 180 torna (0 px) ma il 000 sbaglia 234–256 px (drone 01, 05); drone 02 torna con qualsiasi riferimento. Conclusione: la camera spiega *perché* lo sfasamento = `degree`, ma il campo resta incoerente tra degree per 6 droni → va comunque corretto/ignorato. Script: `scripts/audit/ref_test.py`.
   - Separare due questioni: (a) coerenza → Kabsch; (b) dove sta il "davanti" → definire frame del modello 3D per drone guardando il gimbal (conferma visiva dell'utente).
2. **Ordine keypoint non uniforme**: Z-order (01, 04, 05, 06, 08) vs circolare (02, 03). → decisione: circolare (sezione 2).
3. **Due sistemi di unità** (world = camera × 1.718). → decisione: frame camera.
4. **Bbox larghe**: 1.64× l'estensione x dei keypoint (mediana); area hull keypoint = 20% bbox.
5. **Nessun flag di visibilità**: eliche auto-occluse comunque annotate → `v=2` per tutte in YOLO.

### 3.5 Altri fatti utili
- ~~Modello 3D per drone (frame body, unità camera), ricavabile dai dati: spaziatura eliche da ~0.10 (drone 08) a ~0.45 diagonale (drone 06). Leggermente > specifiche reali (7–25%).~~ (sostituito 2026-10-07 dalla voce sotto, con valori FLU esatti)
- **Modello 3D FLU per drone** (2026-10-07, `scripts/audit/flu_model.py`): ricavato invertendo la trasformazione rigida `m = R^T (k_cam − t_cam)` (idea dell'utente). Funziona **solo con R corretta**: dispersione tra frame 2.5e-5 con R corretta vs 0.05–0.16 con R annotata (drone 02: 0 in entrambi i casi). Passaggio assi body Blender → FLU: `[[0,-1,0],[1,0,0],[0,0,1]]` (rotazione pura); verificato key1/key2 con x > 0 (davanti), key1 y < 0 (destra). Origine = origine oggetto Blender.
  Coordinate FLU [x avanti, y sinistra, z su], unità camera:

  | drone | key1 FR | key2 FL | key3 RL | key4 RR |
  |---|---|---|---|---|
  | 01 | (0.081, −0.154, −0.015) | (0.081, 0.151, −0.015) | (−0.119, 0.110, 0.020) | (−0.119, −0.107, 0.020) |
  | 02 | (0.059, −0.109, 0.016) | (0.059, 0.108, 0.016) | (−0.096, 0.105, −0.011) | (−0.096, −0.104, −0.011) |
  | 03 | (0.095, −0.180, 0.007) | (0.095, 0.180, 0.007) | (−0.170, 0.153, −0.001) | (−0.167, −0.153, −0.001) |
  | 04 | (0.080, −0.138, 0.018) | (0.079, 0.136, 0.018) | (−0.113, 0.123, −0.003) | (−0.113, −0.121, −0.003) |
  | 05 | (0.125, −0.164, 0.073) | (0.125, 0.162, 0.073) | (−0.114, 0.132, 0.044) | (−0.114, −0.137, 0.044) |
  | 06 | (0.115, −0.190, 0.000) | (0.115, 0.193, 0.000) | (−0.169, 0.154, −0.010) | (−0.169, −0.152, −0.012) |
  | 08 | (0.044, −0.050, 0.013) | (0.044, 0.050, 0.013) | (−0.050, 0.049, 0.013) | (−0.050, −0.050, 0.013) |

  Dimensioni (unità camera): larghezza anteriore / posteriore / lunghezza / diagonale — 01: 0.306/0.216/0.200/0.33 · 02: 0.217/0.209/0.155/0.26 · 03: 0.360/0.305/0.264/0.43 · 04: 0.274/0.244/0.193/0.32 · 05: 0.326/0.270/0.238/0.38 · 06: 0.382/0.306/0.284/0.45 · 08: 0.099/0.099/0.094/0.14. Dislivello z anteriore−posteriore fino a 3.5 cm. Origine Blender distante dal centroide delle eliche 1.4–5.9 cm.
  ~~Da fare: confronto con le dimensioni dichiarate nel paper IEEE 11696925~~ → il paper non le riporta; confronto fatto 2026-10-07 con le **specifiche reali DJI fornite dall'utente** (diagonale rotori): Mini 3 Pro 247 mm, Mini 2 213, Air 3 ~362, Air 2 302, Mavic 2 Pro 354, Mavic 3 380.1, Tello "~155" (stima utente, **non plausibile**: l'intero Tello con protezioni ha diagonale ~135 mm → interasse motori ~100–110 mm).
  Rapporto diagonale dataset / reale: **01 1.34 · 02 1.24 · 03 1.17 · 04 1.07 · 05 1.08 · 06 1.17 · 08 ~1.3 (0.88 con la stima 155)**. → I modelli 3D del dataset **non sono in scala reale, né in modo uniforme**. Le "unità camera" sono metri solo nominali. Specifiche complete (pesi, ingombri aperto/chiuso) nel messaggio utente del 2026-10-07; usarle per il modello 3D dei droni **reali**.
- Profondità drone: 1.5–5.8 unità camera (mediana 2.6).
- Assetto: scene 01–02 quasi livellate; scena 11 estrema (roll fino −144°, pitch −122°).
- **Solo 91 traiettorie uniche** (13 scene × 7 droni): le 3 sequenze con background diversi di una stessa (scena, drone) sono la stessa traiettoria.
- 22 panorami di sfondo riutilizzati su tutte le scene, incluso test (con yaw diversi).

### 3.6 Dataset universitario (96 GB, `/Volumes/MakeMePasta/drone_dataset`)
- Parti: Augmentation, IR, RGB, YOLO runs, Videos (Anti-UAV-RGBT, Comina, Fused, Povoletto, Purdue1026, USC thermal, USC youtube, VideDataset).
- Immagini non sequenziali con **sole bbox YOLO** (classe unica). Nessun keypoint, nessuna GT di posa, nessun intrinseco.
- Video Comina/Povoletto: solo overlay di box YOLO, senza label → praticamente inutili.
- Sample `aligned/`: coppie IR/RGB registrate 640×512; droni misti (quad, esacottero). Igiene: un `.png` che è un JPEG, un IR 637×508.
- Uso previsto: detector (stage 0), mini test-set reale annotato (300–500 frame, solo quadricotteri) per misurare il gap sim-to-real; demo qualitativa.

### 3.7 Riferimenti
- DroneKey: arXiv 2508.17746 — https://github.com/kkanuseobin/DroneKey — dataset 2DroneKey/3DronePose (sintetico, Air2S + Mini2). Baseline nel paper: YOLOv8 AP 70.8%, DroneKey 99.68%.
- DroneKey++ / 6DroneSyn: arXiv 2602.06211. Niente PnP (decoder ray-based + class embedding). Split ufficiale: train 01–02, 04–05, 08, 10–12; val 03, 09, 13; test 06–07. Contatto autori: cnu.cvl.hsb@gmail.com.
- PDF in `docs/papers/`: DroneKey = IROS 2025 (DOI 10.1109/IROS60139.2025.11247556); DroneKey++ = ICRA 2026 (DOI 10.1109/ICRA57385.2026.11696925 — è il link IEEE 11696925 indicato dall'utente).

### 3.8 Cosa dicono i paper (letti 2026-10-07)

**Attenzione:** il dataset su disco (6DroneSyn) è quello di **DroneKey++**. Il paper DroneKey (IROS) usa un altro dataset (2DroneKey + 3DronePose: Air2S + Mini2, 10 seq × 1K frame + seq 11–13), **non presente su disco**. DroneKey = metodo da confrontare; DroneKey++ = documentazione del nostro dataset.

**Dataset 6DroneSyn (DroneKey++, Tab. II e Sez. IV):**
- **30 FPS**. Scene #01–#09: 4 s (120 frame); #10–#13: 12 s (360 frame). → conferma lo split A (cartelle 11–14 = paper 10–13). Curiosità: in Fig. 1c gli autori scrivono "Scene 14 – Tello", quindi anche loro usano internamente la numerazione delle cartelle.
- Moto: #01–03 lineare, #04–06 non lineare, #07–09 lineare, #10–13 non lineare.
- `degree` = ambiente 360° **ruotato di 90° attorno a Z** (22 ambienti × 4 = 88 sfondi). Conferma l'intuizione dell'utente sulla causa; il campo rotazione resta comunque incoerente per 6 droni.
- 91 sequenze = 13 scene × 7 droni; ogni scena ha 21 sottosequenze = 7 droni × 3 sfondi (conferma: 3 sfondi = stessa traiettoria).
- Ordine classi in Tab. III: Mini3 (Pro), Mini2, Air3, Air2, Mavic2 (Pro), Mavic3, Tello. Mapping probabile per ordine: **01 Mini3 Pro, 02 Mini2, 03 Air3, 04 Air2, 05 Mavic 2 Pro, 06 Mavic 3, 08 Tello**; plausibile per dimensioni (02 il DJI più piccolo, 08 molto più piccolo di tutti, 06 il più grande). **Verificato 2026-10-07** confrontando le traiettorie con la Fig. 4 (`docs/figures/drone_mapping.jpg`, script `scripts/audit/traj_map.py` e `traj_map_xz.py`): forma unica e identica al paper per **02 = Mini2** (arco a parabola, scena 09), **04 = Air2** (arco, scena 07), **06 = Mavic 3** (zig-zag a 3 curve, scena 06). Scena 03 (Mini3) non discrimina: tutti i droni fanno la stessa linea verticale. Gli altri da ordine + dimensioni + aspetto visivo: **08 = Tello** (certo, dimensioni), 01 = Mini3 Pro, 03 = Air3, 05 = Mavic 2 Pro (coerenti, non verificati con le traiettorie).
- Traslazione = "coordinate 3D del centro del drone" nel **sistema camera**, in **metri** → numeri del paper in unità camera, come le nostre. (Unità world = ×1.72, non usate dal paper.)
- **Nessuna tabella con le dimensioni dei droni** in nessuno dei due paper (DroneKey dice solo "considering the drone sizes" / "specifiche del drone").
- Nessuna data augmentation. Validazione sul reale solo qualitativa (video supplementare).

**Metodo e numeri DroneKey++ (sullo stesso test 06–07):**
- Encoder DroneKey + token [cls] per la classe; decoder ray-based `v = A⁻¹[y,1]` + class embedding; rotazione predetta come 3 angoli normalizzati in [0,1] con sigmoid e loss circolare → **usano angoli Euler, probabilmente dal campo del JSON**.
- Training: ResNet ImageNet, Adam 1e-5 cosine, batch 32, 100 epoche, A100.
- Metriche: rotazione = angolo geodetico `arccos((tr(RpᵀRgt)−1)/2)` (come la nostra), MAE e MedAE; traslazione = distanza euclidea, MAE e MedAE (niente RMSE: "sensibile agli outlier").
- Risultati medi (R MAE / MedAE °, t MAE / MedAE m): YOLOv8-keypoint + PnP 79.25 / 87.55, 0.391 / 0.105 · **DroneKey (con dimensioni GT) 20.17 / 21.28, 0.533 / 0.370** · DronePose 79.06 / 82.63, 0.163 / 0.317 · DroneKey++ 17.34 / 17.10, 0.135 / 0.242. Risultati anche per drone e scena.
- Stranezze da tenere a mente: (1) MAE t (0.135) < MedAE t (0.242), insolito; (2) il baseline PnP ha errori di rotazione di 60–108°, compatibili (ipotesi, non provato) con GT ruotata male dal bug `degree`; (3) PnP citato come [16] = RANSAC (Fischler & Bolles), senza senso con soli 4 punti.
- **Nessuna metrica 2D dei keypoint (AP/OKS) su 6DroneSyn** → il nostro confronto YOLO-Pose vs DroneKey sui keypoint di 6DroneSyn è un contributo nuovo.
- Hanno addestrato sull'ordine keypoint grezzo del JSON (Z-order per 5 droni, circolare per 2) → identità incoerenti tra classi; la nostra canonicalizzazione è un miglioramento da dichiarare.

**Metodo e numeri DroneKey (IROS, sul loro dataset):**
- CNN + encoder transformer (6 layer, 8 head, d = 1024), testa gated-sum sulle rappresentazioni di tutti i layer, loss Mahalanobis pose-adaptive (α = 5, D = 10). Adam lr 1e-5, batch 8, 100 epoche, RTX 3060, 44 FPS. Split 7K / 2K / 1K.
- **Input = immagine; il paper non dice se si usa un crop** né la risoluzione di input → da verificare nel codice (`train.py`).
- OKS "drone": `exp(−d² / (0.2·β²·γ²))`, β = √area dell'oggetto, γ = 1 per tutte le eliche. Metriche SR90, SR95 (successo a OKS ≥ 0.90 / 0.95) e AP.
- Baseline "YOLOv8" = **detector** (non pose) con 4 classi per codificare l'ordine delle eliche → AP 70.8%. **YOLO-Pose non è mai stato confrontato** → il nostro confronto è nuovo anche qui.
- Posa: PnP = EPnP; poi la rotazione viene ricostruita da `v1 = Y2 − YO`, `v2 = Y3 − YO`, `v3 = v1 × v2` (vettori non ortonormalizzati), traslazione = **centroide delle 4 eliche** YO; R e t filtrati con **filtro di Kalman**. Risultati su 3DronePose: MAE-angle 10.62°, RMSE 0.221 m, MAE-abs 0.076 m.
- Lavoro futuro dichiarato: dati reali, numero di eliche diverso da 4, decoder 3D.

### 3.9 Compromessi imposti dal dataset (da dichiarare in tesi, nei limiti)

**Fase 2 (posa):**
1. Rotazione GT ricalcolata con Kabsch: esatta, ma **non confrontabile alla lettera** con i numeri di DroneKey++ (che probabilmente usano il campo Euler bacato).
2. Traslazione in "unità del dataset (≈ m)", non metri veri: scala dei modelli errata del 7–34% per drone.
3. **Due modelli 3D**: sul sintetico il modello ricavato dai dati (con le specifiche reali la profondità sarebbe sbagliata del 7–34% già in partenza); sul reale le specifiche DJI. Il gap sim-to-real include anche un fattore di scala.
   Ipotesi collegata (da verificare/chiedere): DroneKey in DroneKey++ usa "dimensioni fisiche reali" → errore di traslazione alto (0.533 m) compatibile con la scala sbagliata; il drone con scarto minore (Air 2, 1.07) ha l'errore più basso (0.017–0.211 m).
4. Origine nel centroide ≠ "centro del drone" di DroneKey++ (non definito nel paper) → confronto con offset.
5. Ambiguità IPPE a 2 soluzioni intrinseca: lo schema del dataset ha solo 4 eliche complanari, nessun punto fuori piano.
6. Reale: niente GT di posa né `K` → solo qualitativo.

**Fase 1 (keypoint):**
1. Ordine normalizzato (correzione senza costo), ma i pesi pre-addestrati degli autori usano l'ordine misto → DroneKey va riallenato.
2. Nessun flag di visibilità → tutte `v=2`; niente metriche separate visibili/occluse.
3. Bbox larghe (1.64×) → regola di crop nostra + jitter; nessuna bbox "stretta" di riferimento.
4. Diversità effettiva minore: 91 traiettorie (non 273); stessi 22 sfondi in train e test.
5. Nessuna metrica keypoint pubblicata su 6DroneSyn → DroneKey va rilanciato da noi (più lavoro, confronto alla pari).
6. Reale: test piccolo annotato a mano (300–500 frame), IR senza controparte sintetica.

### 3.10 Risposta degli autori (Seobin Hwang, ricevuta 2026-10-09, testo in `docs/2response.txt`)

Confronto punto per punto con i nostri problemi. **Stato verifica: eseguita il 2026-10-09** con `scripts/audit/reply_check.py` (disco collegato); risultati nelle righe sotto (✅ verificato sui dati).

| Tema | Cosa dicono | Confronto con i nostri dati | Esito |
|---|---|---|---|
| Rotazione | Il campo `3Drotation` non include lo yaw `degree`. Correzione `R = euler_xyz(rot) @ Rz(degree)`, Blender 'XYZ' (R = Rz·Ry·Rx), ricostruzione entro 0.1 mm. Aggiunta sezione "Annotation Conventions" nel README del dataset (non visibile nel README GitHub al 2026-10-09). | Stessa formula e convenzione trovate da noi (Euler `xyz` scipy = Rz·Ry·Rx). **Differenza:** loro la applicano a **tutti** i droni; noi abbiamo misurato che il **drone 02** è già corretto senza Rz (0 px a ogni degree, con qualsiasi riferimento) → con la loro formula il 02 verrebbe ruotato di 90/180°. | ✅ confermato il bug. ✅ **Conflitto sul drone 02 verificato con la loro formula esatta** (`reply_check.py` A, su `3Drotation(world)`): dispersione del modello body tra frame 0.04 mm per i droni 01, 03, 04, 05, 06, 08 con `Rz(degree)` (la loro formula è perfetta), ma **172.6 mm per il drone 02** con `Rz(degree)` contro 0.045 mm senza. Chi applica la loro correzione a tutti i droni sbaglia il 02 di `degree`. Kabsch rende la questione irrilevante per la nostra GT; da segnalare nel seguito della mail. |
| Fattore 1.718 | Viene dalla `view_matrix` (scala ≈ 0.582). `R_w2c = V[:3,:3] / scale` (convenzione OpenCV, +Z avanti). Camera nell'origine. | Coincide con le nostre misure (3DKey(camera) = 0.582 · R · 3DKey(world)). | ✅ spiegato |
| Quale frame è "metrico" | **Consigliano il frame world** per valori metrici; la loro pipeline usa il world. | **In conflitto con i dati e col loro paper** (numeri verificati 2026-10-09, `reply_check.py` B). Diagonali eliche vs specifiche DJI: frame camera 1.07–1.34×, frame **world 1.84–2.30×** (01 2.30, 02 2.13, 03 2.02, 04 1.84, 05 1.86, 06 2.02). Lo scarto 1.07–1.34× che loro stessi attribuiscono al posizionamento dei mesh è misurato nel frame camera. Il paper DroneKey++ scrive che `t` è nel sistema camera. | ⚠️ **da chiarire**: i numeri di DroneKey++ (0.135 m) potrebbero essere in unità world (1.72× le nostre). Riportare entrambe le unità. |
| `t` | È l'**origine body del mesh**, non il centroide delle eliche. | Coerente con l'offset 1.4–5.9 cm misurato. | ✅ confrontare DroneKey++ con `t_origine = t_centroide + R·offset` (già previsto) |
| Dimensioni mesh | Modelli posizionati con bbox da specifiche DJI, mesh non calibrati → da qui lo scarto 1.07–1.34×. Per valutazioni metriche: **usare i keypoint 3D delle annotazioni** come dimensioni di riferimento. | Coincide col nostro modello ricavato dai dati (`flu_model.py`). | ✅ confermato l'approccio |
| Solver PnP | `cv2.solvePnP` **default** (= `SOLVEPNP_ITERATIVE`, nessuna gestione dell'ambiguità a 2 soluzioni). | Con 4 punti complanari il solver iterativo può convergere sulla soluzione ribaltata → spiegazione plausibile degli errori di 60–108° del baseline, insieme al bug di rotazione. | ✅ → aggiungere `SOLVEPNP_ITERATIVE` come riga di baseline "alla DroneKey++" |
| Ordine keypoint | Segue key1–key4 delle annotazioni; stesso ordine in training. | Conferma che hanno allenato sull'**ordine misto** (Z-order per 5 droni, circolare per 2). Non hanno risposto sull'incoerenza. | ✅ la nostra canonicalizzazione resta un miglioramento |
| Pesi addestrati | **Persi** (guasto del server). Dataloader non pubblico. | Repo DroneKey incompleto (vedi §4). | ❌ DroneKey va **reimplementato e riaddestrato** da noi |
| Bbox | Estensione dei keypoint 2D + padding per lato `pad = max(0.4w, 0.4h, 20px)`, poi clip all'immagine. | La formula dà larghezza ≥ 1.8× l'estensione dei keypoint; noi abbiamo misurato **mediana 1.64×, p5 1.37×** → non torna esattamente. | ⚠️ **Verificato: la formula dichiarata NON riproduce le bbox** (`reply_check.py` D): errore massimo per box p50 = 61 px, p95 = 119 px, **0.0% delle box entro 1 px**; frame con bbox a bordo immagine solo 6.5%, quindi non è clipping. Padding reale (frame non clippati, campione 677): orizzontale ≈ 0.32·w per lato (≈ 84–87 px), verticale **asimmetrico** (alto ≈ 54 px, basso ≈ 73 px); un fit lineare lascia residui di 14–19 px → nessuna regola semplice sui keypoint. Ipotesi (non verificata): bbox ricavata anche dal mesh proiettato. Nessun impatto: usiamo la nostra regola di crop con jitter; domanda da porre nel seguito della mail. |
| Licenza | Uso non commerciale, tesi ok citando il paper; **vietata la redistribuzione** del dataset. | — | ✅ non mettere dati 6DroneSyn nel repo; rimandare alla pagina ufficiale |
| Senza risposta | Q2 (effetto del bug sul baseline PnP), Q7 (dimensioni usate nel baseline DroneKey), Q8 (MAE < MedAE), Q10 (crop/risoluzione di DroneKey). | Q10 parzialmente risolta dal `train.py` (input 224×224). | Promesso un seguito dopo le loro scadenze |

### 3.11 La fisica del moto nel dataset (verificato 2026-10-09, `scripts/audit/physics_check.py`)
- In un quadricottero reale l'accelerazione orizzontale segue l'inclinazione (`a_h ≈ g·tan(tilt)`). Nel dataset il coseno tra accelerazione orizzontale e direzione d'inclinazione è **≈ 0 in tutte le scene** (reale ≈ +1). Esempio: scena 03, inclinazioni fino a 42° con accelerazione nulla.
- → Traiettorie **animate a keyframe in Blender, non simulate con la fisica**. Assetto e moto sono indipendenti.
- Velocità mediana 0.2–0.8 m/s, massima 2.9 m/s; accelerazione massima 10.5 m/s² (unità dataset).
- Conseguenze per la fase 3: il "modello fisico" utile qui è solo **cinematico** (CV/CA/Kalman); un modello dinamico del quadricottero non è valido su questi dati; la rotazione in input non dovrebbe aiutare sul sintetico (da dimostrare con ablation) mentre aiuterebbe sul reale → altro gap sim-to-real da dichiarare.

---

## 4. Domande aperte

- [x] ~~Conferma approccio Kabsch~~ → confermato 2026-10-07.
- [x] ~~Definizione semantica di "key1"~~ → risolto 2026-10-07: fronte = lato 1-2 (vedi sezione 2).
- [x] ~~Assi del frame body~~ → risolto 2026-10-07: FLU (sezione 2).
- [x] ~~Mapping scene 11–14~~ → risolto 2026-10-07: opzione A (sezione 2).
- [x] ~~Unità dei numeri del paper DroneKey++~~ → risolto 2026-10-07: traslazione nel sistema camera, in metri (sez. 3.8).
- [x] ~~FPS delle sequenze~~ → risolto 2026-10-07: **30 FPS** (sez. 3.8).
- [x] ~~Mapping drone_type → modello DJI~~ → verificato 2026-10-07: 01 Mini3 Pro, 02 Mini2, 03 Air3, 04 Air2, 05 Mavic 2 Pro, 06 Mavic 3, 08 Tello (sez. 3.8; 02/04/06 confermati dalle traiettorie, gli altri per ordine + dimensioni).
- [x] ~~Paper IEEE 11696925~~ → è DroneKey++, letto 2026-10-07. **Nessuna tabella di dimensioni** → scala metrica non verificabile dai paper; si assume unità camera = metri come dichiarato dagli autori. Fonte esterna alternativa: specifiche DJI (diagonale) → rapporto modello/specifica 1.08–1.34, dipende dal drone.
- [ ] Come DroneKey usa l'input (crop o immagine intera, risoluzione): non scritto nel paper. **Letto `train.py` il 2026-10-09**: transform `Resize(300) + CenterCrop(224)` + normalizzazione ImageNet, quindi input 224×224; il CenterCrop suggerisce immagini già ritagliate sul drone, ma il caricamento dati (`cocoprocessing_.py`) **non è nel repo** → resta da confermare con gli autori. ResNet50 + 6 layer transformer (d=1024, 8 head), batch 8.
- **Stato del repo DroneKey (2026-10-09)**: contiene solo `README.md`, `assets/`, `train.py`, `method/loss_key.py`. `loss_key.py` è uno **stub** (91 byte, classe senza corpo); mancano `earlystop.py` e `cocoprocessing_.py` (importati da `train.py`); **nessun peso, nessuna release, nessuno script di inferenza** (sezione "Test" del README vuota, `requirements.txt` commentato). Quindi non si può eseguire su video senza riscrivere le parti mancanti e riaddestrare. Percorsi annotazioni nel codice: `../TEST/annotations/re_key/{train,valid,test}.json`.
- [x] ~~Origine del frame drone~~ → deciso 2026-10-07: **centroide delle 4 eliche** (vedi sezione 2).
- [x] Classe drone nel PnP → deciso 2026-10-07: **due esperimenti, oracle + realistica** (classe predetta). Decisione sezione 2 non ancora definitiva: l'utente vuole prima chiarire i punti sopra.
- [x] **Mail agli autori inviata 2026-10-09** (testo in `docs/1mailDraft.txt`, 14 domande). **Risposta ricevuta lo stesso giorno** → analisi in §3.10.
- [x] **Drone 02 e la correzione degli autori**: verificato 2026-10-09 con la loro formula esatta: il 02 con `Rz(degree)` ha dispersione 172.6 mm contro 0.045 mm senza (gli altri 6 droni 0.04 mm con `Rz`). Da segnalare nel seguito della mail (bozza in `docs/3response.txt`).
- [x] **Unità per il report** → deciso 2026-10-10 (sezione 2): tre tabelle (camera, world, metri veri stimati). Gli autori dicono "world è metrico", i dati dicono il contrario (world 1.84–2.30× le specifiche, camera 1.07–1.34×); quale unità usino i loro 0.135 m resta da chiarire (domanda 2 nel seguito della mail).
- [x] **Formula bbox**: verificato 2026-10-09 (vedi §3.10): la formula dichiarata non riproduce le bbox (0.0% entro 1 px, padding reale asimmetrico). Da chiedere nel seguito della mail, come dubbio nostro sull'applicazione della formula.
- [x] **M1** (voli reali Blackbird / EuRoC / UZH-FPV per la Fase 3) → deciso 2026-10-10: **dopo l'MVP** (tempo limitato).
- [x] **M3** (split con sfondi mai visti) → deciso 2026-10-10: dopo l'MVP, come misura dell'effetto del copy-paste degli sfondi.
- [x] **M11** (video propri) → deciso 2026-10-10: dopo l'MVP; l'utente ha già video di droni.
- [x] ~~Soglia "85" per la fedeltà di DroneKey~~ → **ritirata dall'utente** 2026-10-12: fedeltà valutata solo con le tre basi (test unitari, overfit, tendenze dell'ablation).
- [x] ~~Accesso remoto alle 2× RTX 3090 del lab~~ → 2026-10-12: **non disponibile**. Calcolo su **Kaggle** (lab solo in caso di problemi, andando fisicamente).
- [x] ~~Remoto git e cloud~~ → 2026-10-12: **GitHub, repo privata**; **OneDrive** come copia cloud, gestita **manualmente dall'utente**.
- [x] ~~Revisione esterna del piano~~ → discussa e chiusa 2026-10-12: esito punto per punto in `docs/PIANO_DI_LAVORO.md` §1.3 (v3).
- [ ] **Modello esatto del drone per M11 minimo**: un DJI, non ancora noto. La registrazione la farà **un'altra persona, non esperta**: istruzioni passo passo in `docs/registrazione_M11/ISTRUZIONI_REGISTRAZIONE.md` (+ `.pdf`) e scacchiera `docs/registrazione_M11/scacchiera_A4_25mm.pdf` (generata da `scripts/make_checkerboard.py`, 9×6 angoli interni verificati con OpenCV). Il modello 3D vero arriva dalle misure dei motori richieste nelle istruzioni.
- [ ] 2DroneKey (dataset del paper DroneKey) ancora scaricabile? Serve per il controllo di fedeltà della reimplementazione (M2).
- [x] Seguito della mail scritto: `docs/3response.txt` (drone 02, frame metrico, formula bbox). **In attesa di risposta.** Restano senza risposta anche Q2/Q7/Q8/Q10 della prima mail.

## 5. Da vedere più avanti ("stesse carte" sintetico ↔ reale)

- Intrinseci camera reale (RGB e IR 640×512): assenti; calibrazione con scacchiera se il lab ha le camere. Senza `K` → sul reale solo metriche 2D.
- Angoli in frame camera = orientamento *rispetto alla camera*. Assetto vero (roll/pitch rispetto alla gravità) richiede orientamento camera nel mondo: noto nel sintetico (estrinseci), ignoto nel reale salvo camera in bolla o IMU.
- Annotazione reale con stessa convenzione: ordine circolare, eliche occluse annotate (posizione stimata), solo quadricotteri. Metrica anche "a meno di simmetria" (fronte ambiguo, specie IR).
- Regola di crop unica tra domini + bbox jitter in training (sintetico 1.64× vs YOLO reale probabilmente più stretto: misurarlo).
- Gap di scala: sintetico drone 170–1100 px su 1920×1080; reale ~80 px su 640×512 → augmentation downscale + blur + rumore.
- IR senza controparte sintetica → test cross-modale puro; calo atteso, è un risultato.
- Modello 3D reale in metri per PnP su droni reali di modello noto.
- Fase 3: deduplicare le 91 traiettorie, split per traiettoria; dati pochi e corti → baseline Kalman / fisica obbligatoria.
- Leak di sfondo train/test (stessi panorami): citarlo nei limiti.

## 6. Stato del piano

- [x] Sezione 0 — Fondamenta (repo, dati, audit, formato canonico, mini test-set reale) — approvata 2026-09-16
- [x] Sezione 1 — Keypoint (modelli, interfaccia, crop, training, metriche incl. accuratezza identità, sim-to-real) — approvata 2026-10-07
- [x] Sezione 2 — Posa / PnP — **approvata 2026-10-09**: IPPE (2 soluzioni) + raffinamento LM sul modello 3D vero; baseline EPnP/SQPnP; disambiguazione (a) riproiezione, (b) + continuità temporale, (c) rifiuto se ambiguo, con tasso di flip; keypoint a bassa conf pesati, non scartati; esperimento di sensibilità con rumore σ 0.5–10 px sui keypoint GT; metriche geodetica + per asse ZYX, errore traslazione con z a parte e in %, successo 5°/5cm e 10°/10cm, riproiezione, FPS; output JSON grezzo + temporale; classe drone oracle + realistica.
- [x] Sezione 3 — Predizione traiettoria — **approvata 2026-10-09**:
  - Dati: 91 traiettorie uniche (deduplicate), 30 FPS (dt = 1/30 s); split per scena: train 56, val 21, test 14 (scena 06 non lineare, 07 lineare). Frame allineato alla gravità (x destra, y avanti, z su) centrato nella camera; ogni finestra relativa all'ultima posizione osservata.
  - Finestre: osservazione N = 15 frame (0.5 s; ablation 10 e 30), previsione H = 30 frame (1 s), valutazione a 0.1 / 0.33 / 0.5 / 1 s, passo 1 frame (~7–8K finestre train).
  - Input: posizione del centroide, velocità (differenze finite), rotazione 6D. Output: posizioni future + orientamento futuro (secondario).
  - Modelli: (1) posizione costante, CV, CA; (2) Kalman CV/CA tarato su val = "modello fisico" **solo cinematico** (§3.11); (3) GRU (richiesto) + LSTM, encoder sul passato e uscita diretta degli H passi, 50–200K parametri; (4) **GRU residuale** sulla previsione CV; (5) ablation GRU con/senza rotazione.
  - Augmentation: rotazione attorno alla verticale, specchiatura sinistra/destra, inversione temporale (lecita: moto cinematico), ricampionamento velocità ×0.8–1.2, rumore sulle pose con la statistica d'errore misurata in fase 2.
  - Valutazione: (a) oracle con pose GT; (b) pipeline completa con pose dalla fase 2 — le 3 copie con sfondi diversi diventano 3 osservazioni rumorose indipendenti della stessa traiettoria.
  - Metriche: ADE/FDE per orizzonte, errore geodetico dell'orientamento, separati per moto lineare/non lineare e per drone, **skill vs CV** (% di miglioramento).
  - Test: CV con errore 0 su moto rettilineo uniforme (scena 03); nessuna finestra a cavallo di traiettorie diverse; augmentation che preservano la cinematica.
  - Output pipeline: posa attuale + traiettoria prevista a 1 s, sovrapposta al video nella demo.
  - **Esperimento aggiunto (decisione utente 2026-10-09, opzione A): predizione 2D su video reali** del dataset universitario (Anti-UAV RGBT e altre sequenze con bbox): prevedere posizione del centro/bbox nell'immagine a 0.5–1 s. Servono solo le tracce, non `K` né pose. Stessi modelli (baseline CV/CA/Kalman + GRU) con input 2D. È l'unica valutazione **quantitativa reale** della fase 3. Dettagli da definire nel piano: quali sequenze hanno bbox per frame, FPS reale, normalizzazione per dimensione della bbox, gestione dei frame senza rilevamento.
- [ ] Spec scritta + piano di implementazione — **2026-10-10: piano di lavoro scritto in `docs/PIANO_DI_LAVORO.md`** (valutazione, punti deboli D1–D7, miglioramenti M1–M11, passi 0.1–4.3 + fase R, parallelismo, traguardi T0–T5, rischi). **v3 il 2026-10-12** dopo la revisione esterna (scheletro, flatness, M11 minimo, omografia, archivio errori, budget, bootstrap appaiato); v2 il 2026-10-10 con le osservazioni dell'utente (§0 del piano: MVP prima, scala di ablation, generatore di traiettorie, copy-paste sfondi, backup 0.9, checkpoint 0.10, M1/M3/M11 dopo l'MVP). **2026-10-12: piano di implementazione della Fase 0 minima + scheletro scritto** in `docs/superpowers/plans/2026-10-12-fase0-scheletro.md` (12 task, TDD, dataset finto per i test, test `dataset` con `SIXDRONESYN_ROOT`). In attesa di revisione e scelta del modo di esecuzione.

## 7bis. Promemoria per l'inizio della fase 3 (ricordare all'utente!)

- **Precisazione 2026-10-08 (chiarita con l'utente):** il PnP a runtime **non scopre** la dimensione del drone: riceve il modello 3D (distanze tra rotori) come input fisso e ne ricava `t`. Un drone in scala 1:2 a metà distanza dà gli stessi pixel → la distanza assoluta richiede **classe nota + dimensioni vere**; classe ignota = solo direzione e forma della traiettoria. In **fase 2 sul sintetico** si usa il modello ricavato dal dataset (`flu_model.py`): `t` esce già nelle unità della GT, nessun errore di scala. L'errore "34 vs 28 cm" (PnP compensa con la distanza, fattore 28/34) nasce solo se il modello PnP ha scala diversa da quella di rendering: sintetico con specifiche DJI al posto del modello dati (**da non fare**) oppure reale con specifiche sbagliate. La conversione in metri è solo una moltiplicazione per classe, applicata quando servono metri veri (fase 3, reale).
- **Conversione unità dataset → metri veri** (decisione 2026-10-08, sezione 2). Prima di allenare la predizione di traiettoria:
  1. Moltiplicare l'intera `t` per `S_reale/S_dataset` di ogni classe (rapporti in §3.5, diagonale dataset / reale: 01 1.34, 02 1.24, 03 1.17, 04 1.07, 05 1.08, 06 1.17, 08 incerto). Rotazione invariata.
  2. Confrontare la stessa grandezza: interasse motori reale vs distanza tra mozzi (i keypoint stanno sui mozzi, le specifiche DJI danno la diagonale rotori). Verificare prima di applicare.
  3. **Tello (08): dimensione reale non affidabile** ("~155" dell'utente non plausibile, ~100–110 mm di interasse). Serve un dato verificato.
  4. Sul reale: PnP con modello 3D da specifiche (già in metri) e `K` della camera reale; niente riscalatura.
  5. Dichiarare in tesi: la conversione allinea la scala, non le proporzioni dei mesh.

## 7. Idee / possibili integrazioni

- Metrica "accuratezza identità" dei keypoint (scambi 1↔3 per simmetria): invisibile all'OKS, rompe il PnP.
- Disambiguazione IPPE con continuità temporale + errore di riproiezione.
- ~~Fallback keypoint se DroneKey non gira: RTMPose (mmpose).~~ (2026-10-09: DroneKey non gira di sicuro — repo incompleto e pesi persi — e si reimplementa; RTMPose resta un terzo modello possibile se avanza tempo.)
- Predizione 2D su reale: confrontare la skill vs CV sul reale con quella sul sintetico proiettato in 2D (stessa metrica in pixel normalizzati per la bbox) → misura quanto i movimenti reali sono più o meno prevedibili di quelli animati a keyframe.
- Bootstrap annotazione reale: blob caldi dei motori in IR → proposta keypoint → correzione in CVAT, trasferita su RGB registrata.
- Script di audit esplorativi in `scripts/audit/` (`inventory.py`, `geom_audit.py`, `rot_audit.py`, `rot_hyp.py`, `layout.py`, `temporal.py`, `visual.py`, `show_rot_bug.py`) — da ripulire in un unico `audit.py` quando si crea il repo.
