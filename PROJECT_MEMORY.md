# Project memory — Drone pose estimation & trajectory prediction (tesi magistrale)

Memoria viva del progetto: decisioni prese, fatti verificati sui dati, problemi aperti, idee future.
Aggiornare a ogni decisione. Non cancellare voci: marcarle come superate (~~barrato~~ + data + motivo).

Ultimo aggiornamento: 2026-10-16

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
| 2026-10-16 | **Fase 3, baseline tarate sul train** (CV/CA: lunghezza del fit `k`; Kalman: `q`, `r`), non sulla val come nella Sezione 3 del §6 e nel piano 3.2. Valutazione sulla val, test una sola volta alla fine. | Val pulita per confrontare baseline e reti sullo stesso terreno; le baseline hanno pochi parametri. Design: `docs/superpowers/specs/2026-10-16-fase3-dataset-baseline-design.md`. |
| 2026-10-16 | **Fase 3, codice separato** in `src/dronepose/forecast/` (+ `scripts/forecast/`, `tests/forecast/`, `configs/forecast.json`); il codice esistente (`trajectory/`, scheletro) non si modifica. Drone 08 nel dataset ma fuori dalle medie in metri (riga a parte in unità del dataset) finché manca una dimensione verificata; ingressi indipendenti dall'imbardata = asse "su" del drone nel frame della gravità. | Richiesta dell'utente; §6octies, §7bis. |
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
  - Modelli: (1) posizione costante, CV, CA; (2) Kalman CV/CA ~~tarato su val~~ tarato sul train (decisione 2026-10-16, §2) = "modello fisico" **solo cinematico** (§3.11); (3) GRU (richiesto) + LSTM, encoder sul passato e uscita diretta degli H passi, 50–200K parametri; (4) **GRU residuale** sulla previsione CV; (5) ablation GRU con/senza rotazione.
  - Augmentation: rotazione attorno alla verticale, specchiatura sinistra/destra, inversione temporale (lecita: moto cinematico), ricampionamento velocità ×0.8–1.2, rumore sulle pose con la statistica d'errore misurata in fase 2.
  - Valutazione: (a) oracle con pose GT; (b) pipeline completa con pose dalla fase 2 — le 3 copie con sfondi diversi diventano 3 osservazioni rumorose indipendenti della stessa traiettoria.
  - Metriche: ADE/FDE per orizzonte, errore geodetico dell'orientamento, separati per moto lineare/non lineare e per drone, **skill vs CV** (% di miglioramento).
  - Test: CV con errore 0 su moto rettilineo uniforme (scena 03); nessuna finestra a cavallo di traiettorie diverse; augmentation che preservano la cinematica.
  - Output pipeline: posa attuale + traiettoria prevista a 1 s, sovrapposta al video nella demo.
  - **Esperimento aggiunto (decisione utente 2026-10-09, opzione A): predizione 2D su video reali** del dataset universitario (Anti-UAV RGBT e altre sequenze con bbox): prevedere posizione del centro/bbox nell'immagine a 0.5–1 s. Servono solo le tracce, non `K` né pose. Stessi modelli (baseline CV/CA/Kalman + GRU) con input 2D. È l'unica valutazione **quantitativa reale** della fase 3. Dettagli da definire nel piano: quali sequenze hanno bbox per frame, FPS reale, normalizzazione per dimensione della bbox, gestione dei frame senza rilevamento.
- [ ] Spec scritta + piano di implementazione — **2026-10-10: piano di lavoro scritto in `docs/PIANO_DI_LAVORO.md`** (valutazione, punti deboli D1–D7, miglioramenti M1–M11, passi 0.1–4.3 + fase R, parallelismo, traguardi T0–T5, rischi). **v3 il 2026-10-12** dopo la revisione esterna (scheletro, flatness, M11 minimo, omografia, archivio errori, budget, bootstrap appaiato); v2 il 2026-10-10 con le osservazioni dell'utente (§0 del piano: MVP prima, scala di ablation, generatore di traiettorie, copy-paste sfondi, backup 0.9, checkpoint 0.10, M1/M3/M11 dopo l'MVP). **2026-10-12: piano di implementazione della Fase 0 minima + scheletro scritto** in `docs/superpowers/plans/2026-10-12-fase0-scheletro.md` (12 task, TDD, dataset finto per i test, test `dataset` con `SIXDRONESYN_ROOT`). In attesa di revisione e scelta del modo di esecuzione.

## 6bis. Esito dello scheletro (2026-10-13)

Scheletro end-to-end completato (S.1–S.4): GT → crop → YOLO-Pose (yolo11n-pose, 10 epoche, Kaggle) → IPPE → velocità costante → video. Codice nel pacchetto `dronepose`, un comando (`scripts/run_skeleton.py`). Sequenza di prova: scena 14, drone 03, sfondo 09 (360 frame, val). **Setting A** (crop dal box dei keypoint GT, quindi ottimistico: la detection non è ancora testata).

| Predittore | Errore rotazione (mediana) | Errore traslazione (unità camera) | Flip (>90°) | ADE / FDE mediani (1 s) |
|---|---|---|---|---|
| `gt` | 0.019° | 1.4e-4 | 0% | 0.042 / 0.103 |
| `gt+noise:2` (2 px) | 0.76° | 0.014 | 2.2% | 0.050 / 0.118 |
| `yolo` (10 epoche) | 3.4° (p90 8.1°) | 0.067 | 7.2% (26/360) | 0.114 / 0.176 |

Training (val, tutte le scene di val): `mAP50(P)` 0.90, `mAP50-95(P)` 0.89, plateau dall'epoca 8. Loss di posa: train 0.031 vs val 0.53 (da epoca 6 la val non scende più, il train sì: divario di generalizzazione tra scene, non sotto-addestramento).

Errore keypoint YOLO nel frame originale: mediana 9 px (p90 16 px) su un drone largo ~340 px (~2.6%). Nei frame in flip l'errore in px è lo stesso (5–10 px) dei frame senza flip: **il flip non dipende da keypoint peggiori**, ma dall'ambiguità di IPPE con 4 punti quasi complanari. Un tratto di 8 frame consecutivi (57–65) è in flip: l'ambiguità è temporalmente correlata → la scelta per continuità temporale (§2, strategia b) è la leva giusta, non più training.

Problemi d'integrazione trovati e come sono stati chiusi:
1. `.gitignore` con `data/` non ancorato escludeva anche `src/dronepose/data` → ora `/data/`.
2. Cache Kaggle impacchettata prima che il codice esistesse (codice vecchio nello zip) → due dataset separati: `dronepose-cache` (dati) e `dronepose-code` (~50 KB, da ricaricare a ogni modifica).
3. Kaggle monta i dataset in `/kaggle/input/datasets/<utente>/<nome>/` e il `path:` di `data.yaml` era fisso → `scripts/kaggle/kaggle_paths.py` trova i dataset e riscrive `data.yaml` (anche prima di una ripresa).
4. `geodesic_deg` via `arccos` ha rumore ~1e-6° vicino a zero: tre test hanno soglia 1e-4° invece di 1e-6°. Per le metriche di errori molto piccoli valutare una versione con `atan2` (non urgente: gli errori reali sono ≥0.01°).
5. Il piano importava `pytest` in `export_yolo.py` (inutilizzato) → tolto.
6. Warning `libpng: eXIf: invalid` sui PNG di 6DroneSyn: innocui.
7. Immagini mancanti confermate: 120 (02_04, sfondo 14), saltate e contate dal converter.

Da fare / aperto dopo lo scheletro:
- Setting B e C (crop da detector, frame intero): non ancora testati. Una **detection** non esiste ancora → per video reali serve un detector del drone.
- Una sola sequenza di val: l'errore va misurato su tutte le scene di val e di test, con più semi (budget §Piano 6).
- Quota Kaggle privata osservata dall'utente: fino a 200 GB (decisione "≤ 20 GB" del 2026-10-10 da rivedere: **in attesa di conferma dell'utente**).
- Passi del piano che prendono in carico i problemi: 2.x (disambiguazione temporale), 1.x (scala di ablation, copy-paste sfondi), E5/R.x (reale).

## 6ter. Come si lavora e dove sta cosa (2026-10-14)

**Repo e ambiente**
- Repo: `https://github.com/JuicyTomato/DroneKey.git` (privato). Pacchetto `dronepose` (layout `src/`), venv `.venv` (Python 3.14), dipendenze bloccate in `requirements-lock.txt`.
- Test: `.venv/bin/python -m pytest -q` (senza disco: 66 passed, 3 skipped). Con disco: `SIXDRONESYN_ROOT=/Volumes/MakeMePasta/6DroneSyn .venv/bin/python -m pytest -q` (69 passed). Marker `dataset` = serve il disco o `data/processed`.
- Dati derivati (tutti fuori da git, rigenerabili): `data/manifest/` (SHA-256, 52891 file), `data/processed/` (canonici JSONL 35160/12600/5040, crop YOLO, frame della sequenza `14_03_09`), `data/kaggle_cache/` + `data/dronepose-cache.zip` (0.97 GB) + `data/dronepose-code.zip` (~50 KB). Rigenerare: `build_drone_models.py` → `build_canonical.py` → `build_yolo_crops.py` → `package_cache.py` (README del repo).
- `configs/drone_models.json` (7 modelli FLU) è versionato. Risultati in `experiments/` (fuori da git): `skeleton_gt`, `skeleton_noise2`, `skeleton_out` (YOLO), `skeleton_run` (pesi `best.pt`/`last.pt`, curve).

**Regole di lavoro concordate con l'utente**
- Esecuzione del piano: modo **Native** (un solo contesto), revisione finale con un revisore a contesto pulito.
- **Commit solo dopo consenso, uno per task**; messaggi **molto corti** (es. `feat: geometry lib`); **nessun trailer Co-Authored-By**. **Push solo se chiesto esplicitamente.**
- Nessun file di 6DroneSyn né PDF IEEE in git (licenza). Le figure in `docs/figures/` (ritagli di frame) sono state **tolte dalla storia con rebase il 2026-10-14**, restano su disco e sono ignorate.
- Soglia test di rotazione 1e-4° (non 1e-6°): rumore di `arccos` vicino a 0 (~1e-6°). Se servono errori <1e-4°, usare una versione `atan2` di `geodesic_deg`.
- Chiavi `frame_id` = `scena_drone_sfondo_degree_numero` (univoche); prefisso di sequenza con `_` finale (es. `14_03_09_`). `run_sequence` rifiuta record di più sequenze.

**Kaggle**
- Due dataset privati: `dronepose-cache` (dati) e `dronepose-code` (codice, ricaricare come "New Version" a ogni modifica con `package_cache.py`).
- Kaggle monta in `/kaggle/input/datasets/<utente>/<nome>/` (non `/kaggle/input/<nome>/`). `scripts/kaggle/kaggle_paths.py` trova i dataset e riscrive `data.yaml`; le celle `%%bash` del README cercano `CODE`/`CACHE` da sole.
- Sessione interattiva: scaricare i file da Output → Data prima di chiudere (zippare con `zip -rq`); "Save Version" non serve.
- Quota dataset privati osservata: fino a 200 GB (decisione "≤ 20 GB" da rivedere: **in attesa di conferma**).
- **Trappola inferenza ultralytics:** `model.predict(lista_di_path, batch=N, stream=True)` su Kaggle ha ignorato `batch` e ha caricato l'intero split (12600 crop) in un solo batch → `CUDA out of memory. Tried to allocate 12.30 GiB` (= 12600 × 16×128×128 × 4 B) al warmup, su T4 da 14.56 GiB; il problema non è la GPU. Sempre spezzare a mano in blocchi (`export_keypoints.py` lo fa dal 2026-10-14). Segnale diagnostico: l'allocazione richiesta è ≈ N_immagini × dimensione di un'attivazione, non un batch normale.
- **Trappole delle celle Kaggle (2026-10-15, Task 5):**
  1. **Export bloccato 9 ore senza output.** Prima versione lanciata con Save & Run All: nessuna riga stampata, nemmeno l'`echo "pesi: ..."`; fermata a mano → Kaggle **scarta l'output** di una versione annullata. Causa **non ancora verificata**; sospetto: `find /kaggle/input -name best.pt` che percorre i ~106k file della cache su un mount di rete lento.
  2. **I dataset sono mount di rete** (cartelle `nobody:nogroup`, dimensione 0, non link simbolici). Mai usare `find /kaggle/input`: lento sulla cache, e con `-type d` ha restituito un percorso vuoto (`CODE=` → `can't open file '/scripts/...'`). Usare `ls -d /kaggle/input/datasets/*/<nome>` (glob) come nel README.
  3. **Input mancante senza errore:** in una sessione era montato `dronepose-code` e `dronepose-weights` ma **non** `dronepose-cache`. La cella del README §5 ora stampa `codice`/`cache`/`pesi` e si ferma se uno è vuoto. Primo controllo in caso di dubbio: `ls -la /kaggle/input/datasets/*/`.
  4. **Output invisibile:** sotto `%%bash` lo stdout di Python è in pipe e bufferizzato → nessuna stampa fino alla fine. Usare `python -u` e `print(..., flush=True)`; `export_keypoints.py` stampa ogni passo con il tempo trascorso.
  5. **Regola:** prima prova breve in **sessione interattiva** (`--limit 256`, `timeout`), poi l'esecuzione completa; Save & Run All solo per lavori già provati.
- YOLO: `yolo11n-pose`, 10 epoche, batch 64, imgsz 256, ~22 min su GPU Kaggle. `mAP50(P)` 0.90 (plateau dall'epoca 8). Più epoche da sole servono poco (divario train/val); leve migliori: sfondi (passo 1.8), modello più grande, più semi.

**Prossimi passi (nessun piano di implementazione scritto oltre lo scheletro)**
1. Disambiguazione temporale IPPE (flip 7.2% con YOLO, a blocchi: continuità temporale, §2 strategia b).
   **2026-10-14: piano di implementazione scritto** in `docs/superpowers/plans/2026-10-14-disambiguazione-sensibilita.md` (6 task: strategie (a)/(b)/(c) in `pose/disambiguation.py`, pipeline a due passate, replay offline, sensibilità 2.3 inclusa, export keypoint val/test su Kaggle, taratura soglie sulla sola val). Replay del 2026-10-14 su `14_03_09` con il codice del piano: (a) 26 flip, (b) 6 = oracolo, (c) 1 flip con copertura 94.7%; passo di rotazione GT max 4.6°/frame. Esecuzione nell'altra sessione.
2. Misurare su tutte le scene di val e di test (setting A/B/C), più semi, bootstrap appaiato.
3. Detector del drone nel frame (per setting B e per video reali) e fase R (dati reali, M11).
4. Fase 3 (traiettorie) e promemoria §7bis sulle unità in metri veri.
5. Rinviati dalla revisione finale: `INTER_AREA` inefficace in `warpAffine`; `package_cache.py` da `git archive HEAD`; confidenza per keypoint; ripresa con `--data` personalizzato.

## 6quater. Avanzamento del piano disambiguazione + sensibilità (aggiornato 2026-10-14)

Piano: `docs/superpowers/plans/2026-10-14-disambiguazione-sensibilita.md`. Eseguito inline (Native), **ordine: Task 5, 1, 2**, poi 3, 4, 6 (il 5 prima perché la parte Kaggle va lanciata subito e non dipende dal 4).
- [x] Task 5 — export keypoint YOLO (`keypoints/export.py`, `scripts/kaggle/export_keypoints.py`, README Kaggle §5). **Passo utente aperto:** lanciare su Kaggle (~~consigliato "Save & Run All"~~ 2026-10-15: prima prova breve in sessione interattiva, vedi trappole Kaggle in §6ter; GPU + Internet On) con input `dronepose-cache`, `dronepose-code` (New Version dopo il 2026-10-14) e `dronepose-weights` (zip di `best.pt`: `data/dronepose-weights.zip`, fuori da git). Output atteso: `keypoints.zip` con `val: 12600` e `test: 5040`; scaricarlo dalla pagina della versione (Versions → Output) in `experiments/keypoints/`. Lo script **non è mai stato eseguito** (ultralytics non installato in locale): se la versione fallisce, leggere il log.
- [x] Task 1 — `pose/disambiguation.py` (strategie `reproj`, `temporal`, `temporal_reject`; `TemporalParams`; `Choice`).
- [x] Task 2 — pipeline a due passate (`ippe_candidates`, `assemble_results`, `run_sequence(strategy=..., params=...)`); default `reproj`, riepilogo `gt` identico a prima. Nuovi campi per frame: `solution_index`, `rejected`, `reject_reason`, `reproj_ratio` (`null` se infinito), `oracle_rot_err_deg`; `summarize` in più: `n_rejected`, `coverage`, `p90_rot_err_deg`, `oracle_flip_rate`. `no_detection()` ora è pubblica (rename fatto col Task 5).
- [x] Task 3 — `pose/replay.py` + `scripts/replay_disambiguation.py` (su `14_03_09` con keypoint YOLO: reproj 7.22%, temporal 1.67% = oracolo, temporal_reject 0.29% con copertura 94.7%).
- [x] Task 4 — `pose/sensitivity.py` + `scripts/sensitivity_pnp.py` (matplotlib come extra `plots`; val, 3 semi, ~2 min, output in `experiments/sensitivity/`). Su keypoint GT + rumore σ: flip reproj/temporal/reject = 1.1/0.5/2.0% (σ 0.5), 7.0/2.2/3.4% (σ 2), 17.3/10.6/9.0% (σ 5), 30.1/29.1/19.0% (σ 10); rotazione mediana ≈ 0.26° (σ 0.5), 1.0° (σ 2), 7° (σ 10); copertura reject 98%→80%. Profondità: errore relativo 0.15% (σ 0.5) … 4.5% (σ 10), cresce poco con la distanza. Modello in scala sbagliata: profondità stimata/vera = scala (0.8…1.2), rotazione invariata (~0.016°). **Da indagare nel Task 6:** a rumore basso (σ ≤ 2) `temporal_reject` ha **più** flip di `temporal` (2.0 vs 0.5% a σ 0.5; 3.0 vs 1.1% a σ 1): il rifiuto sembra rompere la catena di riferimento dopo frame scartati.
- [x] Task 5 passo 6 e Task 6 (valutazione su tutta la val, soglie = default, registrazione): vedi §6quinquies. Keypoint in `experiments/keypoints/` (spostati da `results/`). Resta solo il video su Kaggle (utente).

Da ricordare:
- **Regola test:** le scene 06 e 07 non si usano per tarare nulla; gli script di replay rifiutano record di test senza `--allow-test`. Soglie scelte solo sulla val.
- Con keypoint GT le tre strategie danno lo stesso risultato (nessuna ambiguità): il confronto vero si vede solo con i keypoint YOLO.
- Le sezioni di piano non ancora fatte potrebbero richiedere conferma dei parametri (`reanchor_ratio` 3.0, `reanchor_frames` 3, `max_gap` 5, `max_jump_deg` 20, `max_reproj_px` 20): sono punti di partenza, da tarare sulla val nel Task 6.
- File non committati dell'utente: il piano stesso e `scripts/make_checkerboard.py`.

## 6quinquies. Disambiguazione IPPE e sensibilità: risultati su tutta la val (2026-10-15)

Keypoint YOLO (yolo11n-pose, 10 epoche, **setting A**: crop dal box GT) esportati su Kaggle per val (12600) e test (5040): `experiments/keypoints/`; 0 frame senza rilevazione, confidenza media 0.999. Replay: `experiments/replay_val/`; sensibilità: `experiments/sensitivity/`. Il **test non è stato toccato** (soglie scelte solo sulla val). Passo di rotazione GT per frame (val): mediana 0.86°, p99 4.5°, max 6.6°.

| Strategia | Flip (tutta la val) | Copertura | Rot. mediana | Rot. p90 | Errore traslazione mediano |
|---|---|---|---|---|---|
| (a) `reproj` | 15.5% | 100% | 3.99° | 155° | 0.062 |
| (b) `temporal` | 11.3% | 100% | 3.87° | 109° | 0.062 |
| (c) `temporal_reject` | 6.2% | 95.2% (611 scartati) | 3.67° | 11.0° | 0.059 |
| oracolo (migliore delle 2 soluzioni) | 6.8% | | | | |

**Per drone** (flip % a/b/c, oracolo, copertura di c): 01: 25.4/29.1/0.0, 1.2, 95.2% · 02: 10.7/3.1/5.2, 2.6, 93.4% · 03: 10.3/3.0/0.0, 1.4, 95.7% · 04: 9.6/1.3/0.0, 1.3, 95.6% · 05: 5.6/1.9/0.0, 1.9, 95.9% · 06: 5.2/1.3/0.0, 0.6, 97.8% · **08: 42.0/39.1/39.2, 38.6, 92.5%** (rotazione mediana 66° con (a), 19° con (c)).
Media semplice dei 6 droni senza il 08: (a) 11.1%, (b) 6.6%, (c) 0.9%, oracolo 1.5%.

**Il drone 08 (Tello) è un problema di identità dei keypoint, non di IPPE.** Sul 08 la permutazione dei 4 keypoint che minimizza l'errore in pixel non è l'identità nel **54%** dei frame (scambi sinistra/destra e rotazioni di ordine), contro 0–2.4% degli altri droni; errore keypoint mediano 31.8 px contro 5–11 px. Con i 4 punti quasi a quadrato (modello 0.047 × 0.05) YOLO non distingue le eliche davanti/dietro. Nessuna scelta tra le due soluzioni IPPE lo corregge (oracolo 38.6%). **Da affrontare nella Fase 1** (metrica di accuratezza identità, più dati/augmentation per il 08, eventuale testa di orientamento) e da dichiarare in tesi.

**Altri fatti:**
- (b) può *peggiorare* rispetto a (a): drone 01, 25.4% → 29.1% (la continuità si aggancia alla soluzione sbagliata per intere sequenze); (c) lo evita (0.0%). Ma (c) peggiora sul 02 rispetto a (b) (5.2% contro 3.1%) e, con keypoint GT + rumore basso (σ ≤ 2), ha più flip di (b): il rifiuto interrompe la catena di riferimento. Da indagare, non risolto.
- **Soglie scelte** (griglia di 36 combinazioni su val, regola del piano: copertura ≥ 95%, poi flip più basso, poi `max_jump_deg` più grande): `reanchor_ratio` 3.0, `max_jump_deg` 20, `max_reproj_px` 20 = i **default**, quindi nessuna modifica al codice. `max_jump_deg` 20 è sopra il p99 del passo di rotazione GT (4.5°). Con `max_reproj_px` 10 la copertura scende sotto il 95%; con 40 i flip salgono a ~7%. 16 combinazioni su 36 hanno copertura ≥ 95%.
- **Sensibilità (keypoint GT + rumore, 3 semi, val):** flip oltre l'1% a σ ≈ 0.5 (a), ≈ 1 (b), ≈ 0.5 (c); oltre il 5% tra σ 1 e 2 (a), tra 3 e 5 (b) e (c). A σ = 10: 30.1/29.1/19.0%. Rotazione mediana ≈ 1° a σ = 2 e ≈ 7° a σ = 10. Errore relativo di profondità 0.15% (σ 0.5) … 4.5% (σ 10), poco dipendente dalla distanza. Scala sbagliata del modello: profondità stimata/vera = scala (0.8…1.2), rotazione invariata.
- Il rumore reale di YOLO (≈ 9 px mediani) è oltre l'intervallo dove i flip restano < 5% con le sole strategie temporali: il margine vero sta nel migliorare i keypoint.
- Controllo di coerenza dell'export: su `14_03_09` con i keypoint dell'export (crop JPEG) (a) dà 8.33%, (b) 1.67%, (c) 0.00% (lo scheletro su frame interi: 7.2/1.7/0.3%): simile ma non identico, come previsto dal piano.

Da fare: (Task 6 passo 4, utente) video dello scheletro con `--strategy temporal_reject` su Kaggle; metrica di identità nella Fase 1; ripetere la valutazione sul test solo alla fine; più semi di YOLO.

## 6sexies. Difetto di `temporal_reject`: diagnosi e passaggio di sessione (2026-10-15)

Dopo i video di confronto (`scripts/compare_strategies.py`, `experiments/compare/`) è emerso un difetto di (c): dopo una raffica di scarti (> `max_gap` = 5 frame) il riferimento viene dimenticato, il primo frame accettato sceglie per riproiezione con rapporto ≈ 1.1 (lancio di moneta), e la continuità poi si auto-rinforza sulla soluzione sbagliata (esempio: `14_02_07_000`, frame 287–360 a 172–179°). Evidenze: `max_gap` ≥ 15 porta i flip senza il drone 08 da 0.86% a 0.03% (drone 02: 5.2% → 0.2%) a copertura invariata; un gate di gravità (inclinazione ≤ 70°) da solo dà 0.27% senza storia, perché il 94% delle candidate in flip è "a testa in giù"; il drone 08 resta a ~39% (confusione di identità, problema separato). **Tutto il contesto, le idee candidate (A–G), i criteri di successo e il prompt per la nuova sessione sono in `docs/superpowers/handoff/2026-10-15-fix-temporal-reject.md`.** Diagnostica riproducibile: `scripts/diagnose_disambiguation.py {trace|maxgap|gravity}`. **2026-10-15 (stessa sessione):** scelto l'approccio 2 (decisione globale, programmazione dinamica) come nuova strategia `global`, con gravità solo opzionale (`up` esplicito, spenta di default); design in corso in `docs/superpowers/specs/` (spec) e poi piano.

**Idea dell'utente da fare a fine di questo pezzo:** cartella `PerTesi/` con sottocartelle numerate **in ordine di processo** (`1_...`, `2_...`): ognuna con **copie** dei file e dei dati, e una spiegazione dell'obiettivo, di cosa si è trovato/scoperto/costruito; la successiva = passi seguenti o sistemazione della precedente. Serve solo ad agevolare la scrittura della tesi in un secondo momento. **È locale e fuori da git** (`/PerTesi/` è in `.gitignore`): duplica file già versionati altrove, quindi non va committata. Da progettare insieme (cosa copiare; niente frame di 6DroneSyn nel repo).

## 6septies. Strategia `global` (decisione globale): risultati (2026-10-15/16)

Nuova strategia (d) `global` in `src/dronepose/pose/global_disambiguation.py`: sceglie per ogni frame tra le due soluzioni IPPE (o "scartato") minimizzando sull'**intera sequenza** la somma di riproiezione (`reproj_px / 10`), salto angolare tra pose accettate normalizzato sul tempo vero (`min(z, z_cap)²`, `z = angolo / (5° · Δt)`) e costo di scarto (`reject_cost` per frame); programmazione dinamica, **offline**; gravità opzionale (`up` esplicito, spenta di default: i droni possono fare evoluzioni o accelerazioni forti che superano la soglia di 70°). Spec: `docs/superpowers/specs/2026-10-15-global-disambiguation-design.md`; piano: `docs/superpowers/plans/2026-10-15-global-disambiguation.md` (verificato in un worktree isolato prima dell'esecuzione). Risolve il difetto di (c): dopo una raffica di scarti non perde il riferimento e le pose successive "votano" sulle scelte ambigue precedenti.

**Tutta la val** (keypoint YOLO, setting A; `experiments/replay_val_global/`):

| Strategia | Flip | Copertura | Senza drone 08: flip / copertura |
|---|---|---|---|
| (a) `reproj` | 15.52% | 100% | 11.11% / 100% |
| (b) `temporal` | 11.27% | 100% | 6.63% / 100% |
| (c) `temporal_reject` | 6.16% | 95.2% | 0.84% / 95.6% |
| **(d) `global`** | **5.34%** | 95.3% | **0.00% / 95.9%** |

Oracolo (migliore delle due soluzioni) 6.81%. Per sequenza (`per_sequence.csv`): **0** sequenze dei droni 01–06 in cui `global` ha più di 2 punti di flip in più di (a) e (b); 2 sequenze del drone 08 (`03_08_04_180` +2.9, `03_08_05_180` +2.0), dove domina la confusione di identità (sul drone 08 `global` resta ~39%, nelle sequenze peggiori ~58%).

**Sequenze di riferimento** (flip % a / b / c / d): `09_03_01_000` 20.8 / 0.0 / 0.0 / 0.0 · `03_01_07_180` 20.8 / 38.3 / 0.0 / 0.0 · **`14_02_07_000` 14.2 / 1.4 / 20.1 / 0.0** (343 pose su 360) · `14_08_03_000` (drone 08) 59.7 / 56.1 / 58.0 / 57.7. Video e fotogrammi a quattro pannelli in `experiments/compare/` (script `scripts/compare_strategies.py`).

**Parametri** (`GlobalParams`): fissi `step_deg` 5° (p99 GT 4.5°), `max_reproj_px` 20, `reproj_scale` 10, `max_span` 60, `max_tilt_deg` 70 (solo con `up`); tarati `reject_cost` **12** e `z_cap` **6**. Griglia 3 × 4 con leave-one-scene-out sulle scene di val 03, 09, 14 (`scripts/tune_global.py`, `experiments/tune_global/`): scelta finale 12 / 6 (0.00% senza il 08, copertura 95.9%; **questo 0.00% è misurato sugli stessi dati usati per tarare**). Il numero onesto è l'aggregato *held-out* dei tre giri: con la griglia originale (`reject_cost` 3, 6, 12) **1.35%** di flip senza il 08 (140/10359, tutti sulla scena 14) a copertura 95.9%; con la griglia estesa (`reject_cost` 3, 6, 12, 24, 48) **0.19%** (20/10387) a copertura 96.2%. La taratura è **instabile tra i giri**: tenendo fuori la scena 14 la taratura su 03 e 09 sceglie `z_cap` 4 (unica combinazione sopra la soglia di copertura del 95%, 94.7–95.0% contro 94.3–94.7% delle altre), che sulla 14 dà 2.23% di flip; tenendo fuori la 03 la copertura scende a 93.6%. Plateau: `z_cap` ≥ 6 (con `z_cap` 2 il drone 01 arrivava al 23.9% di flip). Con sole tre scene di val è un limite da dichiarare.

**Sensibilità** (keypoint GT + rumore σ, 3 semi, val; `experiments/sensitivity/`), flip % a / b / c / d: σ 0.5: 1.1 / 0.5 / 2.0 / 0.0 · σ 2: 7.0 / 2.2 / 3.4 / 0.0 · σ 5: 17.3 / 10.6 / 9.0 / 0.0 · σ 7: 23.2 / 17.5 / 13.4 / 0.6 · σ 10: 30.1 / 29.1 / 19.0 / 2.1. A rumore basso `global` non peggiora rispetto a (b), a differenza di (c). Il rumore reale di YOLO (~9 px) è più alto dei σ in cui questo test è facile: la sensibilità è con keypoint GT, quindi non include la confusione di identità.

**Griglia estesa (revisione finale):** `reject_cost` fino a 48; l'altopiano a 0.00% senza il 08 è `reject_cost` 3–12 con `z_cap` ≥ 6; oltre 24 i flip tornano (0.12% a 24/6, 0.87% a 48/6) per +0.1 punti di copertura e +0.7 punti di flip con il 08. La regola "copertura più alta" spinge `reject_cost` verso il bordo della griglia (sceglierebbe 24/12), quindi si **mantengono i default 12 / 6**: interni all'altopiano, differenza di copertura trascurabile. `experiments/tune_global_ext/`.

**Limiti dichiarati:** memoria del riferimento **breve**: un salto di 180° su `dt` frame costa ~`(36/dt)²`, quindi dopo una raffica di scarti di ~20–30 frame le due catene tornano quasi equivalenti (misurato dal revisore: 14 frame sì, 30 no), molto prima di `max_span` (alternativa da valutare: scala `√dt`); offline (serve tutta la sequenza; versione a finestra scorrevole non fatta); drone 08 non risolto (identità dei keypoint: piano separato); con un buco più lungo le due catene tornano equivalenti e decide la riproiezione; copertura di `global` con keypoint GT + rumore: scende a ~78% a σ 10 (come (c), 80%); gravità non valutata (`--up`); taratura su tre scene di val, instabile tra i giri; test (scene 06, 07) mai toccato; parametri tarati su keypoint YOLO del setting A (crop dal box GT).

**Da fare dopo:** revisione finale del piano; cartella locale `PerTesi/` (vedi §6sexies); piano per il drone 08 (correzione di identità col tracking); valutare la gravità e la versione causale; test finale una sola volta alla fine.

## 6octies. Drone 08 (Tello): sonda sul problema di identità (2026-10-16)

Sonda locale (usa-e-getta, non nel repo; val, keypoint YOLO setting A, strategia `global`, 1800 frame del 08 contro il 01 di controllo). Il drone 08 ha il modello quasi quadrato (0.047 × 0.05) e piccolo (~156 px di estensione contro 240–390 degli altri).
- **Identità dei keypoint** (la permutazione migliore dei 4 punti è l'identità): 08 = 100% con il muso verso la camera (angolo < 30°), 63% tra 30° e 60°, **0% da 60° in poi**; gli altri droni ≥ 97.6% a qualunque angolo. Errore sistematico, costante nella sequenza (il campo `degree` imposta lo yaw di partenza 0/90/180/270): né il tracking né `global` possono correggerlo.
- **Il moto non dice dov'è il davanti:** angolo tra asse frontale e velocità, mediana 56° sul 08 (entro 45°: 45% dei frame); 95° nelle sequenze a `degree` 90. Il drone non vola di muso.
- **Con `global` sul 08:** flip 38.9%, ma **a meno della simmetria D4 del quadrato** (4 imbardate a multipli di 90° e i 4 ribaltamenti di 180°) **0.0% di flip**, rotazione mediana 7.8° (60% entro 15°); asse "su" mediana 3.8° (97.8% entro 30°); errore relativo di profondità mediano 3.65% (p90 11.1%) contro 2.08% del 01; pose 92%.
- **La gravità (`--up`) non cambia nulla** sul 08: la confusione è di imbardata, non di drone capovolto.

**Correttivo implementato (2026-10-16):** `eval/metrics.py` (`SYMMETRIC_DRONES = {08}`, `BODY_SYMMETRIES`, `symmetric_rotation_error_deg`); campo per frame `rot_err_sym_deg` in `assemble_results` (= errore grezzo per i droni non simmetrici) e chiavi `median_rot_err_sym_deg`, `p90_rot_err_sym_deg`, `flip_rate_sym` in `summarize`; colonna in `replay_disambiguation.py`. **Intera val, flip a meno della simmetria:** (a) 9.52%, (b) 5.68%, (c) 0.73%, **(d) global 0.00%** (flip grezzi: 15.52 / 11.27 / 6.16 / 5.34%). Non tocca il modello né le strategie.

**Conclusione:** per il 08 la posizione e la verticale sono utilizzabili; solo l'imbardata è ambigua a multipli di 90°. Il segnale per risolverla (la piccola camera frontale) è visibile solo nell'immagine, quindi serve un modello migliore, non una regola geometrica o temporale. **Implicazioni:** (i) in valutazione e in tesi riportare per i droni quasi simmetrici l'errore di rotazione **a meno della simmetria**, accanto a quello grezzo; (ii) in Fase 3, per il 08 usare ingressi indipendenti dall'imbardata (asse "su", roll/pitch, posizione e velocità) o l'imbardata ridotta modulo 90°; (iii) un riaddestramento mirato (risoluzione, peso sul 08, cue dedicato al davanti) è un esperimento della Fase 1, di esito incerto e da pianificare a parte.

## 6nonies. Stato e promemoria per la sessione successiva (2026-10-16)

**Fatto e committato** (ultimo commit al momento della scrittura: `feat: symmetry-aware rotation error`): scheletro end-to-end (piano 2026-10-12); strategie IPPE (a) riproiezione, (b) continuità, (c) continuità + rifiuto, replay, sensibilità, export keypoint (piano 2026-10-14); strategia (d) `global` con taratura e revisione finale (piano 2026-10-15); sonda sul drone 08 e metrica "a meno della simmetria" (`rot_err_sym_deg`). Revisioni finali fatte per tutti e tre i piani; Minor rinviati nel registro locale `.superpowers/sdd/2026-10-12-fase0-scheletro/progress.md`.

**Passi dell'utente ancora aperti** (non li può fare Claude):
1. Task 12 dello scheletro: prova di ripristino da un clone pulito (`git clone` di `https://github.com/JuicyTomato/DroneKey.git`, `pip install -e ".[dev]"`, `pytest`; poi `verify_manifest.py` con il disco) e copia manuale su OneDrive di `data/dronepose-cache.zip`, `data/manifest/`, `configs/drone_models.json`, `experiments/`.
2. Video dello scheletro con YOLO e la strategia scelta su Kaggle (README Kaggle §4 con `--strategy global`); prima rigenerare `data/dronepose-code.zip` con `scripts/package_cache.py` e ricaricarlo come New Version (lo zip su Kaggle è precedente a `global`).
3. Quota privata Kaggle fino a 200 GB: la decisione "≤ 20 GB" (2026-10-10) è **da confermare**.
4. `scripts/make_checkerboard.py` è dell'utente, non committato.

**2026-10-16, aggiornamento:** 3.1 e 3.2 fatti, vedi §6decies; prossimi passi della Fase 3 in fondo a §6decies.

~~**Prossima sessione: Fase 3 (predizione della traiettoria)**~~ (fatto 3.1 + 3.2, §6decies) — nel piano di lavoro è il ramo che "parte subito" (non dipende dai keypoint): T1 = 2.3 (fatto) + 3.2 baseline + 3.4 valutazione oracle (E4a) + 3.6 ablation. Si parte con il brainstorming sul **3.1** (dataset di traiettorie: 91 deduplicate, finestre N = 15 / H = 30, frame allineato alla gravità) e sul **3.2** (posizione costante, CV, CA, Kalman, baseline fisica), poi 3.1b (generatore con differential flatness) e 3.3 (GRU/LSTM). Promemoria §7bis: conversione in **metri veri** per classe solo in Fase 3 (Tello escluso finché manca una dimensione verificata). **Per il drone 08 usare ingressi indipendenti dall'imbardata** (asse "su", roll/pitch, posizione e velocità): la sua imbardata è ambigua a multipli di 90° (§6octies). In parallelo, quando c'è budget: Fase 1 (E1: YOLO-Pose contro DroneKey, 3 semi) su Kaggle (~30 h/settimana); il piano di lavoro dice che la Fase 0 completa (0.6, 0.7, 0.8, 0.10) e la Fase R (dati reali) non sono iniziate.

**Rinviato / da riprendere dopo** (decisione dell'utente: ora correzioni piccole, quelle approfondite in un secondo tempo):
- Drone 08 a fondo: riaddestramento mirato (risoluzione, peso sul 08, cue sul davanti), esperimento della Fase 1.
- `global`: scala `√dt` nel salto angolare (memoria del riferimento oggi ~20–30 frame), versione causale/a finestra scorrevole, valutazione della gravità (`--up`) su dati reali, guardia NaN (una riga: un `reproj_px` o una `R` NaN fa scartare tutti i frame), motivo "gravity" separato, test di ottimalità per forza bruta.
- Altri Minor: docstring di `compare_strategies.py` ("tre strategie"), copertura di `global` nella sintesi della sensibilità, definizione unica di "senza 08" (media dei droni 0.86% contro aggregato 0.84%), `check_no_test` in `run_skeleton.py`.
- Il test (scene 06, 07) non è mai stato toccato: si usa **una sola volta**, alla fine.

**Cartella `PerTesi/`** (locale, fuori da git): copie in ordine di processo per scrivere la tesi (vedi `PerTesi/README.md`).

**Regole di lavoro invariate:** commit solo dopo il consenso, uno per task, messaggi molto corti, nessun trailer Co-Authored-By, push solo se richiesto; esecuzione dei piani in modo Native con revisione finale a contesto pulito.

## 6decies. Fase 3: dataset di traiettorie (3.1) e baseline (3.2) — risultati sulla val (2026-10-16)

Spec: `docs/superpowers/specs/2026-10-16-fase3-dataset-baseline-design.md`; piano: `docs/superpowers/plans/2026-10-16-fase3-dataset-baseline.md` (10 task, Native). Codice **separato** in `src/dronepose/forecast/` (+ `scripts/forecast/`, `tests/forecast/`, `configs/forecast.json`); il codice esistente non è stato modificato. Dati: `data/processed/forecast/` (da `scripts/forecast/build_trajectories.py`, 1 s, solo dai JSONL canonici). Risultati: `experiments/forecast/tuning/`, `experiments/forecast/baselines_val/` (`table.md`, `table.json`, `per_window.csv`, `ade_fde_vs_horizon.png`).

**Fatti verificati:** camera in bolla esatta su tutti i 91 file (`y_cam = −0.582·z_world`, scarto 0 su 211.680 keypoint; test su disco `test_raw_camera_is_level`) → frame della gravità `G = C·camera` con `C = [[1,0,0],[0,0,1],[0,−1,0]]`; copie per sfondo identiche al bit → 56 / 21 / 14 traiettorie; finestre N 15 / H 30: 9296 / 3276 / 1064 (in metri 7968 / 2808 / 912); `s_c` = 01 0.7463 · 02 0.8053 · 03 0.8525 · 04 0.9334 · 05 0.9253 · 06 0.8521 (calcolati dal modello, 08 escluso); **la scena 03 non è moto uniforme** (ease-in/out), quindi il criterio "CV esatta sulla scena 03" è stato sostituito da un test su moto sintetico.

**Taratura sul train** (`tune_baselines.py`, 11 s): CV `k` = 2, CA `k` = 3 (i minimi possibili: il fit più corto vince su traiettorie GT senza rumore), Kalman CV `q` = 1e6 `r` = 1e-6 (**plateau** per `q ≥ 100`, differenze < 1e-5: l'avviso di bordo su `q` è innocuo; griglia spostata a 1e-1…1e6), Kalman CA `q` = 1e3 `r` = 1e-6. Gli avvisi di bordo coprono anche `r` (dalla revisione finale): `r` = 1e-6 è al bordo basso, ma **sul GT senza rumore si identifica solo il rapporto `q/r`** (con la griglia estesa a `r` = 1e-8 il CA sceglie `q` = 10, `r` = 1e-8, stesso rapporto 1e9, numeri sulla val identici; a `q` = 1e3, `r` = 1e-6 è un minimo interno): griglia lasciata come nella spec. In E4b, con pose rumorose, la `r` diventa identificabile: allargare la griglia verso l'alto (≥ 1e-1). Riferimento fisso della skill per tutta la Fase 3: **CV-2**.

**Val, gruppo in metri (18 traiettorie, 2808 finestre), FDE media a 0.5 / 1 s (m) e skill ADE a 1 s vs CV-2 (IC 95% bootstrap appaiato per traiettoria, stratificato per scena, 10 000 repliche):**

| Metodo | FDE 0.5 s | FDE 1 s | Skill ADE 1 s |
|---|---|---|---|
| posizione costante | 0.174 | 0.328 | −161% [−237, −114] |
| CV-15 (scheletro) | 0.095 | 0.243 | −62% [−65, −60] |
| CV-2 | 0.052 | 0.179 | 0 |
| CA-3 | 0.027 | 0.156 | +29.8% [+21.8, +39.9] |
| Kalman CV | 0.050 | 0.176 | +1.9% [+1.8, +1.9] |
| **Kalman CA** | 0.027 | **0.155** | **+30.5% [+22.6, +40.6]** |
| fisica (inclinazione) | 0.369 | 1.466 | −666% [−1062, −370] |

Skill **FDE** a 1 s (gruppo in metri): CA +12.6% [+2.2, +25.2], Kalman CA +13.2% [+2.9, +25.8]. Per tipo di moto, Kalman CA: lineare ADE +19.8% [+11.6, +31.3] ma **FDE −1.6% [−12.2, +12.6]** (nessun guadagno a 1 s; scene 03, 09); non lineare ADE +51.2% [+42.5, +59.8], FDE +40.1% [+29.6, +50.9] (solo la scena 14). Sulla scena 09 (lineare ma veloce) la FDE a 1 s è 0.661 (CV), 0.689 (CA), 0.684 (Kalman CA): CA leggermente peggio. Drone 08 (unità del dataset, riga a parte): FDE 1 s CV 0.110, Kalman CA 0.083, skill +39%. Orientamento (gruppo in metri, errore a 1 s): costante 30.7° geodetico / 8.5° asse su; velocità angolare costante 12.8° / 5.0°. Controllo di coerenza: FDE 1 s mediana di CV-15 su `09_06` = 1.8935 m (atteso 2.2223 × 0.8521 = 1.894).

**Lettura:** (1) sul GT senza rumore vincono i fit più corti e l'accelerazione: le traiettorie a keyframe sono lisce; CA/Kalman CA riducono l'**ADE** a 1 s di circa il 30% e l'**FDE** a 1 s di circa il 13% rispetto a CV, con il guadagno concentrato sul moto non lineare (sulle scene lineari l'FDE a 1 s non migliora); il Kalman CV a `q` grande si comporta come un filtro α-β e guadagna un +1.9% costante su CV-2 (ritardo della velocità minore); (2) la **baseline fisica è molto peggio di CV**, come previsto dal §3.11 (assetto scollegato dal moto: inclinazioni fino a 60° con accelerazione nulla); il suo valore va misurato sulle traiettorie generate (3.1b); (3) CV-15 dello scheletro è troppo lenta a reagire (−62%).

**Limiti:** val di 3 scene (non lineare = una sola scena); pose GT senza rumore (con pose stimate, E4b, i `k` e le `r` scelti cambieranno: si ritarano sulle pose rumorose del train); 08 in unità del dataset; nessun test toccato.

**Video delle previsioni** (`scripts/forecast/make_forecast_video.py`, serve il disco; vista camera con inset ingrandito + pannello dall'alto; futuro vero verde sotto le previsioni): `experiments/forecast/videos/` per `14_02_07` (non lineare: CA ≈ vero, FDE media 1 s CV 0.07 / CA 0.03 m), `09_06_10` (virata brusca a keyframe: tutte sbagliano, CV 1.37 / CA 1.60 m), `11_06_12` (train, inclinazione fino a 65°: fisica 4.27 m contro CA 0.08 m). Ogni video ha un fotogramma di esempio `*_frameNNNN.jpg`.

**Prossimi passi della Fase 3:** 3.1b generatore con differential flatness (dove la baseline fisica ha senso), 3.3 reti GRU/LSTM (riferimento CV-2), 3.6 ablation; 3.5 dopo l'export delle pose (2.6).

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
