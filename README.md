# Aerial Object Detector — YOLOv8n + MLflow

Fine-tuning YOLOv8n sur un sous-ensemble DIOR-R (3 classes : airplane, ship, vehicle) avec suivi d'expériences MLflow. Projet d'entraînement d'une après-midi.

## Résultats

| Classe | mAP@0.5 | Precision | Recall | Instances val |
|---|---|---|---|---|
| airplane | 0.893 | 0.811 | 0.845 | 289 |
| ship | 0.917 | 0.877 | 0.884 | 2243 |
| vehicle | 0.517 | 0.843 | 0.368 | 691 |
| **All** | **0.775** | 0.844 | 0.699 | 3223 |

*20 epochs, imgsz 640, batch 16, 764 train / 192 val images, YOLOv8n (~3M params, T4 GPU Colab.*

Exemple qualitatif (GT en vert, prédictions en rouge) :

![GT vs Pred sample](assets/gt_vs_pred_sample.png)

## Analyse d'erreurs

Cinq modes d'échec identifiés (détails complets dans [NOTES.md](NOTES.md)) :

1. **Petits objets (recall vehicle 0.37)** : véhicules < 15 px non détectés à cause du stride 32 de YOLOv8n. Confirmé visuellement sur images haute altitude.
2. **FP airplane sur marquages au sol** (croix, T) : signature de forme insuffisamment discriminante.
3. **FP vehicle sur carrés contrastés** (parkings, containers) : feature bas niveau « carré compact = vehicle ».
4. **Localization imprecise sur vehicles partiellement visibles** : régression bbox peu supervisée sur troncatures.
5. **Double détection NMS sur gros ships** : seuil IoU-NMS (0.7) inadapté aux objets larges → recall ship probablement sur-estimé.

Une hypothèse initiale (biais de contexte port sur ship) a été **infirmée** par l'analyse GT vs Pred — voir NOTES.md.

## Quick start

```bash
git clone <this-repo>
cd aerial-detector
uv sync

# Télécharger DIOR-R depuis Kaggle (~7 Go)
uv run kaggle datasets download -d redzapdos123/dior-r-dataset-yolov11-obb-format -p data --unzip

# Pipeline complet
uv run python -m aerial_detector.data       # prépare le subset 3 classes (~2 min)
uv run python -m aerial_detector.train      # training (GPU nécessaire, ou Colab)
uv run python -m aerial_detector.evaluate   # métriques par classe → outputs/
uv run python -m aerial_detector.visualize  # GT vs Pred → outputs/
```

Tests :
```bash
uv run pytest -v
uv run ruff check .
```

## Structure
aerial-detector/
├── src/aerial_detector/ # data.py, train.py, evaluate.py, visualize.py
├── tests/ # tests pytest sur les fonctions pures
├── configs/dataset.yaml # config dataset Ultralytics
├── scripts/ # sanity check + viz qualitative
├── outputs/ # métriques JSON + images annotées
├── NOTES.md # analyse détaillée + prépa entretien
└── pyproject.toml # deps + config ruff + pytest


## Choix techniques et trade-offs

| Choix | Justification | Trade-off |
|---|---|---|
| YOLOv8n (~3M params) | Entraînement rapide sur T4 gratuit, API mature | Sous-optimal sur petits objets vs modèles plus profonds |
| DIOR-R subset (3/20 classes) | Focus 3 classes distinctes, itération rapide | Généralisation à 20 classes non validée |
| OBB → HBB conversion | Compatibilité YOLOv8n standard | Info d'orientation perdue, boîtes légèrement plus grandes |
| MLflow local (filesystem) | Zéro config | Backend déprécié en MLflow 3.x (`MLFLOW_ALLOW_FILE_STORE=true`) |
| uv pour l'env | `uv sync` en < 2s, lockfile déterministe | Écosystème plus jeune que pip/poetry |
| Compute Colab T4 | Pas de GPU CUDA local (AMD Radeon) | Non-persistance des artefacts, uploads manuels |

## Contraintes rencontrées

- **Dataset Kaggle incomplet** : le split `train/labels/` du zip était vide. J'ai réutilisé le split `test/` (2347 images) refiltré à 956 images contenant airplane/ship/vehicle, re-splité 80/20.
- **Bug de chemin Ultralytics** : `project` relatif est préfixé par `runs_dir` global → double `runs/detect/`. Fix : chemins absolus.
- **AMD Radeon local** : pas de CUDA, entraînement délocalisé sur Colab.

## Ce que je ferais avec plus de temps

1. **Fix vehicle recall** : tiling + imgsz 1024 (impact estimé +0.15 mAP vehicle)
2. **Vraie stratification** du split train/val par classe (le random peut sous-représenter airplane en val)
3. **Data augmentation ciblée** : mosaic + copy-paste pour les petits objets
4. **Passer sur YOLOv8-OBB** pour récupérer l'info d'orientation (cohérent avec cas d'usage IMINT satellite)
5. **Soft-NMS ou DIoU-NMS** pour corriger les double-détections sur gros ships
6. **CI GitHub Actions** qui bloque tout PR avec mAP < seuil (garde-fou anti-régression)
7. **MLflow serveur distant** (Databricks, W&B, ou serveur MLflow self-hosted) au lieu du filesystem

## License

MIT — voir [LICENSE](LICENSE).