## Notes — Aerial Detector

## Résultats baseline (YOLOv8n, 20 epochs, imgsz 640, 764 train / 192 val)

| Classe | mAP@0.5 | Precision | Recall | Instances |
|---|---|---|---|---|
| airplane | 0.893 | 0.811 | 0.845 | 289 |
| ship | 0.917 | 0.877 | 0.884 | 2243 |
| vehicle | 0.517 | 0.843 | **0.368** | 691 |
| **All** | **0.775** | 0.844 | 0.699 | 3223 |

## Mode d'échec principal : petits objets (vehicle)

**Diagnostic chiffré** : recall vehicle 0.37 → 63% des véhicules ratés.

**Validation visuelle** (10 images qualitatives) : 2 images entièrement ratées, toutes deux prises à haute altitude où véhicules < 15 px. Sur images à altitude modérée, détection quasi complète.

**Cause mécanique** : YOLOv8n stride 32 en sortie → un objet de 15 px n'active qu'un demi-pixel de feature map, insuffisant pour la tête de détection.

## Modes d'échec secondaires

- **Faux positifs airplane sur marquages au sol** (croix / T) : le modèle a appris la signature de forme "T depuis le dessus" plus que le concept aircraft. Solution : hard negatives + augmentation rotation.
- **Faux positifs ship en contexte port** : ratio 2243 ships / 66 images = biais de contexte. Le modèle sur-prédit ship dès qu'une forme allongée apparaît dans un port. Solution : rebalancer le sampling par image, pas par instance.

**Solutions par coût croissant** :
1. Augmenter `imgsz 640 → 1024` (aucun changement de code, doubler la mémoire GPU)
2. Tiling en tuiles 640×640 avec overlap 20% (script de découpe + fusion NMS globale)
3. Passer à YOLOv8s ou v8m (~3× plus de paramètres, ~2× plus lent)
4. Ajouter la head P2 de YOLOv8 (feature map stride 4, dédiée petits objets, +30% compute)

## Trade-offs

- **DIOR-R au lieu de DIOR standard** : format d'annotation OBB (rotated boxes) converti en HBB axis-aligned. Info d'orientation perdue, boîtes légèrement plus grandes que les vraies sur objets tournés. Avec plus de temps : YOLOv8-OBB natif.
- **Dataset Kaggle incomplet** : le split `train/labels/` du zip Kaggle était vide. J'ai réutilisé le split `test/` (2347 images), refiltré à 956 images contenant airplane/ship/vehicle, re-splitté 80/20.
- **MLflow backend filesystem déprécié** : fix avec `MLFLOW_ALLOW_FILE_STORE=true`. En vrai projet : SQLite ou serveur MLflow distant.
- **Compute** : Colab T4 (AMD Radeon local sans CUDA). Workflow code local + train remote, aligné pratique industrie.

## Ce que je ferais avec plus de temps

1. Corriger le vehicle recall via tiling + imgsz 1024 (impact estimé : +0.15 mAP vehicle)
2. Vraie stratification du split par classe (le split random peut sous-représenter airplane en val)
3. Data augmentation ciblée sur les petits objets (mosaic + copy-paste)
4. Passer sur YOLOv8-OBB pour récupérer l'info d'orientation, cohérent avec les cas d'usage IMINT
5. Ajouter un CI GitHub Actions qui bloque tout PR avec un mAP < seuil