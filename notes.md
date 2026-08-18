## Notes — Aerial Detector

## Résultats baseline (YOLOv8n, 20 epochs, imgsz 640, 764 train / 192 val)

| Classe | mAP@0.5 | Precision | Recall | Instances |
|---|---|---|---|---|
| airplane | 0.893 | 0.811 | 0.845 | 289 |
| ship | 0.917 | 0.877 | 0.884 | 2243 |
| vehicle | 0.517 | 0.843 | **0.368** | 691 |
| **All** | **0.775** | 0.844 | 0.699 | 3223 |

## Modes d'échec — analyse consolidée

Chaque mode a été **chiffré** (métriques) et/ou **confirmé visuellement** (comparaison GT en vert vs Pred en rouge sur 10 images de val).

### 1. Petits objets — recall vehicle catastrophique
- **Chiffres** : recall vehicle 0.368 (mAP@0.5 = 0.517)
- **Visuel** : sur les images haute altitude, véhicules < 15 px entièrement ratés. Sur images altitude modérée, détection quasi complète.
- **Cause** : YOLOv8n a un stride final de 32 → un objet de 15 px n'active plus qu'un demi-pixel de feature map, insuffisant pour la tête de détection.
- **Solutions par coût croissant** :
  1. `imgsz 640 → 1024` (0 changement de code, ×2 mémoire GPU)
  2. Tiling en tuiles 640×640 avec overlap 20% + NMS globale de fusion
  3. Passer à YOLOv8s/v8m (~3× params, ~2× temps d'inférence)
  4. Ajouter la head P2 (feature map stride 4, +30% compute)

### 2. Faux positifs airplane sur marquages au sol
- **Visuel** : croix et T peints sur le tarmac déclenchent des détections airplane
- **Cause** : le modèle a appris **la signature de forme** (silhouette T/croix depuis le dessus) plus que le concept sémantique d'aircraft. Sans context awareness, un CNN se cale sur les patterns bas niveau qui discriminent le mieux ses classes d'entraînement.
- **Solutions** : (1) hard negative mining sur des tuiles de tarmac vide, (2) augmentation rotation qui casse la signature de croix, (3) modèle plus profond pour capter le context au-delà de la forme locale.

### 3. Faux positifs vehicle sur carrés compacts contrastés
- **Visuel** : détections vehicle sur des patches carrés (sombres ou clairs) — parkings marqués, panneaux au sol, containers, ombres compactes
- **Cause** : le modèle a appris **« carré compact contrasté = vehicle »**, feature bas niveau qui généralise mal à des objets d'appearance similaire mais sémantiquement différents.
- **Solutions** : hard negatives sur ces patches, ou modèle avec plus de contextes (backbone plus grand, ou attention).

### 4. Localization imprecise sur vehicles partiellement visibles
- **Visuel** : sur un camion, boîte prédite plus petite que le vrai objet
- **Cause** : la régression de bbox est peu supervisée sur les objets partiellement visibles (peu d'exemples de troncatures dans le train set). Le modèle prédit une boîte « safe » plus petite.
- **Solutions** : augmentation par cropping aléatoire pour simuler des troncatures, ou loss de régression plus stricte (CIoU).

### 5. Double détection NMS sur gros ships
- **Visuel** : deux boîtes légèrement décalées mais principalement superposées sur un même gros bateau
- **Cause** : seuil IoU-NMS default d'Ultralytics = 0.7. Deux prédictions décalées sur un objet large peuvent avoir un IoU < 0.7 et donc échapper à la fusion.
- **Impact biais métrique** : les double-détections **gonflent artificiellement le recall apparent** (2 boîtes matchant 1 GT = 1 TP + 1 FP). Le recall ship 0.884 est probablement sur-estimé.
- **Solutions** : (1) baisser `iou_threshold` à 0.5 en inference (risque : fusionne à tort deux objets réellement adjacents), (2) **soft-NMS** (décrémente le score au lieu de supprimer), (3) **DIoU-NMS** (prend en compte la distance des centres).

### Hypothèse initialement soupçonnée puis infirmée
- **FP ship en contexte port** : au vu du ratio 2243 instances / 66 images (34 ships par image), j'avais soupçonné un biais de contexte port. **La comparaison GT vs Pred m'a détrompée** : les détections en port correspondent majoritairement à de vrais bateaux.

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