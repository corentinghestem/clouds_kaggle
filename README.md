# Understanding Clouds

Petit projet pour la compétition Kaggle "Understanding Clouds from Satellite
Images" : segmenter 4 types d'organisation nuageuse (Fish, Flower, Gravel,
Sugar) sur des images satellites.

Le modèle est un Swin-UNet : un encodeur Swin Transformer pré-entraîné
(Hugging Face) branché sur un décodeur U-Net classique.

## Installation

```
pip install torch transformers albumentations opencv-python pandas numpy matplotlib tqdm mlflow
```

## Données

Mettre `train.csv` et `train_images/` dans `./data`. 

## Lancer

```
python main.py
```

Les checkpoints, les logs, un exemple de prédiction et `submission.csv`
atterrissent dans `swin_unet_clouds/`.

## Suivi MLflow

Chaque run logue ses hyperparamètres et, à chaque époque, `train_loss`,
`val_loss`, `mIoU` et l'IoU par classe. Le tracking écrit en local dans
`swin_unet_clouds/mlruns/` (pas de serveur à lancer).

Pour consulter les runs :

```
mlflow ui --backend-store-uri swin_unet_clouds/mlruns
```

puis ouvrir http://localhost:5000. Sur Kaggle, télécharge le dossier
`mlruns/` depuis les outputs du notebook et lance la commande ci-dessus en
local.

Si l'entraînement reprend depuis un checkpoint (`RESUME = True`), il
continue de logger dans le même run MLflow plutôt que d'en recréer un.

## Code

Tout est dans `src/` :

- `config.py` : chemins, hyperparamètres
- `dataset.py`, `rle.py`, `transforms.py` : chargement des images et des masques
- `model/swin_unet.py` : l'architecture
- `losses.py`, `metrics.py` : loss (CE + Dice) et mIoU
- `engine.py` : boucles d'entraînement / validation
- `train.py` : le script principal, logue tout dans MLflow
- `predict.py` : génère `submission.csv` sur le test set à partir du meilleur modèle

`main.py` ne fait qu'appeler `src/train.py`. `predict.py` (à la racine)
permet de regénérer une soumission sans réentraîner.
