# Understanding Clouds

Petit projet pour la compétition Kaggle "Understanding Clouds from Satellite
Images" : segmenter 4 types d'organisation nuageuse (Fish, Flower, Gravel,
Sugar) sur des images satellites.

Le modèle est un Swin-UNet : un encodeur Swin Transformer pré-entraîné
(Hugging Face) branché sur un décodeur U-Net classique.

## Installation

```
pip install torch transformers albumentations opencv-python pandas numpy matplotlib tqdm
```

## Données

Mettre `train.csv` et `train_images/` dans `./data`. 

## Lancer

```
python main.py
```

Les checkpoints, les logs et un exemple de prédiction atterrissent dans
`swin_unet_clouds/`.

## Code

Tout est dans `src/` :

- `config.py` : chemins, hyperparamètres
- `dataset.py`, `rle.py`, `transforms.py` : chargement des images et des masques
- `model/swin_unet.py` : l'architecture
- `losses.py`, `metrics.py` : loss (CE + Dice) et mIoU
- `engine.py` : boucles d'entraînement / validation
- `train.py` : le script principal

`main.py` ne fait qu'appeler `src/train.py`.
