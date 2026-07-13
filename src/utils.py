import random

import numpy as np
import pandas as pd
import torch


def stratified_split(df: pd.DataFrame, stratify_key: pd.Series, val_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for value in np.unique(stratify_key.values):
        idx = np.where(stratify_key.values == value)[0]
        rng.shuffle(idx)
        n_val = int(round(len(idx) * val_frac))
        val_idx.extend(idx[:n_val])
        train_idx.extend(idx[n_val:])
    train_idx, val_idx = np.array(train_idx), np.array(val_idx)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
