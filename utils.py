import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from enum import Enum
from dataclasses import dataclass
import math
from tqdm import tqdm
import os
from PIL import Image
from pathlib import Path

from tiles import *

def plot_solve(solve):
    fig, ax = plt.subplots(1,solve.size(-1),figsize=(6*5, 6))
    for i, grid in enumerate(solve.permute(2,0,1)):
        ax[i].matshow(grid.detach().exp().numpy(),vmin=0,vmax=1)
        ax[i].set
        ax[i].set_title(Tile(i))
    plt.show()


def render_solve(solve: torch.Tensor):
    y_size = solve.size(0)
    x_size = solve.size(1)
    assets = Path('assets').iterdir()
    assets = {k.name[0:-4]:Image.open(k) for k in assets}

    fig, axes = plt.subplots(y_size, x_size, figsize=(8, 8),gridspec_kw={"wspace": 0, "hspace": 0})

    for ax, v in zip(axes.ravel(), solve.reshape(y_size*x_size,-1)):
        tile = Tile(v.argmax().item())
        # print(tile)
        img = assets.get(tile.name.lower())
        if img:
            ax.imshow(img)
        ax.axis("off")
    plt.show()

    for i in assets.values():
        i.close()