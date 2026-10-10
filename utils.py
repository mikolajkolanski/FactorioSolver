import torch
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from PIL import Image
from pathlib import Path
import sys, inspect
import seaborn as sns

from tiles import *

def plot_solve(solve):
    fig, ax = plt.subplots(1,solve.size(-1),figsize=(6*5, 6))
    for i, grid in enumerate(solve.permute(2,0,1)):
        ax[i].matshow(grid.detach().exp().numpy(),vmin=0,vmax=1)
        ax[i].set
        ax[i].set_title(Tile(i))
    plt.show()


def plot_grads(logits, losses:dict):
    fig, ax = plt.subplots(1,5,figsize=(3.2*5, 3*1))

    grads = {}
    for name,loss in losses.items():
        grads[name] = torch.autograd.grad(loss, logits, 
                                            retain_graph=True, allow_unused=True)[0][...,1::].sum(-1).detach().cpu()
    vmin = np.min(list(grads.values()))
    vmax = np.max(list(grads.values()))

    
    for i,(name,grad) in enumerate(grads.items()):
        sns.heatmap(grad,ax=ax[i],vmin=vmin,vmax=vmax,cbar=False).set_title(f'Grad/{name}')
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


def plot_schedule(schedule):
    methods = inspect.getmembers(schedule, inspect.ismethod)
    methods = [(i,j) for i,j in methods if not i.startswith('_')]

    com_ax = plt.subplot()
    com_ax.set_title('Combined')

    fig, ax = plt.subplots(1, len(methods), figsize=(3.5*len(methods), 3))

    for i, (m_name, m_func) in enumerate(methods):
        g = list(map(m_func, np.arange(0,schedule.max_iter))) 
        ax[i].plot(g)
        ax[i].set_title(f'{m_name} max:{max(g):.3f} min:{min(g):.3f}')

        com_ax.plot(g)
    plt.plot()