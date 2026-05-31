import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from enum import Enum
from dataclasses import dataclass
import math
from tqdm import tqdm

from tiles import *
from utils import *

class FactorioSolver:
    NEG_INF = -500.0
    
    def __init__(self,box_size,tile_data):
        self.BOX_SIZE = box_size
        self.TILE_DATA = tile_data
    
    def _log1m_sum_exp(self, log_prob, dim=-1):
        m = log_prob.max(dim=dim, keepdim=True).values
        log_sum_exp = m + torch.log(torch.exp(log_prob - m).sum(dim=dim, keepdim=True))
        p_sum = torch.exp(log_sum_exp)
        eps = torch.finfo(log_prob.dtype).eps
        p_sum = torch.clamp(p_sum, max=1 - eps)
        return torch.log1p(-p_sum).squeeze(dim)

    def eval(self, solve, iter_add, verbose=0,max_iter=67):
        items_old = torch.full((self.BOX_SIZE, self.BOX_SIZE, len(Tile)), self.NEG_INF)
        
        items_old = torch.logaddexp(items_old, torch.log(iter_add).unsqueeze(-1).repeat(1,1,len(Tile)))
        item_balance = iter_add.sum()

        for iter in range(round(max_iter)):
            log_total_mass = items_old.max(dim=-1)[0]
            log_total_mass = torch.clamp(log_total_mass, min=self.NEG_INF)

            log_pull = torch.full((self.BOX_SIZE, self.BOX_SIZE, len(Tile)), self.NEG_INF, device=solve.device)
            for i in range(len(Tile)):
                t_data = self.TILE_DATA[Tile(i)]

                if not t_data.can_pull():
                    continue

                inp_y = slice(max(t_data._pull_pos[1],0), self.BOX_SIZE-max(-t_data._pull_pos[1],0))
                inp_x = slice(max(-t_data._pull_pos[0],0), self.BOX_SIZE-max(t_data._pull_pos[0],0))
                out_y = slice(max(-t_data._pull_pos[1],0), self.BOX_SIZE-max(t_data._pull_pos[1],0))
                out_x = slice(max(t_data._pull_pos[0],0), self.BOX_SIZE-max(-t_data._pull_pos[0],0))

                log_pull[out_y, out_x, i] = solve[inp_y, inp_x, i] + np.log(t_data._pull_max_th)

            log_total_demand = torch.logsumexp(log_pull, dim=-1)
            
            log_norm_scale = -torch.clamp(log_total_demand - log_total_mass, min=0.0) # [Y,X] % of mass that will be pulled

            log_pulled_mass = log_pull + log_norm_scale.unsqueeze(-1) # [Y,X,T]

            log_fraction_pulled = log_pulled_mass - log_total_mass.unsqueeze(-1)
            log_left_fraction = self._log1m_sum_exp(log_fraction_pulled)

            mul_self = items_old + solve + log_left_fraction.unsqueeze(-1)

            items = torch.full((self.BOX_SIZE, self.BOX_SIZE, len(Tile)), self.NEG_INF)
            for i in range(len(Tile)):
                tile_type = Tile(i)
                t_data: TileData = self.TILE_DATA[tile_type]
                # if tile_type == Tile.EMPTY: continue
                if len(t_data._output) != 0:
                    idx = t_data.can_give_to
                    if not t_data.can_pull():
                        inp_y = slice(max(t_data._output[1],0), self.BOX_SIZE-max(-t_data._output[1],0))
                        inp_x = slice(max(-t_data._output[0],0), self.BOX_SIZE-max(t_data._output[0],0))
                        out_y = slice(max(-t_data._output[1],0), self.BOX_SIZE-max(t_data._output[1],0))
                        out_x = slice(max(t_data._output[0],0), self.BOX_SIZE-max(-t_data._output[0],0))

                        src = mul_self[inp_y,inp_x, i].unsqueeze(-1).repeat(1, 1, len(idx))
                    else:
                        dy = t_data._output[1]-t_data._pull_pos[1]
                        dx = t_data._output[0]-t_data._pull_pos[0]

                        inp_y = slice(max(dy,0), self.BOX_SIZE-max(-dy,0))
                        inp_x = slice(max(-dx,0), self.BOX_SIZE-max(dx,0))
                        out_y = slice(max(-dy,0), self.BOX_SIZE-max(dy,0))
                        out_x = slice(max(dx,0), self.BOX_SIZE-max(-dx,0))

                        src = (log_pulled_mass[inp_y, inp_x, i]).unsqueeze(-1).repeat(1, 1, len(idx))

                    items[out_y,out_x, idx] = torch.logaddexp(items[out_y,out_x, idx], src)

            if verbose>5:
                plt.matshow(items.detach().max(-1)[0].exp()/item_balance,vmin=0,vmax=1)
                plt.title('Items, normalized')
                plt.show()
            
            sum_items = items.max(-1)[0].exp().sum().item()
            assert sum_items < item_balance*1.05, f'Sum should be <= {item_balance}. ({sum_items})'
            
            items_old = items.clone()
            
        return items # - torch.log(item_balance) # Normalize.