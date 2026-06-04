import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from enum import Enum
from dataclasses import dataclass
import math
import unittest

from tiles import *
from utils import *
from solver import FactorioSim

class TestFactorioSim(unittest.TestCase):
    def test_inserter(self, debug=False):
        sim = FactorioSim(5, DEFAULT_TILE_DATA)
        logits = torch.full((5,5,len(Tile)), -5000.0)
        logits[:,:,Tile.EMPTY.value] = 0

        logits[2, 0, Tile.BELT_RIGHT.value] = 10000
        logits[2, 1, Tile.BELT_RIGHT.value] = 10000
        logits[2, 2, Tile.BELT_RIGHT.value] = 10000

        logits[1, 1, Tile.INSERTER_UP.value] = 10000
        logits[1, 2, Tile.INSERTER_UP.value] = 10000

        logits[3, 2, Tile.INSERTER_DOWN.value] = 10000

        
        logits[0, 1, Tile.BELT_RIGHT.value] = 10000
        logits[0, 2, Tile.BELT_RIGHT.value] = 10000
        logits[0, 3, Tile.BELT_RIGHT.value] = 10000

        logits[4, 2, Tile.BELT_RIGHT.value] = 10000

        add_tensor = torch.zeros(5,5)
        add_tensor[2,0] = 100

        solve = torch.log_softmax(logits, dim=-1)
        if debug:   
            plot_solve(solve)
            render_solve(solve)
        items = sim.eval(solve,add_tensor,verbose=10 if debug else 0,max_iter=4)
        if debug: 
            plt.matshow(items.detach().exp().numpy(),vmin=0,vmax=1)
            plt.show()
        assert abs(items[0,3].exp() -0.4) < 0.01
        assert abs(items[4,3].exp() -0.2) < 0.01
        print('Test passed!')
        # print(f'Output pulled: {items[0,4].max(-1)[0].exp()}, Output left: {items[14,4].max(-1)[0].exp()}')


    def test_belts_q(self, debug=False):
        sim = FactorioSim(3, DEFAULT_TILE_DATA)
        logits = torch.full((3,3,len(Tile)), -5000.0)
        logits[:,:,Tile.EMPTY.value] = 0

        logits[1, 0, Tile.BELT_RIGHT.value] = 10000
        logits[1, 1, Tile.BELT_DOWN.value] = 10000

        add_tensor = torch.zeros(3,3)
        add_tensor[1,0] = 1

        solve = torch.log_softmax(logits, dim=-1)
        if debug:   
            plot_solve(solve)
            render_solve(solve)
        items = sim.eval(solve,add_tensor,verbose=10 if debug else 0,max_iter=3)
        if debug: 
            plt.matshow(items.detach().exp().numpy(),vmin=0,vmax=1)
            plt.show()
        assert abs(items[1,1].exp() - 0) < 0.01
        assert abs(items[1,0].exp() - 1) < 0.01
        assert abs(items[2,1].exp() - 0) < 0.01
        print('Test passed!')


    def _test_belts_c(self, debug=False):
        # This test isnt passing because im not really sure what should be the output of this sim.

        sim = FactorioSim(3, DEFAULT_TILE_DATA)
        solve = torch.zeros((3,3,len(Tile)))

        solve[1, 0, Tile.BELT_RIGHT.value] = 1
        solve[1, 1, Tile.BELT_DOWN.value] = 0.7
        solve[1, 1, Tile.BELT_RIGHT.value] = 0.3

        solve[:,:,Tile.EMPTY.value] = 1 - solve[:,:,1::].sum(-1)
        add_tensor = torch.zeros(3,3)
        add_tensor[1,0] = 1

        solve_log = torch.log(solve+0.000001)
        if debug:   
            plot_solve(solve_log)
            render_solve(solve_log)
        items = sim.eval(solve_log,add_tensor,verbose=10 if debug else 0,max_iter=2)
        if debug: 
            plt.matshow(items.detach().exp().numpy(),vmin=0,vmax=1)
            plt.show()
        assert abs(items[1,2].exp() - 0.51) < 0.01
        assert abs(items[1,0].exp() - 0.49) < 0.01
        assert abs(items[2,1].exp() - 0) < 0.01
        print('Test passed!')

    def test_grad(self, debug=False):
        sim = FactorioSim(3, DEFAULT_TILE_DATA)
        logits = 5 * torch.randn(3,3,len(Tile))
        logits.requires_grad_(True)

        add_tensor = torch.zeros(3,3)
        add_tensor[1,0] = 67

        solve = torch.log_softmax(logits, dim=-1)

        items = sim.eval(solve,add_tensor,verbose=0,max_iter=50)

        loss = -( items[2, 2] - np.log(67) )

        loss.backward()

        print('Test passed')


if __name__=='__main__':
    unittest.main()
    