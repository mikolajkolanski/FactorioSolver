import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from enum import Enum
from dataclasses import dataclass
import math

class Tile(Enum):
    EMPTY = 0

    BELT_UP = 1
    BELT_RIGHT = 2
    BELT_DOWN = 3
    BELT_LEFT = 4

    BELT_RIGHT_UP = 5
    BELT_DOWN_RIGHT = 6
    BELT_LEFT_DOWN = 7
    BELT_UP_LEFT = 8 

    BELT_UP_RIGHT = 9
    BELT_RIGHT_DOWN = 10
    BELT_DOWN_LEFT = 11
    BELT_LEFT_UP = 12

    INSERTER_UP = 13
    INSERTER_RIGHT = 14
    INSERTER_DOWN = 15
    INSERTER_LEFT = 16

    @staticmethod
    def INSERTERS():
        return [Tile.INSERTER_UP,
                Tile.INSERTER_RIGHT,
                Tile.INSERTER_DOWN,
                Tile.INSERTER_LEFT]
    
    @staticmethod
    def ALL():
        return [i for i in Tile]

    @staticmethod
    def BELTS():
        return [Tile(i) for i in range(1,12+1)]
    
class Direction(Enum):
    DIR_NONE = 0
    UP = 1
    RIGHT = 2
    DOWN = 3
    LEFT = 4
    

DIR_NONE = Direction.DIR_NONE
UP = Direction.UP
RIGHT = Direction.RIGHT
DOWN = Direction.DOWN
LEFT = Direction.LEFT

RelPos = tuple[int,int] # x (right+), y(up+) 

class TileData:
    def __init__(self,output:RelPos,can_give_to:list[Tile]):
        self._output = output
        self._can_give_to = can_give_to

        self._pull_pos = None
        self._pull_max_th = 1

    def pull(self,pos:RelPos,max_th=1):
        self._pull_pos = pos
        self._pull_max_th = max_th

        return self
    
    def can_pull(self):
        return self._pull_pos is not None
    
    @property
    def can_give_to(self):
        return [i.value for i in self._can_give_to]
    
DEFAULT_TILE_DATA: dict[Tile, TileData] = {
    Tile.EMPTY:
        TileData((0, 0), Tile.ALL()),

    Tile.BELT_UP:
        TileData((0, 1), [
            Tile.BELT_UP,
            Tile.BELT_DOWN_LEFT,
            Tile.BELT_DOWN_RIGHT,
            Tile.EMPTY,
        ] + Tile.INSERTERS()),

    Tile.BELT_RIGHT:
        TileData((1, 0), [
            Tile.BELT_RIGHT,
            Tile.BELT_LEFT_UP,
            Tile.BELT_LEFT_DOWN,
            Tile.EMPTY,
        ] + Tile.INSERTERS()),

    Tile.BELT_DOWN:
        TileData((0, -1), [
            Tile.BELT_DOWN,
            Tile.BELT_UP_LEFT,
            Tile.BELT_UP_RIGHT,
            Tile.EMPTY,
        ] + Tile.INSERTERS()),

    Tile.BELT_LEFT:
        TileData((-1, 0), [
            Tile.BELT_LEFT,
            Tile.BELT_RIGHT_UP,
            Tile.BELT_RIGHT_DOWN,
            Tile.EMPTY,
        ] + Tile.INSERTERS()),

    Tile.BELT_RIGHT_UP:
        TileData((0, 1), [
            Tile.BELT_UP,
            Tile.BELT_DOWN_LEFT,
            Tile.BELT_DOWN_RIGHT,
            Tile.EMPTY,
        ] + Tile.INSERTERS()),

    Tile.BELT_DOWN_RIGHT:
        TileData((1, 0), [
            Tile.BELT_RIGHT,
            Tile.BELT_LEFT_UP,
            Tile.BELT_LEFT_DOWN,
            Tile.EMPTY,
        ] + Tile.INSERTERS()),

    Tile.BELT_LEFT_DOWN:
        TileData((0, -1), [
            Tile.BELT_DOWN,
            Tile.BELT_UP_LEFT,
            Tile.BELT_UP_RIGHT,
            Tile.EMPTY,
        ] + Tile.INSERTERS()),

    Tile.BELT_UP_LEFT:
        TileData((-1, 0), [
            Tile.BELT_LEFT,
            Tile.BELT_RIGHT_UP,
            Tile.BELT_RIGHT_DOWN,
            Tile.EMPTY,
        ] + Tile.INSERTERS()),

    Tile.BELT_UP_RIGHT:
        TileData((1, 0), [
            Tile.BELT_RIGHT,
            Tile.BELT_LEFT_UP,
            Tile.BELT_LEFT_DOWN,
            Tile.EMPTY,
        ] + Tile.INSERTERS()),

    Tile.BELT_RIGHT_DOWN:
        TileData((0, -1), [
            Tile.BELT_DOWN,
            Tile.BELT_UP_LEFT,
            Tile.BELT_UP_RIGHT,
            Tile.EMPTY,
        ] + Tile.INSERTERS()),

    Tile.BELT_DOWN_LEFT:
        TileData((-1, 0), [
            Tile.BELT_LEFT,
            Tile.BELT_RIGHT_UP,
            Tile.BELT_RIGHT_DOWN,
            Tile.EMPTY,
        ] + Tile.INSERTERS()),

    Tile.BELT_LEFT_UP:
        TileData((0, 1), [
            Tile.BELT_UP,
            Tile.BELT_DOWN_LEFT,
            Tile.BELT_DOWN_RIGHT,
            Tile.EMPTY,
        ] + Tile.INSERTERS()),

    Tile.INSERTER_UP:
        TileData((0, 1), Tile.ALL()).pull((0, -1), 0.2),

    Tile.INSERTER_RIGHT:
        TileData((1, 0), Tile.ALL()).pull((-1, 0), 0.2),

    Tile.INSERTER_DOWN:
        TileData((0, -1), Tile.ALL()).pull((0, 1), 0.2),

    Tile.INSERTER_LEFT:
        TileData((-1, 0), Tile.ALL()).pull((1, 0), 0.2),
}