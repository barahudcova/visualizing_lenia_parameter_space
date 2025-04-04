import numpy as np
import torch
from textwrap import dedent
import pygame
from easydict import EasyDict


class Automaton:
    """
    Class that internalizes the rules and evolution of
    an Alife model. By default, the world tensor has shape
    (3,H,W) and should contain floats with values in [0.,1.].
    """

    def __init__(self, size):
        """
        Parameters :
        size : 2-uple (H,W)
            Shape of the CA world
        """
        self.h, self.w = size
        self.size = size

        self._worldmap = torch.zeros((3, self.h, self.w), dtype=float)  # (3,H,W), contains a 2D 'view' of the CA world

    def step(self):
        return NotImplementedError('Please subclass "Automaton" class, and define self.step')

    def draw(self):
        """
        This method should be overriden. It should update the self._worldmap tensor,
        drawing the current state of the CA world. self._worldmap is a torch tensor of shape (3,H,W).
        If you choose to use another format, you should override the worldmap property as well.
        """
        return NotImplementedError('Please subclass "Automaton" class, and define self.draw')

    @property
    def worldmap(self):
        """
        Converts self._worldmap to a numpy array, and returns it in a pygame-plottable format (W,H,3).

        Should be overriden only if you use another format for self._worldmap, instead of a torch (3,H,W) tensor.
        """
        return (255 * self._worldmap.permute(2, 1, 0)).detach().cpu().numpy().astype(dtype=np.uint8)

    @property
    def worldsurface(self):
        """
            Converts self.worldmap to a pygame surface.

            Can be overriden for more complex drawing operations, 
            such as blitting sprites.
        """
        return pygame.surfarray.make_surface(self.worldmap)