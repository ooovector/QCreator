from .core import DesignElement, DesignTerminal, LayerConfiguration
from .. import transmission_line_simulator as tlsim
import numpy as np
import gdspy
from typing import Tuple


class AlignmentMarkers(DesignElement):
    def __init__(self, edge_distance: Tuple[float, float],chip_dimensions: Tuple[float, float], size: float,
                 layer_configuration: LayerConfiguration):
        """
        Element for creating four squares as markers for alignment purposes durin chip fabrication
        chip_dimensions : (x,y) size of qubit chip,
        edge_distance : (dx,dy) distance from chip edge,
        size : size of the square
        """
        super().__init__('chip_edge_ground', 'chip_edge_ground')
        self.position = edge_distance
        self.chip_dimensions = chip_dimensions
        self.size = size
        self.layer_configuration = layer_configuration

    def render(self):
        pos = self.position
        chip_d = self.chip_dimensions
        size = np.asarray((self.size/2,self.size/2))
        size_restrcited = np.asarray((50,50))
        markers = None
        Restricted = None
        for edge in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            sq = gdspy.Rectangle(-np.asarray(pos)*edge+np.asarray(chip_d)*edge/2-size+np.asarray(chip_d)/2,-np.asarray(pos)*edge+np.asarray(chip_d)*edge/2+size+np.asarray(chip_d)/2)
            R  = gdspy.Rectangle(-np.asarray(pos)*edge+np.asarray(chip_d)*edge/2-size_restrcited+np.asarray(chip_d)/2,-np.asarray(pos)*edge+np.asarray(chip_d)*edge/2+size_restrcited+np.asarray(chip_d)/2)
            markers = gdspy.boolean(markers, sq, 'or', layer=self.layer_configuration.inverted)
            Restricted = gdspy.boolean(Restricted, R, 'or', layer=self.layer_configuration.restricted_area_layer)

        return {'positive': markers, 'restrict': Restricted}

    def get_terminals(self) -> dict:
        return {}

    def add_to_tls(self, tls_instance: tlsim.TLSystem, terminal_mapping: dict, track_changes: bool = True,
                   cutoff: float = np.inf, epsilon: float = 11.45) -> list:
        return []











