from .core import DesignElement, DesignTerminal, LayerConfiguration
from .. import transmission_line_simulator as tlsim
import numpy as np
import gdspy
from typing import Tuple


class AirbridgeAlignmentMarks(DesignElement):
    def __init__(self, position: Tuple[float, float], window_size: float, square_size: float, gap: float,
                 layer_configuration: LayerConfiguration):
        """
        Element for creating a wide and solid ground electrode around the edges of the chip to wirebond for wirebonding
        to PCB.
        :param chip_geometry:
        :param layer_configuration:
        :param pads:
        """
        super().__init__('chip_edge_ground', 'chip_edge_ground')
        self.position = position
        self.window_size = window_size
        self.square_size = square_size
        self.gap = gap
        self.layer_configuration = layer_configuration

    def render(self):
        """
        Draws edge g metallization on chip
        :return:
        """

        contact_window = gdspy.Rectangle(np.asarray(self.position) - self.window_size/2,
                                         np.asarray(self.position) + self.window_size/2,
                                         layer=self.layer_configuration.airbridges_pad_layer)

        squares = None
        for sqc in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            sqc = gdspy.Rectangle(np.asarray(self.position) + np.asarray(sqc) * (self.gap/2 + self.square_size),
                                  np.asarray(self.position) + np.asarray(sqc) * self.gap/2)
            squares = gdspy.boolean(squares, sqc, 'or', layer=self.layer_configuration.total_layer)

        return {'positive': squares, 'restrict': contact_window, 'airbridges_pads': contact_window}

    def get_terminals(self) -> dict:
        return {}

    def add_to_tls(self, tls_instance: tlsim.TLSystem, terminal_mapping: dict, track_changes: bool = True,
                   cutoff: float = np.inf, epsilon: float = 11.45) -> list:
        return []
