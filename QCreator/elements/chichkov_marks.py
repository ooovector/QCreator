from .core import DesignElement, LayerConfiguration
from .. import transmission_line_simulator as tlsim
import numpy as np
import gdspy
from typing import Tuple, List
import os


class ChichkovMarks(DesignElement):
    def __init__(self, cross_position: Tuple, layers_configuration: LayerConfiguration, cross_type=None):
        super().__init__('chichkov_crosses', 'chichkov_crosses')
        self.cross_position = cross_position
        self.cross_type = cross_type
        path = os.getcwd()
        self.path_mask = path[:path.rindex('QCreator')] + 'QCreator\QCreator\elements\masks'
        self.layers_configuration = layers_configuration

    def render(self):
        filename = "\/navigation_cross_misis.gds"
        cell_name = 'Navigation_cross'

        cross_positive = gdspy.GdsLibrary().read_gds(infile=self.path_mask +
                                                            filename).cells[cell_name].remove_polygons(
            lambda pts, layer,
                   datatype: layer != self.layers_configuration['total'])

        cross_inverted = gdspy.GdsLibrary().read_gds(infile=self.path_mask +
                                                            filename).cells[cell_name].remove_polygons(
            lambda pts, layer,
                   datatype: layer != self.layers_configuration['inverted'])

        for elements_layer in [cross_inverted.polygons, cross_positive.polygons]:
            for element in elements_layer:
                element.translate(self.cross_position[0], self.cross_position[1])

        restricted_object = gdspy.boolean(cross_inverted.get_polygons(), cross_positive.get_polygons(), 'or')

        return {'positive': cross_positive,
                'restrict': restricted_object
                }
    def get_terminals(self) -> dict:
        return {}

    def add_to_tls(self, tls_instance: tlsim.TLSystem, terminal_mapping: dict, track_changes: bool = True,
                   cutoff: float = np.inf, epsilon: float = 11.45) -> list:
        return []