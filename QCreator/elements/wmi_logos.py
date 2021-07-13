from .core import DesignElement, LayerConfiguration
from .. import transmission_line_simulator as tlsim
import numpy as np
import gdspy
from typing import Tuple


class WMILogos(DesignElement):
    def __init__(self, position_logo_one: Tuple[float, float], position_logo_two: Tuple[float, float], path_mask ,layers_configuration: LayerConfiguration):
        """
        Element for creating a WMI logos on a chip
        :param position_logo_one:
        :param position_logo_two:
        :param layer_configuration:
        """
        super().__init__('wmi_logos','wmi_logos')
        self.position_logo_one = position_logo_one
        self.position_logo_two = position_logo_two
        self.path_mask = path_mask
        self.logo_mcqst = gdspy.GdsLibrary(infile = path_mask+"\logo_mcqst.gds")
        self.logo_wmi = gdspy.GdsLibrary(infile = path_mask+"\logo_wmi.gds")
        self.layers_configuration = layers_configuration

    def render(self):
        """
        Draws WMI logos
        :return:
        """
        mcqst_negative_cell=gdspy.GdsLibrary()
        mcqst_negative = mcqst_negative_cell.read_gds(infile=self.path_mask+"\logo_mcqst.gds").cells['MCQST']
        mcqst_negative.remove_polygons(lambda pts, layer, datatype: layer != self.layers_configuration['inverted'])

        mcqst_positive_cell = gdspy.GdsLibrary()
        mcqst_positive = mcqst_positive_cell.read_gds(infile=self.path_mask + "\logo_mcqst.gds").cells['MCQST']
        mcqst_positive.remove_polygons(lambda pts, layer, datatype: layer != self.layers_configuration['total'])

        ############### add wmi logos
        wmi_negative_cell = gdspy.GdsLibrary()
        wmi_negative = wmi_negative_cell.read_gds(infile=self.path_mask + "\logo_wmi.gds").cells['WMI']
        wmi_negative.remove_polygons(lambda pts, layer, datatype: layer != self.layers_configuration['inverted'])

        wmi_positive_cell = gdspy.GdsLibrary()
        wmi_positive = wmi_positive_cell.read_gds(infile=self.path_mask + "\logo_wmi.gds").cells['WMI']
        wmi_positive.remove_polygons(lambda pts, layer, datatype: layer != self.layers_configuration['total'])

        return {'positive': gdspy.boolean(mcqst_positive.get_polygons(),wmi_positive.get_polygons(),'or',layer=self.layers_configuration['total']),
                'inverted': gdspy.boolean(mcqst_negative.get_polygons(),wmi_negative.get_polygons(),'or',layer=self.layers_configuration['inverted'])}

    def get_terminals(self) -> dict:
        return {}

    def add_to_tls(self, tls_instance: tlsim.TLSystem, terminal_mapping: dict, track_changes: bool = True,
                   cutoff: float = np.inf, epsilon: float = 11.45) -> list:
        return []
