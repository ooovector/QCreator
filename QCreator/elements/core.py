from .. import transmission_line_simulator as tlsim
from abc import *
from typing import Mapping, Iterable, Union
import gdspy
import numpy as np


class DesignTerminal:
    """
    A Terminal is an output/input of a DesignElement. Most terminals are CPW-type, as they can be connected to
    CPW-compatible transmission lines. The properties of the terminal determine what type of CPW can be connected to it.
    """
    def __init__(self, position: Iterable[float], orientation: float, type: str, w: Union[float, Iterable[float]],
                 s: Union[float, Iterable[float]], g: float, disconnected: str = 'short', order: bool = True):
        """
        Create terminal with specific connection type (e.g. cpw) and cpw geometry
        :param position: CPW end position
        :param orientation: CPW end orientation in radians
        :param type: only 'cpw' is currently supported (someday maybe also ``wire'' and ``fcb'' or ``tsv'')
        :param w: central conductor width of cpw in microns
        :param s: s width of cpw in width
        :param g: finite g width of cpw in microns
        :param disconnected: terimnal behaviour in transmission line model if disconnected, either 'short' or 'open'
        :param order: True if wires in straight order, False if reversed
        """
        self.position = position
        self.orientation = orientation
        self.type = type
        self.w = w
        self.s = s
        self.g = g
        self.disconnected = disconnected
        self.order = order

    def __repr__(self):
        return '''Type: {}, Position: ({}, {}), Orientation: {}
w: {}, s: {}, g: {}
reversed: {}'''.format(self.type, np.round(self.position[0], 3), np.round(self.position[1], 3),
                       np.round(self.orientation, 3), self.w, self.s, self.g, self.order)

class DesignElement:
    """
    Abstract class for design elements that defines the interface to draw it on a gds and its tlsim model
    """
    def __init__(self, type: str, name: str):
        self.type = type
        self.name = name
        self.resource = None
        self.modifiers = []

    def get(self):
        if self.resource is None:
            self.resource = self.render()
        return self.resource

    @abstractmethod
    def render(self) -> Mapping[str, Mapping[str, Union[Iterable, gdspy.PolygonSet]]]:
        """
        Draw the element on the design gds. Pure with respect to self: when implementing in subclasses, make sure not to
        change properties of self.
        """
        pass

    @abstractmethod
    def get_terminals(self) -> Mapping[str, DesignTerminal]:
        """
        Returns a list of terminals for the transmission line model of the system. Pure wrt to self
        :return:
        """
        pass

    @abstractmethod
    def add_to_tls(self, tls_instance: tlsim.TLSystem, terminal_mapping: Mapping[str, int], track_changes: bool = True,
                   cutoff: float = np.inf, epsilon: float = 11.45) -> list:
        """
        Adds the circuit to a transmission line system model
        :param cutoff:
        :param tls_instance: transmission_line_system class instance to add the model elements to
        :param terminal_mapping: dict that maps terminal names of this element to transmission line system node ids
        :param track_changes: add element to tracked so that its z0 gets automatically changed
        :return: list of circuit elements
        """
        pass


class LayerConfiguration:
    """
    Layer configuration for superconducting qubits technology. TODO: inherited by technology-specific classes?
    """
    def __init__(self, **layer_configurations):
        self.layer_configurations = layer_configurations
        self.total_layer = layer_configurations['total']
        self.restricted_area_layer = layer_configurations['restricted area']
        self.layer_to_remove = layer_configurations['for removing']
        self.jj_layer = layer_configurations['JJs']
        self.airbridges_layer = layer_configurations['air bridges']
        self.airbridges_pad_layer = layer_configurations['air bridge pads']
        if 'air bridge sm pads' in layer_configurations:
            self.airbridges_sm_pad_layer = layer_configurations['air bridge sm pads']
        if 'air bridge dielectric' in layer_configurations:
            self.airbridges_dielectric_layer = layer_configurations['air bridge dielectric']
        self.gridline_x_layer = layer_configurations['vertical gridlines']
        self.gridline_y_layer = layer_configurations['horizontal gridlines']
        self.inverted = layer_configurations['inverted']
        self.bandages_layer = layer_configurations['bandages']

class ChipGeometry:
    """
    Chip geometry class. TODO: Inherited by sample holder specific classes
    """
    def __init__(self, **chip_geometry):
        if 'sample_vertical_size' in chip_geometry:
            self.sample_vertical_size = chip_geometry['sample_vertical_size']
        if 'sample_horizontal_size' in chip_geometry:
            self.sample_horizontal_size = chip_geometry['sample_horizontal_size']
