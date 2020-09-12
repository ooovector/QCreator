from .core import DesignElement, DesignTerminal, LayerConfiguration
from .. import transmission_line_simulator as tlsim
import numpy as np
import gdspy
from typing import Tuple, Mapping, Any
from .drawing import combine
from .cpw_primitives import Trapezoid


class AirBridge:
    def __init__(self, width: float, length: float, padsize: float, min_spacing: float,
                 layer_configuration: LayerConfiguration):
        """
        Airbridge crossover element geometry settings, defined by technology
        :param width: crossover width
        :param length: crossover length
        :param padsize: crossover contact pad size
        :param min_spacing: minimal distance between neighboring crossovers
        :param layer_configuration: LayerConfiguration object
        """
        self.padsize = padsize
        self.width = width
        self.length = length
        self.layer_configuration = layer_configuration

    def render(self):
        # first the two contacts
        contact_1 = gdspy.Rectangle((- self.padsize / 2, self.length / 2 - self.padsize / 2),
                                    (  self.padsize / 2, self.length / 2 + self.padsize / 2))
        contact_2 = gdspy.Rectangle((- self.padsize / 2, - self.length / 2 - self.padsize / 2),
                                    (  self.padsize / 2, - self.length / 2 + self.padsize / 2))
        contacts = gdspy.boolean(contact_1, contact_2, 'or', layer=self.layer_configuration.airbridges_pad_layer)
        # add restricted area for holes
        restricted_area = gdspy.Rectangle(( -self.padsize / 2, - self.length / 2 - self.padsize / 2),
                                          (  self.padsize / 2, self.length / 2 + self.padsize / 2))
        # now the bridge itself
        bridge = gdspy.Rectangle((-self.width / 2, -self.length / 2),
                                 (self.width / 2,   self.length / 2),
                                 layer=self.layer_configuration.airbridges_layer)

        return {'airbridges_pad_layer': [contacts], 'airbridges_layer': [bridge], 'restrict': (restricted_area,)}


class CPWGroundAirBridge(DesignElement):
    def __init__(self, name: str, position: Tuple[float, float], orientation: float, geometry: AirBridge,
                 w: float, s: float, g: float, layer_configuration: LayerConfiguration,
                 l: float = 0, c: float = 0):
        """
        Airbridge crossover element
        :param name: Design element name
        :param position: position on chip
        :param orientation: contact pad orientation
        :param geometry: AirBridge object containing the geometry of the air bridge
        :param w: signal line width of CPW beneath air bridge
        :param s: s width of CPW beneath air bridge
        :param g: finite g of CPW beneath air bridge (unused)
        :param layer_configuration: LayerConfiguration object
        :param l: inductance of CPW line beneath
        :param c: capacitance of CPW line beneath
        """
        super().__init__('airbridge', name)

        self.position = np.asarray(position)
        self.orientation = orientation
        self.geometry = geometry
        self.w = w
        self.s = s
        self.g = g
        self.layer_configuration = layer_configuration
        self.l = l
        self.c = c
        self.tls_cache = []
        p = self.geometry.padsize / 2 * np.asarray([np.cos(self.orientation), np.sin(self.orientation)])

        self.terminals = {'port1': DesignTerminal(position=self.position - p, orientation=self.orientation, type='cpw',
                                               w=self.w, s=self.s, g=self.g, disconnected='short'),
                          'port2': DesignTerminal(position=self.position + p, orientation=self.orientation+np.pi, type='cpw',
                                                w=self.w, s=self.s, g=self.g, disconnected='short')}

    def render(self):
        airbridge = self.geometry.render()
        under_cpw = Trapezoid(w1=self.w, s1=self.s, g1=self.g, w2=self.w, s2=self.s, g2=self.g,
                              length=self.geometry.padsize, layer_configuration=self.layer_configuration).render()

        result = combine([(airbridge, (0, 0)), (under_cpw, (0, 0))])
        #result = {}

        for layer_name, polygons in result.items():
            result[layer_name].rotate(self.orientation, (0, 0))
            result[layer_name].translate(*self.position)

        return result

    def get_terminals(self) -> dict:
        return self.terminals

    def add_to_tls(self, tls_instance: tlsim.TLSystem, terminal_mapping: Mapping[str, int],
                   track_changes: bool = True) -> list:
        l = tlsim.Inductor(l=self.l)
        c1 = tlsim.Capacitor(c=self.c/2)
        c2 = tlsim.Capacitor(c=self.c / 2)

        elements = [l, c1, c2]
        if track_changes:
            self.tls_cache.append(elements)

        tls_instance.add_element(l,  [terminal_mapping['port1'], terminal_mapping['port2']])
        tls_instance.add_element(c1, [terminal_mapping['port1'], 0])
        tls_instance.add_element(c2, [terminal_mapping['port2'], 0])

        return elements