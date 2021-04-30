from .core import DesignElement, DesignTerminal, LayerConfiguration
from .. import transmission_line_simulator as tlsim
import numpy as np
import gdspy
from typing import Tuple, Mapping, Any
from .drawing import combine
from .cpw_primitives import Trapezoid
from .. import conformal_mapping as cm
from scipy.constants import epsilon_0


class AirBridgeGeometry:
    def __init__(self, pad_width: float, pad_length: float,
                 bridge_width: float, bridge_length: float, pad_distance: float,
                 layer_configuration: LayerConfiguration):
        """
        Airbridge crossover element geometry settings, defined by technology
        :param width: crossover width
        :param length: crossover length
        :param padsize: crossover contact pad size
        :param min_spacing: minimal distance between neighboring crossovers
        :param layer_configuration: LayerConfiguration object
        """
        self.pad_width = pad_width
        self.pad_length = pad_length
        self.bridge_width = bridge_width
        self.bridge_length = bridge_length
        self.pad_distance = pad_distance
        self.layer_configuration = layer_configuration


class AirbridgeOverCPW(DesignElement):
    def __init__(self, name: str, position: Tuple[float, float], orientation: float,
                 w: float, s: float, g: float, geometry: AirBridgeGeometry):
        """
        Airbridge crossover element over CPW. Some parameters of this element depend on CPW structure.
        Here y axe is parallel to CPW.
        :param name: element name
        :param position: position of element (center position)
        :param orientation: orientation of element
        :param w: CPW signal conductor under bridge
        :param s:
        :param g:
        :param distance_between_pads: distance between two pads of airbridge
        :param pads_geometry: rectangle parameters of airbridge's pads width (first parameter) and length (second parameter)
        :bridge_geometry: rectangle parameters of airbridge
        :layer_configuration

        """
        super().__init__('airbridge', name)

        self.position = np.asarray(position)
        self.orientation = orientation
        self.w = w
        self.s = s
        self.g = g

        self.geometry = geometry

        h = 2 * 1e-6  # bridge height 2 mu m # TODO: CONSTANTS IN CODE OMG REMOVE THIS
        s = (self.geometry.bridge_width * 1e-6) * (self.geometry.bridge_length * 1e-6) # TODO: CONSTANTS IN CODE OMG REMOVE THIS

        epsilon = 1 # TODO: CONSTANTS IN CODE OMG REMOVE THIS

        self.bridge_capacitance = epsilon_0 * epsilon * s / h

        self.tls_cache = []

        cpw_width = 2 * self.s + self.w + 2 * self.g

        self.p = self.geometry.pad_width / 2 * np.asarray([np.cos(self.orientation), np.sin(self.orientation)])

        if self.geometry.pad_distance > cpw_width:
            raise ValueError('Distance between pads is larger that CPW width!')
        if self.geometry.pad_distance < 2 * self.s + self.w:
            raise ValueError('Distance between pads is too small!')

        if self.geometry.bridge_width > self.geometry.pad_width:
            raise ValueError('Airbridge width is larger than width of airbridge pad!')
        if self.geometry.bridge_length < self.geometry.pad_distance:
            raise ValueError('Airbridge length is less than distance between pads!')

        self.terminals = {'port1': DesignTerminal(position=self.position - self.p, orientation=self.orientation,
                                                  type='cpw', w=self.w, s=self.s, g=self.g, disconnected='short'),
                          'port2': DesignTerminal(position=self.position + self.p, orientation=self.orientation + np.pi,
                                                  type='cpw', w=self.w, s=self.s, g=self.g, disconnected='short')}

    def render(self):
        bend_radius = self.g
        precision = 0.001

        # create CPW line under airbridge
        cpw_line = gdspy.FlexPath(points=[self.position - self.p, self.position + self.p],
                                  width=[self.g, self.w, self.g],
                                  offset=[- self.w / 2 - self.s - self.g / 2, 0, self.w / 2 + self.s + self.g / 2],
                                  ends='flush',
                                  corners='natural', bend_radius=bend_radius, precision=precision,
                                  layer=self.geometry.layer_configuration.total_layer)
        # create pads of bridge
        pad1 = gdspy.Rectangle(
            (self.position[0] + self.geometry.pad_width / 2, self.position[1] + self.geometry.pad_distance / 2),
            (self.position[0] - self.geometry.pad_width / 2,
             self.position[1] + self.geometry.pad_distance / 2 + self.geometry.pad_length),
            layer=self.geometry.layer_configuration.airbridges_pad_layer)

        pad2 = gdspy.Rectangle(
            (self.position[0] - self.geometry.pad_width / 2, self.position[1] - self.geometry.pad_distance / 2),
            (self.position[0] + self.geometry.pad_width / 2,
             self.position[1] - self.geometry.pad_distance / 2 - self.geometry.pad_length),
            layer=self.geometry.layer_configuration.airbridges_pad_layer)

        contacts = gdspy.boolean(pad1, pad2, 'or', layer=self.geometry.layer_configuration.airbridges_pad_layer)

        contacts.rotate(self.orientation, self.position)

        # create bridge
        bridge = gdspy.Rectangle(
            (self.position[0] - self.geometry.bridge_width / 2, self.position[1] + self.geometry.bridge_length / 2),
            (self.position[0] + self.geometry.bridge_width / 2, self.position[1] - self.geometry.bridge_length / 2),
            layer=self.geometry.layer_configuration.airbridges_layer)

        bridge.rotate(self.orientation, self.position)

        restrict_total = gdspy.FlexPath(points=[self.position - self.p, self.position + self.p],
                                        width=self.w + 2 * self.s + 2 * self.g,
                                        corners='natural', ends='flush',
                                        layer=self.geometry.layer_configuration.restricted_area_layer)

        return {'positive': cpw_line, 'airbridges_pads': contacts, 'airbridges': bridge, 'restrict': restrict_total}

    def get_terminals(self) -> dict:
        return self.terminals

    def cm(self, epsilon):
        cross_section = [self.s, self.w, self.s]

        cl, ll = cm.ConformalMapping(cross_section, epsilon).cl_and_Ll()

        if not self.terminals['port1'].order:
            ll, cl = ll[::-1, ::-1], cl[::-1, ::-1]

        return cl, ll

    def add_to_tls(self, tls_instance: tlsim.TLSystem, terminal_mapping: Mapping[str, int],
                   track_changes: bool = True, cutoff: float = np.inf, epsilon: float = 11.45) -> list:
        """
        In this model an airbridge is a capacitor with C = epsilon_0 * epsilon * (S / h),
        where S - area of the airbridge, h - height of the airbridge upon the surface of a chip
        """

        h = 2*1e-6   # bridge height 2 mu m # TODO: OMG CONSTANTS IN CODE
        s = (self.geometry.bridge_length*1e-6) * (self.geometry.bridge_width*1e-6)

        epsilon_bridge = 1 # TODO: OMG CONSTANTS IN CODE

        bridge_capacitance = epsilon_0 * epsilon_bridge * s / h

        cl, ll = self.cm(epsilon)

        c = (bridge_capacitance + cl[0, 0] * 1e-6 * self.geometry.pad_width)
        l = ll[0, 0] * 1e-6 * self.geometry.pad_width

        c1 = tlsim.Capacitor(c=c/2, name=self.name + '_c1')
        c2 = tlsim.Capacitor(c=c/2, name=self.name + '_c2')
        l = tlsim.Inductor(l=l, name=self.name + '_l')

        elements = [c1, c2, l]
        if track_changes:
            self.tls_cache.append(elements)

        tls_instance.add_element(l, [terminal_mapping['port1'], terminal_mapping['port2']])
        tls_instance.add_element(c1, [terminal_mapping['port1'], 0])
        tls_instance.add_element(c2, [terminal_mapping['port2'], 0])

        return elements

    def __repr__(self):
        return "AirbridgeOverCPW {} C = {}".format(self.name, self.bridge_capacitance)

