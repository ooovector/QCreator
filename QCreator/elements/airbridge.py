from .core import DesignElement, DesignTerminal, LayerConfiguration
from .. import transmission_line_simulator as tlsim
import numpy as np
import gdspy
from typing import Tuple, Mapping, Any
from .drawing import combine
from .cpw_primitives import Trapezoid
from .. import conformal_mapping as cm


class AirbridgeOverCPW(DesignElement):
    def __init__(self, name: str, position: Tuple[float, float], orientation: float,
                 w: float, s: float, g: float, pads_geometry: Tuple[float, float],
                 bridge_geometry: Tuple[float, float], layer_configuration: LayerConfiguration,
                 distance_between_pads: float = None):
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

        self.position = position
        self.orientation = orientation
        self.w = w
        self.s = s
        self.g = g

        if not distance_between_pads:
            self.distance_between_pads = self.w + 2 * self.s + self.g / 2
        else:
            self.distance_between_pads = distance_between_pads

        self.pads_geometry = pads_geometry
        self.bridge_geometry = bridge_geometry

        self.layer_configuration = layer_configuration
        self.tls_cache = []

        cpw_width = 2 * self.s + self.w + 2 * self.g

        self.p = self.pads_geometry[1] / 2 * np.asarray([np.cos(self.orientation), np.sin(self.orientation)])

        if self.distance_between_pads > cpw_width:
            raise ValueError('Distance between pads is larger that CPW width!')
        if self.distance_between_pads < 2 * (self.w / 2 + self.s):
            raise ValueError('Distance between pads is too small!')

        if self.bridge_geometry[1] > self.pads_geometry[1]:
            raise ValueError('Airbridge length is larger than length of airbridge pad!')
        if self.bridge_geometry[0] < self.distance_between_pads:
            raise ValueError('Airbridge width is less than distance between pads!')

        self.terminals = {'port1': DesignTerminal(position=self.position - self.p, orientation=self.orientation,
                                                  type='cpw', w=self.w, s=self.s, g=self.g, disconnected='short'),
                          'port2': DesignTerminal(position=self.position + self.p, orientation=self.orientation + np.pi,
                                                  type='cpw', w=self.w, s=self.s, g=self.g, disconnected='short'),
                          'port': DesignTerminal(position=self.position, orientation=self.orientation + np.pi,
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
                                  layer=self.layer_configuration.total_layer)
        # create pads of bridge
        pad1 = gdspy.Rectangle(
            (self.position[0] + self.pads_geometry[1] / 2, self.position[1] + self.distance_between_pads / 2),
            (self.position[0] - self.pads_geometry[1] / 2,
             self.position[1] + self.distance_between_pads / 2 + self.pads_geometry[0]),
            layer=self.layer_configuration.airbridges_pad_layer)

        pad2 = gdspy.Rectangle(
            (self.position[0] - self.pads_geometry[1] / 2, self.position[1] - self.distance_between_pads / 2),
            (self.position[0] + self.pads_geometry[1] / 2,
             self.position[1] - self.distance_between_pads / 2 - self.pads_geometry[0]),
            layer=self.layer_configuration.airbridges_pad_layer)

        contacts = gdspy.boolean(pad1, pad2, 'or', layer=self.layer_configuration.airbridges_pad_layer)

        contacts.rotate(self.orientation, self.position)

        # create bridge
        bridge = gdspy.Rectangle(
            (self.position[0] - self.bridge_geometry[1] / 2, self.position[1] + self.bridge_geometry[0] / 2),
            (self.position[0] + self.bridge_geometry[1] / 2, self.position[1] - self.bridge_geometry[0] / 2),
            layer=self.layer_configuration.airbridges_layer)

        bridge.rotate(self.orientation, self.position)

        return {'positive': cpw_line, 'airbridges_pads': contacts, 'airbridges': bridge}

    def get_terminals(self) -> dict:
        return self.terminals

    def cm(self):
        cross_section = [self.s, self.w, self.s]

        ll, cl = cm.ConformalMapping(cross_section).cl_and_Ll()

        if not self.terminals['port1'].order:
            ll, cl = ll[::-1, ::-1], cl[::-1, ::-1]

        return ll, cl

    def add_to_tls(self, tls_instance: tlsim.TLSystem, terminal_mapping: Mapping[str, int], track_changes: bool = True,
                   cutoff: float = np.inf) -> list:
        # TODO: calculate bridge_capacitance???
        bridge_capacitance = 1e-15

        c1 = tlsim.Capacitor(c=bridge_capacitance)
        cl, ll = self.cm()
        line1_length = self.pads_geometry[1] / 2
        line1 = tlsim.TLCoupler(n=1,
                                l=line1_length,
                                cl=cl,
                                ll=ll,
                                rl=np.zeros((1, 1)),
                                gl=np.zeros((1, 1)),
                                name=self.name + '_line1',
                                cutoff=cutoff)
        line2 = tlsim.TLCoupler(n=1,
                                l=self.p,
                                cl=cl,
                                ll=ll,
                                rl=np.zeros((1, 1)),
                                gl=np.zeros((1, 1)),
                                name=self.name + '_line2',
                                cutoff=cutoff)

        elements = [c1, line1, line2]
        if track_changes:
            self.tls_cache.append(elements)

        tls_instance.add_element(line1, [terminal_mapping['port1'], terminal_mapping['port']])
        tls_instance.add_element(line2, [terminal_mapping['port'], terminal_mapping['port2']])

        tls_instance.add_element(c1, [terminal_mapping['port'], 0])

        return elements

    def __repr__(self):
        return "AirbridgeOverCPW {}".format(self.name)


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
                                    (self.padsize / 2, self.length / 2 + self.padsize / 2))
        contact_2 = gdspy.Rectangle((- self.padsize / 2, - self.length / 2 - self.padsize / 2),
                                    (self.padsize / 2, - self.length / 2 + self.padsize / 2))
        contacts = gdspy.boolean(contact_1, contact_2, 'or', layer=self.layer_configuration.airbridges_pad_layer)
        # add restricted area for holes
        restricted_area = gdspy.Rectangle((-self.padsize / 2, - self.length / 2 - self.padsize / 2),
                                          (self.padsize / 2, self.length / 2 + self.padsize / 2))
        # now the bridge itself
        bridge = gdspy.Rectangle((-self.width / 2, -self.length / 2),
                                 (self.width / 2, self.length / 2),
                                 layer=self.layer_configuration.airbridges_layer)

        # return {'airbridges_pad_layer': [contacts], 'airbridges_layer': [bridge], 'restrict': (restricted_area,)}
        return {'airbridges_pad_layer': contacts, 'airbridges_layer': bridge, 'restrict': restricted_area}


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

        self.terminals = {'port1': DesignTerminal(position=self.position - p, orientation=self.orientation,
                                                  type='cpw', w=self.w, s=self.s, g=self.g, disconnected='short'),
                          'port2': DesignTerminal(position=self.position + p, orientation=self.orientation + np.pi,
                                                  type='cpw', w=self.w, s=self.s, g=self.g, disconnected='short')}

    def render(self):
        airbridge = self.geometry.render()
        under_cpw = Trapezoid(w1=self.w, s1=self.s, g1=self.g, w2=self.w, s2=self.s, g2=self.g,
                              length=self.geometry.padsize, layer_configuration=self.layer_configuration).render()

        result = combine([(airbridge, (0, 0)), (under_cpw, (0, 0))])
        # result = {}

        for layer_name, polygons in result.items():
            result[layer_name].rotate(self.orientation, (0, 0))
            result[layer_name].translate(*self.position)

        return result

    def get_terminals(self) -> dict:
        return self.terminals

    def add_to_tls(self, tls_instance: tlsim.TLSystem, terminal_mapping: Mapping[str, int], track_changes: bool = True,
                   cutoff: float = np.inf) -> list:
        l = tlsim.Inductor(l=self.l)
        c1 = tlsim.Capacitor(c=self.c / 2)
        c2 = tlsim.Capacitor(c=self.c / 2)

        elements = [l, c1, c2]
        if track_changes:
            self.tls_cache.append(elements)

        tls_instance.add_element(l, [terminal_mapping['port1'], terminal_mapping['port2']])
        tls_instance.add_element(c1, [terminal_mapping['port1'], 0])
        tls_instance.add_element(c2, [terminal_mapping['port2'], 0])

        return elements
