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
                          'port2': DesignTerminal(position=self.position + p, orientation=self.orientation, type='cpw',
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


# TODO: CPW-CPW crossing element
'''
# TODO: create some model for capacitances and inductances in tlsim
class CPWCrossingAirBridge(DesignElement):
    def __init__(self, name: str, position: Tuple[float, float], orientation: float, width: float, length: float,
                 padsize: float, padspacing: float, layer_configuration: LayerConfiguration, line_type: str = None,
                 l_over: float = 0, l_under: float = 0, c_over: float = 0, c_under: float = 0, c_over_under: float = 0):
        """
        Airbridge crossover element
        :param name: Design element name
        :param position: position on chip
        :param orientation: contact pad orientation
        :param width: air bridge width
        :param length: air bridge length
        :param padsize: air bridge square contact pad edge size
        :param padspacing: distance between contact pads of the three-pad device
        :param layer_configuration: LayerConfiguration object
        :param line_type: TODO: what does this do?
        :param l_over: indutance of top electrode for tlsim
        :param l_under: indutance of bottom electrode for tlsim
        :param c_over: capacitance of top electrode for tlsim
        :param c_under: capacitance of bottom electrode for tlsim
        :param c_over_under: mutual capacitance between top and bottom electrodes for tlsim
        """
        super().__init__('cpw_airbridge', name)
        self.position = np.asarray(position)
        self.padspacing = padspacing
        self.orientation = orientation

        offset = (padsize+padspacing)*np.asarray([np.sin(self.orientation), np.cos(self.orientation)])

        self.g1 = AirBridge(name=name+'g1', position=self.position+offset, orientation=orientation, width=width,
                            length=length, padsize=padsize, layer_configuration=layer_configuration, line_type=line_type,
                            l_over=0, l_under=l_under, c_over=0, c_under=c_under, c_over_under=0)

        self.g1 = AirBridge(name=name+'s', position=self.position, orientation=orientation, width=width,
                            length=length, padsize=padsize, layer_configuration=layer_configuration, line_type=line_type,
                            l_over=0, l_under=l_under, c_over=0, c_under=c_under, c_over_under=0)

        self.g2 = AirBridge(name=name+'g1', position=self.position-offset, orientation=orientation, width=width,
                            length=length, padsize=padsize, layer_configuration=layer_configuration, line_type=line_type,
                            l_over=0, l_under=l_under, c_over=0, c_under=c_under, c_over_under=0)


        self.padsize = padsize
        self.width = width
        self.length = length
        self.line_type = line_type
        self.layer_configuration = layer_configuration
        self.l_over = l_over
        self.l_under = l_under
        self.c_over = c_over
        self.c_under = c_under
        self.c_over_under = c_over_under
        #self.start = (None, None)
        #self.end = (None, None)
        self.tls_cache = []

    def render(self):
        x, y = self.position
        #TODO: wat is this
        if self.line_type == 'line':
            x += (self.length/2 + self.padsize/2)*np.cos(self.orientation)
            y += (self.length/2 + self.padsize/2)*np.sin(self.orientation)

        # first the two contacts
        contact_1 = gdspy.Rectangle((x - self.length/2 - self.padsize/2, y-self.padsize/2),
                                    (x - self.length/2 + self.padsize/2, y + self.padsize/2))
        contact_2 = gdspy.Rectangle((x + self.length/2 - self.padsize/2, y-self.padsize/2),
                                    (x + self.length/2 + self.padsize/2, y + self.padsize/2))
        contacts = gdspy.boolean(contact_1, contact_2, 'or', layer=self.layer_configuration.airbridges_pad_layer)
        contacts.rotate(self.orientation, (x, y))
        # add restricted area for holes
        restricted_area = gdspy.Rectangle((x - self.length / 2 - self.padsize / 2, y - self.padsize / 2),
                            (x + self.length / 2 + self.padsize / 2, y + self.padsize / 2))
        # now the bridge itself
        bridge = gdspy.Rectangle((x - self.length / 2, y - self.width / 2),
                                 (x + self.length / 2, y + self.width / 2),
                                 layer=self.layer_configuration.airbridges_layer)
        bridge.rotate(self.orientation, (x, y))

        return {'positive': [contacts, bridge], 'restrict': restricted_area}

    def get_terminals(self) -> Mapping[str, Mapping:[str, Any]]:
        return {'over_in': {'position': self.position - self.length/2 - self.padsize/2, 'orientation': self.orientation},
                'over_out': {'position': self.position + self.length/2 + self.padsize/2, 'orientation': self.orientation+np.pi},
                'under_in': {'position': self.position - self.padsize/2, 'orientation': self.orientation + np.pi/2},
                'under_out': {'position': self.position + self.padsize/2, 'orientation': self.orientation - np.pi/2}} #['over_in', 'over_out', 'under_in', 'under_out']

    def add_to_tls(self, tls_instance: tlsim.TLSystem, terminal_mapping: Mapping[str, int], track_changes: bool = True) -> list:
        l_over = tlsim.Inductor(l=self.l_over)
        l_under = tlsim.Inductor(l=self.l_under)
        c_over_in = tlsim.Capacitor(c=self.c_over / 2)
        c_over_out = tlsim.Capacitor(c=self.c_over / 2)
        c_under_in = tlsim.Capacitor(c=self.c_under / 2)
        c_under_out = tlsim.Capacitor(c=self.c_under / 2)
        c_over_in_under_in = tlsim.Capacitor(c=self.c_over_under / 4)
        c_over_out_under_in = tlsim.Capacitor(c=self.c_over_under / 4)
        c_over_in_under_out = tlsim.Capacitor(c=self.c_over_under / 4)
        c_over_out_under_out = tlsim.Capacitor(c=self.c_over_under / 4)

        elements = [l_over, l_under, c_over_in, c_over_out, c_under_in, c_under_out,
                    c_over_in_under_in, c_over_out_under_in, c_over_in_under_out, c_over_out_under_out]
        if track_changes:
            self.tls_cache.append(elements)

        tls_instance.add_element(l_over, [terminal_mapping['over_in'], terminal_mapping['over_out']])
        tls_instance.add_element(l_under, [terminal_mapping['under_in'], terminal_mapping['under_out']])
        tls_instance.add_element(c_over_in, [terminal_mapping['over_in'], 0])
        tls_instance.add_element(c_over_out, [terminal_mapping['over_out'], 0])
        tls_instance.add_element(c_under_in, [terminal_mapping['under_in'], 0])
        tls_instance.add_element(c_under_out, [terminal_mapping['under_out'], 0])
        tls_instance.add_element(c_over_in_under_in, [terminal_mapping['over_in'], terminal_mapping['under_in']])
        tls_instance.add_element(c_over_out_under_in, [terminal_mapping['over_out'], terminal_mapping['over_in']])
        tls_instance.add_element(c_over_in_under_out, [terminal_mapping['over_in'], terminal_mapping['under_out']])
        tls_instance.add_element(c_over_out_under_out, [terminal_mapping['over_out'], terminal_mapping['under_out']])

        return elements
'''
