from .core import DesignElement, LayerConfiguration
from .. import transmission_line_simulator as tlsim
import numpy as np
import gdspy

# TODO: create some model for capacitances and inductances in tlsim
class Airbridge(DesignElement):
    def __init__(self, name: str, width: float, length: float, padsize: float, position: tuple, angle: float,
                 layer_configuration: LayerConfiguration, line_type = None,
                 l_over=0, l_under=0, c_over=0, c_under=0, c_over_under=0):
        """

        :param name: Design element name
        :param width: air bridge width
        :param length: air bridge length
        :param padsize: ait bridge square contact pad edge size
        :param position: position on chip
        :param angle: contact pad orientation
        :param layer_configuration: LayerConfiguration object
        :param line_type:
        """
        super().__init__('airbridge', name)
        self.orientation = angle
        self.position = np.asarray(position)
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

        return {'positive':[contacts, bridge], 'restrict': restricted_area}

    def get_terminals(self) -> dict:
        return {'over_in': (self.position - self.length/2 - self.padsize/2, self.orientation),
                'over_out': (self.position + self.length/2 + self.padsize/2, self.orientation+np.pi),
                'under_in': (self.position - self.padsize/2, self.orientation + np.pi/2),
                'under_out': (self.position + self.padsize/2, self.orientation - np.pi/2)} #['over_in', 'over_out', 'under_in', 'under_out']

    def add_to_tls(self, tls_instance: tlsim.TLSystem, terminal_mapping: dict, track_changes: bool = True) -> list:
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
