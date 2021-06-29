import gdspy
from . import JJ4q
from .core import DesignElement, LayerConfiguration, DesignTerminal
import numpy as np
from .. import transmission_line_simulator as tlsim
from scipy.constants import epsilon_0

class JJ_in_line(DesignElement):
    def __init__(self,name, cpw_port,w, g, s, length, jj_params, layer_configuration):
        super().__init__(type='JJ in line', name=name)
        self.cpw_port = cpw_port
        self.orientation = cpw_port.orientation + np.pi
        self.w = w
        self.g = g
        self.s = s
        self.length = length
        self.jj_params = jj_params
        self.layer_configuration = layer_configuration
        h = 2 * 1e-9
        s = 1e-12*self.jj_params['a1']*self.jj_params['a2']
        epsilon = 10
        self.jj_capacitance = epsilon_0 * epsilon * s / h
        self.terminals = {'port1': DesignTerminal(position=self.cpw_port.position, orientation=self.orientation,
                                                  type='cpw', w=self.w, s=self.s, g=self.g, disconnected='short'),
                          'port2': DesignTerminal(
                              position=[self.cpw_port.position[0]+np.cos(self.orientation)*self.length,
                                        self.cpw_port.position[1]+np.sin(self.orientation)*self.length],
                              orientation=self.orientation-np.pi,
                                                  type='cpw', w=self.w, s=self.s, g=self.g, disconnected='short')}

    def render(self):
        self.jj_params['x'] = self.cpw_port.position[0] + np.cos(self.orientation)*self.length/2
        self.jj_params['y'] = self.cpw_port.position[1] + np.abs(np.cos(self.orientation))*self.jj_params['indent']/2 +\
                              np.abs(np.sin(self.orientation))*(np.sin(self.orientation)*self.length+self.jj_params['indent'])/2
        JJ = JJ4q.JJ_1(self.jj_params['x'], self.jj_params['y'],
                            self.jj_params['a1'], self.jj_params['a2'])
        jj = JJ.generate_jj()
        jj = gdspy.boolean(jj, jj, 'or', layer=self.layer_configuration.jj_layer)
        angle = self.jj_params['angle']
        jj.rotate(angle, (self.jj_params['x'], self.jj_params['y']))
        indent = 1 # overlap between the JJ's layer and the ground layer
        pad_up = gdspy.Rectangle((self.jj_params['x'] - JJ.contact_pad_a / 2,
                                 self.jj_params['y'] + indent),
                                (self.jj_params['x'] + JJ.contact_pad_a / 2,
                                 self.jj_params['y'] - JJ.contact_pad_b + indent),
                                layer=self.layer_configuration.total_layer)
        pad_down = gdspy.Rectangle((JJ.x_end - JJ.contact_pad_a / 2,
                                 JJ.y_end - 1),
                                (JJ.x_end + JJ.contact_pad_a / 2,
                                 JJ.y_end - JJ.contact_pad_b - indent),
                                layer=self.layer_configuration.total_layer)


        if np.round(np.sin(self.orientation),3)==0: # for horizontal based couplers
            poly1 = gdspy.Polygon([(self.jj_params['x'] - JJ.contact_pad_a / 2,
                                    self.jj_params['y'] + indent),
                                   (self.jj_params['x'] - JJ.contact_pad_a / 2,
                                    self.jj_params['y'] + indent - JJ.contact_pad_b),
                                   (self.jj_params['x'] - 3*JJ.contact_pad_a / 2,
                                    self.cpw_port.position[1] - self.w / 2),
                                   (self.jj_params['x'] - self.length/2,
                                    self.cpw_port.position[1] - self.w / 2),
                                   (self.jj_params['x'] - self.length/2,
                                    self.cpw_port.position[1] + self.w / 2),
                                   (self.jj_params['x'] - 3 * JJ.contact_pad_a / 2,
                                    self.cpw_port.position[1] + self.w / 2)],
                                  layer=self.layer_configuration.total_layer)
            poly2 = gdspy.Polygon([(JJ.x_end + JJ.contact_pad_a / 2,
                                    JJ.y_end - indent - JJ.contact_pad_b),
                                   (JJ.x_end + JJ.contact_pad_a / 2,
                                    JJ.y_end - indent),
                                   (JJ.x_end + 3*JJ.contact_pad_a / 2,
                                    self.cpw_port.position[1] + self.w / 2),
                                   (JJ.x_end + self.length/2,
                                    self.cpw_port.position[1] + self.w / 2),
                                   (JJ.x_end + self.length/2,
                                    self.cpw_port.position[1] - self.w / 2),
                                   (JJ.x_end + 3 * JJ.contact_pad_a / 2,
                                    self.cpw_port.position[1] - self.w / 2)],
                                  layer=self.layer_configuration.total_layer)
        elif np.round(np.cos(self.orientation),3)==0:
            poly1 = gdspy.Rectangle((self.jj_params['x'] - self.w/ 2,
                                    self.jj_params['y']),
                                   (self.jj_params['x'] + self.w/2,
                                    self.jj_params['y'] + (self.length -self.jj_params['indent'])/2),
                                    layer=self.layer_configuration.total_layer)
            poly2 = gdspy.Rectangle((JJ.x_end + self.w/ 2,
                                    JJ.y_end - JJ.contact_pad_b),
                                   (JJ.x_end - self.w/ 2,
                                    JJ.y_end - JJ.contact_pad_b - (self.length -self.jj_params['indent'])/2-indent),
                                    layer=self.layer_configuration.total_layer)


        ground_x1 = self.cpw_port.position[0]+np.sin(self.orientation)*(self.w/2+self.g+self.s)
        ground_y1 = self.cpw_port.position[1] + np.cos(self.orientation) * (self.w /2 + self.g + self.s)
        ground_x2 = self.cpw_port.position[0]+np.sin(self.orientation)*(self.w/2+self.s) +\
            np.cos(self.orientation)*self.length
        ground_y2 = self.cpw_port.position[1] + np.cos(self.orientation) * (self.w /2 + self.s) + \
             np.sin(self.orientation) * self.length
        ground1 = gdspy.Rectangle((ground_x1,ground_y1), (ground_x2,ground_y2), layer=self.layer_configuration.total_layer)
        ground2 = gdspy.copy(ground1)
        ground2 = ground2.mirror((self.cpw_port.position[0], self.cpw_port.position[1]),
                                 (self.cpw_port.position[0]+ np.cos(self.orientation)*self.length,
                                 self.cpw_port.position[1]+np.sin(self.orientation)*self.length))

        line = gdspy.boolean(pad_up, [pad_down, poly1, poly2, ground1, ground2], 'or',
                             layer=self.layer_configuration.total_layer)
        line.rotate(angle, (self.jj_params['x'], self.jj_params['y']))
        return {'positive': line, 'JJ':jj}

    def get_terminals(self):
        return self.terminals

    def add_to_tls(self, tls_instance: tlsim.TLSystem, terminal_mapping: dict, track_changes: bool = True,
                   cutoff: float = np.inf, epsilon: float = 11.45) -> list:
        from scipy.constants import hbar, e

        jj = tlsim.JosephsonJunction(self.jj_params['ic'] * hbar / (2 * e), name=self.name + ' jj')
        c = tlsim.Capacitor(self.jj_capacitance, name=self.name + ' jj-ground')
        cache = [jj, c]

        tls_instance.add_element(jj, [terminal_mapping['port1'], terminal_mapping['port2']])
        tls_instance.add_element(c, [terminal_mapping['port1'], terminal_mapping['port2']])

        return cache

