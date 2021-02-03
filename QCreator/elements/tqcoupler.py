from .core import DesignElement, LayerConfiguration, DesignTerminal
from .. import conformal_mapping as cm
from .. import transmission_line_simulator as tlsim
import gdspy
import numpy as np
from typing import List, Tuple, Mapping, Dict, AnyStr
from . import JJ4q #TODO make this qubit class suitable for any squid types
from . import squid3JJ #TODO make this qubit class suitable for any squid types

from copy import deepcopy

class MMCoupler(DesignElement):
    """
    Coaxmon qubit consists of several parts:
    1) Central part - central circuit
    params: center = center of the circle/qubit, center_radius = radius of the center part=qubit
    2) Couplers - 5 couplers, 4 are used to connect two-qubit couplers. 1 for a flux line or a microwave line.
    They are arcs, you can read more about them in CoaxmonCoupler description
    params: inner_couplers_radius,outer_couplers_radius
    3) Ground = grounded 6th coupler which is used for a flux line and a microwave line
    params: inner_ground_radius,outer_ground_radius
    4)layer_configuration
    5)Couplers - coupler classes
    6) jj_params - parameters of the SQUID which here is 3JJ SQUID.#TODO add more information
    """
    def __init__(self, name: str, qubit1, coupler1_name: AnyStr,
                 qubit2, coupler2_name: AnyStr, core: float,
                 gap: float, ground: float,
                 layer_configuration: LayerConfiguration, jj_params: Dict, squid_params: Dict, fluxline: Dict):
        super().__init__(type='qubit coupler', name=name)
        #coupler parameters
        self.qubit1 = qubit1
        self.qubit2 = qubit2
        self.coupler1_name = coupler1_name
        self.coupler2_name = coupler2_name
        self.connection1 = qubit1.get_terminals()[coupler1_name].position
        self.connection2 = qubit2.get_terminals()[coupler2_name].position
        self.orientation = np.arctan2(self.connection1[0]-self.connection2[0], self.connection1[1]-self.connection2[1])
        self.core = core
        self.gap = gap
        self.ground = ground
        # jj parameters
        self.jj_params = jj_params
        self.squid_params = squid_params
        self.JJ = None
        self.fluxline = fluxline
        if self.fluxline is not None:
            self.w = self.fluxline['w']
            self.s = self.fluxline['g']
            self.g = self.fluxline['s']
        #layers
        self.layer_configuration = layer_configuration
        # coupler terminals
        self.terminals = {'coupler1': qubit1.get_terminals()[coupler1_name],
                          'coupler2': qubit2.get_terminals()[coupler2_name],
                          'flux line': None}
        self.layers = []

    def render(self):
        """
        This function draw everything
        """
        ground1= self.generate_rect(self.core/2+self.ground+self.gap, self.core/2+self.ground,layer=self.layer_configuration.total_layer)
        qubit = self.generate_rect(self.core/2, -self.core/2, layer=self.layer_configuration.total_layer)
        ground2 = self.generate_rect(-self.core/2-self.ground-self.gap, -self.core/2-self.ground,layer=self.layer_configuration.total_layer)
        restricted = self.generate_rect(-self.core/2-self.ground-self.gap, +self.core/2+self.ground+self.gap,layer=self.layer_configuration.restricted_area_layer)
        positive = gdspy.boolean(qubit, [ground2, ground1], 'or', layer=self.layer_configuration.total_layer)

        qubit_cap_parts=[positive,gdspy.boolean(ground1,ground2,'or')] # TODO: add cap.calc. option

        # add JJs at the ends
        if self.jj_params != {}:
            for connection in [self.connection1,self.connection2]:
                JJ, rect , to_remove= self.generate_JJ(connection) #TODO change it in a new manner, probably one day
                result = gdspy.boolean(positive, to_remove, 'not', layer=self.layer_configuration.total_layer)
                positive = gdspy.boolean(result, rect, 'or', layer=self.layer_configuration.total_layer)
                self.JJ = gdspy.boolean(JJ, self.JJ, 'or', layer=self.layer_configuration.jj_layer)

        # add flux line and squid
        if self.squid_params !={}:
            squid, rect, to_remove = self.generate_squid(self.squid_params['point'])
            self.JJ = gdspy.boolean(squid, self.JJ, 'or', layer=self.layer_configuration.jj_layer)
            positive = gdspy.boolean(positive, to_remove, 'not', layer=self.layer_configuration.total_layer)
            positive = gdspy.boolean(positive, rect, 'or', layer=self.layer_configuration.total_layer)
            fluxline = self.connection_to_ground()
            positive = gdspy.boolean(positive, fluxline['positive'], 'or', layer=self.layer_configuration.total_layer)
            positive = gdspy.boolean(positive, fluxline['remove'], 'not', layer=self.layer_configuration.total_layer)

        # # set terminals for couplers
        # self.set_terminals()
        return {'positive': positive,
                'restricted': restricted,
                'JJ': self.JJ
                }



    def set_terminals(self):
        pass
        # for id, coupler in enumerate(self.couplers):
        #     self.terminals['coupler'+str(id)] = DesignTerminal(coupler.connection,
        #                                                        coupler.phi, g=coupler.w, s=coupler.g,
        #                                                         w=coupler.w, type='cpw')
        return True
    def get_terminals(self):
        return self.terminals

    def generate_JJ(self, connection_part):
        JJ_part_length=39 # TODO: maybe we need to find a way to remove this value from here, but for now it's fine
        max_x=max(self.connection1[0],self.connection2[0])
        min_x=min(self.connection1[0],self.connection2[0])
        coeff_x=1 if max_x>connection_part[0] + self.jj_params['indent']*np.cos(self.orientation+np.pi/2)>min_x else -1
        self.jj_params['x'] = connection_part[0] + coeff_x*self.jj_params['indent']*np.cos(self.orientation+np.pi/2)
        max_y = max(self.connection1[1], self.connection2[1])
        min_y = min(self.connection1[1], self.connection2[1])
        if max_y!=min_y:
            coeff_y = 1 if max_y > connection_part[1]+((self.core +self.ground)/ 2 +self.jj_params['indent'])*np.sin(self.orientation + np.pi / 2) > min_y else -1
            self.jj_params['y'] = connection_part[1] +coeff_y*(-coeff_y*JJ_part_length/2+ self.jj_params['indent']) * np.sin(self.orientation + np.pi / 2)
        if max_y==min_y:
            self.jj_params['y'] = connection_part[1]+JJ_part_length/2
        JJ = JJ4q.JJ_1(self.jj_params['x'], self.jj_params['y'],
                            self.jj_params['a1'], self.jj_params['a2'],
                            )
        result = JJ.generate_jj()
        result = gdspy.boolean(result, result, 'or', layer=self.layer_configuration.jj_layer)
        angle = self.jj_params['angle']
        result.rotate(angle, (self.jj_params['x'], self.jj_params['y']))
        indent = 1 # overlap between the JJ's layer and the ground layer
        rect1 = gdspy.Rectangle((self.jj_params['x'] - JJ.contact_pad_a / 2,
                                 self.jj_params['y'] + indent),
                                (self.jj_params['x'] + JJ.contact_pad_a / 2,
                                 self.jj_params['y'] - JJ.contact_pad_b + indent), layer=6)
        rect2 = gdspy.Rectangle((JJ.x_end - JJ.contact_pad_a / 2,
                                 JJ.y_end - 1),
                                (JJ.x_end + JJ.contact_pad_a / 2,
                                 JJ.y_end - JJ.contact_pad_b - indent), layer=6)
        if self.connection1[0] != self.connection2[0]: # for horizontal based couplers
            poly1 = gdspy.Polygon([(self.jj_params['x'] - JJ.contact_pad_a / 2,
                                    self.jj_params['y'] + indent),
                                   (self.jj_params['x'] - JJ.contact_pad_a / 2,
                                    self.jj_params['y'] + indent - JJ.contact_pad_b),
                                   (self.jj_params['x'] - JJ.contact_pad_a - indent,
                                    self.connection1[1] - self.core / 2),
                                   (self.jj_params['x'] - JJ.contact_pad_a - indent,
                                    self.connection1[1] + self.core / 2)
                                   ])
            poly2 = gdspy.Polygon([(JJ.x_end + JJ.contact_pad_a / 2,
                                    JJ.y_end - indent - JJ.contact_pad_b),
                                   (JJ.x_end + JJ.contact_pad_a / 2,
                                    JJ.y_end - indent),
                                   (JJ.x_end + JJ.contact_pad_a + indent,
                                    self.connection1[1] + self.core / 2),
                                   (JJ.x_end + JJ.contact_pad_a + indent,
                                    self.connection1[1] - self.core / 2)
                                   ])
        else:
            poly1, poly2 = [], []
        rect = gdspy.boolean(rect1, [rect2, poly1, poly2], 'or')
        rect.rotate(angle, (self.jj_params['x'], self.jj_params['y']))
        to_remove = gdspy.Polygon(JJ.points_to_remove, layer=self.layer_configuration.layer_to_remove)
        to_remove.rotate(angle, (self.jj_params['x'], self.jj_params['y']))
        return result, rect, to_remove

    def generate_squid(self, point):
        self.squid_params['x'] = point[0]
        self.squid_params['y'] = point[1]
        self.squid = squid3JJ.JJ_2(self.squid_params['x'],
                                   self.squid_params['y'],
                self.squid_params['a1'], self.squid_params['a2'],
                self.squid_params['b1'], self.squid_params['b2'],
                self.squid_params['c1'], self.squid_params['c2'])
        squid = self.squid.generate_jj()
        rect = gdspy.Rectangle((self.squid_params['x'] - self.squid.contact_pad_a_outer / 2,
                                self.squid_params['y'] + 0*self.squid.contact_pad_b_outer/2),
                               (self.squid_params['x'] + self.squid.contact_pad_a_outer / 2,
                                self.squid_params['y'] - self.squid.contact_pad_b_outer*np.cos(self.squid_params['angle'])),
                               layer=self.layer_configuration.total_layer)
        if self.connection1[0] == self.connection2[0]:
            indent=2
            coeff = 1 if self.squid_params['side']=='right' else -1
            path1=gdspy.Polygon([(self.squid_params['x']+ self.squid.contact_pad_a_outer / 2*coeff, self.squid_params['y'] +indent),
                                 (self.connection1[0], self.squid_params['y'] +indent),
                                 (self.connection1[0], self.squid_params['y'] - self.squid.contact_pad_b_outer),
                                 (self.squid_params['x']+ self.squid.contact_pad_a_outer / 2*coeff, self.squid_params['y'] - self.squid.contact_pad_b_outer)])
            rect=gdspy.boolean(rect,path1,'or',layer=self.layer_configuration.total_layer)
        # create a polygon to remove
        to_remove = gdspy.Rectangle((self.squid_params['x'] + self.squid_params['removing']['right'],
                                     self.squid_params['y'] + self.squid_params['removing']['up']),
                                    (self.squid_params['x'] - self.squid_params['removing']['left'],
                                     self.squid_params['y'] - self.squid_params['removing']['down']))
        to_remove.rotate(self.squid_params['angle'], (self.squid_params['x'],
                                                      self.squid_params['y']))
        squid=gdspy.boolean(squid,squid,'or',layer=self.layer_configuration.jj_layer)
        squid.rotate(self.squid_params['angle'],(self.squid_params['x'],
                                   self.squid_params['y']))
        self.squid.rect1 = rotate_point(self.squid.rect1,self.squid_params['angle'],(self.squid_params['x'],
                                   self.squid_params['y']))
        self.squid.rect2 = rotate_point(self.squid.rect2, self.squid_params['angle'], (self.squid_params['x'],
                                                                                       self.squid_params['y']))
        return squid, rect, to_remove



    def connection_to_ground(self):
        """
        This function generate a connection from squid rectangulars to a flux line output. Should be changed if you want
        to use another type of JJ or a flux line
        """
        result = None
        width = self.fluxline['width']
        if 'side' in self.squid_params:
            length_x = self.fluxline['length_x']
            length_y = self.fluxline['length_y']
            (coeff, squid_pad1, squid_pad2) = (1,self.squid.rect2,self.squid.rect1) if self.squid_params['side']=='right' \
                else (-1,self.squid.rect1,self.squid.rect2)
            connection_length = self.squid_params['removing']['right'] if self.squid_params['side'] == 'right' \
                else self.squid_params['removing']['left']
            indent =1
            rect1 = gdspy.Rectangle((squid_pad1[0]+length_x*coeff, squid_pad1[1]+indent/2),
                                    (squid_pad1[0]-coeff*self.squid.rect_size_a/2, squid_pad1[1]-width+indent/2))
            result = gdspy.boolean(rect1,result,'or',layer=self.layer_configuration.total_layer)
            connection = (self.squid_params['x']+connection_length*coeff, squid_pad1[1] + (indent-width)/2)
            # add cpw from
            flux_line_output = (self.connection1[0]+(self.core/2+self.gap+self.ground)*coeff, connection[1] )
            remove = gdspy.FlexPath(deepcopy([connection, flux_line_output]), [self.w, self.w],
                                    offset=[-self.s, self.s], layer=self.layer_configuration.total_layer)
            self.terminals['flux line'] = DesignTerminal(flux_line_output, np.pi if coeff==-1 else 0,
                                                         g=self.g, s=self.s,
                                                         w=self.w, type='cpw')
            # add connection to the ground
            first_turn = (squid_pad2[0], squid_pad2[1]-length_y)
            ground_connection = (first_turn[0]+coeff*self.gap, first_turn[1])
            rect2 = gdspy.FlexPath(deepcopy([squid_pad2, first_turn,ground_connection]), [width],
                                    offset=0, layer=self.layer_configuration.total_layer)
            result = gdspy.boolean(rect2, result, 'or', layer=self.layer_configuration.total_layer)
        else: # for horizontal base couplers
            length = self.fluxline['length']
            for squid_pad in [self.squid.rect1, self.squid.rect2]:
                rect1 = gdspy.Rectangle((squid_pad[0]+width/2, squid_pad[1]-length*np.cos(self.squid_params['angle'])),
                                        (squid_pad[0]-width/2, squid_pad[1]))
                result = gdspy.boolean(rect1,result,'or',layer=self.layer_configuration.total_layer)

            connection = (self.squid.rect2[0], self.squid_params['y'] -self.squid_params['removing']['down']*np.cos(self.squid_params['angle']))
            # add cpw from
            flux_line_output = (connection[0], self.squid_params['y']-(self.gap+self.ground)*np.cos(self.squid_params['angle']) )
            remove = gdspy.FlexPath(deepcopy([connection, flux_line_output]), [self.w, self.w],
                                    offset=[-self.s, self.s], layer=self.layer_configuration.total_layer)
            self.terminals['flux line'] = DesignTerminal(flux_line_output, self.squid_params['angle']-np.pi/2,
                                                         g=self.g, s=self.s,
                                                         w=self.w, type='cpw')
        return {'positive': result,
                'remove': remove,
                }

    def generate_rect(self,offset1,offset2,layer):
        return gdspy.Rectangle((self.connection1[0]+offset1*np.cos(self.orientation),
                               self.connection1[1]+offset1*np.sin(self.orientation)),
                              (self.connection2[0]+offset2*np.cos(self.orientation),
                               self.connection2[1]+offset2*np.sin(self.orientation)),layer=layer)


def rotate_point(point, angle, origin):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point
    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return qx, qy