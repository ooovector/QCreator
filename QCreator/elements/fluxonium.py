import gdspy
from .core import DesignElement, LayerConfiguration, DesignTerminal
import numpy as np
from . import squid3JJ
from typing import List, Tuple, Mapping, Dict, AnyStr
from copy import deepcopy
from .. import conformal_mapping as cm
from .. import transmission_line_simulator as tlsim
from scipy.constants import hbar, e
import pandas as pd 


class Fluxonium(DesignElement):
    """
                   :param center: center of fluxonium like (x_coordinate, y_coordinate)
                   :param distance: distance from center to borders of inner rectangles
                   :param rectang_params: parameters like (width_rectang,height_rectang) for big inner rectangles
                   :param gap: distance between inner rectangles and ground
                   :param ground_width: width of ground
                   :param slit_width: width of the small area at the top of fluxonium
                   :param rect_in_slit_params: parameters like (width_rectang,height_rectang) for rectangle in slit
                   :param ledge: the depth of penetration of the rectangle into the cavity
                   :param groove_params: {'width': 2,
                           'height': 1,
                           'distance_from_center': 5}: parameters of the cavity for couplers in the ground
                   :param port_params:   {'width': 0.5,
                           'distance_from_center': 7}: parameters of the cavity for the ports
                   :param couplers: {'left': True,
                       'right': False}: indicates if there should be left or(and) right coupler(s)
                   :param couplers_params = {"height": 2,
                             "width": 5,
                             "distance_from_center": 17}: parameters of the couplers
                   :param left(right)_cpw_params = {'w' = w
                                             'g' = g
                                             's' = s}: cpw parameters

           """
    def __init__(self,
                 name: str,
                 layer_configuration: LayerConfiguration,
                 center,
                 distance,
                 rectang_params,
                 gap,
                 ground_width,
                 slit_width,
                 rect_in_slit_params,
                 ledge,
                 groove_params,
                 port_params,
                 couplers=None,
                 couplers_params=None,
                 calculate_capacitance=False,
                 left_cpw_params=None,
                 right_cpw_params=None,
                 qubit_cpw_params=None,
                 transformations=None):

        super().__init__(type='fluxonium', name=name)
        self.center = center
        self.distance = distance
        self.rectang_params = rectang_params
        self.gap = gap
        self.ground = ground_width
        self.slit_width = slit_width
        self.rect_in_slit_params = rect_in_slit_params
        self.ledge = ledge
        self.layer_configuration = layer_configuration
        self.groove_params = groove_params
        self.port_params = port_params
        self.transformations = transformations

        self.half_length_x = self.distance + self.rectang_params[0] + self.gap + self.ground
        self.half_length_y = self.rectang_params[1] / 2 + self.gap + self.ground


        # set default parameters for couplers

        if couplers is not None:
            self.couplers = couplers
        else:
            self.couplers = {"left": False,
                             "right": False}

        if couplers_params is not None:
            self.couplers_height = couplers_params["height"]
            self.couplers_width = couplers_params["width"]
            self.couplers_distance_from_center = couplers_params["distance_from_center"]

        # self. 'lm': 3.3e-12,
        self.JJ_params = {'ic': 1e-6, #1e-6 # uA/um^2
                          'lm': 3.3e-12}
        self.tls_cache = []

        # calculate capacitance:
        self.calculate_capacitance = calculate_capacitance
        self.capacitance_matrix = None 

        self.C = {'C_lg': None,
                  'C_rg': None,
                  'C_l1': None,
                  'C_r2': None,
                  'qubit': None} # TODO: fix this constant 



        self.layers = []

        self.terminals = {"qubit": None,
                          "left_coupler": None,
                          "right_coupler": None}


        if self.couplers["left"] is True:
            self.left_cpw_params = left_cpw_params

        if self.couplers["right"] is True:
            self.right_cpw_params = right_cpw_params

        self.qubit_cpw_params=qubit_cpw_params


    def _generate_ground(self):

        half_length_x = self.distance + self.rectang_params[0] + self.gap + self.ground
        half_length_y = self.rectang_params[1] / 2 + self.gap + self.ground

        ground = gdspy.Rectangle((self.center[0] - half_length_x, self.center[1] - half_length_y),
                                 (self.center[0] + half_length_x, self.center[1] + half_length_y))
        
        empty_rectangle = gdspy.Rectangle((self.center[0] - self.distance - self.rectang_params[0] - self.gap,
                                           self.center[1] - self.rectang_params[1] / 2 - self.gap),
                                          (self.center[0] + self.distance + self.rectang_params[0] + self.gap,
                                           self.center[1] + self.rectang_params[1] / 2 + self.gap))

        empty_top_rectangle = gdspy.Rectangle(
            (self.center[0] - self.slit_width / 2, self.center[1] + self.gap + self.rectang_params[1] / 2),
            (self.center[0] + self.slit_width / 2,
             self.center[1] + self.gap + self.rectang_params[1] / 2 + self.ground))

        additional_rectangles = list(map(lambda sign: gdspy.Rectangle(
            (self.center[0] + sign * self.slit_width / 2,
             self.center[1] + self.gap + self.rectang_params[1] / 2 - self.ledge),
            (self.center[0] + sign * (self.slit_width / 2 - self.rect_in_slit_params[0]),
             self.center[1] + self.gap + self.rectang_params[1] / 2 - self.ledge + self.rect_in_slit_params[1])
        ), [+1, -1]))

        result = gdspy.boolean(ground, empty_rectangle, 'not')
        result = gdspy.boolean(result, empty_top_rectangle, 'not')
        result = gdspy.boolean(result, additional_rectangles[0], 'or')
        result = gdspy.boolean(result, additional_rectangles[1], 'or')

        #  add some place for the second coupler

        if self.couplers["left"] is True:
            empty_rectangle_for_groove = gdspy.Rectangle(
                (self.center[0] - self.groove_params["distance_from_center"] - self.groove_params["width"],
                 self.center[1] - self.gap - self.rectang_params[1] / 2 - self.groove_params["height"]),

                (self.center[0] - self.groove_params["distance_from_center"],
                 self.center[1] - self.gap - self.rectang_params[1] / 2))

            result = gdspy.boolean(result, empty_rectangle_for_groove, 'not')

            empty_rectangle_for_port = gdspy.Rectangle(
                (self.center[0] - self.port_params["distance_from_center"] - self.port_params["width"] / 2,
                 self.center[1] - self.gap - self.rectang_params[1] / 2 - self.ground),

                (self.center[0] - self.port_params["distance_from_center"] + self.port_params["width"] / 2,
                 self.center[1] - self.gap - self.rectang_params[1] / 2 - self.groove_params["height"]))

            result = gdspy.boolean(result, empty_rectangle_for_port, 'not')

        if self.couplers["right"] is True:
            empty_rectangle_for_groove = gdspy.Rectangle(
                (self.center[0] + self.groove_params["distance_from_center"],
                 self.center[1] - self.gap - self.rectang_params[1] / 2 - self.groove_params["height"]),

                (self.center[0] + self.groove_params["distance_from_center"] + self.groove_params["width"],
                 self.center[1] - self.gap - self.rectang_params[1] / 2))

            result = gdspy.boolean(result, empty_rectangle_for_groove, 'not')

            empty_rectangle_for_port = gdspy.Rectangle(
                (self.center[0] + self.port_params["distance_from_center"] - self.port_params["width"] / 2,
                 self.center[1] - self.gap - self.rectang_params[1] / 2 - self.ground),

                (self.center[0] + self.port_params["distance_from_center"] + self.port_params["width"] / 2,
                 self.center[1] - self.gap - self.rectang_params[1] / 2 - self.groove_params["height"]))

            result = gdspy.boolean(result, empty_rectangle_for_port, 'not')


        ground = result

        return ground

    def _generate_left_coupler(self):
        left_coupler = gdspy.Rectangle(
            (self.center[0] - self.couplers_distance_from_center - self.couplers_width,
             self.center[1] - self.gap - self.rectang_params[1] / 2 - self.couplers_height),

            (self.center[0] - self.couplers_distance_from_center,
             self.center[1] - self.gap - self.rectang_params[1] / 2))

        mini_rectangle = gdspy.Rectangle(
            (self.center[0] - self.port_params["distance_from_center"] - self.left_cpw_params["w"] / 2,
             self.center[1] - self.gap - self.rectang_params[1] / 2 - self.groove_params["height"]),

            (self.center[0] - self.port_params["distance_from_center"] + self.left_cpw_params["w"] / 2,
             self.center[1] - self.gap - self.couplers_height))

        result = gdspy.boolean(left_coupler, mini_rectangle, 'or')

        return result

    def _generate_right_coupler(self):
        right_coupler = gdspy.Rectangle(
            (self.center[0] + self.couplers_distance_from_center,
             self.center[1] - self.gap - self.rectang_params[1] / 2 - self.couplers_height),

            (self.center[0] + self.couplers_distance_from_center + self.couplers_width,
             self.center[1] - self.gap - self.rectang_params[1] / 2))

        mini_rectangle = gdspy.Rectangle(
            (self.center[0] + self.port_params["distance_from_center"] - self.right_cpw_params["w"] / 2,
             self.center[1] - self.gap - self.rectang_params[1] / 2 - self.groove_params["height"]),

            (self.center[0] + self.port_params["distance_from_center"] + self.right_cpw_params["w"] / 2,
             self.center[1] - self.gap - self.couplers_height))

        result = gdspy.boolean(right_coupler, mini_rectangle, 'or')



        return result

    def _generate_couplers(self):
        #TODO: fix warning referenced before assignment

        # couplers = None

        if self.couplers["left"] is True:
            left_coupler = self._generate_left_coupler()
            if self.couplers["right"] is False:
                return left_coupler

        if self.couplers["right"] is True:
            right_coupler = self._generate_right_coupler()
            if self.couplers["left"] is False:
                return right_coupler

        couplers = gdspy.boolean(left_coupler, right_coupler, 'or')

        return couplers

    def _generate_inner_left_rectangle(self):
        rectangle = gdspy.Rectangle(
                (self.center[0] - self.distance - self.rectang_params[0],
                 self.center[1] - self.rectang_params[1] / 2),

                (self.center[0] - self.distance,
                 self.center[1] + self.rectang_params[1] / 2))

        return rectangle

    def _generate_inner_right_rectangle(self):
        rectangle = gdspy.Rectangle(
                (self.center[0] + self.distance,
                 self.center[1] - self.rectang_params[1] / 2),

                (self.center[0] + self.distance + self.rectang_params[0],
                 self.center[1] + self.rectang_params[1] / 2))

        return rectangle

    def generate_fluxonium(self):

        result = self._generate_ground()
        
        inner_rectangles = [
            self._generate_inner_left_rectangle(),
            self._generate_inner_right_rectangle()
        ]

        result = gdspy.boolean(result, inner_rectangles[0], 'or')
        result = gdspy.boolean(result, inner_rectangles[1], 'or')

        couplers = self._generate_couplers()

        result = gdspy.boolean(result, couplers, 'or')

        return result

    def render(self):
        ## add positive mask
        result = gdspy.boolean(self.generate_fluxonium(),
                                     self.generate_fluxonium(),
                                     'and',
                                     layer=self.layer_configuration.total_layer)
        ## add negative mask
        half_length_x = self.distance + self.rectang_params[0] + self.gap + self.ground
        half_length_y = self.rectang_params[1] / 2 + self.gap + self.ground
        restricted_area = gdspy.Rectangle(
            (self.center[0] - half_length_x, self.center[1] - half_length_y),
            (self.center[0] + half_length_x, self.center[1] + half_length_y),
            layer=self.layer_configuration.restricted_area_layer)
        negative = gdspy.boolean(restricted_area,result,'not',layer=self.layer_configuration.inverted)
        self.set_terminals()
        if self.calculate_capacitance is False:
            return {'positive': result.rotate(self.transformations['rotate'][0], self.transformations['rotate'][1]),
                    'restrict': restricted_area.rotate(self.transformations['rotate'][0], self.transformations['rotate'][1]),
                    'inverted': negative.rotate(self.transformations['rotate'][0], self.transformations['rotate'][1]),
                    'qubit_cap': None}

        else:
            qubit_cap_parts = []

            # add ground
            ground = self._generate_ground()

            qubit_cap_parts.append(gdspy.boolean(ground, ground, 'or', layer=9))
            self.layers.append(9)

            last_step_cap = [ground]
            self.layers.append(self.layer_configuration.total_layer)


            # add inner rectangulars
            inner_left_rectangle = self._generate_inner_left_rectangle()
            qubit_cap_parts.append(gdspy.boolean(inner_left_rectangle, inner_left_rectangle, 'or', layer=10))

            inner_right_rectangle = self._generate_inner_right_rectangle()
            qubit_cap_parts.append(gdspy.boolean(inner_right_rectangle, inner_right_rectangle, 'or', layer=11))

            last_step_cap = [inner_right_rectangle]

            if self.couplers["left"] is True:
                left_coupler = self._generate_left_coupler()
                qubit_cap_parts.append(gdspy.boolean(left_coupler, left_coupler, 'or', layer=12))
                last_step_cap = [left_coupler]

            if self.couplers["right"] is True:
                right_coupler = self._generate_right_coupler()
                qubit_cap_parts.append(gdspy.boolean(right_coupler, right_coupler, 'or', layer=13))
                last_step_cap = [right_coupler]

            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # qubit_cap_parts.append(gdspy.boolean(result, last_step_cap, 'not'))

            return {'positive': result.rotate(self.transformations['rotate'][0], self.transformations['rotate'][1]),
                    'restrict': restricted_area.rotate(self.transformations['rotate'][0], self.transformations['rotate'][1]),
                    'inverted': negative.rotate(self.transformations['rotate'][0], self.transformations['rotate'][1]),
                    'qubit_cap': qubit_cap_parts}

    def set_terminals(self):
        if self.couplers["left"] is True:
            left_coupler_connection = (self.center[0] - self.port_params["distance_from_center"],
                                       self.center[1] - self.gap - self.rectang_params[1] / 2 - self.groove_params["height"])
            left_coupler_connection = rotate_point(deepcopy(left_coupler_connection), self.transformations['rotate'][0], self.transformations['rotate'][1])
            coupler_phi = np.pi/2 + self.transformations['rotate'][0]
            self.terminals['left_coupler'] = DesignTerminal(tuple(left_coupler_connection), coupler_phi, g=self.left_cpw_params["g"], s=self.left_cpw_params["s"], w=self.left_cpw_params["w"], type='cpw')


        if self.couplers["right"] is True:
            right_coupler_connection = (self.center[0] + self.port_params["distance_from_center"],
                                        self.center[1] - self.gap - self.rectang_params[1] / 2 - self.groove_params["height"])
            right_coupler_connection = rotate_point(deepcopy(right_coupler_connection), self.transformations['rotate'][0], self.transformations['rotate'][1])

            coupler_phi = np.pi/2 + self.transformations['rotate'][0]
            self.terminals['right_coupler'] = DesignTerminal(tuple(right_coupler_connection), coupler_phi, g=self.right_cpw_params["g"], s=self.right_cpw_params["s"], w=self.right_cpw_params["w"], type='cpw')
        
        
        coupler_phi = np.pi/2 + self.transformations['rotate'][0]

        qubit_cpw_connection = (self.center[0], 
                                self.center[1]+ self.gap + self.rectang_params[1] / 2)

        self.terminals['qubit'] = DesignTerminal(tuple(qubit_cpw_connection), 2*np.pi - coupler_phi, g=self.qubit_cpw_params["g"], s=self.qubit_cpw_params["s"], w=self.qubit_cpw_params["w"], type='cpw')


    def get_terminals(self):
        return self.terminals

    # def add_to_tls(self, tls_instance: tlsim.TLSystem, terminal_mapping: dict, track_changes: bool = True,
    #                    cutoff: float = np.inf, epsilon: float = 11.45) -> list:
    #     return list(None)

    def fill_capacitance_matrix(self, matrix):
        if matrix.shape != (4, 4):
            raise ValueError("Incorrect capacitance matrix shape ")
        else:
            self.capacitance_matrix = matrix
            
        self.C = {'C_lg': self.capacitance_matrix['l']['l'] - self.capacitance_matrix['l']['1'],
                      'C_rg': self.capacitance_matrix['r']['r'] - self.capacitance_matrix['r']['2'],
                      'C_l1': -self.capacitance_matrix['l']['1'],
                      'C_r2': -self.capacitance_matrix['r']['2'],
                      'qubit': 0.5} # TODO: fix this constant 
        
    
    def _fill_C(self):
        if self.capacitance_matrix is None:
            raise ValueError("Capacitance matrix not filled, use 'fill_capacitance_matrix' method in notebook")
        else: 
            self.C = {'C_lg': self.capacitance_matrix['l']['l'] - self.capacitance_matrix['l']['1'],
                      'C_rg': self.capacitance_matrix['r']['r'] - self.capacitance_matrix['r']['2'],
                      'C_l1': -self.capacitance_matrix['l']['1'],
                      'C_r2': -self.capacitance_matrix['r']['2'],
                      'qubit': 0.5} # TODO: fix this constant 


    def add_to_tls(self, tls_instance: tlsim.TLSystem, terminal_mapping: dict, track_changes: bool = True,
                   cutoff: float = np.inf, epsilon: float = 11.45) -> list:

        #          2                  1
        #          |                  |
        #          |                  |
        #         ---       c        ---
        #         ---   ---||---     ---
        #          |    |       |     |
        # 0        |    |  jj   |     |        0
        # g----||----------|x|------------||---g
        #          r    |       |     l
        #               x       x
        #               x       x jj_chain
        #               x       x
        #               x   m   x
        #         qubit ---mmm--- inductance
        #               |       |
        #               /\      g 0
        #              |  |    
        #               ---      

        scal_C = 1e-15

        # TODO: add self.JJ_params

        # self.terminals = {"qubit": None,
        #                   "left_coupler": None,
        #                   "right_coupler": None}

        node_1 = terminal_mapping['left_coupler']
        node_2 = terminal_mapping['right_coupler']
        node_qubit = terminal_mapping['qubit']
        node_r = 'node_r'
        node_l = 'node_l'
        node_g = 'node_g'


        jj = tlsim.JosephsonJunction(self.JJ_params['ic'] * hbar / (2 * e), name=self.name + ' jj')
        jj_chain1 = tlsim.Inductor(self.JJ_params['lm'], name=self.name + ' qubit-l1')
        jj_chain2 = tlsim.Inductor(self.JJ_params['lm'], name=self.name + ' qubit-l2')
        m = tlsim.Inductor(3.3e-12, name=self.name + ' qubit-g')
        c = tlsim.Capacitor(c=self.C['qubit'] * scal_C, name=self.name + ' lr')

        tls_instance.add_element(jj, [node_r, node_l])
        tls_instance.add_element(c, [node_r, node_l])
        tls_instance.add_element(jj_chain1, [node_qubit, node_r])
        tls_instance.add_element(m, [node_qubit, node_g])
        tls_instance.add_element(jj_chain2, [node_g, node_l])


        c_lg = tlsim.Capacitor(c=self.C['C_lg'] * scal_C, name=self.name + ' lg')
        c_l1 = tlsim.Capacitor(c=self.C['C_l1'] * scal_C, name=self.name + ' l1')
        c_rg = tlsim.Capacitor(c=self.C['C_rg'] * scal_C, name=self.name + ' rg')
        c_r2 = tlsim.Capacitor(c=self.C['C_r2'] * scal_C, name=self.name + ' r2')

        tls_instance.add_element(c_lg, [node_l, node_g])
        tls_instance.add_element(c_rg, [node_r, node_g])
        tls_instance.add_element(c_l1, [node_l, node_1])
        tls_instance.add_element(c_r2, [node_r, node_2])

        # GND = tlsim.Short()
        # tls_instance.add_element(GND, [node_g])

        # line_end = tlsim.Port(50, name='flux line')
        # tls_instance.add_element(line_end, [node_qubit])
        # current = 0.13e-3
        # line_end.idc = current


        cache = [jj, jj_chain1, jj_chain2, m, c, c_lg, c_l1, c_rg, c_r2]
        #self.tls_cache = cache

        # mut_cap = []
        # cap_g = []

        # for id, coupler in enumerate(self.couplers):
        #     if coupler.coupler_type == 'coupler':
        #         c0 = tlsim.Capacitor(c=self.C['coupler' + str(id)][1] * scal_C,
        #                              name=self.name + ' qubit-coupler' + str(id))
        #         c0g = tlsim.Capacitor(c=self.C['coupler' + str(id)][0] * scal_C,
        #                               name=self.name + ' coupler' + str(id) + '-ground')
        #         tls_instance.add_element(c0, [terminal_mapping['qubit'], terminal_mapping['coupler' + str(id)]])
        #         tls_instance.add_element(c0g, [terminal_mapping['coupler' + str(id)], 0])
        #         mut_cap.append(c0)
        #         cap_g.append(c0g)
            # elif coupler.coupler_type =='grounded':
            #     tls_instance.add_element(tlsim.Short(), [terminal_mapping['flux line'], 0])

        if track_changes:
            self.tls_cache.append(cache)
        return cache 



# TODO: check whether we need it
class FluxoniumCoupler:
    def __init__(self, coupler_type=None, w=None, g=None,s=None):
        self.w = w
        self.g = g
        self.s = s
        self.coupler_type = coupler_type
        self.connection = None
        self.result_coupler = None



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



































