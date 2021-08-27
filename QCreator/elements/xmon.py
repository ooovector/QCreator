from .core import DesignElement, LayerConfiguration, DesignTerminal
# from .. import conformal_mapping as cm
from .. import transmission_line_simulator as tlsim
import gdspy
import numpy as np
from typing import List, Tuple, Mapping, Dict
# from . import squid3JJ
# from copy import deepcopy


class Xmon(DesignElement):

    def __init__(self, name: str, center: Tuple[float, float],
                 length: float, width_gap: float, center_width: float,
                 crab_position: Tuple[str, str, str, str], crab_shoulder: float,
                 crab_thickness: float, crab_terminals: Dict,ground_thickness: float, delete_ground: str,
                 jj_position: str, jj_params1: Dict, jj_params2: Dict, aux_jj_params : Dict,
                 layer_configuration: LayerConfiguration, hole_in_squid_pad=True):
        super().__init__(type='qubit', name=name)
        # qubit parameters
        self.center = center
        self.length = length
        self.w = width_gap
        self.s = center_width
        self.a = crab_shoulder
        self.b = crab_thickness
        self.pos = crab_position
        self.g = ground_thickness
        self.jjpos = jj_position
        self.delg = delete_ground
        self.ct = crab_terminals
        self.jgeom = jj_params1
        self.jj = jj_params2
        self.auxjj = aux_jj_params
        # layers
        self.layer_configuration = layer_configuration
        self.hole_in_squid_pad = hole_in_squid_pad

        self.terminals = {'crab_left': None,
                          'crab_right': None,
                          'crab_up': None,
                          'crab_down': None,
                          'flux': None,
                          'qubit': None}
        self.couplers = {}
        self.tls_cache = []
        self.M = 12e-12
        self.C = {'crab_left': None,
                  'crab_right': None,
                  'crab_up': None,
                  'crab_down': None,
                  'qubit': None}
        self.layers = []

    def render(self):
        """
        I've no clue if it is right or just strait up dumb BUT:
            Every element of the Xmon here is drawn by subtracting two rectangles from each other:
                Everything that is marked *_out or *_o means that it's the outer rectangle
                Same goes for *_in or *_i -- they're inner ones

            I figured this strange (read stupid) way of drawing parts would be easier than making a bunch of complex polygons.


        """
        qubit_cap_parts = []
        # draw center cross:
        # auxillary:
        cross_hor = gdspy.Rectangle((self.center[0] - self.s / 2 - self.length, self.center[1] - self.s / 2),
                                    (self.center[0] + self.s / 2 + self.length, self.center[1] + self.s / 2))

        cross_ver = gdspy.Rectangle((self.center[0] - self.s / 2, self.center[1] - self.s / 2 - self.length),
                                    (self.center[0] + self.s / 2, self.center[1] + self.s / 2 + self.length))

        # the cross itself
        cross = gdspy.boolean(cross_hor, cross_ver, "or", layer=self.layer_configuration.total_layer)
        cross_core = cross
        qubit_cap_parts.append(gdspy.boolean(cross_core, cross_core, 'or', layer=9))
        self.layers.append(9)
        # draw ground cross
        # auxillary
        cross_gnd_hor_out = gdspy.Rectangle((self.center[0] - self.s / 2 - self.length - self.w - self.g,
                                             self.center[1] - self.s / 2 - self.w - self.g),
                                            (self.center[0] + self.s / 2 + self.length + self.w + self.g,
                                             self.center[1] + self.s / 2 + self.w + self.g))
        cross_gnd_hor_in = gdspy.Rectangle(
            (self.center[0] - self.s / 2 - self.length - self.w, self.center[1] - self.s / 2 - self.w),
            (self.center[0] + self.s / 2 + self.length + self.w, self.center[1] + self.s / 2 + self.w))
        cross_gnd_ver_out = gdspy.Rectangle((self.center[0] - self.s / 2 - self.w - self.g,
                                             self.center[1] - self.s / 2 - self.length - self.w - self.g),
                                            (self.center[0] + self.s / 2 + self.w + self.g,
                                             self.center[1] + self.s / 2 + self.length + self.w + self.g))
        cross_gnd_ver_in = gdspy.Rectangle(
            (self.center[0] - self.s / 2 - self.w, self.center[1] - self.s / 2 - self.length - self.w),
            (self.center[0] + self.s / 2 + self.w, self.center[1] + self.s / 2 + self.length + self.w))

        cross_gnd_out = gdspy.boolean(cross_gnd_hor_out, cross_gnd_ver_out, "or",
                                      layer=self.layer_configuration.total_layer)
        cross_restrict = cross_gnd_out
        cross_gnd_in = gdspy.boolean(cross_gnd_hor_in, cross_gnd_ver_in, "or",
                                     layer=self.layer_configuration.total_layer)

        # the cross ground (things will be added or deleted from this later on

        cross_gnd = gdspy.boolean(cross_gnd_out, cross_gnd_in, "not", layer=self.layer_configuration.total_layer)

        # the following draws a crab around the upper end of the cross
        for word in self.pos:
            crab_out = gdspy.Rectangle((self.center[0] - self.s / 2 - self.w - self.g - self.w - self.b,
                                        self.center[1] + self.s / 2 + self.length - self.a),
                                       (self.center[0] + self.s / 2 + self.w + self.g + self.w + self.b,
                                        self.center[1] + self.s / 2 + self.length + self.g + self.w + self.w + self.b))
            crab_in = gdspy.Rectangle((self.center[0] - self.s / 2 - self.w - self.g - self.w,
                                       self.center[1] + self.s / 2 + self.length - self.a),
                                      (self.center[0] + self.s / 2 + self.w + self.g + self.w,
                                       self.center[1] + self.s / 2 + self.length + self.w + self.g + self.w))
            crab_u = gdspy.boolean(crab_out, crab_in, "not", layer=self.layer_configuration.total_layer)

            crab_cap_u = crab_u

            ognd_u_out = gdspy.Rectangle((self.center[
                                              0] - self.s / 2 - self.w - self.g - self.w - self.b - self.w - self.g,
                                          self.center[1] + self.s / 2 + self.length - self.a - self.w - self.g),
                                         (self.center[
                                              0] + self.s / 2 + self.w + self.g + self.w + self.b + self.w + self.g,
                                          self.center[
                                              1] + self.s / 2 + self.length + self.g + self.w + self.w + self.b + self.w + self.g))
            orestrict = ognd_u_out
            ognd_u_in = gdspy.Rectangle((self.center[0] - self.s / 2 - self.w - self.g - self.w - self.b - self.w,
                                         self.center[1] + self.s / 2 + self.length - self.a - self.w),
                                        (self.center[0] + self.s / 2 + self.w + self.g + self.w + self.b + self.w,
                                         self.center[
                                             1] + self.s / 2 + self.length + self.w + self.g + self.w + self.b + self.w))
            aux = gdspy.Rectangle((self.center[0] - self.s / 2 - self.w,
                                   self.center[1] + self.s / 2 + self.length - self.a - self.w - self.g),
                                  (self.center[0] + self.s / 2 + self.w,
                                   self.center[1] + self.s / 2 + self.length - self.a - self.w))
            ognd_u_in = gdspy.boolean(ognd_u_in, aux, "or", layer=self.layer_configuration.total_layer)
            ognd_u = gdspy.boolean(ognd_u_out, ognd_u_in, "not", layer=self.layer_configuration.total_layer)
            crab_u = gdspy.boolean(crab_u, ognd_u, "or", layer=self.layer_configuration.total_layer)

            crab_connection = (
            self.center[0], self.center[1] + self.s / 2 + self.length + self.w + self.g + self.w + self.b + self.w)

            # rotate the crab depending on it's respective position

            if word == 'right':
                delete = gdspy.Rectangle((self.center[0] - self.ct['right_w'] / 2 - self.ct['right_s'], self.center[
                    1] + self.s / 2 + self.length + self.w + self.g + self.w + self.b + self.w),
                                         (self.center[0] + self.ct['right_w'] / 2 + self.ct['right_s'], self.center[
                                             1] + self.s / 2 + self.length + self.w + self.g + self.w + self.b + self.w + self.g))
                crab_u = gdspy.boolean(crab_u, delete, 'not', layer=self.layer_configuration.total_layer)
                crab_r = crab_u.rotate(-np.pi / 2, self.center)

                connect = gdspy.Rectangle((self.center[0] - self.ct['right_w'] / 2, self.center[
                    1] + self.s / 2 + self.length + self.w + self.g + self.w + self.b),
                                          (self.center[0] + self.ct['right_w'] / 2, self.center[
                                              1] + self.s / 2 + self.length + self.w + self.g + self.w + self.b + self.w))
                connect = connect.rotate(-np.pi / 2, self.center)
                crab_r = gdspy.boolean(crab_r, connect, "or", layer=self.layer_configuration.total_layer)

                cross = gdspy.boolean(cross, crab_r, 'or', layer=self.layer_configuration.total_layer)
                orestrict = orestrict.rotate(-np.pi / 2, self.center)
                cross_restrict = gdspy.boolean(cross_restrict, orestrict, "or",
                                               layer=self.layer_configuration.restricted_area_layer)
                crab_connection = rotate_point(crab_connection, -np.pi / 2, self.center)
                crab_connection_angle = -np.pi
                self.terminals['crab_right'] = DesignTerminal(crab_connection, crab_connection_angle,
                                                              g=self.ct['right_g'], s=self.ct['right_s'],
                                                              w=self.ct['right_w'], type='cpw')

                crab_cap_r = crab_cap_u.rotate(-np.pi / 2, self.center)
                crab_cap_r = gdspy.boolean(crab_cap_r, connect, "or")
                qubit_cap_parts.append(gdspy.boolean(crab_cap_r, crab_cap_r, 'or', layer=10))
                self.layers.append(10)

            if word == 'up':
                delete = gdspy.Rectangle((self.center[0] - self.ct['up_w'] / 2 - self.ct['up_s'], self.center[
                    1] + self.s / 2 + self.length + self.w + self.g + self.w + self.b + self.w),
                                         (self.center[0] + self.ct['up_w'] / 2 + self.ct['up_s'], self.center[
                                             1] + self.s / 2 + self.length + self.w + self.g + self.w + self.b + self.w + self.g))
                crab_u = gdspy.boolean(crab_u, delete, 'not', layer=self.layer_configuration.total_layer)

                cross_restrict = gdspy.boolean(cross_restrict, orestrict, "or",
                                               layer=self.layer_configuration.restricted_area_layer)

                connect = gdspy.Rectangle((self.center[0] - self.ct['up_w'] / 2, self.center[
                    1] + self.s / 2 + self.length + self.w + self.g + self.w + self.b),
                                          (self.center[0] + self.ct['up_w'] / 2, self.center[
                                              1] + self.s / 2 + self.length + self.w + self.g + self.w + self.b + self.w))
                crab_u = gdspy.boolean(crab_u, connect, "or", layer=self.layer_configuration.total_layer)
                crab_cap_u = gdspy.boolean(crab_cap_u, connect, "or")

                cross = gdspy.boolean(cross, crab_u, 'or', layer=self.layer_configuration.total_layer)
                crab_connection_angle = -np.pi / 2
                self.terminals['crab_up'] = DesignTerminal(crab_connection, crab_connection_angle, g=self.ct['up_g'],
                                                           s=self.ct['up_s'], w=self.ct['up_w'], type='cpw')
                qubit_cap_parts.append(gdspy.boolean(crab_cap_u, crab_cap_u, 'or', layer=11))
                self.layers.append(11)

            if word == 'down':
                delete = gdspy.Rectangle((self.center[0] - self.ct['down_w'] / 2 - self.ct['down_s'], self.center[
                    1] + self.s / 2 + self.length + self.w + self.g + self.w + self.b + self.w),
                                         (self.center[0] + self.ct['down_w'] / 2 + self.ct['down_s'], self.center[
                                             1] + self.s / 2 + self.length + self.w + self.g + self.w + self.b + self.w + self.g))
                crab_u = gdspy.boolean(crab_u, delete, 'not', layer=self.layer_configuration.total_layer)

                crab_d = crab_u.rotate(np.pi, self.center)

                connect = gdspy.Rectangle((self.center[0] - self.ct['down_w'] / 2, self.center[
                    1] + self.s / 2 + self.length + self.w + self.g + self.w + self.b),
                                          (self.center[0] + self.ct['down_w'] / 2, self.center[
                                              1] + self.s / 2 + self.length + self.w + self.g + self.w + self.b + self.w))
                connect = connect.rotate(np.pi, self.center)
                crab_d = gdspy.boolean(crab_d, connect, "or", layer=self.layer_configuration.total_layer)

                cross = gdspy.boolean(cross, crab_d, 'or', layer=self.layer_configuration.total_layer)
                orestrict = orestrict.rotate(np.pi, self.center)
                cross_restrict = gdspy.boolean(cross_restrict, orestrict, "or",
                                               layer=self.layer_configuration.restricted_area_layer)
                crab_connection = rotate_point(crab_connection, np.pi, self.center)
                crab_connection_angle = np.pi / 2
                self.terminals['crab_down'] = DesignTerminal(crab_connection, crab_connection_angle,
                                                             g=self.ct['down_g'], s=self.ct['down_s'],
                                                             w=self.ct['down_w'], type='cpw')

                crab_cap_d = crab_cap_u.rotate(np.pi, self.center)
                crab_cap_d = gdspy.boolean(crab_cap_d, connect, "or")
                qubit_cap_parts.append(gdspy.boolean(crab_cap_d, crab_cap_d, 'or', layer=12))
                self.layers.append(12)

            if word == 'left':
                delete = gdspy.Rectangle((self.center[0] - self.ct['left_w'] / 2 - self.ct['left_s'], self.center[
                    1] + self.s / 2 + self.length + self.w + self.g + self.w + self.b + self.w),
                                         (self.center[0] + self.ct['left_w'] / 2 + self.ct['left_s'], self.center[
                                             1] + self.s / 2 + self.length + self.w + self.g + self.w + self.b + self.w + self.g))
                crab_u = gdspy.boolean(crab_u, delete, 'not', layer=self.layer_configuration.total_layer)

                crab_l = crab_u.rotate(np.pi / 2, self.center)

                connect = gdspy.Rectangle((self.center[0] - self.ct['left_w'] / 2, self.center[
                    1] + self.s / 2 + self.length + self.w + self.g + self.w + self.b),
                                          (self.center[0] + self.ct['left_w'] / 2, self.center[
                                              1] + self.s / 2 + self.length + self.w + self.g + self.w + self.b + self.w))
                connect = connect.rotate(np.pi / 2, self.center)
                crab_l = gdspy.boolean(crab_l, connect, "or", layer=self.layer_configuration.total_layer)

                cross = gdspy.boolean(cross, crab_l, 'or', layer=self.layer_configuration.total_layer)
                orestrict.rotate(np.pi / 2, self.center)
                cross_restrict = gdspy.boolean(cross_restrict, orestrict, "or",
                                               layer=self.layer_configuration.restricted_area_layer)
                crab_connection = rotate_point(crab_connection, np.pi / 2, self.center)
                crab_connection_angle = 0
                self.terminals['crab_left'] = DesignTerminal(crab_connection, crab_connection_angle,
                                                             g=self.ct['left_g'], s=self.ct['left_s'],
                                                             w=self.ct['left_w'], type='cpw')

                crab_cap_l = crab_cap_u.rotate(np.pi / 2, self.center)
                crab_cap_l = gdspy.boolean(crab_cap_l, connect, "or")
                qubit_cap_parts.append(gdspy.boolean(crab_cap_l, crab_cap_l, 'or', layer=13))
                self.layers.append(13)

        # if deleting ground between the crab and the cross is needed:

        if self.delg != '':
            cross_gnd = self.delete_ground(cross_gnd)

        # create a pad for jj
        jpad, cross_gnd, jrestrict, cross_restrict = self.create_jpad(cross_gnd, cross_restrict)
        # if self.jj['type'] == 2:
        #         if self.jgeom['bandages'] == True:
        #             jj, band = self.generate_jj()
        #         else:
        #             jj = self.generate_jj()
        #             band = None
        # else:
        #     jj = self.generate_3jj()
        #     band = None
        cross_gnd = gdspy.boolean(cross_gnd, jpad, "or", layer=self.layer_configuration.total_layer)

        # create a fluxline near jj pad
        fgnd, frestrict = self.generate_fluxline()
        cross_gnd = gdspy.boolean(cross_gnd, fgnd, "or", layer=self.layer_configuration.total_layer)

        # parts of the restricted area
        cross_restrict = gdspy.boolean(cross_restrict, frestrict, "or",
                                       layer=self.layer_configuration.restricted_area_layer)
        cross_restrict = gdspy.boolean(cross_restrict, jrestrict, "or",
                                       layer=self.layer_configuration.restricted_area_layer)

        result = gdspy.boolean(cross_gnd, cross, "or", layer=self.layer_configuration.total_layer)

        gnd_cap = result
        for word in self.pos:
            if word == 'down':
                gnd_cap = gdspy.boolean(gnd_cap, crab_cap_d, 'not', layer=self.layer_configuration.total_layer)
            elif word == 'left':
                gnd_cap = gdspy.boolean(gnd_cap, crab_cap_l, 'not', layer=self.layer_configuration.total_layer)
            elif word == 'right':
                gnd_cap = gdspy.boolean(gnd_cap, crab_cap_r, 'not', layer=self.layer_configuration.total_layer)
            elif word == 'up':
                gnd_cap = gdspy.boolean(gnd_cap, crab_cap_u, 'not', layer=self.layer_configuration.total_layer)

        gnd_cap = gdspy.boolean(gnd_cap, cross_core, 'not', layer=self.layer_configuration.total_layer)
        qubit_cap_parts.append(gdspy.boolean(gnd_cap, gnd_cap, 'or'))

        if self.jj['type'] == 3:
            self.terminals['squid_intermediate'] = None


        if self.jj['type'] == 2:
                if self.jgeom['bandages'] == True:
                    jj, band = self.generate_jj()
                    return {'positive': result,
                             'qubit': result,
                             'restrict': cross_restrict,
                             'JJ': jj,
                             'qubit_cap': qubit_cap_parts,
                             'bandages': band
                             }
                else:
                    jj = self.generate_jj()
                    return {'positive': result,
                            'qubit': result,
                            'restrict': cross_restrict,
                            'JJ': jj,
                            'qubit_cap': qubit_cap_parts,
                            }

        else:
            jj, band = self.generate_3jj()
            return {'positive': result,
                    'qubit': result,
                    'restrict': cross_restrict,
                    'JJ': jj,
                    'qubit_cap': qubit_cap_parts,
                    'bandages': band
                    }

        # result_dict = {'positive': result,
        #             'qubit': result,
        #             'restrict': cross_restrict,
        #             'JJ': jj,
        #             'qubit_cap': qubit_cap_parts,
        #             }
        #
        # if self.jgeom['bandages'] == True:
        #      result_dict = {'positive': result,
        #                     'qubit': result,
        #                     'restrict': cross_restrict,
        #                     'JJ': jj,
        #                     'qubit_cap': qubit_cap_parts,
        #                     'bandages': band
        #                     }
        #
        # return result_dict

    def create_jpad(self, cross_gnd, cross_restrict):
        # variables
        a = self.jgeom['gwidth']
        b = self.jgeom['gheight']
        k = self.jgeom['iwidth']
        h = self.jgeom['iheight']
        t = self.jgeom['ithick']
        op = self.jgeom['iopen']

        ignd_u = gdspy.Rectangle((self.center[0] - self.s/2 - self.w - self.g, self.center[1] + self.s/2 + self.length),
                                 (self.center[0] + self.s/2 + self.w + self.g, self.center[1] + self.s/2 + self.length + self.w + self.g))

        jgnd_o = gdspy.Rectangle((self.center[0] - a/2 - self.g, self.center[1] - self.s/2 - self.length - b - self.g),
                                 (self.center[0] + a/2 + self.g, self.center[1] - self.s/2 - self.length + self.g))
        jrestrict = jgnd_o
        jgnd_i = gdspy.Rectangle((self.center[0] - a/2, self.center[1] - self.s/2 - self.length - b),
                                 (self.center[0] + a/2, self.center[1] - self.s/2 - self.length))
        aux_2 = gdspy.Rectangle((self.center[0] - self.s/2 - self.w, self.center[1] - self.s/2 - self.length),
                                (self.center[0] + self.s/2 + self.w, self.center[1] - self.s/2 - self.length + self.g))
        jgnd_o = gdspy.boolean(jgnd_o, aux_2, "not", layer=self.layer_configuration.total_layer)
        jgnd = gdspy.boolean(jgnd_o, jgnd_i, "not", layer=self.layer_configuration.total_layer)

        jpad_o = gdspy.Rectangle((self.center[0] - k/2, self.center[1] - self.s/2 - self.length - b),
                                 (self.center[0] + k/2, self.center[1] - self.s/2 - self.length - (b - h - t)))
        jpad_i = gdspy.Rectangle((self.center[0] - k/2 + t, self.center[1] - self.s/2 - self.length - b),
                                 (self.center[0] + k/2 - t, self.center[1] - self.s/2 - self.length - (b - h)))
        jpad = gdspy.boolean(jpad_o, jpad_i, "not", layer=self.layer_configuration.total_layer)

        # aux_3 = gdspy.Rectangle((self.center[0] - self.s/2 - self.w, self.center[1] - self.s/2 - self.length - (b - h)),
        #                         (self.center[0] + self.s/2 + self.w, self.center[1] - self.s/2 - self.length - (b - h - t)))
        aux_3 = gdspy.Rectangle((self.center[0] - op/2, self.center[1] - self.s/2 - self.length - (b - h)),
                                (self.center[0] + op / 2, self.center[1] - self.s / 2 - self.length - (b - h - t)))

        jpad = gdspy.boolean(jpad, aux_3, "not", layer=self.layer_configuration.total_layer)

        jpad = gdspy.boolean(jpad, jgnd, "or", layer=self.layer_configuration.total_layer)

        aux_4 = gdspy.Rectangle((self.center[0] - a/2 - self.g, self.center[1] - self.s/2 - self.length - b - self.g),
                                (self.center[0] + a/2 + self.g, self.center[1] - self.s/2 - self.length - b))
        jpad = gdspy.boolean(jpad, aux_4, "not", layer=self.layer_configuration.total_layer)


        if self.jjpos == 'up':
            cross_gnd = gdspy.boolean(cross_gnd, ignd_u, "not", layer=self.layer_configuration.total_layer)
            jpad = jpad.rotate(np.pi, self.center)
            jrestrict = jrestrict.rotate(np.pi, self.center)
            cross_restrict = gdspy.boolean(cross_restrict, ignd_u, "not", layer=self.layer_configuration.restricted_area_layer)
        elif self.jjpos == 'down':
            ignd_d = ignd_u.rotate(np.pi, self.center)
            cross_gnd = gdspy.boolean(cross_gnd, ignd_d, "not", layer=self.layer_configuration.total_layer)

            cross_restrict = gdspy.boolean(cross_restrict, ignd_d, "not", layer=self.layer_configuration.restricted_area_layer)
        elif self.jjpos == 'left':
            ignd_l = ignd_u.rotate(np.pi/2, self.center)
            cross_gnd = gdspy.boolean(cross_gnd, ignd_l, "not", layer=self.layer_configuration.total_layer)
            jpad = jpad.rotate(-np.pi/2, self.center)
            jrestrict = jrestrict.rotate(-np.pi/2, self.center)
            cross_restrict = gdspy.boolean(cross_restrict, ignd_l, "not", layer=self.layer_configuration.restricted_area_layer)
        elif self.jjpos == 'right':
            ignd_r = ignd_u.rotate(-np.pi/2, self.center)
            cross_gnd = gdspy.boolean(cross_gnd, ignd_r, "not", layer=self.layer_configuration.total_layer)
            jpad = jpad.rotate(np.pi/2, self.center)
            jrestrict = jrestrict.rotate(np.pi/2, self.center)
            cross_restrict = gdspy.boolean(cross_restrict, ignd_r, "not", layer=self.layer_configuration.restricted_area_layer)
        return jpad, cross_gnd, jrestrict, cross_restrict


    def delete_ground(self, cross_gnd):
        gnd_out = gdspy.Rectangle((self.center[0] - self.s/2 - self.w - self.g, self.center[1] + self.s/2 + self.length - self.a - self.w),
                                  (self.center[0] + self.s/2 + self.w + self.g, self.center[1] + self.s/2 + self.length + self.g + self.w))
        gnd_in = gdspy.Rectangle((self.center[0] - self.s/2 - self.w, self.center[1] + self.s/2 + self.length - self.a - self.w),
                                 (self.center[0] + self.s/2 + self.w, self.center[1] + self.s/2 + self.length + self.w))
        gnd = gdspy.boolean(gnd_out, gnd_in, "not", layer=self.layer_configuration.total_layer)
        if self.delg == 'down':
            gnd = gnd.rotate(np.pi, self.center)
        elif self.delg == 'left':
            gnd = gnd.rotate(np.pi/2, self.center)
        elif self.delg == 'right':
            gnd = gnd.rotate(-np.pi/2, self.center)

        cross_gnd = gdspy.boolean(cross_gnd, gnd, "not", layer=self.layer_configuration.total_layer)
        return cross_gnd

    def generate_fluxline(self):
        # variables
        a = self.jgeom['gwidth']
        b = self.jgeom['gheight']
        h1 = self.jgeom['fheight1']
        h2 = self.jgeom['fheight2']
        d = self.jgeom['hdist']
        sh = self.jgeom['fshoulder']
        f = self.jgeom['fcore']
        fw = self.jgeom['fgap']
        gt = self.jgeom['gter']

        fgnd_1 = gdspy.Rectangle((self.center[0] - a/2 - self.g, self.center[1] - self.s/2 - self.length - b - h1),
                                 (self.center[0] + a/2 + self.g, self.center[1] - self.s/2 - self.length - b))
        fgnd_2 = gdspy.Rectangle((self.center[0] - self.s/2 - self.w - self.g, self.center[1] - self.s/2 - self.length - h1 - h2 - b),
                                 (self.center[0] + self.s/2 + self.w + self.g, self.center[1] - self.s/2 - self.length - h1 - b))

        fgnd = gdspy.boolean(fgnd_1, fgnd_2, "or", layer=self.layer_configuration.total_layer)
        frestrict = fgnd
        fhor = gdspy.Rectangle((self.center[0] - a/2, self.center[1] - self.s/2 - self.length - b - d - fw),
                               (self.center[0] + a/2, self.center[1] - self.s/2 - self.length - b - d))
        fgnd = gdspy.boolean(fgnd, fhor, "not", layer=self.layer_configuration.total_layer)

        fver1 = gdspy.Rectangle((self.center[0] - f/2 - fw, self.center[1] - self.s/2 - self.length - b - d - h2 - h1),
                                (self.center[0] - f/2, self.center[1] - self.s/2 - self.length - b - d - fw - f))
        fver1_turn = gdspy.Rectangle((self.center[0] - f/2 - sh, self.center[1] - self.s/2 - self.length - b - d - 2*fw - f),
                                     (self.center[0] - f/2, self.center[1] - self.s/2 - self.length - b - d - fw - f))
        fver1 = gdspy.boolean(fver1, fver1_turn, "or", layer=self.layer_configuration.total_layer)

        fver2 = gdspy.Rectangle((self.center[0] + f/2 + fw, self.center[1] - self.s/2 - self.length - b - d - h2 - h1),
                                (self.center[0] + f/2, self.center[1] - self.s/2 - self.length - b - d))
        fver = gdspy.boolean(fver1, fver2, "or", layer=self.layer_configuration.total_layer)
        fgnd = gdspy.boolean(fgnd, fver, "not", layer=self.layer_configuration.total_layer)
        fluxline_connection = (self.center[0], self.center[1] - self.s/2 - self.length - b - h2 - h1)
        connection_angle = np.pi/2


        if self.jjpos == 'up':
            fgnd = fgnd.rotate(np.pi, self.center)
            fluxline_connection = rotate_point(fluxline_connection, np.pi, self.center)
            frestrict = frestrict.rotate(np.pi, self.center)
            connection_angle = -np.pi/2
        elif self.jjpos == 'left':
            fgnd = fgnd.rotate(-np.pi/2, self.center)
            fluxline_connection = rotate_point(fluxline_connection, -np.pi/2, self.center)
            connection_angle = 0
            frestrict = frestrict.rotate(-np.pi/2, self.center)
        elif self.jjpos == 'right':
            fgnd = fgnd.rotate(np.pi/2, self.center)
            fluxline_connection = rotate_point(fluxline_connection, np.pi/2, self.center)
            connection_angle = np.pi
            frestrict = frestrict.rotate(np.pi/2, self.center)

        self.terminals['flux'] = DesignTerminal(fluxline_connection, connection_angle, g=gt, s=f, w=fw, type='cpw')

        return fgnd, frestrict

    def generate_3jj(self):
        contact_pad_a_outer = 10.5
        contact_pad_b_outer = 6
        self.contact_pad_b_outer = contact_pad_b_outer
        self.contact_pad_a_outer = contact_pad_a_outer
        if self.hole_in_squid_pad==True:
            contact_pad_a_inner = 7.5
            contact_pad_b_inner = 1
        else:
            contact_pad_a_inner = 0
            contact_pad_b_inner = 0

        self._x0 = self.center[0]
        self._y0 = self.center[1]-self.length-self.s/2+ self.contact_pad_b_outer-contact_pad_b_outer/2

        self._parametr1 = 10#Hb

        self.jj1_width = self.jj['up_r_thick']
        self.jj1_height = self.jj['side_r_thick']
        self.jj2_width = self.jj['up_l_thick']
        self.jj2_height = self.jj['side_l_thick']
        self.jj3_width = self.jj['3jj_thick']
        self.jj3_height = self.jj['3jj_height']

        # Add contact pad1
        points0 = [(self._x0 - contact_pad_a_outer / 2, self._y0 - contact_pad_b_outer),
                   (self._x0 - contact_pad_a_outer / 2, self._y0),
                   (self._x0 + contact_pad_a_outer / 2, self._y0),
                   (self._x0 + contact_pad_a_outer / 2, self._y0 - contact_pad_b_outer),

                   (self._x0 - contact_pad_a_outer / 2, self._y0 - contact_pad_b_outer),

                   (self._x0 - contact_pad_a_inner / 2,
                    self._y0 - (contact_pad_b_outer - contact_pad_b_inner) / 2 - contact_pad_b_inner),
                   (self._x0 + contact_pad_a_inner / 2,
                    self._y0 - (contact_pad_b_outer - contact_pad_b_inner) / 2 - contact_pad_b_inner),

                   (self._x0 + contact_pad_a_inner / 2, self._y0 - (contact_pad_b_outer - contact_pad_b_inner) / 2),
                   (self._x0 - contact_pad_a_inner / 2, self._y0 - (contact_pad_b_outer - contact_pad_b_inner) / 2),

                   (self._x0 - contact_pad_a_inner / 2,
                    self._y0 - (contact_pad_b_outer - contact_pad_b_inner) / 2 - contact_pad_b_inner),
                   ]

        x1 = self._x0
        y1 = self._y0 - contact_pad_b_outer

        # parametr1=H_b

        H_a = 0.5  # 1
        H_b = self._parametr1

        L_a = 6  # 10.7
        L_b = 0.75

        h = 0.16  # 0.5

        points1 = [(x1 - 3 * H_a, y1 - H_b / 2),
                   (x1 + H_a, y1 - H_b / 2),
                   (x1 + H_a, y1 - H_b / 2 - self.jj3_height),
                   (x1 - 2 * H_a, y1 - H_b / 2 - self.jj3_height),
                   (x1 - 2 * H_a, y1 - H_b),
                   (x1 - 3 * H_a, y1 - H_b)
                   ]

        points1_1 = [(x1 - H_a, y1),
                     (x1 - H_a / 4, y1),
                     (x1 - H_a / 4, y1 - H_b / 3),
                     (x1 - H_a + self.jj3_width, y1 - H_b / 3),
                     (x1 - H_a + self.jj3_width, y1 - H_b / 2 + h),
                     (x1 - H_a, y1 - H_b / 2 + h)
                     ]

        x2 = x1
        y2 = y1 - H_b

        points2 = [(x2 - L_a / 2, y2),
                   (x2 + L_a / 2, y2),
                   (x2 + L_a / 2, y2 - L_b),
                   (x2 - L_a / 2, y2 - L_b)
                   ]

        H1_a = self.jj2_width#0.8
        H1_b = 2
        H2_a = self.jj1_width#0.8
        H2_b = 2

        x3 = x2 - L_a / 2 + H1_a / 2
        y3 = y2 - L_b

        x4 = x2 + L_a / 2 - H1_a / 2
        y4 = y2 - L_b

        points3 = [(x3 - H1_a / 2, y3),
                   (x3 + H1_a / 2, y3),
                   (x3 + H1_a / 2, y3 - 2 * H1_b),
                   (x3 - H1_a / 2, y3 - 2 * H1_b)
                   ]

        points4 = [(x4 - H2_a / 2, y4),
                   (x4 + H2_a / 2, y4),
                   (x4 + H2_a / 2, y4 - H2_b),
                   (x4 - H2_a / 2, y4 - H2_b)
                   ]

        x5 = x3 + H1_a / 2
        y5 = y3 - H1_b
        x6 = x4 - H2_a / 2
        y6 = y4 - H2_b

        # parametr2=pad1_a
        # parametr3=pad2_a

        pad1_a = self.jj2_width
        pad1_b = 3
        pad2_a = self.jj1_width
        pad2_b = 3

        points5_for_pad1 = [(x5, y5),
                            (x5, y5 - pad1_b),
                            (x5 - pad1_a, y5 - pad1_b),
                            (x5 - pad1_a, y5)
                            ]

        points6_for_pad2 = [(x6, y6),
                            (x6 + pad2_a, y6),
                            (x6 + pad2_a, y6 - pad2_b),
                            (x6, y6 - pad2_b)
                            ]

        contact_pad1_a_outer = 13
        contact_pad1_b_outer = 6.4
        contact_pad1_a_inner = 12
        contact_pad1_b_inner = 5.8

        x7 = self._x0
        y7 = self._y0 - contact_pad_b_outer - H_b - L_b - H1_b - pad1_b - h - contact_pad1_b_outer

        x8 = self._x0 - contact_pad1_a_inner / 2
        y8 = self._y0 - contact_pad_b_outer - H_b - L_b - H1_b - pad1_b - h

        x9 = self._x0 + contact_pad1_a_inner / 2
        y9 = self._y0 - contact_pad_b_outer - H_b - L_b - H1_b - pad1_b - h

        pad3_a = 4.5  # 2.5
        pad3_b = self.jj2_height

        pad4_a = 4.5  # 2.5
        pad4_b = self.jj1_height

        points8_for_pad3 = [(x8, y8),

                            (x8 + pad3_a, y8),

                            (x8 + pad3_a, y8 - pad3_b),

                            (x8, y8 - pad3_b)]

        points9_for_pad4 = [(x9 - pad4_a, y9),

                            (x9, y9),

                            (x9, y9 - pad4_b),

                            (x9 - pad4_a, y9 - pad4_b)]

        delta = 6

        x10 = x8#x7 - contact_pad1_a_outer / 2
        y10 = y8

        x11 = x9#x7 + contact_pad1_a_outer / 2
        y11 = y9

        L1_a = 2.1
        L1_b = 1

        L2_a = 2.1
        L2_b = 1

        rec1_a_outer = 4.8
        rec1_b_outer = 5.8

        if self.hole_in_squid_pad == True:
            rec1_a_inner = 2
            rec1_b_inner = 1
        else:
            rec1_a_inner = 0
            rec1_b_inner = 0

        rec2_a_outer = rec1_a_outer
        rec2_b_outer = rec1_b_outer

        rec2_a_inner = rec1_a_inner
        rec2_b_inner = rec1_b_inner

        self.rect_size_a = rec1_a_outer
        self.rect_size_b = rec1_b_outer

        points10 = [(x10 - L1_a, y10),
                    (x10, y10),
                    (x10, y10 - L1_b),
                    (x10 - L1_a, y10 - L1_b)]

        points11 = [(x11, y11),
                    (x11 + L2_a, y11),
                    (x11 + L2_a, y11 - L2_b),
                    (x11, y11 - L2_b)]

        x12 = x10 - L1_a - (rec1_a_outer / 2)
        y12 = y10 - L1_b / 2 + (rec1_b_outer / 2)

        x13 = x11 + L2_a + (rec2_a_outer / 2)
        y13 = y11 - L2_b / 2 + (rec2_b_outer / 2)
        self.rect1 = (x12, y12)
        self.rect2 = (x13, y13)
        points12 = [(x12 - rec1_a_outer / 2, y12 - rec1_b_outer),
                    (x12 - rec1_a_outer / 2, y12),
                    (x12 + rec1_a_outer / 2, y12),
                    (x12 + rec1_a_outer / 2, y12 - rec1_b_outer),

                    (x12 - rec1_a_outer / 2, y12 - rec1_b_outer),

                    (x12 - rec1_a_inner / 2, y12 - (rec1_b_outer - rec1_b_inner) / 2 - rec1_b_inner),
                    (x12 + rec1_a_inner / 2, y12 - (rec1_b_outer - rec1_b_inner) / 2 - rec1_b_inner),

                    (x12 + rec1_a_inner / 2, y12 - (rec1_b_outer - rec1_b_inner) / 2),
                    (x12 - rec1_a_inner / 2, y12 - (rec1_b_outer - rec1_b_inner) / 2),

                    (x12 - rec1_a_inner / 2, y12 - (rec1_b_outer - rec1_b_inner) / 2 - rec1_b_inner),
                    ]

        points13 = [(x13 - rec2_a_outer / 2, y13 - rec2_b_outer),
                    (x13 - rec2_a_outer / 2, y13),
                    (x13 + rec2_a_outer / 2, y13),
                    (x13 + rec2_a_outer / 2, y13 - rec2_b_outer),

                    (x13 - rec2_a_outer / 2, y13 - rec2_b_outer),

                    (x13 - rec2_a_inner / 2, y13 - (rec2_b_outer - rec2_b_inner) / 2 - rec2_b_inner),
                    (x13 + rec2_a_inner / 2, y13 - (rec2_b_outer - rec2_b_inner) / 2 - rec2_b_inner),

                    (x13 + rec2_a_inner / 2, y13 - (rec2_b_outer - rec2_b_inner) / 2),
                    (x13 - rec2_a_inner / 2, y13 - (rec2_b_outer - rec2_b_inner) / 2),

                    (x13 - rec2_a_inner / 2, y13 - (rec2_b_outer - rec2_b_inner) / 2 - rec2_b_inner),
                    ]

        bandage_to_island = gdspy.Rectangle((self._x0 - self.contact_pad_a_outer / 4,
                                self._y0 + self.contact_pad_b_outer/4),
                               (self._x0 + self.contact_pad_a_outer / 4,
                                self._y0 - 3*self.contact_pad_b_outer/4),
                               layer=self.layer_configuration.bandages_layer)

        bandage_right = gdspy.Rectangle((self.rect2[0] - self.rect_size_a/4,
                                               self.rect2[1] - self.rect_size_b/4),
                                              (self.rect2[0] + 3*self.rect_size_a / 4,
                                               self.rect2[1] - 3*self.rect_size_b/4),
                               layer=self.layer_configuration.bandages_layer)

        bandage_left = gdspy.Rectangle((self.rect1[0] - 3*self.rect_size_a/4,
                                               self.rect1[1] - self.rect_size_b/4),
                                              (self.rect1[0] + self.rect_size_a / 4,
                                               self.rect1[1] - 3*self.rect_size_b/4),
                               layer=self.layer_configuration.bandages_layer)
        bandages = gdspy.boolean(bandage_to_island, [bandage_left, bandage_right], 'or',
                                 layer=self.layer_configuration.bandages_layer)

        squid = gdspy.PolygonSet(
            [points0, points1, points1_1, points2, points3, points4, points5_for_pad1, points6_for_pad2,
             points8_for_pad3, points9_for_pad4, points10, points11, points12, points13])
        jj = gdspy.boolean(squid, squid, "or", layer=self.layer_configuration.jj_layer)
        if self.jjpos == 'up':
            jj = jj.rotate(np.pi, self.center)
            bandages = bandages.rotate(np.pi, self.center)
        elif self.jjpos == 'left':
            jj = jj.rotate(-np.pi / 2, self.center)
            bandages = bandages.rotate(-np.pi / 2, self.center)
        elif self.jjpos == 'right':
            jj = jj.rotate(np.pi / 2, self.center)
            bandages = bandages.rotate(np.pi / 2, self.center)
        return jj, bandages


    def generate_jj(self):
        uh = self.jj['up_rect_h']
        uw = self.jj['up_rect_w']
        sh = self.jj['side_rect_h']
        sw = self.jj['side_rect_w']
        type = self.jj['type']
        th_s1 = self.jj['side_l_thick']
        th_s2 = self.jj['side_r_thick']
        th_u1 = self.jj['up_l_thick']
        th_u2 = self.jj['up_r_thick']
        l_s1 = self.jj['side_l_length']
        l_s2 = self.jj['side_r_length']
        l_u1 = self.jj['up_l_length']
        l_u2 = self.jj['up_r_length']
        sh_u = self.jj['up_rect_shift']
        sh_s = self.jj['side_rect_shift']

        b = self.jgeom['gheight']
        h = self.jgeom['iheight']
        t = self.jgeom['ithick']
        op = self.jgeom['iopen']

        bandage = self.jgeom['bandages']



        up_rect = gdspy.Rectangle((self.center[0] - uw / 2, self.center[1] - self.s / 2 - self.length - sh_u),
                                  (self.center[0] + uw / 2, self.center[1] - self.s / 2 - self.length + uh - sh_u))
        if type == 2:
            # aux_3 = gdspy.Rectangle((self.center[0] - self.s/2 - self.w, self.center[1] - self.s/2 - self.length - (b - h)),
            #                   (self.center[0] + self.s/2 + self.w, self.center[1] - self.s/2 - self.length - (b - h - t)))
            side_rect1 = gdspy.Rectangle(
                (self.center[0] - op / 2 - sw + sh_s, self.center[1] - self.s / 2 - self.length - b + h + t / 2 - sh / 2),
                (self.center[0] - op / 2 + sh_s, self.center[1] - self.s / 2 - self.length - b + h + t / 2 + sh / 2))

            ju1 = gdspy.Rectangle((self.center[0] - uw / 2 + uw / 3 - th_u1, self.center[1] - self.s / 2 - self.length - sh_u),
                                  (self.center[0] - uw / 2 + uw / 3, self.center[1] - self.s / 2 - self.length - l_u1 - sh_u))

            ju2 = gdspy.Rectangle((self.center[0] + uw / 2 - uw / 3, self.center[1] - self.s / 2 - self.length - sh_u),
                                  (self.center[0] + uw / 2 - uw / 3 + th_u2,
                                   self.center[1] - self.s / 2 - self.length - l_u2 - sh_u))
            ju = gdspy.boolean(ju1, ju2, "or", layer=self.layer_configuration.jj_layer)

            js1 = gdspy.Rectangle((self.center[0] - op / 2 + sh_s,
                                   self.center[1] - self.s / 2 - self.length - b + h + t / 2 - sh / 2 + sh / 3 - th_s1),
                                  (self.center[0] - op / 2 + l_s1 + sh_s,
                                   self.center[1] - self.s / 2 - self.length - b + h + t / 2 - sh / 2 + sh / 3))

            js2 = gdspy.Rectangle((self.center[0] + op / 2 - sh_s,
                                   self.center[1] - self.s / 2 - self.length - b + h + t / 2 - sh / 2 + sh / 3 - th_s2),
                                  (self.center[0] + op / 2 - l_s2 - sh_s,
                                   self.center[1] - self.s / 2 - self.length - b + h + t / 2 - sh / 2 + sh / 3))
            js = gdspy.boolean(js1, js2, "or", layer=self.layer_configuration.jj_layer)

            jfull = gdspy.boolean(ju, js, "or", layer=self.layer_configuration.jj_layer)
            side_rect2 = gdspy.Rectangle(
                (self.center[0] + op / 2 - sh_s, self.center[1] - self.s / 2 - self.length - b + h + t / 2 - sh / 2),
                (self.center[0] + op / 2 + sw - sh_s, self.center[1] - self.s / 2 - self.length - b + h + t / 2 + sh / 2))

            side = gdspy.boolean(side_rect1, side_rect2, "or", layer=self.layer_configuration.jj_layer)

        rect = gdspy.boolean(side, up_rect, "or", layer=self.layer_configuration.jj_layer)
        jj = gdspy.boolean(rect, jfull, "or", layer=self.layer_configuration.jj_layer)

        if bandage == True:
            ubh = self.auxjj['bandage_up_height']
            ubw = self.auxjj['bandage_up_width']
            sbh = self.auxjj['bandage_side_height']
            sbw = self.auxjj['bandage_side_width']
            shift_bu = self.auxjj['up_shift']
            shift_bs = self.auxjj['side_shift']

            band_up = gdspy.Rectangle((self.center[0] - ubw/2, self.center[1] - self.s / 2 - self.length - sh_u + uh / 2 - shift_bu),
                                      (self.center[0] + ubw/2, self.center[1] - self.s / 2 - self.length - sh_u + uh / 2 + ubh - shift_bu))

            band_side_1 = gdspy.Rectangle((self.center[0] - op / 2 + sh_s - sw/2 + shift_bs, self.center[1] - self.s / 2 - self.length - b + h + t / 2 - sbh/2),
                                          (self.center[0] - op / 2 + sh_s - sw/2 - sbw + shift_bs, self.center[1] - self.s / 2 - self.length - b + h + t / 2 + sbh/2))

            band_side_2 = gdspy.Rectangle((self.center[0] + op / 2 - sh_s + sw/2 - shift_bs, self.center[1] - self.s / 2 - self.length - b + h + t / 2 - sbh/2),
                                          (self.center[0] + op / 2 - sh_s + sw/2 + sbw - shift_bs, self.center[1] - self.s / 2 - self.length - b + h + t / 2 + sbh/2))

            band_side = gdspy.boolean(band_side_1, band_side_2, "or", layer=self.layer_configuration.bandages_layer)
            band = gdspy.boolean(band_up, band_side, "or", layer=self.layer_configuration.bandages_layer)
            if self.jjpos == 'up':
                band = band.rotate(np.pi, self.center)
                jj = jj.rotate(np.pi, self.center)
            elif self.jjpos == 'left':
                band = band.rotate(-np.pi / 2, self.center)
                jj = jj.rotate(-np.pi / 2, self.center)
            elif self.jjpos == 'right':
                band = band.rotate(np.pi / 2, self.center)
                jj = jj.rotate(np.pi / 2, self.center)
            return jj, band
        else:
            if self.jjpos == 'up':
                jj = jj.rotate(np.pi, self.center)
            elif self.jjpos == 'left':
                jj = jj.rotate(-np.pi / 2, self.center)
            elif self.jjpos == 'right':
                jj = jj.rotate(np.pi / 2, self.center)

            return jj

    def get_terminals(self):
        return self.terminals

    def add_to_tls(self, tls_instance: tlsim.TLSystem, terminal_mapping: dict, track_changes: bool = True,
                   cutoff: float = np.inf, epsilon: float = 11.45) -> list:
        from scipy.constants import hbar, e
        #scaling factor for C
        scal_C = 1e-15

        jj1 = tlsim.JosephsonJunction(e_j=self.jj['ic_r']*hbar/(2*e), name=self.name + ' jj1')
        jj2 = tlsim.JosephsonJunction(e_j=self.jj['ic_l']*hbar/(2*e), name=self.name + ' jj2')
        m = tlsim.Inductor(self.jgeom['lm'], name=self.name + ' flux-wire')
        c = tlsim.Capacitor(c=self.C['qubit']*scal_C, name=self.name+' qubit-ground')
        cache = [jj1, jj2, m, c]
        if self.jj['type'] == 2:
            squid_top = terminal_mapping['qubit']
        else:
            squid_top = terminal_mapping['squid_intermediate']
            jj3 = tlsim.JosephsonJunction(self.jj['ic3'] * hbar / (2 * e), name=self.name + ' jj3')
            tls_instance.add_element(jj3, [squid_top, terminal_mapping['qubit']])
            cache.append(jj3)
        tls_instance.add_element(jj1, [0, squid_top])
        tls_instance.add_element(jj2, [terminal_mapping['flux'], squid_top])
        tls_instance.add_element(m, [0, terminal_mapping['flux']])
        tls_instance.add_element(c, [0, terminal_mapping['qubit']])

        mut_cap = []
        cap_g = []
        for coupler in self.pos:
            print(self.pos)
            c0 = tlsim.Capacitor(c=self.C['crab_'+str(coupler)][1]*scal_C, name=self.name+' qubit-crab'+str(coupler))
            c0g = tlsim.Capacitor(c=self.C['crab_'+str(coupler)][0]*scal_C, name=self.name+' crab'+str(coupler)+'-ground')
            tls_instance.add_element(c0, [terminal_mapping['qubit'], terminal_mapping['crab_'+str(coupler)]])
            tls_instance.add_element(c0g, [terminal_mapping['crab_'+str(coupler)], 0])
            mut_cap.append(c0)
            cap_g.append(c0g)
        if track_changes:
            self.tls_cache.append([jj1, jj2, m, c]+mut_cap+cap_g)
        return cache+mut_cap+cap_g

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
