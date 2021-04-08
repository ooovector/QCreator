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
                 crab_thickness: float, crab_terminals: Dict,ground_thickness: float, delete_ground: str, jj_position: str, jj_params1: Dict, jj_params2: Dict,
                 layer_configuration: LayerConfiguration):
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
        # layers
        self.layer_configuration = layer_configuration

        self.terminals = {'crab_left': None,
                          'crab_right': None,
                          'crab_up': None,
                          'crab_down': None,
                          'flux': None,
                          'qubit': None}
        self.couplers = {}
        self.tls_cache = []
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
        jpad, cross_gnd, jrestrict = self.create_jpad(cross_gnd)
        jj = self.generate_jj()
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

        return {'positive': result,
                'qubit': result,
                'restricted': cross_restrict,
                'JJ': jj,
                'qubit_cap': qubit_cap_parts
                }

    def create_jpad(self, cross_gnd):
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
        elif self.jjpos == 'down':
            ignd_d = ignd_u.rotate(np.pi, self.center)
            cross_gnd = gdspy.boolean(cross_gnd, ignd_d, "not", layer=self.layer_configuration.total_layer)


        elif self.jjpos == 'left':
            ignd_l = ignd_u.rotate(np.pi/2, self.center)
            cross_gnd = gdspy.boolean(cross_gnd, ignd_l, "not", layer=self.layer_configuration.total_layer)
            jpad = jpad.rotate(-np.pi/2, self.center)
            jrestrict = jrestrict.rotate(-np.pi/2, self.center)
        elif self.jjpos == 'right':
            ignd_r = ignd_u.rotate(-np.pi/2, self.center)
            cross_gnd = gdspy.boolean(cross_gnd, ignd_r, "not", layer=self.layer_configuration.total_layer)
            jpad = jpad.rotate(np.pi/2, self.center)
            jrestrict = jrestrict.rotate(np.pi/2, self.center)
        return jpad, cross_gnd, jrestrict


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

        b = self.jgeom['gheight']
        h = self.jgeom['iheight']
        t = self.jgeom['ithick']
        op = self.jgeom['iopen']

        up_rect = gdspy.Rectangle((self.center[0] - uw / 2, self.center[1] - self.s / 2 - self.length),
                                  (self.center[0] + uw / 2, self.center[1] - self.s / 2 - self.length + uh))
        if type == 2:
            # aux_3 = gdspy.Rectangle((self.center[0] - self.s/2 - self.w, self.center[1] - self.s/2 - self.length - (b - h)),
            #                   (self.center[0] + self.s/2 + self.w, self.center[1] - self.s/2 - self.length - (b - h - t)))
            side_rect1 = gdspy.Rectangle(
                (self.center[0] - op / 2 - sw, self.center[1] - self.s / 2 - self.length - b + h + t / 2 - sh / 2),
                (self.center[0] - op / 2, self.center[1] - self.s / 2 - self.length - b + h + t / 2 + sh / 2))

            ju1 = gdspy.Rectangle((self.center[0] - uw / 2 + uw / 3 - th_u1, self.center[1] - self.s / 2 - self.length),
                                  (self.center[0] - uw / 2 + uw / 3, self.center[1] - self.s / 2 - self.length - l_u1))

            ju2 = gdspy.Rectangle((self.center[0] + uw / 2 - uw / 3, self.center[1] - self.s / 2 - self.length),
                                  (self.center[0] + uw / 2 - uw / 3 + th_u2,
                                   self.center[1] - self.s / 2 - self.length - l_u2))
            ju = gdspy.boolean(ju1, ju2, "or", layer=self.layer_configuration.jj_layer)

            js1 = gdspy.Rectangle((self.center[0] - op / 2,
                                   self.center[1] - self.s / 2 - self.length - b + h + t / 2 - sh / 2 + sh / 3 - th_s1),
                                  (self.center[0] - op / 2 + l_s1,
                                   self.center[1] - self.s / 2 - self.length - b + h + t / 2 - sh / 2 + sh / 3))

            js2 = gdspy.Rectangle((self.center[0] + op / 2,
                                   self.center[1] - self.s / 2 - self.length - b + h + t / 2 - sh / 2 + sh / 3 - th_s2),
                                  (self.center[0] + op / 2 - l_s2,
                                   self.center[1] - self.s / 2 - self.length - b + h + t / 2 - sh / 2 + sh / 3))
            js = gdspy.boolean(js1, js2, "or", layer=self.layer_configuration.jj_layer)

            jfull = gdspy.boolean(ju, js, "or", layer=self.layer_configuration.jj_layer)
            side_rect2 = gdspy.Rectangle(
                (self.center[0] + op / 2, self.center[1] - self.s / 2 - self.length - b + h + t / 2 - sh / 2),
                (self.center[0] + op / 2 + sw, self.center[1] - self.s / 2 - self.length - b + h + t / 2 + sh / 2))

            side = gdspy.boolean(side_rect1, side_rect2, "or", layer=self.layer_configuration.jj_layer)

        rect = gdspy.boolean(side, up_rect, "or", layer=self.layer_configuration.jj_layer)
        jj = gdspy.boolean(rect, jfull, "or", layer=self.layer_configuration.jj_layer)
        if self.jjpos == 'up':
            jj = jj.rotate(np.pi, self.center)
        elif self.jjpos == 'left':
            jj = jj.rotate(-np.pi / 2, self.center)
        elif self.jjpos == 'right':
            jj = jj.rotate(np.pi / 2, self.center)

        return jj

    def get_terminals(self):
        return self.terminals

    def add_to_tls(self, tls_instance: tlsim.TLSystem, terminal_mapping: dict, track_changes: bool = True, cutoff: float = np.inf) -> list:
        from scipy.constants import hbar, e
        #scaling factor for C
        scal_C = 1e-15

        jj1 = tlsim.JosephsonJunction(e_j=self.jj['ic_l']*hbar/(2*e), name=self.name + ' jj1')
        jj2 = tlsim.JosephsonJunction(e_j=self.jj['ic_r']*hbar/(2*e), name=self.name + ' jj2')
        m = tlsim.Inductor(self.jgeom['lm'], name=self.name + ' flux-wire')
        c = tlsim.Capacitor(c=self.C['qubit']*scal_C, name=self.name+' qubit-ground')
        tls_instance.add_element(jj1, [0, terminal_mapping['qubit']])
        tls_instance.add_element(jj2, [terminal_mapping['flux'], terminal_mapping['qubit']])
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
        return [jj1, jj2, m, c]+mut_cap+cap_g

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
