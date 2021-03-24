from .core import DesignElement, LayerConfiguration, DesignTerminal
# from .. import conformal_mapping as cm
# from .. import transmission_line_simulator as tlsim
import gdspy
import numpy as np
from typing import List, Tuple, Mapping, Dict
# from . import squid3JJ
# from copy import deepcopy


class Xmon(DesignElement):

    def __init__(self, name: str, center: Tuple[float, float],
                 length: float, width_gap: float, center_width: float,
                 crab_position: Tuple[str, str, str, str], crab_shoulder: float,
                 crab_thickness: float, ground_thickness: float, delete_ground: str, jj_position: str, jj_params1: Dict, jj_params2: Dict,
                 layer_configuration: LayerConfiguration):
        super().__init__(type='qubit', name=name)
        # qubit parameters
        self.center = center
        self.L = length
        self.w = width_gap
        self.s = center_width
        self.a = crab_shoulder
        self.b = crab_thickness
        self.pos = crab_position
        self.g = ground_thickness
        self.jjpos = jj_position
        self.delg = delete_ground

        self.jgeom = jj_params1
        self.jj = jj_params2
        # layers
        self.layer_configuration = layer_configuration

        self.layers = []




    def render(self):
        """
        I've no clue if it is right or just strait up dumb BUT:
            Every element of the Xmon here is drawn by subtracting two rectangles from each other:
                Everything that is marked *_out or *_o means that it's the outer rectangle
                Same goes for *_in or *_i -- they're inner ones

            I figured this strange (read stupid) way of drawing parts would be easier than making a bunch of complex polygons.


        """
        # draw center cross:
        # auxillary:
        cross_hor = gdspy.Rectangle((self.center[0] - self.s/2 - self.L, self.center[1] - self.s/2),
                                    (self.center[0] + self.s/2 + self.L, self.center[1] + self.s/2))

        cross_ver = gdspy.Rectangle((self.center[0] - self.s/2, self.center[1] - self.s/2 - self.L),
                                    (self.center[0] + self.s/2, self.center[1] + self.s/2 + self.L))

        # the cross itself
        cross = gdspy.boolean(cross_hor, cross_ver, "or", layer=self.layer_configuration.total_layer)

        # draw ground cross
        # auxillary
        cross_gnd_hor_out = gdspy.Rectangle((self.center[0] - self.s/2 - self.L - self.w - self.g, self.center[1] - self.s/2 - self.w - self.g),
                                            (self.center[0] + self.s/2 + self.L + self.w + self.g, self.center[1] + self.s/2 + self.w + self.g))
        cross_gnd_hor_in = gdspy.Rectangle((self.center[0] - self.s/2 - self.L - self.w, self.center[1] - self.s/2 - self.w),
                                           (self.center[0] + self.s/2 + self.L + self.w, self.center[1] + self.s/2 + self.w))
        cross_gnd_ver_out = gdspy.Rectangle((self.center[0] - self.s/2 - self.w - self.g, self.center[1] - self.s/2 - self.L - self.w - self.g),
                                            (self.center[0] + self.s/2 + self.w + self.g, self.center[1] + self.s/2 + self.L + self.w + self.g))
        cross_gnd_ver_in = gdspy.Rectangle((self.center[0] - self.s/2 - self.w, self.center[1] - self.s/2 - self.L - self.w),
                                           (self.center[0] + self.s/2 + self.w, self.center[1] + self.s/2 + self.L + self.w))

        cross_gnd_out = gdspy.boolean(cross_gnd_hor_out, cross_gnd_ver_out, "or", layer=self.layer_configuration.total_layer)
        cross_gnd_in = gdspy.boolean(cross_gnd_hor_in, cross_gnd_ver_in, "or", layer=self.layer_configuration.total_layer)


        # the cross ground (things will be added or deleted from this later on

        cross_gnd = gdspy.boolean(cross_gnd_out, cross_gnd_in, "not", layer=self.layer_configuration.total_layer)


        # the following draws a crab around the upper end of the cross
        for word in self.pos:
            crab_out = gdspy.Rectangle((self.center[0] - self.s/2 - self.w - self.g - self.w - self.b, self.center[1] + self.s/2 + self.L - self.a),
                                       (self.center[0] + self.s/2 + self.w + self.g + self.w + self.b, self.center[1] + self.s/2 + self.L + self.g + self.w + self.w + self.b))
            crab_in = gdspy.Rectangle((self.center[0] - self.s/2 - self.w - self.g - self.w, self.center[1] + self.s/2 + self.L - self.a),
                                      (self.center[0] + self.s/2 + self.w + self.g + self.w, self.center[1] + self.s/2 + self.L + self.w + self.g + self.w))
            crab_u = gdspy.boolean(crab_out, crab_in, "not", layer=self.layer_configuration.total_layer)

            ognd_u_out = gdspy.Rectangle((self.center[0] - self.s/2 - self.w - self.g - self.w - self.b - self.w - self.g, self.center[1] + self.s/2 + self.L - self.a - self.w - self.g),
                                         (self.center[0] + self.s/2 + self.w + self.g + self.w + self.b + self.w + self.g, self.center[1] + self.s/2 + self.L + self.g + self.w + self.w + self.b + self.w + self.g))
            ognd_u_in = gdspy.Rectangle((self.center[0] - self.s/2 - self.w - self.g - self.w - self.b - self.w, self.center[1] + self.s/2 + self.L - self.a - self.w),
                                        (self.center[0] + self.s/2 + self.w + self.g + self.w + self.b + self.w, self.center[1] + self.s/2 + self.L + self.w + self.g + self.w + self.b + self.w))
            aux = gdspy.Rectangle((self.center[0] - self.s/2 - self.w, self.center[1] + self.s/2 + self.L - self.a - self.w - self.g),
                                  (self.center[0] + self.s/2 + self.w, self.center[1] + self.s/2 + self.L - self.a - self.w))
            ognd_u_in = gdspy.boolean(ognd_u_in, aux, "or", layer=self.layer_configuration.total_layer)
            ognd_u = gdspy.boolean(ognd_u_out, ognd_u_in, "not", layer=self.layer_configuration.total_layer)
            crab_u = gdspy.boolean(crab_u, ognd_u, "or", layer=self.layer_configuration.total_layer)

            # rotate the crab depending on it's respective position

            if word == 'right':
                crab_r = crab_u.rotate(-np.pi/2, self.center)
                cross = gdspy.boolean(cross, crab_r, 'or', layer=self.layer_configuration.total_layer)

            if word == 'up':

                cross = gdspy.boolean(cross, crab_u, 'or', layer=self.layer_configuration.total_layer)
            if word == 'down':
                crab_d = crab_u.rotate(np.pi, self.center)
                cross = gdspy.boolean(cross, crab_d, 'or', layer=self.layer_configuration.total_layer)

            if word == 'left':
                crab_l = crab_u.rotate(np.pi/2, self.center)
                cross = gdspy.boolean(cross, crab_l, 'or', layer=self.layer_configuration.total_layer)

        # if deleting ground between the crab and the cross is needed:

        if self.delg != '':
            cross_gnd = self.delete_ground(cross_gnd)

        # create a pad for jj
        jpad, cross_gnd = self.create_jpad(cross_gnd)
        jj = self.generate_jj()
        cross_gnd = gdspy.boolean(cross_gnd, jpad, "or", layer=self.layer_configuration.total_layer)

        # parts of the restricted area
        cross_hor_restrict = gdspy.Rectangle((self.center[0] - self.s/2 - self.L - self.w, self.center[1] - self.s/2 - self.w),
                                             (self.center[0] + self.s/2 + self.L + self.w, self.center[1] + self.s/2 + self.w))
        cross_ver_restrict = gdspy.Rectangle((self.center[0] - self.s/2 - self.w, self.center[1] - self.s/2 - self.L - self.w),
                                             (self.center[0] + self.s/2 + self.w, self.center[1] + self.s/2 + self.L + self.w))

        # create a fluxline near jj pad
        fgnd = self.generate_fluxline()
        cross_gnd = gdspy.boolean(cross_gnd, fgnd, "or", layer=self.layer_configuration.total_layer)
        cross_restrict = gdspy.boolean(cross_hor_restrict, cross_ver_restrict, "or",  layer=self.layer_configuration.restricted_area_layer)
        result = gdspy.boolean(cross_gnd, cross, "or", layer=self.layer_configuration.total_layer)
        return {'positive': result,
                'qubit': result,
                'restricted': cross_restrict,
                'JJ': jj
                }


    def create_jpad(self, cross_gnd):
        # variables
        a = self.jgeom['gwidth']
        b = self.jgeom['gheight']
        k = self.jgeom['iwidth']
        h = self.jgeom['iheight']
        t = self.jgeom['ithick']


        ignd_u = gdspy.Rectangle((self.center[0] - self.s/2 - self.w - self.g, self.center[1] + self.s/2 + self.L),
                                 (self.center[0] + self.s/2 + self.w + self.g, self.center[1] + self.s/2 + self.L + self.w + self.g))

        jgnd_o = gdspy.Rectangle((self.center[0] - a/2 - self.g, self.center[1] - self.s/2 - self.L - b - self.g),
                                 (self.center[0] + a/2 + self.g, self.center[1] - self.s/2 - self.L + self.g))
        jgnd_i = gdspy.Rectangle((self.center[0] - a/2, self.center[1] - self.s/2 - self.L - b),
                                 (self.center[0] + a/2, self.center[1] - self.s/2 - self.L))
        aux_2 = gdspy.Rectangle((self.center[0] - self.s/2 - self.w, self.center[1] - self.s/2 - self.L),
                                (self.center[0] + self.s/2 + self.w, self.center[1] - self.s/2 - self.L + self.g))
        jgnd_o = gdspy.boolean(jgnd_o, aux_2, "not", layer=self.layer_configuration.total_layer)
        jgnd = gdspy.boolean(jgnd_o, jgnd_i, "not", layer=self.layer_configuration.total_layer)

        jpad_o = gdspy.Rectangle((self.center[0] - k/2, self.center[1] - self.s/2 - self.L - b),
                                 (self.center[0] + k/2, self.center[1] - self.s/2 - self.L - (b - h - t)))
        jpad_i = gdspy.Rectangle((self.center[0] - k/2 + t, self.center[1] - self.s/2 - self.L - b),
                                 (self.center[0] + k/2 - t, self.center[1] - self.s/2 - self.L - (b - h)))
        jpad = gdspy.boolean(jpad_o, jpad_i, "not", layer=self.layer_configuration.total_layer)

        aux_3 = gdspy.Rectangle((self.center[0] - self.s/2 - self.w, self.center[1] - self.s/2 - self.L - (b - h)),
                                (self.center[0] + self.s/2 + self.w, self.center[1] - self.s/2 - self.L - (b - h - t)))
        jpad = gdspy.boolean(jpad, aux_3, "not", layer=self.layer_configuration.total_layer)

        jpad = gdspy.boolean(jpad, jgnd, "or", layer=self.layer_configuration.total_layer)

        aux_4 = gdspy.Rectangle((self.center[0] - a/2 - self.g, self.center[1] - self.s/2 - self.L - b - self.g),
                                (self.center[0] + a/2 + self.g, self.center[1] - self.s/2 - self.L - b))
        jpad = gdspy.boolean(jpad, aux_4, "not", layer=self.layer_configuration.total_layer)


        if self.jjpos == 'up':
            cross_gnd = gdspy.boolean(cross_gnd, ignd_u, "not", layer=self.layer_configuration.total_layer)
            jpad = jpad.rotate(np.pi, self.center)

        elif self.jjpos == 'down':
            ignd_d = ignd_u.rotate(np.pi, self.center)
            cross_gnd = gdspy.boolean(cross_gnd, ignd_d, "not", layer=self.layer_configuration.total_layer)


        elif self.jjpos == 'left':
            ignd_l = ignd_u.rotate(np.pi/2, self.center)
            cross_gnd = gdspy.boolean(cross_gnd, ignd_l, "not", layer=self.layer_configuration.total_layer)
            jpad = jpad.rotate(-np.pi/2, self.center)

        elif self.jjpos == 'right':
            ignd_r = ignd_u.rotate(-np.pi/2, self.center)
            cross_gnd = gdspy.boolean(cross_gnd, ignd_r, "not", layer=self.layer_configuration.total_layer)
            jpad = jpad.rotate(np.pi/2, self.center)

        return jpad, cross_gnd


    def delete_ground(self, cross_gnd):
        gnd_out = gdspy.Rectangle((self.center[0] - self.s/2 - self.w - self.g, self.center[1] + self.s/2 + self.L - self.a - self.w),
                                  (self.center[0] + self.s/2 + self.w + self.g, self.center[1] + self.s/2 + self.L + self.g + self.w))
        gnd_in = gdspy.Rectangle((self.center[0] - self.s/2 - self.w, self.center[1] + self.s/2 + self.L - self.a - self.w),
                                 (self.center[0] + self.s/2 + self.w, self.center[1] + self.s/2 + self.L + self.w))
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
        f = self.jgeom['fthick']

        fgnd_1 = gdspy.Rectangle((self.center[0] - a/2 - self.g, self.center[1] - self.s/2 - self.L - b - h1),
                                 (self.center[0] + a/2 + self.g, self.center[1] - self.s/2 - self.L - b))
        fgnd_2 = gdspy.Rectangle((self.center[0] - self.s/2 - self.w - self.g, self.center[1] - self.s/2 - self.L - h1 - h2 - b),
                                 (self.center[0] + self.s/2 + self.w + self.g, self.center[1] - self.s/2 - self.L - h1 - b))

        fgnd = gdspy.boolean(fgnd_1, fgnd_2, "or", layer=self.layer_configuration.total_layer)
        fhor = gdspy.Rectangle((self.center[0] - a/2, self.center[1] - self.s/2 - self.L - b - d - self.w),
                               (self.center[0] + a/2, self.center[1] - self.s/2 - self.L - b - d))
        fgnd = gdspy.boolean(fgnd, fhor, "not", layer=self.layer_configuration.total_layer)

        fver1 = gdspy.Rectangle((self.center[0] - f/2 - self.w, self.center[1] - self.s/2 - self.L - b - d - h2 - h1),
                                (self.center[0] - f/2, self.center[1] - self.s/2 - self.L - b - d - self.w - f))
        fver1_turn = gdspy.Rectangle((self.center[0] - f/2 - sh, self.center[1] - self.s/2 - self.L - b - d - 2*self.w - f),
                                     (self.center[0] - f/2, self.center[1] - self.s/2 - self.L - b - d - self.w - f))
        fver1 = gdspy.boolean(fver1, fver1_turn, "or", layer=self.layer_configuration.total_layer)

        fver2 = gdspy.Rectangle((self.center[0] + f/2 + self.w, self.center[1] - self.s/2 - self.L - b - d - h2 - h1),
                                (self.center[0] + f/2, self.center[1] - self.s/2 - self.L - b - d))
        fver = gdspy.boolean(fver1, fver2, "or", layer=self.layer_configuration.total_layer)
        fgnd = gdspy.boolean(fgnd, fver, "not", layer=self.layer_configuration.total_layer)

        if self.jjpos == 'up':
            fgnd = fgnd.rotate(np.pi, self.center)
        elif self.jjpos == 'left':
            fgnd = fgnd.rotate(-np.pi/2, self.center)
        elif self.jjpos == 'right':
            fgnd = fgnd.rotate(np.pi/2, self.center)

        return fgnd

    def generate_jj(self):
        uh = self.jj['up_rect_h']
        uw = self.jj['up_rect_w']
        sh = self.jj['side_rect_h']
        sw = self.jj['side_rect_w']
        type = self.jj['type']
        th_s1 = self.jj['side_l_thick']
        th_s2 = self.jj['side_r_thick']
        th_u1 = self.jj['up_r_thick']
        th_u2 = self.jj['up_l_thick']

        b = self.jgeom['gheight']
        h = self.jgeom['iheight']
        t = self.jgeom['ithick']

        up_rect = gdspy.Rectangle((self.center[0] - uw/2, self.center[1] - self.s/2 - self.L),
                                  (self.center[0] + uw/2, self.center[1] - self.s/2 - self.L + uh))
        if type == 2:
            # aux_3 = gdspy.Rectangle((self.center[0] - self.s/2 - self.w, self.center[1] - self.s/2 - self.L - (b - h)),
            #                   (self.center[0] + self.s/2 + self.w, self.center[1] - self.s/2 - self.L - (b - h - t)))
            side_rect1 = gdspy.Rectangle((self.center[0] - self.s/2 - self.w - sw, self.center[1] - self.s/2 - self.L - b + h + t/2 - sh/2),
                                         (self.center[0] - self.s/2 - self.w, self.center[1] - self.s/2 - self.L - b + h + t/2 + sh/2))

            ju1 = gdspy.Rectangle((self.center[0] - uw/2 + uw/3 - th_u1, self.center[1] - self.s/2 - self.L),
                                  (self.center[0] - uw/2 + uw/3, self.center[1] - self.s/2 - self.L - b + h + t/2 - sh/2))

            ju2 = gdspy.Rectangle((self.center[0] + uw/2 - uw/3, self.center[1] - self.s/2 - self.L),
                                  (self.center[0] + uw/2 - uw/3 + th_u2, self.center[1] - self.s/2 - self.L - b + h + t/2 - sh/2))
            ju = gdspy.boolean(ju1, ju2, "or", layer=self.layer_configuration.jj_layer)

            js1 = gdspy.Rectangle((self.center[0] - self.s/2 - self.w, self.center[1] - self.s/2 - self.L - b + h + t/2 - sh/2 + sh/3 - th_s1),
                                  (self.center[0] - uw/2 + uw/3, self.center[1] - self.s/2 - self.L - b + h + t/2 - sh/2 + sh/3))

            js2 = gdspy.Rectangle((self.center[0] + self.s/2 + self.w, self.center[1] - self.s/2 - self.L - b + h + t/2 - sh/2 + sh/3 - th_s2),
                                  (self.center[0] + uw/2 - uw/3, self.center[1] - self.s/2 - self.L - b + h + t/2 - sh/2 + sh/3))
            js = gdspy.boolean(js1, js2, "or", layer=self.layer_configuration.jj_layer)

            jfull = gdspy.boolean(ju, js, "or", layer=self.layer_configuration.jj_layer)
            side_rect2 = gdspy.Rectangle((self.center[0] + self.s/2 + self.w, self.center[1] - self.s/2 - self.L - b + h + t/2 - sh/2),
                                         (self.center[0] + self.s/2 + self.w + sw, self.center[1] - self.s/2 - self.L - b + h + t/2 + sh/2))

            side = gdspy.boolean(side_rect1, side_rect2, "or", layer=self.layer_configuration.jj_layer)

        rect = gdspy.boolean(side, up_rect, "or", layer=self.layer_configuration.jj_layer)
        jj = gdspy.boolean(rect, jfull, "or", layer=self.layer_configuration.jj_layer)
        if self.jjpos == 'up':
            jj = jj.rotate(np.pi, self.center)
        elif self.jjpos == 'left':
            jj = jj.rotate(-np.pi/2, self.center)
        elif self.jjpos == 'right':
            jj = jj.rotate(np.pi/2, self.center)

        return jj
