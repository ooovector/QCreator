import gdspy
from .core import DesignElement, LayerConfiguration, DesignTerminal
import numpy as np
from . import squid3JJ
from typing import List, Tuple, Mapping, Dict, AnyStr
from copy import deepcopy
from .. import conformal_mapping as cm
from .. import transmission_line_simulator as tlsim
from .functions import find_normal_point


class SquidInLine(DesignElement):
    def __init__(self, name: str, center: tuple, core: float, gap: float, ground: float,
                 layer_configuration: LayerConfiguration, squid_params: Dict, fluxline: Dict, coil: str = 'old',
                 import_jj=False, file_jj=None, cell_jj=None, bandages=False):
        super().__init__(type='squid in line', name=name)
        self.core = core
        self.gap = gap
        self.ground = ground
        self.center = center
        self.squid_params = squid_params
        self.JJ = None
        self.fluxline = fluxline
        if self.fluxline is not None:
            self.w = self.fluxline['w']
            self.s = self.fluxline['g']
            self.g = self.fluxline['s']
        self.layer_configuration = layer_configuration
        # coupler terminals
        self.terminals = {'port1': None,
                          'port2': None,
                          'flux': None}
        self.tls_cache = []

        self.coil = coil

        self.import_jj = import_jj
        self.file_jj = file_jj
        self.cell_jj = cell_jj

        self.bandages = bandages
        if self.bandages:
            self.imported_bandages = None

    def render(self):
        if 'side' in self.squid_params:
            coeff = 1 if self.squid_params['side'] == 'right' else -1
            squid_point = (self.center[0] + coeff * self.squid_params['shift'], self.center[1])
        else:
            squid_point = (self.center[0],
                           self.center[1] - np.cos(self.squid_params['angle']) * self.squid_params['shift'])
        if self.import_jj:
            squid_polygons, rect, to_remove = self.import_squid(squid_point, file_name=self.file_jj, cell_name=self.cell_jj)
        else:
            squid_polygons, rect, to_remove = self.generate_squid(squid_point)
        # self.JJ = gdspy.boolean(squid_polygons, squid_polygons, 'or', layer=self.layer_configuration.jj_layer)
        self.JJ = squid_polygons

        if 'side' in self.squid_params:
            self.line_length = 2 * (self.center[1] - self.squid.rect2[1] - self.squid.rect_size_b / 2 +
                                    self.fluxline['length_y'] + 2 * self.fluxline['width'] + self.s + self.w)
            self.port1_position = (self.center[0], self.center[1] + self.line_length / 2)
            self.port2_position = (self.center[0], self.center[1] - self.line_length / 2)
        else:
            self.line_length = 2 * (np.abs(self.center[0] - self.squid.rect2[0] - self.squid.rect_size_a / 2) +
                                    self.fluxline['w'] + 2 * self.fluxline['s'])
            self.port1_position = (self.center[0] - self.line_length / 2, self.center[1])
            self.port2_position = (self.center[0] + self.line_length / 2, self.center[1])
        self.orientation = np.arctan2(self.port1_position[1] - self.port2_position[1],
                                      self.port1_position[0] - self.port2_position[0])

        ground1 = self.generate_rect(self.core / 2 + self.ground + self.gap, self.core / 2 + self.gap,
                                     layer=self.layer_configuration.total_layer)
        line = self.generate_rect(self.core / 2, -self.core / 2, layer=self.layer_configuration.total_layer)
        ground2 = self.generate_rect(-self.core / 2 - self.ground - self.gap, -self.core / 2 - self.gap,
                                     layer=self.layer_configuration.total_layer)
        restricted = self.generate_rect(-self.core / 2 - self.ground - self.gap,
                                        +self.core / 2 + self.ground + self.gap,
                                        layer=self.layer_configuration.restricted_area_layer)
        positive = gdspy.boolean(line, [ground2, ground1], 'or', layer=self.layer_configuration.total_layer)

        if self.coil == 'old':
            # positive = gdspy.boolean(positive, to_remove, 'not', layer=self.layer_configuration.total_layer)
            positive = gdspy.boolean(positive, rect, 'or', layer=self.layer_configuration.total_layer)
            fluxline = self.connection_to_ground()
            positive = gdspy.boolean(positive, fluxline['positive'], 'or', layer=self.layer_configuration.total_layer)
            positive = gdspy.boolean(positive, fluxline['remove'], 'not', layer=self.layer_configuration.total_layer)
        elif self.coil == 'new':
            positive = gdspy.boolean(positive, to_remove, 'not', layer=self.layer_configuration.total_layer)
            positive = gdspy.boolean(positive, rect, 'or', layer=self.layer_configuration.total_layer)
            fluxline = self.connection_to_ground()
            positive = gdspy.boolean(positive, fluxline['positive'], 'or', layer=self.layer_configuration.total_layer)
            positive = gdspy.boolean(positive, fluxline['remove'], 'not', layer=self.layer_configuration.total_layer)
        else:
            raise ValueError('Coil type of flux line is not defined!')

        self.terminals['port1'] = DesignTerminal(position=self.port1_position, orientation=self.orientation + np.pi,
                                                 type='cpw', w=self.core, s=self.gap, g=self.ground,
                                                 disconnected='short')
        self.terminals['port2'] = DesignTerminal(
            position=self.port2_position,
            orientation=self.orientation,
            type='cpw', w=self.core, s=self.gap, g=self.ground,
            disconnected='short')
        render_result = {'positive': positive,
                         'restrict': restricted,
                         'JJ': self.JJ
                         }
        if self.bandages:
            bandages = self.imported_bandages
            render_result.update({'bandages': bandages})
        return render_result

    def generate_squid(self, point):
        self.squid_params['x'] = point[0]
        self.squid_params['y'] = point[1]
        self.squid = squid3JJ.JJ_2(self.squid_params['x'],
                                   self.squid_params['y'],
                                   self.squid_params['a1'], self.squid_params['a2'],
                                   self.squid_params['b1'], self.squid_params['b2'],
                                   self.squid_params['c1'], self.squid_params['c2'], 0, 0)
        squid = self.squid.generate_jj()
        rect = gdspy.Rectangle((self.squid_params['x'] - self.squid.contact_pad_a_outer / 2,
                                self.squid_params['y']),
                               (self.squid_params['x'] + self.squid.contact_pad_a_outer / 2,
                                self.squid_params['y'] - self.squid.contact_pad_b_outer * np.cos(
                                    self.squid_params['angle'])),
                               layer=self.layer_configuration.total_layer)
        if 'side' in self.squid_params:
            indent = 2
            coeff = 1 if self.squid_params['side'] == 'right' else -1
            path1 = gdspy.Polygon(
                [(self.squid_params['x'] + self.squid.contact_pad_a_outer / 2 * coeff, self.squid_params['y'] + indent),
                 (self.center[0], self.squid_params['y'] + indent),
                 (self.center[0], self.squid_params['y'] - self.squid.contact_pad_b_outer),
                 (self.squid_params['x'] + self.squid.contact_pad_a_outer / 2 * coeff,
                  self.squid_params['y'] - self.squid.contact_pad_b_outer)])
            rect = gdspy.boolean(rect, path1, 'or', layer=self.layer_configuration.total_layer)
        # create a polygon to remove
        to_remove = gdspy.Rectangle((self.squid_params['x'] + self.squid_params['removing']['right'],
                                     self.squid_params['y'] + self.squid_params['removing']['up']),
                                    (self.squid_params['x'] - self.squid_params['removing']['left'],
                                     self.squid_params['y'] - self.squid_params['removing']['down']))
        to_remove.rotate(self.squid_params['angle'], (self.squid_params['x'],
                                                      self.squid_params['y']))
        squid = gdspy.boolean(squid, squid, 'or', layer=self.layer_configuration.jj_layer)
        squid.rotate(self.squid_params['angle'], (self.squid_params['x'],
                                                  self.squid_params['y']))
        self.squid.rect1 = rotate_point(self.squid.rect1, self.squid_params['angle'], (self.squid_params['x'],
                                                                                       self.squid_params['y']))
        self.squid.rect2 = rotate_point(self.squid.rect2, self.squid_params['angle'], (self.squid_params['x'],
                                                                                       self.squid_params['y']))
        return squid, rect, to_remove

    def import_squid(self, point, file_name, cell_name):
        """
        Import SQUID topology for the SQUID in line from GDS file. Transmission line model it defined be user itself.
        """
        self.squid_params['x'] = point[0]
        self.squid_params['y'] = point[1]
        self.squid = squid3JJ.JJ_2(self.squid_params['x'],
                                   self.squid_params['y'],
                                   self.squid_params['a1'], self.squid_params['a2'],
                                   self.squid_params['b1'], self.squid_params['b2'],
                                   self.squid_params['c1'], self.squid_params['c2'], 0, 0)
        self.squid.generate_jj()

        import os
        path = os.getcwd()
        path_for_file = path[:path.rindex('QCreator')] + 'QCreator\QCreator\elements\junctions' + file_name
        # import cell
        squid = gdspy.GdsLibrary().read_gds(infile=path_for_file).cells[cell_name].remove_polygons(lambda pts, layer,
                                                                                                          datatype: layer not in [
            self.layer_configuration.jj_layer])
        # convert to polygonset
        squid_polygons = []
        squid_layers = []
        for p_id, p in enumerate(squid.polygons):
            points = p.polygons[0]
            l = p.layers[0]
            squid_polygons.append(points)
            squid_layers.append(l)
        squid_polygonset = gdspy.PolygonSet(squid_polygons, layer=self.layer_configuration.jj_layer)
        squid_polygonset.translate(self.squid_params['x'], self.squid_params['y'])
        squid_polygonset.rotate(self.squid_params['angle'], (self.squid_params['x'], self.squid_params['y']))
        rect = gdspy.Rectangle((self.squid_params['x'] - self.squid.contact_pad_a_outer / 2,
                                self.squid_params['y']),
                               (self.squid_params['x'] + self.squid.contact_pad_a_outer / 2,
                                self.squid_params['y'] - self.squid.contact_pad_b_outer * np.cos(
                                    self.squid_params['angle'])),
                               layer=self.layer_configuration.total_layer)
        if 'side' in self.squid_params:
            indent = 2
            coeff = 1 if self.squid_params['side'] == 'right' else -1
            path1 = gdspy.Polygon(
                [(self.squid_params['x'] + self.squid.contact_pad_a_outer / 2 * coeff, self.squid_params['y'] + indent),
                 (self.center[0], self.squid_params['y'] + indent),
                 (self.center[0], self.squid_params['y'] - self.squid.contact_pad_b_outer),
                 (self.squid_params['x'] + self.squid.contact_pad_a_outer / 2 * coeff,
                  self.squid_params['y'] - self.squid.contact_pad_b_outer)])
            rect = gdspy.boolean(rect, path1, 'or', layer=self.layer_configuration.total_layer)
        # create a polygon to remove
        to_remove = gdspy.Rectangle((self.squid_params['x'] + self.squid_params['removing']['right'],
                                     self.squid_params['y'] + self.squid_params['removing']['up']),
                                    (self.squid_params['x'] - self.squid_params['removing']['left'],
                                     self.squid_params['y'] - self.squid_params['removing']['down']))
        to_remove.rotate(self.squid_params['angle'], (self.squid_params['x'],
                                                      self.squid_params['y']))

        self.squid.rect1 = rotate_point(self.squid.rect1, self.squid_params['angle'], (self.squid_params['x'],
                                                                                       self.squid_params['y']))
        self.squid.rect2 = rotate_point(self.squid.rect2, self.squid_params['angle'], (self.squid_params['x'],
                                                                                       self.squid_params['y']))

        if self.bandages:
            bandages = gdspy.GdsLibrary().read_gds(infile=path_for_file).cells[cell_name].remove_polygons(
                lambda pts, layer,
                       datatype: layer not in [
                    self.layer_configuration.bandages_layer])
            # convert bandages to polygonset
            bandages_polygons = []
            for p_id, p in enumerate(bandages.polygons):
                points = p.polygons[0]
                bandages_polygons.append(points)
            bandages_polygonset = gdspy.PolygonSet(bandages_polygons, layer=self.layer_configuration.bandages_layer)
            bandages_polygonset.translate(self.squid_params['x'], self.squid_params['y'])
            bandages_polygonset.rotate(self.squid_params['angle'], (self.squid_params['x'], self.squid_params['y']))
            self.imported_bandages = bandages_polygonset

        return squid_polygonset, rect, to_remove

    def connection_to_ground(self):
        """
        This function generate a connection from squid rectangulars to a flux line output. Should be changed if you want
        to use another type of JJ or a flux line
        """
        result = None
        remove = None
        width = self.fluxline['width']
        if self.coil == 'old':
            if 'side' in self.squid_params:
                length_x = self.fluxline['length_x']
                length_y = self.fluxline['length_y']
                coeff = 1 if self.squid_params['side'] == 'right' else -1
                if self.squid_params['side'] == 'right':
                    squid_pad1 = self.squid.rect2
                    squid_pad2 = self.squid.rect1
                else:
                    squid_pad1 = self.squid.rect1
                    squid_pad2 = self.squid.rect2

                # rect1 = gdspy.Rectangle(
                #     (squid_pad1[0] - width / 2 + coeff * length_x * np.cos(self.squid_params['angle']), squid_pad1[1]),
                #     (squid_pad1[0] + width / 2, squid_pad1[1] - width))

                rect1 = gdspy.Rectangle(
                    (squid_pad1[0] - width / 2 - np.abs((coeff - 1) / 2) * length_x * np.cos(
                        self.squid_params['angle']),
                     squid_pad1[1]),
                    (squid_pad1[0] + width / 2 + np.abs((coeff + 1) / 2) * length_x * np.cos(
                        self.squid_params['angle']),
                     squid_pad1[1] - width))

                result = gdspy.boolean(rect1, result, 'or', layer=self.layer_configuration.total_layer)

                rect1 = gdspy.Rectangle(
                    (squid_pad2[0] - width / 2, squid_pad2[1] - length_y * np.cos(self.squid_params['angle'])),
                    (squid_pad2[0] + width / 2, squid_pad2[1]))
                result = gdspy.boolean(rect1, result, 'or', layer=self.layer_configuration.total_layer)

                rect1 = gdspy.Rectangle(
                    (squid_pad2[0] - coeff * width / 2, squid_pad2[1] - length_y * np.cos(self.squid_params['angle'])),
                    (squid_pad2[0] + width / 2 + coeff * length_x * np.cos(self.squid_params['angle']),
                     squid_pad2[1] - length_y * np.cos(self.squid_params['angle']) - width))
                result = gdspy.boolean(rect1, result, 'or', layer=self.layer_configuration.total_layer)

                connection_length = self.squid_params['removing']['right'] if self.squid_params['side'] == 'right' \
                    else self.squid_params['removing']['left']

                connection = (
                    squid_pad2[0] + coeff * connection_length,
                    self.squid.rect2[1] - np.abs((coeff - 1) / 2) * length_y * np.cos(
                        self.squid_params['angle']) - width)

                # add cpw from
                flux_line_output = (
                    self.center[0] + coeff * (self.core / 2 + self.gap + self.ground),
                    connection[1])

                connection_0 = find_normal_point(connection, flux_line_output, 20,
                                                 reverse=True)
                path1 = gdspy.FlexPath(deepcopy([connection_0, connection, flux_line_output]), [self.w, self.w],
                                       offset=[-self.s / 2 - self.w / 2, self.s / 2 + self.w / 2],
                                       layer=self.layer_configuration.total_layer)

                remove = gdspy.boolean(path1, remove, 'or', layer=self.layer_configuration.total_layer)

                remove_extra = gdspy.FlexPath(deepcopy([find_normal_point(connection, flux_line_output, 25),
                                                        find_normal_point(connection, flux_line_output, 25,
                                                                          reverse=True)]), [self.w],
                                              offset=[-self.s / 2 - self.w / 2])

                remove = gdspy.boolean(remove_extra, remove, 'or', layer=self.layer_configuration.total_layer)

                self.terminals['flux'] = DesignTerminal(flux_line_output, np.pi if coeff == 1 else 0,
                                                        g=self.g, s=self.s,
                                                        w=self.w, type='cpw')
            else:
                length = self.fluxline['length']
                for squid_pad in [self.squid.rect1, self.squid.rect2]:
                    rect1 = gdspy.Rectangle(
                        (squid_pad[0] + width / 2, squid_pad[1] - length * np.cos(self.squid_params['angle'])),
                        (squid_pad[0] - width / 2, squid_pad[1]))
                    result = gdspy.boolean(rect1, result, 'or', layer=self.layer_configuration.total_layer)

                    connection = (
                        self.squid.rect2[0], self.squid_params['y'] - self.squid_params['removing']['down'] * np.cos(
                            self.squid_params['angle']))
                    # add cpw from
                    flux_line_output = (
                        connection[0],
                        self.squid_params['y'] - (self.gap + self.ground) * np.cos(self.squid_params['angle']))

                    connection_0 = find_normal_point(connection, flux_line_output, 20, reverse=True)
                    path1 = gdspy.FlexPath(deepcopy([connection_0, connection, flux_line_output]), [self.w, self.w],
                                           offset=[-self.s / 2 - self.w / 2, self.s / 2 + self.w / 2],
                                           layer=self.layer_configuration.total_layer)

                    remove = gdspy.boolean(path1, remove, 'or', layer=self.layer_configuration.total_layer)

                    remove_extra = gdspy.FlexPath(deepcopy([find_normal_point(connection, flux_line_output, 25),
                                                            find_normal_point(connection, flux_line_output, 25,
                                                                              reverse=True)]), [self.w],
                                                  offset=[-self.s / 2 - self.w / 2])

                    remove = gdspy.boolean(remove_extra, remove, 'or', layer=self.layer_configuration.total_layer)

                    self.terminals['flux'] = DesignTerminal(flux_line_output, self.squid_params['angle'] + np.pi / 2,
                                                            g=self.g, s=self.s,
                                                            w=self.w, type='cpw')
        elif self.coil == 'new':
            if 'side' in self.squid_params:
                length_x = self.fluxline['length_x']
                length_y = self.fluxline['length_y']
                (coeff, squid_pad1, squid_pad2) = (1, self.squid.rect2, self.squid.rect1) if self.squid_params[
                                                                                                 'side'] == 'right' \
                    else (-1, self.squid.rect1, self.squid.rect2)
                connection_length = self.squid_params['removing']['right'] if self.squid_params['side'] == 'right' \
                    else self.squid_params['removing']['left']
                indent = 1
                rect1 = gdspy.Rectangle((squid_pad1[0] + length_x * coeff, squid_pad1[1] + indent / 2),
                                        (squid_pad1[0] - coeff * self.squid.rect_size_a / 2,
                                         squid_pad1[1] - width + indent / 2))
                result = gdspy.boolean(rect1, result, 'or', layer=self.layer_configuration.total_layer)
                connection = (self.squid_params['x'] + connection_length * coeff, squid_pad1[1] + (indent - width) / 2)
                # add cpw from
                flux_line_output = (self.center[0] + (self.core / 2 + self.gap + self.ground) * coeff, connection[1])
                remove = gdspy.FlexPath(deepcopy([connection, flux_line_output]), [self.w, self.w],
                                        offset=[-self.s, self.s], layer=self.layer_configuration.total_layer)
                self.terminals['flux'] = DesignTerminal(flux_line_output, np.pi if coeff == 1 else 0,
                                                        g=self.g, s=self.s,
                                                        w=self.w, type='cpw')
                # add connection to the ground
                first_turn = (squid_pad2[0], squid_pad2[1] - length_y)
                ground_connection = (first_turn[0] + coeff * self.gap + self.fluxline['width'], first_turn[1])
                rect2 = gdspy.FlexPath(deepcopy([squid_pad2, first_turn, ground_connection]), [width],
                                       offset=0, layer=self.layer_configuration.total_layer)
                result = gdspy.boolean(rect2, result, 'or', layer=self.layer_configuration.total_layer)
            else:  # for horizontal base couplers
                length = self.fluxline['length']
                for squid_pad in [self.squid.rect1, self.squid.rect2]:
                    rect1 = gdspy.Rectangle(
                        (squid_pad[0] + width / 2, squid_pad[1] - length * np.cos(self.squid_params['angle'])),
                        (squid_pad[0] - width / 2, squid_pad[1]))
                    result = gdspy.boolean(rect1, result, 'or', layer=self.layer_configuration.total_layer)

                connection = (
                self.squid.rect2[0], self.squid_params['y'] - self.squid_params['removing']['down'] * np.cos(
                    self.squid_params['angle']))
                # add cpw from
                flux_line_output = (
                    connection[0],
                    self.squid_params['y'] - (self.gap + self.ground) * np.cos(self.squid_params['angle']))
                remove = gdspy.FlexPath(deepcopy([connection, flux_line_output]), [self.w, self.w],
                                        offset=[-self.s, self.s], layer=self.layer_configuration.total_layer)
                self.terminals['flux'] = DesignTerminal(flux_line_output, self.squid_params['angle'] + np.pi / 2,
                                                        g=self.g, s=self.s,
                                                        w=self.w, type='cpw')
        else:
            raise ValueError('Coil type of flux line is not defined!')
        # return {'positive': result,
        #         'remove': remove,
        #         }
        return {'positive': result,
                'remove': remove
                }

    def get_terminals(self):
        return self.terminals

    def cm(self, epsilon):
        cross_section = [self.gap, self.core, self.gap]

        cl, ll = cm.ConformalMapping(cross_section, epsilon).cl_and_Ll()

        if not self.terminals['port1'].order:
            ll, cl = ll[::-1, ::-1], cl[::-1, ::-1]

        return cl, ll

    def add_to_tls(self, tls_instance: tlsim.TLSystem, terminal_mapping: dict, track_changes: bool = True,
                   cutoff: float = np.inf, epsilon: float = 11.45) -> list:
        from scipy.constants import hbar, e
        jj1 = tlsim.JosephsonJunction(self.squid_params['ic1'] * hbar / (2 * e), name=self.name + ' jj1')
        jj2 = tlsim.JosephsonJunction(self.squid_params['ic2'] * hbar / (2 * e), name=self.name + ' jj2')
        m = tlsim.Inductor(self.squid_params['lm'], name=self.name + ' flux-wire')

        cl, ll = self.cm(epsilon)
        c = cl[0, 0] * self.line_length
        l = ll[0, 0] * self.line_length
        c1 = tlsim.Capacitor(c=c / 2, name=self.name + '_c1')
        c2 = tlsim.Capacitor(c=c / 2, name=self.name + '_c2')
        l = tlsim.Inductor(l=l, name=self.name + '_l')

        cache = [jj1, jj2, m, l, c1, c2]

        tls_instance.add_element(jj1, [0, terminal_mapping['port1']])
        tls_instance.add_element(jj2, [terminal_mapping['flux'], terminal_mapping['port2']])
        tls_instance.add_element(m, [0, terminal_mapping['flux']])
        tls_instance.add_element(l, [terminal_mapping['port1'], terminal_mapping['port2']])
        tls_instance.add_element(c1, [terminal_mapping['port1'], 0])
        tls_instance.add_element(c2, [terminal_mapping['port2'], 0])

        if track_changes:
            self.tls_cache.append(cache)
        return cache

    def generate_rect(self, offset1, offset2, layer):
        return gdspy.Rectangle((self.port1_position[0] + offset1 * np.sin(self.orientation),
                                self.port1_position[1] + offset1 * np.cos(self.orientation)),
                               (self.port2_position[0] + offset2 * np.sin(self.orientation),
                                self.port2_position[1] + offset2 * np.cos(self.orientation)), layer=layer)


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
