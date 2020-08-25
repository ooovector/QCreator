import gdspy
import numpy as np
#import libraries.squid3JJ as squid3JJ
#import libraries.JJ4q as JJ4q
from . import transmission_line_simulator as tlsim
from . import conformal_mapping as cm
from copy import deepcopy
from typing import Tuple, List
from abc import *
#from . import conformal_mapping as cm


class DesignTerminal:
    def __init__(self, position, orientation, type, core, gap, ground):
        self.position = position
        self.orientation = orientation
        self.type = type
        self.core = core
        self.gap = gap
        self.ground = ground


class DesignElement:
    def __init__(self, type, name):
        self.type = type
        self.name = name
        self.resource = None
        self.modifiers = []

    def get(self):
        if self.resource is None:
            self.resource = self.render()
        return self.resource

    @abstractmethod
    def render(self):
        """
        Draw the element on the design gds, by getting dependencies
        """
        pass

    @abstractmethod
    def get_terminals(self) -> dict:
        """
        Returns a list of terminals for the transmission line model of the system
        :return:
        """
        pass

    @abstractmethod
    def add_to_tls(self, tls_instance: tlsim.TLSystem,
                   terminal_mapping: dict, track_changes: bool = True) -> list:
        """
        Adds the circuit to a transmission line system model
        :param tls_instance: transmission_line_system class instance to add the model elements to
        :param terminal_mapping: dict that maps terminal names of this element to transmission line system node ids
        :param track_changes: add element to tracked so that its z0 gets automatically changed
        :return: list of circuit elements
        """
        pass


class LayerConfiguration(DesignElement):
    def __init__(self, **layer_configurations):
        super().__init__('layout_configuration', 'layout_configuration')
        self.layer_configurations = layer_configurations
        self.total_layer = layer_configurations['total']
        self.restricted_area_layer = layer_configurations['restricted area']
        self.layer_to_remove = layer_configurations['for removing']
        self.jj_layer = layer_configurations['JJs']
        self.airbridges_layer = layer_configurations['air bridges']
        self.airbridges_pad_layer = layer_configurations['air bridge pads']
        self.gridline_x_layer = layer_configurations['vertical gridlines']
        self.gridline_y_layer = layer_configurations['horizontal gridlines']

    def render(self):
        return {}

    def get_terminals(self) -> dict:
        return {}

    def add_to_tls(self, tls_instance: tlsim.TLSystem,
                   terminal_mapping: dict, track_changes: bool = True) -> list:
        return []


class ChipGeometry(DesignElement):
    def __init__(self, **chip_geometry):
        super().__init__('chip_geometry', 'chip_geometry')
        if 'sample_vertical_size' in chip_geometry:
            self.sample_vertical_size = chip_geometry['sample_vertical_size']
        if 'sample_horizontal_size' in chip_geometry:
            self.sample_horizontal_size = chip_geometry['sample_horizontal_size']

    def render(self):
        return {}

    def get_terminals(self) -> dict:
        return {}

    def add_to_tls(self, tls_instance: tlsim.TLSystem,
                   terminal_mapping: dict, track_changes: bool = True) -> list:
        return []


class ChipEdgeGround(DesignElement):
    def __init__(self, chip_geometry, layer_configuration, pads):
        super().__init__('chip_edge_ground', 'chip_edge_ground')
        self.chip_geometry = chip_geometry
        self.layer_configuration = layer_configuration
        self.pads = pads

    def render(self):
        """
        Draws edge ground metallization on chip
        :return:
        """
        self.chip_geometry.render()
        self.layer_configuration.render()
        chip_geometry = self.chip_geometry
        edge = 600 #  fundamental constant - edge length
        r1 = gdspy.Rectangle((0, 0), (chip_geometry.sample_horizontal_size, chip_geometry.sample_vertical_size))
        r2 = gdspy.Rectangle((edge, edge), (chip_geometry.sample_horizontal_size - edge, chip_geometry.sample_vertical_size - edge))
        result = gdspy.boolean(r1, r2, 'not')

        for pad in self.pads.items():
        #pads = gdspy.polygon.PolygonSet(pads)
            pad = pad.get()
            to_bool = gdspy.Rectangle(pad['positive'].get_bounding_box()[0].tolist(), pad['positive'].get_bounding_box()[1].tolist())
            result = gdspy.boolean(result, to_bool, 'not')

        return {'positive': result}

    def get_terminals(self) -> dict:
        return {}

    def add_to_tls(self, tls_instance: tlsim.TLSystem,
                   terminal_mapping: dict, track_changes: bool = True) -> list:
        return []


class Pads(DesignElement):
    def __init__(self, object_list):
        super().__init__('pads', 'pads')
        self.object_list = object_list

    def render(self):
        result = []
        restricted_area = []
        #for object_ in self.object_list:
        #    if type(object_) is Pad:
        #        r, ra = object_.get()
        #        result.append(r)
        #        restricted_area.append(ra)
        return {}

    def items(self):
        for object_ in self.object_list:
            if type(object_) is Pad:
                yield object_

    def get_terminals(self) -> dict:
        return {}

    def add_to_tls(self, tls_instance: tlsim.TLSystem,
                   terminal_mapping: dict, track_changes: bool = True) -> list:
        return []


class Pad(DesignElement):
    """
    Contact pad for bonding the chip to the PCB
    """
    def __init__(self, name: str, w: float, s: float, g: float, position, z0: float, orientation: float,
                 layer_configuration: LayerConfiguration, chip_geometry: ChipGeometry):
        """

        :param name: Design element name
        :param w: CPW signal conductor width
        :param s: CPW signal-ground gap
        :param g: Ground conductor width
        :param position: Position on the chip
        :param orientation: Orientation on chip in radians; 0 is along x positive direction (right-looking)
        :param z0: characteristic impedance of port for transmission line system simulation
        """
        super().__init__(self, name)
        self._z0 = z0
        self.tls_cache = []
        self.layer_configuration = layer_configuration
        self.chip_geometry = chip_geometry
        self.terminal = DesignTerminal(position=position, orientation=orientation, type='cpw', core=w, gap=s, ground=g)

    @property
    def z0(self):
        return self._z0

    @z0.setter
    def z0(self, value):
        if self._z0 != value:
            for port in self.tls_cache:
                port.Z0 = value
        self._z0 = value

    def render(self):
        self.chip_geometry.get()
        self.layer_configuration.get()
        coord_init_x, coord_init_y = self.terminal.position
        w = self.terminal.core
        s = self.terminal.gap
        g = self.terminal.ground
        outer_tl_width = w + 2 * (g + s)
        x, y = (coord_init_x - outer_tl_width / 2, coord_init_y)

        pad_core = 250 #  to make pad with 50 Om impedance
        pad_vacuum = 146 #  to make pad with 50 Om impedance
        pad_ground = g
        pad_length = 600
        pad_indent = 50
        edge_indent = 100
        narrowing = 160
        outer_pad_width = (pad_core + (pad_vacuum + pad_ground) * 2)
        inner_pad_width = (pad_core + pad_vacuum * 2)
        outer_width = 2 * (g + s) + w
        inner_width = 2 * s + w

        r1 = gdspy.Polygon([(x, y), (x + outer_width, y),
                            (x + (outer_width + outer_pad_width) / 2, y - narrowing),
                            (x + (outer_width + outer_pad_width) / 2, y - (narrowing + pad_length)),
                            (x - (-outer_width + outer_pad_width) / 2, y - (narrowing + pad_length)),
                            (x - (-outer_width + outer_pad_width) / 2, y - narrowing)])
        x += g
        r2 = gdspy.Polygon([(x, y), (x + inner_width, y),
                            (x + (inner_width + inner_pad_width) / 2, y - narrowing),
                            (x + (inner_width + inner_pad_width) / 2, y - (narrowing + pad_length - edge_indent)),
                            (x - (-inner_width + inner_pad_width) / 2, y - (narrowing + pad_length - edge_indent)),
                            (x - (-inner_width + inner_pad_width) / 2, y - narrowing)])
        x += s
        r3 = gdspy.Polygon([(x, y), (x + w, y),
                            (x + (pad_core + w) / 2, y - narrowing),
                            (x + (pad_core + w) / 2, y - (narrowing + pad_length - pad_indent - edge_indent)),
                            (x - (pad_core - w) / 2, y - (narrowing + pad_length - pad_indent - edge_indent)),
                            (x - (pad_core - w) / 2, y - narrowing)])
        pad, restricted_pad = gdspy.boolean(gdspy.boolean(r1, r2, 'not'), r3, 'or'), r1

        pad.rotate(self.terminal.orientation, [coord_init_x, coord_init_y])
        restricted_pad.rotate(self.terminal.orientation, [coord_init_x, coord_init_y])

        return {'positive': pad, 'restrict': restricted_pad}

    def get_terminals(self) -> dict:
        return {'port': self.terminal}

    def add_to_tls(self, tls_instance: tlsim.TLSystem, terminal_mapping: dict,
                   track_changes: bool = True) -> list:
        p = tlsim.Port(z0=self.z0)
        if track_changes:
            self.tls_cache.append(p)

        tls_instance.add_element(p, [terminal_mapping['port']])
        return [p]


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


class CPW(DesignElement):
    def __init__(self, name, points, core, gap, ground, nodes, layer_configuration: LayerConfiguration, R,
                 corner_type: str='round'):
        super().__init__('cpw', name)
        self.core = core
        self.ground = ground
        self.gap = gap
        self.points = points
        self.nodes = nodes
        self._R = R # fundamental constant
        self.restricted_area = None
        self.layer_configuration = layer_configuration
        self.end = self.points[-1]
        self.angle = None
        self.corner_type = corner_type
        self.length = None
        self.tls_cache = []

    def render(self):
        offset = self.gap + (self.core + self.ground) / 2

        R1 = self._R - (self.core / 2 + self.gap + self.ground / 2)
        R2 = np.abs(self._R + self.core / 2 + self.gap + self.ground / 2)

        R1_new = self._R - (self.core / 2 + self.gap / 2)
        R2_new = np.abs(self._R + self.core / 2 + self.gap / 2)

        result = None
        result_restricted = None
        result_new = None

        if len(self.points) < 3:
            self.points.insert(1, ((self.points[0][0] + self.points[1][0]) / 2,
                                   (self.points[0][1] + self.points[1][1]) / 2))
        new_points = self.points
        # new_points_restricted_line= self.points

        width_restricted_line = 2 * self.ground + 2 * self.gap + self.core
        width_new = self.gap
        offset_new = self.gap + self.core

        if R1 > 0:
            for i in range(len(self.points) - 2):
                point1 = self.points[i]
                point2 = self.points[i + 1]
                if i < len(self.points) - 3:
                    point3 = ((self.points[i + 1][0] + self.points[i + 2][0]) / 2,
                              (self.points[i + 1][1] + self.points[i + 2][1]) / 2)
                else:
                    point3 = new_points[-1]
                self.points[i + 1] = ((self.points[i + 1][0] + self.points[i + 2][0]) / 2,
                                      (self.points[i + 1][1] + self.points[i + 2][1]) / 2)
                if self.corner_type is 'round':
                    vector1 = np.asarray(new_points[i + 1][:]) - new_points[i][:]
                    vector2 = np.asarray(new_points[i + 2][:]) - new_points[i + 1][:]
                    vector_prod = vector1[0] * vector2[1] - vector1[1] * vector2[0]
                    if vector_prod < 0:
                        line = gdspy.FlexPath([point1, point2, point3],
                                              [self.ground, self.core,
                                               self.ground], offset, ends=["flush", "flush", "flush"],
                                              corners=["circular bend", "circular bend", "circular bend"],
                                              bend_radius=[R1, self._R, R2], precision=0.001, layer=0)
                        restricted_line = gdspy.FlexPath([point1, point2, point3], width_restricted_line, offset=0,
                                                         ends="flush", corners="circular bend", bend_radius=self._R,
                                                         precision=0.001, layer=1)
                        line_new = gdspy.FlexPath([point1, point2, point3], [width_new, width_new], offset=offset_new,
                                                  ends=["flush", "flush"], corners=["circular bend", "circular bend"],
                                                  bend_radius=[R1_new, R2_new], precision=0.001, layer=2)

                        # rectricted_line = gdspy.FlexPath([point1, point2, point3], width_restricted_line, offset=0,  ends="flush",  corners="circular bend",bend_radius= self._R, precision=0.001, layer=1)

                        # self.end = line.x
                    else:
                        line = gdspy.FlexPath([point1, point2, point3],
                                              [self.ground, self.core,
                                               self.ground], offset, ends=["flush", "flush", "flush"],
                                              corners=["circular bend", "circular bend", "circular bend"],
                                              bend_radius=[R2, self._R, R1], precision=0.001, layer=0)
                        restricted_line = gdspy.FlexPath([point1, point2, point3], width_restricted_line, offset=0,
                                                         ends="flush", corners="circular bend", bend_radius=self._R,
                                                         precision=0.001, layer=1)
                        line_new = gdspy.FlexPath([point1, point2, point3], [width_new, width_new], offset=offset_new,
                                                  ends=["flush", "flush"], corners=["circular bend", "circular bend"],
                                                  bend_radius=[R2_new, R1_new], precision=0.001, layer=2)

                        # self.end = line.x
                    result = gdspy.boolean(line, result, 'or', layer=self.layer_configuration.total_layer)
                    result_restricted = gdspy.boolean(restricted_line, result_restricted, 'or', layer=self.layer_configuration.restricted_area_layer)
                    result_new = gdspy.boolean(line_new, result_new, 'or', layer=2)
                    # self.restricted_area = gdspy.boolean(rectricted_line, self.restricted_area, 'or', layer = self.restricted_area_layer)

                    # result= line, restricted_line
                else:
                    line = gdspy.FlexPath([point1, point2, point3],
                                          [self.ground, self.core,
                                           self.ground], offset, ends=["flush", "flush", "flush"],
                                          precision=0.001, layer=0)
                    # self.end = line.x
                    restricted_line = gdspy.FlexPath([point1, point2, point3],
                                                     width_restricted_line, offset=0, ends="flush",
                                                     precision=0.001, layer=1)

                    line_new = gdspy.FlexPath([point1, point2, point3],
                                              [width_new, width_new], offset=offset_new, ends=["flush", "flush"],
                                              precision=0.001, layer=2)

                    # result= line, rectricted_line
                    result = gdspy.boolean(line, result, 'or', layer=self.layer_configuration.total_layer)
                    result_restricted = gdspy.boolean(restricted_line, result_restricted, 'or', layer=self.layer_configuration.restricted_area_layer)
                    result_new = gdspy.boolean(line_new, result_new, 'or', layer=2)

                    # self.restricted_area = gdspy.boolean(rectricted_line, self.restricted_area, 'or', layer = self.restricted_area_layer)

        else:
            print('R small < 0')
            result = 0
            result_restricted = 0
            result_new = 0
        vector2_x = point3[0] - point2[0]
        vector2_y = point3[1] - point2[1]
        if vector2_x != 0 and vector2_x >= 0:
            tang_alpha=vector2_y/vector2_x
            self.angle = np.arctan(tang_alpha)
        elif vector2_x != 0 and vector2_x < 0:
            tang_alpha=vector2_y/vector2_x
            self.angle = np.arctan(tang_alpha)+np.pi
        elif vector2_x == 0 and vector2_y > 0:
            self.angle = np.pi/2
        elif vector2_x == 0 and vector2_y < 0:
            self.angle = -np.pi/2
        else:
            print("something is wrong in angle")
        return {'positive':result, 'restrict':result_restricted, 'remove':result_new}

    def get_terminals(self):
        orientation1 = np.arctan2(self.points[1][1]-self.points[0][1], self.points[1][0]-self.points[0][0])
        orientation2 = np.arctan2(self.points[-1][1] - self.points[-2][1], self.points[-1][0] - self.points[-2][0])
        return {'port1':(self.points[0], orientation1), 'port2':(self.points[1], orientation2)}

    def cm(self):
        return cm.ConformalMapping([self.gap, self.core, self.gap]).cl_and_Ll()

    def add_to_tls(self, tls_instance: tlsim.TLSystem,
                   terminal_mapping: dict, track_changes: bool = True) -> list:
        cl, ll = self.cm()
        line = tlsim.TLCoupler(n=1,
                             l=0,  #TODO: get length
                             cl=cl,
                             ll=ll,
                             rl=[[0]],
                             gl=[[0]])

        if track_changes:
            self.tls_cache.append([line])

        tls_instance.add_element(line, [terminal_mapping['port1'], terminal_mapping['port2']])
        return [line]

    def generate_end(self, end):
        if end['type'] is 'open':
            return self.generate_open_end(end)
        if end['type'] is 'fluxline':
            return self.generate_fluxline_end(end)

    def generate_fluxline_end(self,end):
        jj = end['JJ']
        length = end['length']
        width = end['width']
        point1 = jj.rect1
        point2 = jj.rect2
        result = None
        # rect_to_remove = gdspy.Rectangle((),
        #                                  ())
        for point in [point1, point2]:
            line = gdspy.Rectangle((point[0] - width/2, point[1]),
                                   (point[0] + width/2, point[1] - length))
            result = gdspy.boolean(line, result, 'or', layer=6)
        # result = gdspy.boolean(line1,line2,'or',layer=self.total_layer)
        path1 = gdspy.Polygon([(point1[0] + width / 2, point1[1] - length), (point1[0] - width / 2, point1[1] - length),
                               (self.end[0] + (self.core / 2 + self.gap + width) * np.cos(self.angle + np.pi/2),
                                self.end[1] + (self.core / 2 + self.gap + width) * np.sin(self.angle + np.pi/2)),
                               (self.end[0] + (self.core / 2 + self.gap) * np.cos(self.angle + np.pi/2),
                                self.end[1] + (self.core / 2 + self.gap) * np.sin(self.angle + np.pi/2))])

        result = gdspy.boolean(path1, result, 'or', layer=6)

        path2 = gdspy.Polygon([(point2[0] + width / 2, point2[1] - length), (point2[0] - width / 2, point2[1] - length),
                               (self.end[0] +(self.core / 2)*np.cos(self.angle+np.pi/2), self.end[1]+( self.core / 2)*np.sin(self.angle+np.pi/2)),
                               (self.end[0] + self.core / 2 *np.cos(self.angle+3*np.pi/2), self.end[1]+( self.core / 2)*np.sin(self.angle+3*np.pi/2))])
        result = gdspy.boolean(path2, result, 'or', layer=6)

        # if end['type'] != 'coupler':
        restricted_area = gdspy.Polygon([(point1[0] - width / 2, point1[1]),
                                         (point2[0] + width / 2 + self.gap, point2[1]),
                                         (point2[0] + width / 2 + self.gap, point2[1] - length),
                                         (self.end[0] + (self.core / 2 + self.gap) * np.cos(
                                             self.angle + 3 * np.pi / 2),
                                          self.end[1] + (self.core / 2 + self.gap) * np.sin(
                                              self.angle + 3 * np.pi / 2)),
                                         (self.end[0] + (self.core / 2 + self.gap + width) * np.cos(
                                             self.angle + np.pi / 2),
                                          self.end[1] + (self.core / 2 + self.gap + width) * np.sin(
                                              self.angle + np.pi / 2)),
                                         (point1[0] - width / 2, point1[1] - length)],
                                        layer=self.restricted_area_layer)
        # else:
        #
        return result, restricted_area, restricted_area

    def generate_open_end(self, end):
        end_gap = end['gap']
        end_ground_length = end['ground']
        if end['end']:
            x_begin = self.end[0]
            y_begin = self.end[1]
            additional_rotation = -np.pi/2
        if end['begin']:
            x_begin = self.points[0][0]
            y_begin = self.points[0][1]
            additional_rotation = -np.pi / 2
        restricted_area = gdspy.Rectangle((x_begin-self.core/2-self.gap-self.ground, y_begin),
                                          (x_begin+self.core/2+self.gap+self.ground, y_begin+end_gap+end_ground_length), layer=10)#fix it
        rectangle_for_removing = gdspy.Rectangle((x_begin-self.core/2-self.gap, y_begin),
                                                 (x_begin+self.core/2+self.gap, y_begin+end_gap), layer=2)
        total = gdspy.boolean(restricted_area, rectangle_for_removing, 'not')
        for obj in [total,restricted_area, rectangle_for_removing]:
            obj.rotate(additional_rotation+self.angle, (x_begin, y_begin))
        return total, restricted_area, rectangle_for_removing


class Coaxmon(DesignElement):
    def __init__(self, name, center, r1, r2, r3, R4, outer_ground, total_layer, restricted_area_layer, JJ_layer, Couplers, JJ):
        super().__init__(name)
        self.center = center
        self.R1 = r1
        self.R2 = r2
        self.R3 = r3
        self.R4 = R4
        self.outer_ground = outer_ground
        self.couplers = Couplers
        self.restricted_area_layer = restricted_area_layer
        self.total_layer = total_layer
        self.JJ_layer = JJ_layer
        self.JJ_params = JJ
        self.JJ = None

    def generate_qubit(self):
        ground = gdspy.Round(self.center, self.outer_ground, self.R4, initial_angle=0, final_angle=2*np.pi)
        restricted_area = gdspy.Round(self.center, self.outer_ground,  layer=self.restricted_area_layer)
        core = gdspy.Round(self.center, self.R1, inner_radius=0, initial_angle=0, final_angle=2*np.pi)
        result = gdspy.boolean(ground, core, 'or', layer=self.total_layer)
        if len(self.couplers) != 0:
            for coupler in self.couplers:
                if coupler.grounded == True:
                    result = gdspy.boolean(coupler.generate_coupler(self.center, self.R2, self.outer_ground,self.R4 ), result, 'or', layer=self.total_layer)
                else:
                    result = gdspy.boolean(coupler.generate_coupler(self.center, self.R2, self.R3, self.R4), result, 'or', layer=self.total_layer)
        self.JJ_coordinates = (self.center[0] + self.R1*np.cos(self.JJ_params['angle_qubit']), self.center[1] + self.R1*np.sin(self.JJ_params['angle_qubit']))
        JJ,rect = self.generate_JJ()
        result = gdspy.boolean(result, rect, 'or')
        # self.AB1_coordinates = coordinates(self.center.x, self.center.y + self.R4)
        # self.AB2_coordinates = coordinates(self.center.x, self.center.y - self.outer_ground)
        return result, restricted_area, JJ

    def generate_JJ(self):
        self.JJ = squid3JJ.JJ_2(self.JJ_coordinates[0], self.JJ_coordinates[1],
                    self.JJ_params['a1'], self.JJ_params['a2'],
                    self.JJ_params['b1'], self.JJ_params['b2'],
                    self.JJ_params['c1'], self.JJ_params['c2'])
        result = self.JJ.generate_jj()
        result = gdspy.boolean(result, result, 'or', layer=self.JJ_layer)
        angle = self.JJ_params['angle_JJ']
        # print(self.JJ_coordinates[0],self.JJ_coordinates[1])
        # print((self.JJ_coordinates[0],self.JJ_coordinates[1]))
        result.rotate(angle, (self.JJ_coordinates[0], self.JJ_coordinates[1]))
        rect = gdspy.Rectangle((self.JJ_coordinates[0] - self.JJ.contact_pad_a_outer/2, self.JJ_coordinates[1] + self.JJ.contact_pad_b_outer),
                               (self.JJ_coordinates[0] + self.JJ.contact_pad_a_outer/2, self.JJ_coordinates[1] - self.JJ.contact_pad_b_outer),layer=self.total_layer)
        rect.rotate(angle, (self.JJ_coordinates[0], self.JJ_coordinates[1]))
        return result, rect


class QubitCoupler(DesignElement):
    def __init__(self, name, arc_start, arc_finish, phi, w, grounded=False):
        super().__init__(name)
        self.arc_start = arc_start
        self.arc_finish = arc_finish
        self.phi = phi
        self.w = w
        self.grounded = grounded

    def generate_coupler(self, coordinate, r_init, r_final, rect_end):
        #to fix bug
        bug = 5
        result = gdspy.Round(coordinate, r_init, r_final,
                             initial_angle=self.arc_start * np.pi,
                             final_angle=self.arc_finish * np.pi)
        rect = gdspy.Rectangle((coordinate[0] + r_final - bug, coordinate[1] - self.w / 2),
                               (coordinate[0] + rect_end + bug, coordinate[1] + self.w / 2))
        rect.rotate(self.phi*np.pi, coordinate)
        return gdspy.boolean(result,rect, 'or')




class Narrowing(DesignElement):
    def __init__(self, name: str, x: float, y: float, core1: float, gap1: float, ground1: float,
                 core2: float, gap2: float, ground2: float, h: float, rotation: float):
        super().__init__(name)
        self._x_begin = x
        self._y_begin = y

        self.first_core = core1
        self.first_gap = gap1
        self.first_ground = ground1

        self.second_core = core2
        self.second_gap = gap2
        self.second_ground = ground2
        self._h=h

        self._rotation = rotation

    def generate_narrowing(self):
        x_end = self._x_begin-self._h
        y_end = self._y_begin

        points_for_poly1 = [(self._x_begin,self._y_begin+self.first_core/2+self.first_gap+self.first_ground),
                  (self._x_begin, self._y_begin+self.first_core/2+self.first_gap),
                  (x_end, y_end+self.second_core/2+self.second_gap),
                  (x_end, y_end+self.second_core/2+self.second_gap+self.second_ground)]

        points_for_poly2 = [(self._x_begin,self._y_begin+self.first_core/2),
                  (self._x_begin, self._y_begin-self.first_core/2),
                  (x_end, y_end-self.second_core/2),
                  (x_end, y_end+self.second_core/2)]

        points_for_poly3 = [(self._x_begin,self._y_begin-(self.first_core/2+self.first_gap+self.first_ground)),
                  (self._x_begin, self._y_begin-(self.first_core/2+self.first_gap)),
                  (x_end, y_end-(self.second_core/2+self.second_gap)),
                  (x_end, y_end-(self.second_core/2+self.second_gap+self.second_ground))]

        points_for_restricted_area = [(self._x_begin, self._y_begin+ self.first_core/2 + self.first_gap + self.first_ground),
                                    (x_end, y_end+self.second_core/2+self.second_gap+self.second_ground),
                                    (x_end, y_end-(self.second_core/2+self.second_gap+self.second_ground)),
                                    (self._x_begin, self._y_begin-(self.first_core/2+self.first_gap+self.first_ground))]

        restricted_area = gdspy.Polygon(points_for_restricted_area)

        poly1 = gdspy.Polygon(points_for_poly1)
        poly2 = gdspy.Polygon(points_for_poly2)
        poly3 = gdspy.Polygon(points_for_poly3)

        if self._rotation == 0:
            result = poly1, poly2, poly3
        else:
            poly1_ = poly1.rotate(angle=self._rotation, center=(self._x_begin, self._y_begin))
            poly2_ = poly2.rotate(angle=self._rotation, center=(self._x_begin, self._y_begin))
            poly3_ = poly3.rotate(angle=self._rotation, center=(self._x_begin, self._y_begin))
            restricted_area.rotate(angle=self._rotation, center=(self._x_begin, self._y_begin))
            result = poly1_, poly2_, poly3_

        polygon_to_remove = gdspy.boolean(restricted_area, result, 'not', layer=2)

        return result, restricted_area, polygon_to_remove


class IlyaCoupler(DesignElement):
    def __init__(self, name: str, core: float, gap: float, ground: float, coaxmon1: Coaxmon, coaxmon2: Coaxmon,
                 jj_params: dict, squid_params: dict, total_layer: int, restricted_area_layer: int, jj_layer: int,
                 layer_to_remove: int):
        super().__init__(name)
        self.core = core
        self.gap = gap
        self.ground = ground
        self.coaxmon1 = coaxmon1
        self.coaxmon2 = coaxmon2
        self.total_layer = total_layer
        self.restricted_area_layer = restricted_area_layer
        self.angle = None
        self.jj_layer = jj_layer
        self.layer_to_remove = layer_to_remove
        self.jj_params = jj_params
        self.squid_params = squid_params

    def generate_coupler(self):
        vector2_x = self.coaxmon2.center[0] - self.coaxmon1.center[0]
        vector2_y = self.coaxmon2.center[1] - self.coaxmon1.center[1]
        if vector2_x != 0 and vector2_x >= 0:
            tang_alpha = vector2_y / vector2_x
            self.angle = np.arctan(tang_alpha)
        elif vector2_x != 0 and vector2_x < 0:
            tang_alpha = vector2_y / vector2_x
            self.angle = np.arctan(tang_alpha) + np.pi
        elif vector2_x == 0 and vector2_y > 0:
            self.angle = np.pi / 2
        elif vector2_x == 0 and vector2_y < 0:
            self.angle = -np.pi / 2
        bug = 3# there is a bug
        points = [(self.coaxmon1.center[0] + (self.coaxmon1.R4 - bug) * np.cos(self.angle),
                   self.coaxmon1.center[1] + (self.coaxmon1.R4 - bug) * np.sin(self.angle)),
                  (self.coaxmon2.center[0] + (self.coaxmon1.R4 - bug) * np.cos(self.angle + np.pi),
                   self.coaxmon2.center[1] + (self.coaxmon1.R4 - bug) * np.sin(self.angle + np.pi))]
        line = Feedline(points, self.core, self.gap, self.ground, None, self.total_layer, self.restricted_area_layer, 100)
        line = line.generate_feedline()
        JJ1 = self.generate_jj()
        self.jj_params['indent'] = np.abs(self.coaxmon2.center[1] - self.coaxmon1.center[1]) + \
                                   np.abs(self.coaxmon2.center[0] - self.coaxmon1.center[0]) - \
                                   2 * self.coaxmon1.R4 - self.jj_params['indent']
        JJ2 = self.generate_jj()
        squid = self.generate_squid()
        JJ_0 = gdspy.boolean(JJ1[0], JJ2[0], 'or', layer=self.jj_layer)
        JJ_1 = gdspy.boolean(JJ1[1], JJ2[1], 'or', layer=6)
        JJ_2 = gdspy.boolean(JJ1[2], JJ2[2], 'or', layer=self.layer_to_remove)

        JJ_0 = gdspy.boolean(JJ_0, squid[0], 'or', layer=self.jj_layer)
        JJ_1 = gdspy.boolean(JJ_1, squid[1], 'or', layer=6)
        # result = gdspy.boolean(JJ[1],line,'or',layer=self.total_layer)
        return line, [JJ_0, JJ_1, JJ_2]

    def generate_jj(self):
        self.jj_params['x'] = self.coaxmon1.center[0] + (self.coaxmon1.R4 + self.jj_params['indent']) * np.cos(self.angle)
        if self.coaxmon1.center[0] != self.coaxmon2.center[0]:
            self.jj_params['y'] = self.coaxmon1.center[1] + (self.coaxmon1.R4 +
                                                             self.jj_params['indent']) * np.sin(self.angle) + (self.core / 2 + self.gap / 2)
        else:
            self.jj_params['y'] = self.coaxmon1.center[1] + (self.coaxmon1.R4 + self.jj_params['indent']) * np.sin(self.angle)
        # print(self.angle)
        self.jj = JJ4q.JJ_1(self.jj_params['x'], self.jj_params['y'],
                            self.jj_params['a1'], self.jj_params['a2'],
                            )
        result = self.jj.generate_jj()
        result = gdspy.boolean(result, result, 'or', layer=self.jj_layer)
        angle = self.jj_params['angle_JJ']
        result.rotate(angle, (self.jj_params['x'], self.jj_params['y']))
        indent = 1
        rect1 = gdspy.Rectangle((self.jj_params['x'] - self.jj.contact_pad_a / 2,
                                 self.jj_params['y'] + indent),
                                (self.jj_params['x'] + self.jj.contact_pad_a / 2,
                                 self.jj_params['y'] - self.jj.contact_pad_b + indent), layer=6)
        rect2 = gdspy.Rectangle((self.jj.x_end - self.jj.contact_pad_a / 2,
                                 self.jj.y_end - 1),
                                (self.jj.x_end + self.jj.contact_pad_a / 2 ,
                                 self.jj.y_end - self.jj.contact_pad_b - indent), layer=6)
        if self.coaxmon1.center[0] != self.coaxmon2.center[0]:
            poly1 = gdspy.Polygon([(self.jj_params['x'] - self.jj.contact_pad_a / 2,
                                    self.jj_params['y'] + indent),
                                   (self.jj_params['x'] - self.jj.contact_pad_a / 2,
                                    self.jj_params['y'] + indent - self.jj.contact_pad_b),
                                   (self.jj_params['x'] - self.jj.contact_pad_a - indent, self.coaxmon1.center[1] - self.core / 2),
                                   (self.jj_params['x'] - self.jj.contact_pad_a - indent, self.coaxmon1.center[1] + self.core / 2)
                                   ])
            poly2 = gdspy.Polygon([(self.jj.x_end + self.jj.contact_pad_a / 2,
                                    self.jj.y_end - indent - self.jj.contact_pad_b),
                                   (self.jj.x_end + self.jj.contact_pad_a / 2,
                                    self.jj.y_end - indent),
                                   (self.jj.x_end + self.jj.contact_pad_a + indent,
                                    self.coaxmon1.center[1] + self.core / 2),
                                   (self.jj.x_end + self.jj.contact_pad_a + indent,
                                    self.coaxmon1.center[1] - self.core / 2)
                                   ])
        else:
            poly1 = []
            poly2 = []
        rect = gdspy.boolean(rect1,[rect2,poly1,poly2], 'or', layer=6)
        rect.rotate(angle, (self.jj_params['x'], self.jj_params['y']))
        to_remove = gdspy.Polygon(self.jj.points_to_remove, layer=self.layer_to_remove)
        to_remove.rotate(angle, (self.jj_params['x'], self.jj_params['y']))
        return result, rect, to_remove

    def generate_squid(self):
        # print(self.squid_params)
        self.squid = squid3JJ.JJ_2(self.squid_params['x'],
                                   self.squid_params['y'],
                self.squid_params['a1'], self.squid_params['a2'],
                self.squid_params['b1'], self.squid_params['b2'],
                self.squid_params['c1'], self.squid_params['c2'])
        squid = self.squid.generate_jj()
        rect = gdspy.Rectangle((self.squid_params['x'] - self.squid.contact_pad_a_outer / 2,
                                self.squid_params['y'] + 0*self.squid.contact_pad_b_outer/2),
                               (self.squid_params['x'] + self.squid.contact_pad_a_outer / 2,
                                self.squid_params['y'] - self.squid.contact_pad_b_outer), layer=self.total_layer)

        if self.coaxmon1.center[0] == self.coaxmon2.center[0]:
            path1 = gdspy.Polygon([(self.squid_params['x'], self.squid_params['y'] ),
                                 (self.coaxmon1.center[0], self.squid_params['y']),
                                 (self.coaxmon1.center[0], self.squid_params['y'] - self.squid.contact_pad_b_outer),
                                 (self.squid_params['x'], self.squid_params['y'] - self.squid.contact_pad_b_outer)])
            rect = gdspy.boolean(rect, path1, 'or', layer=self.total_layer)

        # point1 =
        squid = gdspy.boolean(squid, squid, 'or', layer=self.jj_layer)
        squid.rotate(self.squid_params['angle'], (self.squid_params['x'], self.squid_params['y']))
        return squid, rect


class RoundResonator(DesignElement):

    def __init__(self, frequency, initial_point, core, gap, ground, open_end_length,open_end, coupler_length, l1, l2, l3,l4, l5, h_end,corner_type,
                 total_layer, restricted_area_layer):
        self._start_x = initial_point[0] + coupler_length/2
        self._start_y = initial_point[1]-open_end_length
        self.total_layer = total_layer
        self.restricted_area_layer = restricted_area_layer
        self.core = core
        self.ground = ground
        self.gap = gap
        self.open_end_length = open_end_length
        self.open_end = open_end
        self.coupler_length = coupler_length
        self._l1 = l1
        self._l2 = l2
        self._l3 = l3
        self._l4 = l4
        self._l5 = l5
        self.f = frequency
        self.c = 299792458
        self.epsilon_eff = (11.45+1)/2
        self.L = self.c/(4*np.sqrt(self.epsilon_eff)*frequency)*1e6
        self._h_end = h_end
        self.corner_type = corner_type
        self.points = None

    def generate_resonator(self):
        # specify points to generate everything before a meander
        points = [(self._start_x, self._start_y), (self._start_x, self._start_y + self.open_end_length),
                  (self._start_x-self.coupler_length/2, self._start_y),
                  (self._start_x-self.coupler_length/2, self._start_y - self._l1),
                  (self._start_x-self.coupler_length/2 + self._l2,  self._start_y - self._l1),
                  (self._start_x-self.coupler_length/2 + self._l2, self._start_y - self._l1-self._l3)]
        # line0 = Feedline(points, self.core, self.gap, self.ground, None, self.total_layer, self.restricted_area_layer,
        #                  R=40)
        # line = line0.generate_feedline(self.corner_type)
        # open_end = line0.generate_end(self.open_end)
        # generate the meander
        L_meander=self.L - self.open_end_length-self.coupler_length-self._l1-self._l2-self._l3-self._l4-self._l5
        if L_meander <=0:
            print("Error!Meander length for a resonator is less than zero")
        meander_step = self._l4 + self._l5
        N = int(L_meander // meander_step)
        tail = np.floor(L_meander - N * meander_step)
        print(tail)
        # const = self.ground + self.gap + self.core/2
        # offset=self.gap+(self.core+self.ground)/2
        meander_points = deepcopy(points)
        i = 1
        while i < N + 1:
            if i % 2 != 0:
                list1 = [
                    (meander_points[-1][0] - (i - 1) * self._l4, meander_points[-1][1]),
                    (meander_points[-1][0] - i * self._l4, meander_points[-1][1])]
                meander_points.extend(list1)

            else:
                list1 = [(meander_points[-1][0] - (i - 1) * self._l4,
                          self._start_y + self._l1 - self._l3 - self._l5 - (self.ground + self.gap + self.core / 2)),
                         (meander_points[-1][0] - (i - 1) * self._l4,
                          self._start_y + self._l1 - self._l3 - self._l5 - (self.ground + self.gap + self.core / 2))]
                meander_points.extend(list1)
            i = i + 1
        #
        # if (N)%2!=0:
        #         tail1=Number_of_points[2*N][1]+tail-const
        #         list_add_tail1=[(Number_of_points[2*N][0],tail1)]
        #         Number_of_points.extend(list_add_tail1)
        # else:
        #         tail2=Number_of_points[2*N][1]-tail-const
        #         list_add_tail2=[(Number_of_points[2*N][0],tail2) ]
        #         Number_of_points.extend(list_add_tail2)
        # if (N)%2!=0:
        #     end1=(Number_of_points[2*N+1][0]+const, Number_of_points[2*N+1][1])
        #     end2=(Number_of_points[2*N+1][0]+const, Number_of_points[2*N+1][1]+self._h_end)
        #     end3=(Number_of_points[2*N+1][0]-const, Number_of_points[2*N+1][1]+self._h_end)
        #     end4=(Number_of_points[2*N+1][0]-const, Number_of_points[2*N+1][1])
        # else:
        #     end1=(Number_of_points[2*N+1][0]+const, Number_of_points[2*N+1][1])
        #     end2=(Number_of_points[2*N+1][0]+const, Number_of_points[2*N+1][1]-self._h_end)
        #     end3=(Number_of_points[2*N+1][0]-const, Number_of_points[2*N+1][1]-self._h_end)
        #     end4=(Number_of_points[2*N+1][0]-const, Number_of_points[2*N+1][1])
        # end = gdspy.Polygon([end1, end2, end3, end4])

        # open_end_rects = gdspy.Polygon([
        #     (self._x+self.core/2+self.gap, self._y),
        #     (self._x+self.core/2+self.gap, self._y-self.open_end),
        #     (self._x-self.core/2-self.gap, self._y-self.open_end),
        #     (self._x+self.core/2+self.gap, self._y),
        # ])




        line1 = Feedline(deepcopy(Number_of_points), self.core, self.gap, self.ground, None, self.total_layer, self.restricted_area_layer,
                        R=40)
        line1 = line1.generate_feedline(self.corner_type)
        line2 = []
        line2.append(gdspy.boolean(line1[0],end,'or',layer=self.total_layer))
        line2.append(gdspy.boolean(line1[1],end,'or',layer=self.restricted_area_layer))
        line2.append(line1[2])
        # line = gdspy.FlexPath(points,offset,
        #                       corners=["circular bend", "circular bend", "circular bend"],
        #                       bend_radius=[20,20, 20],ends=[end_type, end_type, end_type],precision=0.1)
        # line1 = gdspy.FlexPath(Number_of_points,[self.ground, self.core, self.ground],offset,
        #                        corners=["circular bend", "circular bend", "circular bend"],
        #                        bend_radius=[20,20, 20], ends=[end_type, end_type, end_type])



        # result = gdspy.boolean(line, line1, 'or')
        # result = gdspy.boolean(end, result, 'or')
        # return result.rotate(angle, (self._x,self._y))
        return line,line2,open_end, Number_of_points
