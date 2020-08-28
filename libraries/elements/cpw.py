from .core import DesignElement, DesignTerminal, LayerConfiguration
import numpy as np
import gdspy
from .. import conformal_mapping as cm
from .. import transmission_line_simulator as tlsim
from typing import List, Tuple, Mapping


class CPW(DesignElement):
    def __init__(self, name: str, points: List[Tuple[float, float]], w: float, s: float, g: float,
                 layer_configuration: LayerConfiguration, r:float, corner_type: str = 'round'):
        """
        Create a coplanar waveguide (CPW) through points.
        :param name: element identifier
        :param points: points which the CPW traverses
        :param w: CPW signal conductor
        :param s: CPW signal-g s
        :param g:CPW finite g width
        :param layer_configuration:
        :param r: bend radius
        :param corner_type: 'round' for circular arcs instead of sharp corners, anything else for sharp corners
        """
        super().__init__('cpw', name)
        self.w = w
        self.g = g
        self.s = s
        self.points = points
        self.r = r
        self.restricted_area = None
        self.layer_configuration = layer_configuration
        self.end = self.points[-1]
        self.angle = None
        self.corner_type = corner_type
        self.length = None
        self.tls_cache = []

        orientation1 = np.arctan2(self.points[1][1]-self.points[0][1], self.points[1][0]-self.points[0][0])
        orientation2 = np.arctan2(self.points[-1][1] - self.points[-2][1], self.points[-1][0] - self.points[-2][0])
        self.terminals = {'port1': DesignTerminal(self.points[0], orientation1, g=g, s=s, w=w, type='cpw'),
                          'port2': DesignTerminal(self.points[-1], orientation2, g=g, s=s, w=w, type='cpw')}

    def render(self):
        offset = self.s + (self.w + self.g) / 2

        r1 = self.r - (self.w / 2 + self.s + self.g / 2)
        r2 = np.abs(self.r + self.w / 2 + self.s + self.g / 2)

        r1_new = self.r - (self.w / 2 + self.s / 2)
        r2_new = np.abs(self.r + self.w / 2 + self.s / 2)

        result = None
        result_restricted = None
        result_new = None

        if len(self.points) < 3:
            self.points.insert(1, ((self.points[0][0] + self.points[1][0]) / 2,
                                   (self.points[0][1] + self.points[1][1]) / 2))
        new_points = self.points
        # new_points_restricted_line= self.points

        width_restricted_line = 2 * self.g + 2 * self.s + self.w
        width_new = self.s
        offset_new = self.s + self.w

        assert r1 > 0

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
                                          [self.g, self.w,
                                           self.g], offset, ends=["flush", "flush", "flush"],
                                          corners=["circular bend", "circular bend", "circular bend"],
                                          bend_radius=[r1, self.r, r2], precision=0.001, layer=0)
                    restricted_line = gdspy.FlexPath([point1, point2, point3], width_restricted_line, offset=0,
                                                     ends="flush", corners="circular bend", bend_radius=self.r,
                                                     precision=0.001, layer=1)
                    line_new = gdspy.FlexPath([point1, point2, point3], [width_new, width_new], offset=offset_new,
                                              ends=["flush", "flush"], corners=["circular bend", "circular bend"],
                                              bend_radius=[r1_new, r2_new], precision=0.001, layer=2)

                    # rectricted_line = gdspy.FlexPath([point1, point2, point3], width_restricted_line, offset=0,  ends="flush",  corners="circular bend",bend_radius= self.r, precision=0.001, layer=1)

                    # self.end = line.x
                else:
                    line = gdspy.FlexPath([point1, point2, point3],
                                          [self.g, self.w,
                                           self.g], offset, ends=["flush", "flush", "flush"],
                                          corners=["circular bend", "circular bend", "circular bend"],
                                          bend_radius=[r2, self.r, r1], precision=0.001, layer=0)
                    restricted_line = gdspy.FlexPath([point1, point2, point3], width_restricted_line, offset=0,
                                                     ends="flush", corners="circular bend", bend_radius=self.r,
                                                     precision=0.001, layer=1)
                    line_new = gdspy.FlexPath([point1, point2, point3], [width_new, width_new], offset=offset_new,
                                              ends=["flush", "flush"], corners=["circular bend", "circular bend"],
                                              bend_radius=[r2_new, r1_new], precision=0.001, layer=2)

                    # self.end = line.x
                result = gdspy.boolean(line, result, 'or', layer=self.layer_configuration.total_layer)
                result_restricted = gdspy.boolean(restricted_line, result_restricted, 'or', layer=self.layer_configuration.restricted_area_layer)
                result_new = gdspy.boolean(line_new, result_new, 'or', layer=2)
                # self.restricted_area = gdspy.boolean(rectricted_line, self.restricted_area, 'or', layer = self.restricted_area_layer)

                # result= line, restricted_line
            else:
                line = gdspy.FlexPath([point1, point2, point3],
                                      [self.g, self.w,
                                       self.g], offset, ends=["flush", "flush", "flush"],
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

        self.angle = np.arctan2(point3[1] - point2[1], point3[0] - point2[0]) # TODO: what's is the point of this?
                                                                              # also, render() should not mutate self

        return {'positive': result,
                'restrict': result_restricted,
                'remove': result_new #TODO: do we need this?
                }

    def get_terminals(self):
        return self.terminals

    def cm(self):
        return cm.ConformalMapping([self.s, self.w, self.s]).cl_and_Ll()

    def add_to_tls(self, tls_instance: tlsim.TLSystem,
                   terminal_mapping: Mapping[str, int], track_changes: bool = True) -> list:
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

    '''
    # TODO: remove completely
    def generate_end(self, end):
        if end['type'] is 'open':
            return self.generate_open_end(end)
        if end['type'] is 'fluxline':
            return self.generate_fluxline_end(end)
    '''

    # TODO: create a separate class for this method
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
            result = gdspy.boolean(line, result, 'or', layer=6) #TODO: layer should not be constant
        # result = gdspy.boolean(line1,line2,'or',layer=self.total_layer)
        path1 = gdspy.Polygon([(point1[0] + width / 2, point1[1] - length), (point1[0] - width / 2, point1[1] - length),
                               (self.end[0] + (self.w / 2 + self.s + width) * np.cos(self.angle + np.pi / 2),
                                self.end[1] + (self.w / 2 + self.s + width) * np.sin(self.angle + np.pi / 2)),
                               (self.end[0] + (self.w / 2 + self.s) * np.cos(self.angle + np.pi / 2),
                                self.end[1] + (self.w / 2 + self.s) * np.sin(self.angle + np.pi / 2))])

        result = gdspy.boolean(path1, result, 'or', layer=6) #TODO: layer should not be constant

        path2 = gdspy.Polygon([(point2[0] + width / 2, point2[1] - length), (point2[0] - width / 2, point2[1] - length),
                               (self.end[0] + (self.w / 2) * np.cos(self.angle + np.pi / 2), self.end[1] + (self.w / 2) * np.sin(self.angle + np.pi / 2)),
                               (self.end[0] + self.w / 2 * np.cos(self.angle + 3 * np.pi / 2), self.end[1] + (self.w / 2) * np.sin(self.angle + 3 * np.pi / 2))])
        result = gdspy.boolean(path2, result, 'or', layer=6) #TODO: layer should not be constant

        # if end['type'] != 'coupler':
        restricted_area = gdspy.Polygon([(point1[0] - width / 2, point1[1]),
                                         (point2[0] + width / 2 + self.s, point2[1]),
                                         (point2[0] + width / 2 + self.s, point2[1] - length),
                                         (self.end[0] + (self.w / 2 + self.s) * np.cos(
                                             self.angle + 3 * np.pi / 2),
                                          self.end[1] + (self.w / 2 + self.s) * np.sin(
                                              self.angle + 3 * np.pi / 2)),
                                         (self.end[0] + (self.w / 2 + self.s + width) * np.cos(
                                             self.angle + np.pi / 2),
                                          self.end[1] + (self.w / 2 + self.s + width) * np.sin(
                                              self.angle + np.pi / 2)),
                                         (point1[0] - width / 2, point1[1] - length)],
                                        layer=self.restricted_area_layer)
        # else:
        #
        return result, restricted_area, restricted_area

    # TODO: create separate class for this method
    def generate_open_end(self, end):
        end_gap = end['s']
        end_ground_length = end['g']
        if end['end']:
            x_begin = self.end[0]
            y_begin = self.end[1]
            additional_rotation = -np.pi/2
        if end['begin']:
            x_begin = self.points[0][0]
            y_begin = self.points[0][1]
            additional_rotation = -np.pi / 2
        restricted_area = gdspy.Rectangle((x_begin - self.w / 2 - self.s - self.g, y_begin),
                                          (x_begin + self.w / 2 + self.s + self.g, y_begin + end_gap + end_ground_length), layer=10)#fix it
        rectangle_for_removing = gdspy.Rectangle((x_begin - self.w / 2 - self.s, y_begin),
                                                 (x_begin + self.w / 2 + self.s, y_begin + end_gap), layer=2)
        total = gdspy.boolean(restricted_area, rectangle_for_removing, 'not')
        for obj in [total,restricted_area, rectangle_for_removing]:
            obj.rotate(additional_rotation+self.angle, (x_begin, y_begin))
        return total, restricted_area, rectangle_for_removing


#TODO: make compatible with DesignElement and implement add_to_tls
class Narrowing(DesignElement):
    def __init__(self, name: str, position: Tuple[float, float], orientation: float, w1: float, s1: float, g1: float,
                 w2: float, s2: float, g2: float, layer_configuration: LayerConfiguration, length: float,
                 c: float=0, l: float=0):
        """
        Isosceles trapezoid-form adapter from one CPW to another.
        :param name: Element name
        :param position: position of center
        :param orientation: orientation in radians
        :param w1: signal conductor width of port 1
        :param s1: signal-ground gap of port 1
        :param g1: finite ground width of port 1
        :param w2: signal conductor width of port 2
        :param s2: signal-ground gap of port 2
        :param g2: finite ground width of port 2
        :param layer_configuration:
        :param length: height of trapezoid
        :param c: signal-to-ground capacitance
        :param l: port 1 to port 2 inductance
        """
        super().__init__('narrowing', name)
        self.position = position
        self.orientation = orientation

        self.w1 = w1
        self.s1 = s1
        self.g1 = g1

        self.w2 = w2
        self.s2 = s2
        self.g2 = g2

        self.layer_configuration = layer_configuration
        self.length = length

        self.c = c
        self.l = l

        x_begin = self.position[0] - self.length/2*np.cos(self.orientation)
        x_end = self.position[0] + self.length/2*np.cos(self.orientation)
        y_begin = self.position[1] - self.length/2*np.sin(self.orientation)
        y_end = self.position[1] + self.length/2*np.sin(self.orientation)

        self.terminals = {'port1': DesignTerminal((x_begin, y_begin), self.orientation, w=w1, s=s1, g=g1, type='cpw'),
                          'port2': DesignTerminal((x_end, y_end), self.orientation+np.pi, w=w2, s=s2, g=g2, type='cpw')}

        self.tls_cache = []

    def render(self):
        x_begin = self.position[0] - self.length/2
        x_end = self.position[0] + self.length/2
        y_begin = self.position[1]
        y_end = self.position[1]
        #x_end = self._x_begin-self.length
        #y_end = self._y_begin

        points_for_poly1 = [(x_begin, y_begin + self.w1 / 2 + self.s1 + self.g1),
                            (x_begin, y_begin + self.w1 / 2 + self.s1),
                            (x_end, y_end + self.w2 / 2 + self.s2),
                            (x_end, y_end + self.w2 / 2 + self.s2 + self.g2)]

        points_for_poly2 = [(x_begin, y_begin + self.w1 / 2),
                            (x_begin, y_begin - self.w1 / 2),
                            (x_end, y_end - self.w2 / 2),
                            (x_end, y_end + self.w2 / 2)]

        points_for_poly3 = [(x_begin, y_begin-(self.w1 / 2 + self.s1 + self.g1)),
                            (x_begin, y_begin-(self.w1 / 2 + self.s1)),
                            (x_end, y_end-(self.w2 / 2 + self.s2)),
                            (x_end, y_end-(self.w2 / 2 + self.s2 + self.g2))]

        points_for_restricted_area = [(x_begin, y_begin + self.w1 / 2 + self.s1 + self.g1),
                                      (x_end, y_end + self.w2 / 2 + self.s2 + self.g2),
                                      (x_end, y_end-(self.w2 / 2 + self.s2 + self.g2)),
                                      (x_begin, y_begin-(self.w1 / 2 + self.s1 + self.g1))]

        restricted_area = gdspy.Polygon(points_for_restricted_area)

        poly1 = gdspy.Polygon(points_for_poly1)
        poly2 = gdspy.Polygon(points_for_poly2)
        poly3 = gdspy.Polygon(points_for_poly3)

        if self.orientation == 0:
            result = poly1, poly2, poly3
        else:
            poly1_ = poly1.rotate(angle=self.orientation, center=self.position)
            poly2_ = poly2.rotate(angle=self.orientation, center=self.position)
            poly3_ = poly3.rotate(angle=self.orientation, center=self.position)
            restricted_area.rotate(angle=self.orientation, center=self.position)
            result = poly1_, poly2_, poly3_

        polygon_to_remove = gdspy.boolean(restricted_area, result, 'not',
                                          layer=self.layer_configuration.layer_to_remove)

        return {'positive': result, 'restricted': [restricted_area], 'remove': polygon_to_remove}

    def get_terminals(self):
        return self.terminals

    def add_to_tls(self, tls_instance: tlsim.TLSystem,
                   terminal_mapping: Mapping[str, int], track_changes: bool = True) -> list:
        l = tlsim.Inductor(l=self.l)
        c1 = tlsim.Capacitor(c=self.c / 2)
        c2 = tlsim.Capacitor(c=self.c / 2)

        if track_changes:
            self.tls_cache.append([l, c1, c2])

        tls_instance.add_element(l, [terminal_mapping['port1'], terminal_mapping['port2']])
        tls_instance.add_element(c1, [terminal_mapping['port1'], 0])
        tls_instance.add_element(c2, [terminal_mapping['port2'], 0])

        return [l, c1, c2]
