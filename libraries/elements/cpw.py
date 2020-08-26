from .core import DesignElement, LayerConfiguration
import numpy as np
import gdspy
from .. import conformal_mapping as cm
from .. import transmission_line_simulator as tlsim


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

    # TODO: create separate class for this method
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


#TODO: make compatible with DesignElement and implement add_to_tls
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