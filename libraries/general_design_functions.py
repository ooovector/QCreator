import gdspy
import numpy as np
#import libraries.squid3JJ as squid3JJ
#import libraries.JJ4q as JJ4q

from . import conformal_mapping as cm
from copy import deepcopy
from typing import Tuple, List
from abc import *
#from . import conformal_mapping as cm


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
