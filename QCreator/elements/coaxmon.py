from .core import DesignElement, LayerConfiguration, DesignTerminal
from .. import conformal_mapping as cm
from .. import transmission_line_simulator as tlsim
import gdspy
import numpy as np
from typing import List, Tuple, Mapping, Dict
from . import squid3JJ
from copy import deepcopy

class Coaxmon(DesignElement):
    def __init__(self, name: str, center: Tuple[float, float],
                 center_radius: float, inner_couplers_radius: float,
                 outer_couplers_radius: float, inner_ground_radius: float, outer_ground_radius: float,
                 layer_configuration: LayerConfiguration, Couplers, jj_params: Dict,
                 connection_points: List[Tuple[float, float]]):
        super().__init__(type='qubit', name=name)
        self.center = center
        self.R1 = center_radius
        self.R2 = inner_couplers_radius
        self.R3 = outer_couplers_radius
        self.R4 = inner_ground_radius
        self.outer_ground = outer_ground_radius
        self.layer_configuration = layer_configuration

        self.couplers = Couplers
        # there is one special coupler - for the fluxline
        for coupler in self.couplers:
            if coupler.coupler_type == 'grounded':
                self.grounded = coupler
        self.JJ_params = jj_params
        self.JJ = None
        self.connection_points = connection_points
        self.core = self.grounded.w
        self.gap = self.grounded.g
        self.ground = self.grounded.w
        self.angle = -np.pi / 2 - np.pi / 3
        # DesignTerminal(self.points[0], orientation1, g=g, s=s, w=w, type='cpw')
        # qubit terminals
        self.terminals = {'coupler1': None,
                          'coupler2': None,
                          'coupler3': None,
                          'coupler4': None,
                          'coupler5': None,
                          'flux line': None, }

    def render(self):
        ground = gdspy.Round(self.center, self.outer_ground, self.R4, initial_angle=0, final_angle=2 * np.pi)
        result_restricted = gdspy.Round(self.center, self.outer_ground,
                                        layer=self.layer_configuration.restricted_area_layer)
        core = gdspy.Round(self.center, self.R1, inner_radius=0, initial_angle=0, final_angle=2 * np.pi)
        result = gdspy.boolean(ground, core, 'or', layer=self.layer_configuration.total_layer)
        # add couplers
        if len(self.couplers) != 0:
            for coupler in self.couplers:
                coupler_parts = coupler.render(self.center, self.R2, self.R3, self.R4, self.outer_ground)
                if 'remove' in coupler_parts:
                    # print(coupler_parts['remove'])
                    result = gdspy.boolean(result, coupler_parts['remove'], 'not',
                                           layer=self.layer_configuration.total_layer)
                result = gdspy.boolean(coupler_parts['positive'], result, 'or',
                                       layer=self.layer_configuration.total_layer)
        if self.JJ_params != None:
            self.JJ_coordinates = (self.center[0] + self.R1 * np.cos(self.JJ_params['angle_qubit']),
                                   self.center[1] + self.R1 * np.sin(self.JJ_params['angle_qubit']))
            JJ, rect = self.generate_JJ()
            result = gdspy.boolean(result, rect, 'or')
            flux_line = self.connection_to_ground(self.JJ_params['length'], self.JJ_params['width'])
            result = gdspy.boolean(result, flux_line['remove'], 'not')
            result = gdspy.boolean(result, flux_line['flux line'], 'or', layer=self.layer_configuration.total_layer)
        # self.specify_terminals()
        return {'positive': result,
                'restricted': result_restricted,
                'qubit': result,
                # 'flux line': flux_line['flux line'],
                'JJ': JJ,
                'test': flux_line['test']
                }

    def specify_terminals(self):
        # for coupler in self.couplers:
        pass

    def get_terminals(self):
        return self.terminals

    def generate_JJ(self):
        self.JJ = squid3JJ.JJ_2(self.JJ_coordinates[0], self.JJ_coordinates[1],
                                self.JJ_params['a1'], self.JJ_params['a2'],
                                self.JJ_params['b1'], self.JJ_params['b2'],
                                self.JJ_params['c1'], self.JJ_params['c2'])
        result = self.JJ.generate_jj()
        result = gdspy.boolean(result, result, 'or', layer=self.layer_configuration.jj_layer)
        angle = self.JJ_params['angle_JJ']
        result.rotate(angle, (self.JJ_coordinates[0], self.JJ_coordinates[1]))
        rect = gdspy.Rectangle((self.JJ_coordinates[0] - self.JJ.contact_pad_a_outer / 2,
                                self.JJ_coordinates[1] + self.JJ.contact_pad_b_outer),
                               (self.JJ_coordinates[0] + self.JJ.contact_pad_a_outer / 2,
                                self.JJ_coordinates[1] - self.JJ.contact_pad_b_outer),
                               layer=self.layer_configuration.total_layer)
        rect.rotate(angle, (self.JJ_coordinates[0], self.JJ_coordinates[1]))
        return result, rect

    # TODO: create a separate class for this method, or not?
    def connection_to_ground(self, length, width):
        jj = self.JJ
        point1 = jj.rect1
        point2 = jj.rect2
        result = None
        remove = None
        end = self.connection_points[0]
        start = self.connection_points[1]

        orientation1 = np.arctan2(-(self.center[1] - (point1[1]-length)), -(self.center[0] - point1[0]))
        points =[point1, (point1[0], point1[1] - length),
                 (self.center[0]+self.R2*np.cos(orientation1),self.center[1]+self.R2*np.sin(orientation1))]
        path1 = gdspy.FlexPath(deepcopy(points),self.core,offset=0,layer=self.layer_configuration.jj_flux_lines)
        result = gdspy.boolean(path1, result, 'or', layer=self.layer_configuration.jj_flux_lines)

        orientation2 = np.arctan2(-(self.center[1] - (point2[1]-length)), -(self.center[0] - point2[0]))
        points = [point2, (point2[0], point2[1] - length),
                  (self.center[0]+self.R2*np.cos(orientation2),self.center[1]+self.R2*np.sin(orientation2))]
        path2 = gdspy.FlexPath(deepcopy(points),self.core,offset=0,layer=self.layer_configuration.jj_flux_lines)
        result = gdspy.boolean(path2, result, 'or', layer=self.layer_configuration.jj_flux_lines)

        # remove = gdspy.Polygon([(point1[0] - width / 2, point1[1]),
        #                         (point2[0] + width / 2 + self.grounded.g, point2[1]),
        #                         (point2[0] + width / 2 + self.grounded.g, point2[1] - length),
        #                         (end[0] + (self.grounded.w / 2 + 2 * self.grounded.g + width) * np.cos(
        #                             self.angle + np.pi / 2),
        #                          end[1] + (self.grounded.w / 2 + 2 * self.grounded.g + width) * np.sin(
        #                              self.angle + np.pi / 2)),
        #                         (end[0] + (self.grounded.w / 2) * np.cos(
        #                             self.angle + 3 * np.pi / 2),
        #                          end[1] + (self.grounded.w / 2) * np.sin(
        #                              self.angle + 3 * np.pi / 2)),
        #                         (point1[0] - width / 2, point1[1] - length)],
        #                        layer=self.layer_configuration.test)
        # orientation1 = np.arctan2(end[1] - start[1], end[0] - start[0])

        # cpw_1part = gdspy.Polygon([(end[0] + (self.grounded.w / 2 + self.grounded.g) * np.cos(self.angle + np.pi / 2),
        #                             end[1] + (self.grounded.w / 2 + self.grounded.g) * np.sin(self.angle + np.pi / 2)),
        #                            (end[0] + (self.grounded.w / 2) * np.cos(self.angle + np.pi / 2),
        #                             end[1] + (self.grounded.w / 2) * np.sin(self.angle + np.pi / 2)),
        #                            (start[0] + (self.grounded.w / 2 + self.grounded.g) * np.cos(self.angle + np.pi / 2),
        #                             start[1] + (self.grounded.w / 2 + self.grounded.g) * np.sin(
        #                                 self.angle + np.pi / 2)),
        #                            (start[0] + (self.grounded.w / 2 + self.grounded.g * 2) * np.cos(
        #                                self.angle + np.pi / 2),
        #                             start[1] + (self.grounded.w / 2 + self.grounded.g * 2) * np.sin(
        #                                 self.angle + np.pi / 2))
        #                            ])
        # cpw_2part = gdspy.Polygon(
        #     [(end[0] + (self.grounded.w / 2 * 3 + 2 * self.grounded.g) * np.cos(self.angle + np.pi / 2),
        #       end[1] + (self.grounded.w / 2 * 3 + 2 * self.grounded.g) * np.sin(self.angle + np.pi / 2)),
        #      (end[0] + (self.grounded.w / 2 * 3 + self.grounded.g) * np.cos(self.angle + np.pi / 2),
        #       end[1] + (self.grounded.w / 2 * 3 + self.grounded.g) * np.sin(self.angle + np.pi / 2)),
        #      (start[0] + (self.grounded.w / 2 * 3 + 2 * self.grounded.g) * np.cos(self.angle + np.pi / 2),
        #       start[1] + (self.grounded.w / 2 * 3 + 2 * self.grounded.g) * np.sin(self.angle + np.pi / 2)),
        #      (
        #          start[0] + (self.grounded.w / 2 * 3 + self.grounded.g * 3) * np.cos(self.angle + np.pi / 2),
        #          start[1] + (self.grounded.w / 2 * 3 + self.grounded.g * 3) * np.sin(self.angle + np.pi / 2))
        #      ])
        # remove = gdspy.boolean(remove, cpw_1part, 'or', layer=self.layer_configuration.test)
        # remove = gdspy.boolean(remove, cpw_2part, 'or', layer=self.layer_configuration.test)
        point = (start[0]+(self.grounded.w + self.grounded.g) * np.sin(np.pi-orientation1),
                 start[1]+(self.grounded.w + self.grounded.g) * np.cos(np.pi-orientation1))
        # point = start
        self.terminals['flux line'] = DesignTerminal(point, orientation1, g=self.grounded.w, s=self.grounded.g,
                                                 w=self.grounded.w, type='cpw')
        # TODO: grounded part specified incorrectly
        return {'positive': result,
                'remove': remove,
                'flux line': result,
                'test':gdspy.boolean(path2, path1, 'or', layer=self.layer_configuration.jj_flux_lines)
                }


class CoaxmonCoupler:
    def __init__(self, arc_start, arc_finish, phi, coupler_type=None, w=None, g=None):
        self.arc_start = arc_start
        self.arc_finish = arc_finish
        self.phi = phi
        self.w = w
        self.g = g
        self.coupler_type = coupler_type
        self.connections = None  # TODO: add connections

    def render(self, center, r_init, r_final, rect_end, outer_ground):
        if self.coupler_type is None:
            arc = gdspy.Round(center, r_init, r_final,
                              initial_angle=self.arc_start * np.pi, final_angle=self.arc_finish * np.pi)
            # to fix intersection bug
            bug = 5
            rect = gdspy.Rectangle((center[0] + r_final - bug, center[1] - self.w / 2),
                                   (center[0] + rect_end + bug, center[1] + self.w / 2))
            rect.rotate(self.phi * np.pi, center)
            result = gdspy.boolean(arc, rect, 'or')
        elif self.coupler_type == 'grounded':
            result = gdspy.Round(center, r_init, outer_ground,
                                 initial_angle=self.arc_start * np.pi, final_angle=self.arc_finish * np.pi)
        elif self.coupler_type == 'coupler':
            arc = gdspy.Round(center, r_init, r_final,
                              initial_angle=self.arc_start * np.pi, final_angle=self.arc_finish * np.pi)
            rect = gdspy.Rectangle((center[0] + r_final, center[1] - self.w / 2),
                                   (center[0] + outer_ground, center[1] + self.w / 2))
            rect.rotate(self.phi * np.pi, center)

            part1_remove = gdspy.Rectangle((center[0] + r_final, center[1] - self.w / 2 - self.g),
                                           (center[0] + outer_ground, center[1] + self.w / 2 - self.g))
            part1_remove = part1_remove.rotate(self.phi * np.pi, center)

            part2_remove = gdspy.Rectangle((center[0] + r_final, center[1] - self.w / 2 + self.g),
                                           (center[0] + outer_ground, center[1] + self.w / 2 + self.g))
            part2_remove = part2_remove.rotate(self.phi * np.pi, center)

            remove = gdspy.boolean(part2_remove, part1_remove, 'or')
            result = gdspy.boolean(arc, rect, 'or')
            return {
                'positive': result,
                'remove': remove
            }
        return {
            'positive': result,
        }
