from .core import DesignElement, LayerConfiguration
from .. import conformal_mapping as cm
from .. import transmission_line_simulator as tlsim
import gdspy
import numpy as np
from typing import List, Tuple, Mapping, Dict
from . import squid3JJ

class Coaxmon(DesignElement):
    def __init__(self, name: str, center: Tuple[float, float],
                 center_radius: float, inner_couplers_radius: float,
                 outer_couplers_radius: float, inner_ground_radius: float, outer_ground_radius: float,
                 layer_configuration: LayerConfiguration, Couplers, jj_params: Dict,
                 connection_point: Tuple[float,float]):
        super().__init__(type='qubit', name=name)
        self.center = center
        self.R1 = center_radius
        self.R2 = inner_couplers_radius
        self.R3 = outer_couplers_radius
        self.R4 = inner_ground_radius
        self.outer_ground = outer_ground_radius
        self.layer_configuration = layer_configuration

        self.couplers = Couplers
        self.JJ_params = jj_params
        self.JJ = None
        self.connection_point = connection_point
        self.w =8
        self.g =4
        self.angle=-np.pi/2-np.pi/3

    def render(self):
        ground = gdspy.Round(self.center, self.outer_ground, self.R4, initial_angle=0, final_angle=2*np.pi)
        result_restricted = gdspy.Round(self.center, self.outer_ground,  layer=self.layer_configuration.restricted_area_layer)
        core = gdspy.Round(self.center, self.R1, inner_radius=0, initial_angle=0, final_angle=2*np.pi)
        result = gdspy.boolean(ground, core, 'or', layer=self.layer_configuration.total_layer)
        if len(self.couplers) != 0:
            for coupler in self.couplers:
                if coupler.grounded == True:
                    result = gdspy.boolean(coupler.render(self.center, self.R2, self.outer_ground,self.R4 )['positive'], result, 'or', layer=self.layer_configuration.total_layer)
                else:
                    result = gdspy.boolean(coupler.render(self.center, self.R2, self.R3, self.R4)['positive'], result, 'or', layer=self.layer_configuration.total_layer)
        if self.JJ_params != None:
            self.JJ_coordinates = (self.center[0] + self.R1*np.cos(self.JJ_params['angle_qubit']), self.center[1] + self.R1*np.sin(self.JJ_params['angle_qubit']))
            JJ,rect = self.generate_JJ()
            result = gdspy.boolean(result, rect, 'or')
            flux_line = self.connection_to_ground(self.JJ_params['length'],self.JJ_params['width'])
            result = gdspy.boolean(result, flux_line['test'], 'not')
            result = gdspy.boolean(result, flux_line['flux line'], 'or', layer=self.layer_configuration.total_layer)
        # self.AB1_coordinates = coordinates(self.center.x, self.center.y + self.R4)
        # self.AB2_coordinates = coordinates(self.center.x, self.center.y - self.outer_ground)
        return {#'positive': result,
                'restricted': result_restricted,
                'qubit': result,
                # 'flux line': flux_line['flux line'],
                'JJ':JJ,
                # 'test': flux_line['test']
                }

    def generate_JJ(self):
        self.JJ = squid3JJ.JJ_2(self.JJ_coordinates[0], self.JJ_coordinates[1],
                    self.JJ_params['a1'], self.JJ_params['a2'],
                    self.JJ_params['b1'], self.JJ_params['b2'],
                    self.JJ_params['c1'], self.JJ_params['c2'])
        result = self.JJ.generate_jj()
        result = gdspy.boolean(result, result, 'or', layer=self.layer_configuration.jj_layer)
        angle = self.JJ_params['angle_JJ']
        # print(self.JJ_coordinates[0],self.JJ_coordinates[1])
        # print((self.JJ_coordinates[0],self.JJ_coordinates[1]))
        result.rotate(angle, (self.JJ_coordinates[0], self.JJ_coordinates[1]))
        rect = gdspy.Rectangle((self.JJ_coordinates[0] - self.JJ.contact_pad_a_outer/2, self.JJ_coordinates[1] + self.JJ.contact_pad_b_outer),
                               (self.JJ_coordinates[0] + self.JJ.contact_pad_a_outer/2, self.JJ_coordinates[1] - self.JJ.contact_pad_b_outer),
                               layer=self.layer_configuration.total_layer)
        rect.rotate(angle, (self.JJ_coordinates[0], self.JJ_coordinates[1]))
        return result, rect

    # TODO: create a separate class for this method, or not?
    def connection_to_ground(self,length,width):
        jj = self.JJ
        point1 = jj.rect1
        point2 = jj.rect2
        result = None
        end = self.connection_point
        # rect_to_remove = gdspy.Rectangle((),
        #                                  ())
        for point in [point1, point2]:
            line = gdspy.Rectangle((point[0] - width / 2, point[1]),
                                   (point[0] + width / 2, point[1] - length))
            result = gdspy.boolean(line, result, 'or', layer=self.layer_configuration.jj_flux_lines)
        # result = gdspy.boolean(line1,line2,'or',layer=self.total_layer)
        path1 = gdspy.Polygon(
            [(point1[0] + width / 2, point1[1] - length), (point1[0] - width / 2, point1[1] - length),
             (end[0] + self.w / 2 * np.cos(self.angle + 3 * np.pi / 2),
              end[1] + (self.w / 2) * np.sin(self.angle + 3 * np.pi / 2)),
             (end[0] + (self.w / 2) * np.cos(self.angle + np.pi / 2),
              end[1] + (self.w / 2) * np.sin(self.angle + np.pi / 2)),
             ])

        result = gdspy.boolean(path1, result, 'or', layer=self.layer_configuration.jj_flux_lines)

        path2 = gdspy.Polygon(
            [(point2[0] + width / 2, point2[1] - length), (point2[0] - width / 2, point2[1] - length),
             (end[0] + (self.w / 2 + self.g) * np.cos(self.angle + np.pi / 2),
              end[1] + (self.w / 2 + self.g) * np.sin(self.angle + np.pi / 2)),
             (end[0] + (self.w / 2 + self.g + width) * np.cos(self.angle + np.pi / 2),
              end[1] + (self.w / 2 + self.g + width) * np.sin(self.angle + np.pi / 2))
             ])
        result = gdspy.boolean(path2, result, 'or', layer=self.layer_configuration.jj_flux_lines)


        remove = gdspy.Polygon([(point1[0] - width / 2, point1[1]),
                                         (point2[0] + width / 2 + self.g, point2[1]),
                                         (point2[0] + width / 2 + self.g, point2[1] - length),
                                        (end[0] + (self.w / 2 + 2*self.g + width) * np.cos(
                                            self.angle + np.pi / 2),
                                         end[1] + (self.w / 2 + 2*self.g + width) * np.sin(
                                             self.angle + np.pi / 2)),
                                         (end[0] + (self.w / 2 + self.g*0) * np.cos(
                                             self.angle + 3 * np.pi / 2),
                                          end[1] + (self.w / 2 + self.g*0) * np.sin(
                                              self.angle + 3 * np.pi / 2)),
                                         (point1[0] - width / 2, point1[1] - length)],
                                        layer=self.layer_configuration.test)

        return {'positive': result,
                'test': remove,
                'flux line': result
                }

class CoaxmonCoupler:
    def __init__(self, arc_start, arc_finish, phi, w, grounded=False):
        self.arc_start = arc_start
        self.arc_finish = arc_finish
        self.phi = phi
        self.w = w
        self.grounded = grounded
        self.connections = None # TODO: add connections

    def render(self,coordinate,r_init,r_final,rect_end):
        #to fix intersection bug
        bug=5
        result = gdspy.Round(coordinate, r_init, r_final,
                                  initial_angle=(self.arc_start) * np.pi, final_angle=(self.arc_finish) * np.pi)
        rect = gdspy.Rectangle((coordinate[0]+r_final-bug,coordinate[1]-self.w/2),(coordinate[0]+rect_end+bug, coordinate[1]+self.w/2))
        rect.rotate(self.phi*np.pi, coordinate)
        result = gdspy.boolean(result,rect, 'or')
        return {
            'positive':result
              }