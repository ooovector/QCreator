from .core import DesignElement, LayerConfiguration
from .. import conformal_mapping as cm
from .. import transmission_line_simulator as tlsim
import gdspy
import numpy as np
from typing import List, Tuple, Mapping


class Coaxmon(DesignElement):
    def __init__(self, name: str, center: Tuple[float, float],
                 center_radius: float, inner_couplers_radius: float,
                 outer_couplers_radius: float, inner_ground_radius: float, outer_ground_radius: float,
                 layer_configuration: LayerConfiguration,
                 ):
        super().__init__(type='qubit', name=name)
        self.center = center
        self.R1 = center_radius
        self.R2 = inner_couplers_radius
        self.R3 = outer_couplers_radius
        self.R4 = inner_ground_radius
        self.outer_ground = outer_ground_radius
        self.layer_configuration = layer_configuration

        # self.couplers = Couplers
        # self.JJ_params = JJ
        # self.JJ = None

    def render(self):
        ground = gdspy.Round(self.center, self.outer_ground, self.R4, initial_angle=0, final_angle=2*np.pi)
        result_restricted = gdspy.Round(self.center, self.outer_ground,  layer=self.layer_configuration.restricted_area_layer)
        core = gdspy.Round(self.center, self.R1, inner_radius=0, initial_angle=0, final_angle=2*np.pi)
        result = gdspy.boolean(ground, core, 'or', layer=self.layer_configuration.total_layer)
        if len(self.couplers) != 0:
            for coupler in self.couplers:
                if coupler.grounded == True:
                    result = gdspy.boolean(coupler.generate_coupler(self.center, self.R2, self.outer_ground,self.R4 ), result, 'or', layer=self.layer_configuration.total_layer)
                else:
                    result = gdspy.boolean(coupler.generate_coupler(self.center, self.R2, self.R3, self.R4), result, 'or', layer=self.layer_configuration.total_layer)
        if self.JJ_params != None:
            self.JJ_coordinates = (self.center[0] + self.R1*np.cos(self.JJ_params['angle_qubit']), self.center[1] + self.R1*np.sin(self.JJ_params['angle_qubit']))
            JJ,rect = self.generate_JJ()
            result = gdspy.boolean(result, rect, 'or')
        # self.AB1_coordinates = coordinates(self.center.x, self.center.y + self.R4)
        # self.AB2_coordinates = coordinates(self.center.x, self.center.y - self.outer_ground)
        return {'positive': result,
                'restrict': result_restricted,
                }

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