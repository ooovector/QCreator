from .core import DesignElement
from .. import conformal_mapping as cm
from .. import transmission_line_simulator as tlsim
import gdspy
import numpy as np


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