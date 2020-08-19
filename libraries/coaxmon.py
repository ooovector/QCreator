import numpy as np
import gdspy
from libraries.general_sample_creator import Sample
import libraries.squid3JJ as squid3JJ
import libraries.JJ4q as JJ4q

class SampleCoaxmons(Sample):

    def __init__(self, name, layer_configurations):
        self.name = str(name)
        self.result = gdspy.Cell('result')
        self.total_cell = gdspy.Cell(self.name)
        self.restricted_area_cell = gdspy.Cell(self.name + 'resctricted area')
        self.label_cell = gdspy.Cell(self.name + ' labels')
        self.cell_to_remove = gdspy.Cell(self.name + ' remove')

        self.total_layer = layer_configurations['total']
        self.restricted_area_layer = layer_configurations['restricted area']
        self.layer_to_remove = layer_configurations['for removing']
        self.JJ_layer = layer_configurations['JJs']
        self.AirbridgesLayer = layer_configurations['air bridges']
        self.AirbridgesPadLayer = layer_configurations['air bridge pads']
        self.gridline_x_layer = layer_configurations['vertical gridlines']
        self.gridline_y_layer = layer_configurations['horizontal gridlines']
        self.name = str(name)
        self.pads = []
        self.coaxmons = []
        self.lines = []
        self.bridges = []
        self.couplers = []

        self.MinDist = 4 / np.pi


    #Specific methods
    def add_coaxmon(self, coordinate, r1, r2, r3, r4, outer_ground, Couplers, JJ, angle=0, mirror=False):
        self.coaxmons.append(Coaxmon(coordinate, r1, r2, r3, r4, outer_ground, self.total_layer,
                                         self.restricted_area_layer, self.JJ_layer, Couplers, JJ))
        qubit_total, restricted_area, JJ_total = self.coaxmons[-1].generate_qubit()
        self.total_cell.add(qubit_total.rotate(angle, coordinate))
        self.total_cell.add(JJ_total)  # .rotate(angle,(center_point.x,center_point.y))
        self.restricted_area_cell.add(restricted_area)
        self.numerate("Coax", len(self.coaxmons) - 1, coordinate)

    def add_qubit_coupler(self, core, gap, ground, Coaxmon1, Coaxmon2, JJ, squid):
        coupler = IlyaCoupler(core, gap, ground, Coaxmon1, Coaxmon2, JJ, squid,
                              self.total_layer, self.restricted_area_layer, self.JJ_layer, self.layer_to_remove)
        self.couplers.append(coupler)
        line, JJ = coupler.generate_coupler()
        self.total_cell.add([line[0], JJ[0], JJ[1]])
        #         self.total_cell.add(line[1])
        self.restricted_area_cell.add(line[1])
        self.cell_to_remove.add([line[2], JJ[2]])
        self.numerate("IlCoup", len(self.couplers) - 1, ((Coaxmon1.center[0]+Coaxmon2.center[0])/2, (Coaxmon1.center[1]+Coaxmon2.center[1])/2))




#Specific functions for the Coaxmon qubit
class Coaxmon:
    def __init__(self, center, r1, r2, r3, R4, outer_ground, total_layer, restricted_area_layer,JJ_layer, Couplers, JJ):
        self.center = center
        self.R1 = r1
        self.R2 = r2
        self.R3 = r3
        self.R4 = R4
        self.outer_ground = outer_ground
        self.Couplers = Couplers
        self.restricted_area_layer = restricted_area_layer
        self.total_layer = total_layer
        self.JJ_layer = JJ_layer
        self.JJ_params=JJ
        self.JJ = None
    def generate_qubit(self):
        ground = gdspy.Round(self.center, self.outer_ground, self.R4, initial_angle=0, final_angle=2*np.pi)
        restricted_area = gdspy.Round(self.center, self.outer_ground,  layer=self.restricted_area_layer)
        core = gdspy.Round(self.center, self.R1, inner_radius=0, initial_angle=0, final_angle=2*np.pi)
        result = gdspy.boolean(ground, core, 'or', layer=self.total_layer)
        if len(self.Couplers) != 0:
            for Coupler in self.Couplers:
                if Coupler.grounded == True:
                    result = gdspy.boolean(Coupler.generate_coupler(self.center, self.R2, self.outer_ground,self.R4 ), result, 'or', layer=self.total_layer)
                else:
                    result = gdspy.boolean(Coupler.generate_coupler(self.center, self.R2, self.R3, self.R4), result, 'or', layer=self.total_layer)
        self.JJ_coordinates = (self.center[0] + self.R1*np.cos(self.JJ_params['angle_qubit']), self.center[1] + self.R1*np.sin(self.JJ_params['angle_qubit']))
        JJ,rect = self.generate_JJ()
        result = gdspy.boolean(result, rect, 'or')
        # self.AB1_coordinates = coordinates(self.center.x, self.center.y + self.R4)
        # self.AB2_coordinates = coordinates(self.center.x, self.center.y - self.outer_ground)
        return result, restricted_area, JJ
    def generate_JJ(self):
        self.JJ = squid3JJ.JJ_2(self.JJ_coordinates[0],self.JJ_coordinates[1],
                    self.JJ_params['a1'], self.JJ_params['a2'],
                    self.JJ_params['b1'], self.JJ_params['b2'],
                    self.JJ_params['c1'],self.JJ_params['c2'])
        result = self.JJ.generate_JJ()
        result = gdspy.boolean(result, result, 'or', layer=self.JJ_layer)
        angle = self.JJ_params['angle_JJ']
        # print(self.JJ_coordinates[0],self.JJ_coordinates[1])
        # print((self.JJ_coordinates[0],self.JJ_coordinates[1]))
        result.rotate(angle, (self.JJ_coordinates[0],self.JJ_coordinates[1]))
        rect = gdspy.Rectangle((self.JJ_coordinates[0]-self.JJ.contact_pad_a_outer/2,self.JJ_coordinates[1]+self.JJ.contact_pad_b_outer),
                               (self.JJ_coordinates[0]+self.JJ.contact_pad_a_outer/2,self.JJ_coordinates[1]-self.JJ.contact_pad_b_outer),layer=self.total_layer)
        rect.rotate(angle,(self.JJ_coordinates[0],self.JJ_coordinates[1]))
        return result,rect
class QubitCoupler:
    def __init__(self, arc_start, arc_finish, phi, w, grounded=False):
        self.arc_start = arc_start
        self.arc_finish = arc_finish
        self.phi = phi
        self.w = w
        self.grounded = grounded
    def generate_coupler(self,coordinate,r_init,r_final,rect_end):
        #to fix bug
        bug=5
        result = gdspy.Round(coordinate, r_init, r_final,
                                  initial_angle=(self.arc_start) * np.pi, final_angle=(self.arc_finish) * np.pi)
        rect = gdspy.Rectangle((coordinate[0]+r_final-bug,coordinate[1]-self.w/2),(coordinate[0]+rect_end+bug, coordinate[1]+self.w/2))
        rect.rotate(self.phi*np.pi, coordinate)
        return gdspy.boolean(result,rect, 'or')


class IlyaCoupler:
    def __init__(self,core,gap,ground,Coaxmon1,Coaxmon2,JJ_params,squid_params, total_layer, restricted_area_layer, JJ_layer,layer_to_remove):
        self.core = core
        self.gap = gap
        self.ground = ground
        self.Coaxmon1 = Coaxmon1
        self.Coaxmon2 = Coaxmon2
        self.total_layer = total_layer
        self.restricted_area_layer = restricted_area_layer
        self.angle = None
        self.JJ_layer = JJ_layer
        self.layer_to_remove = layer_to_remove
        self.JJ_params = JJ_params
        self.squid_params = squid_params

    def generate_coupler(self):
        vector2_x = self.Coaxmon2.center[0] - self.Coaxmon1.center[0]
        vector2_y = self.Coaxmon2.center[1] - self.Coaxmon1.center[1]
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
        bug=3# there is a bug
        points=[(self.Coaxmon1.center[0]+(self.Coaxmon1.R4-bug)*np.cos(self.angle),
                 self.Coaxmon1.center[1]+(self.Coaxmon1.R4-bug)*np.sin(self.angle)),
        (self.Coaxmon2.center[0]+(self.Coaxmon1.R4-bug)*np.cos(self.angle+np.pi),
         self.Coaxmon2.center[1]+(self.Coaxmon1.R4-bug)*np.sin(self.angle+np.pi))]
        line = Feedline(points, self.core, self.gap, self.ground, None, self.total_layer, self.restricted_area_layer, 100)
        line = line.generate_feedline()
        JJ1 = self.generate_JJ()
        self.JJ_params['indent'] = np.abs(self.Coaxmon2.center[1] - self.Coaxmon1.center[1])+np.abs(self.Coaxmon2.center[0] -
                                    self.Coaxmon1.center[0]) -2*self.Coaxmon1.R4 -self.JJ_params['indent']
        JJ2 = self.generate_JJ()
        squid = self.generate_squid()
        JJ_0 = gdspy.boolean(JJ1[0],JJ2[0],'or',layer=self.JJ_layer)
        JJ_1 = gdspy.boolean(JJ1[1],JJ2[1],'or',layer=6)
        JJ_2 = gdspy.boolean(JJ1[2],JJ2[2],'or',layer=self.layer_to_remove)

        JJ_0 = gdspy.boolean(JJ_0, squid[0], 'or', layer=self.JJ_layer)
        JJ_1 = gdspy.boolean(JJ_1, squid[1], 'or', layer=6)
        # result = gdspy.boolean(JJ[1],line,'or',layer=self.total_layer)
        return line,[JJ_0,JJ_1,JJ_2]

    def generate_JJ(self):
        self.JJ_params['x'] = self.Coaxmon1.center[0] + (self.Coaxmon1.R4+self.JJ_params['indent'])*np.cos(self.angle)
        if self.Coaxmon1.center[0] != self.Coaxmon2.center[0]:
            self.JJ_params['y'] = self.Coaxmon1.center[1] + (self.Coaxmon1.R4+
                                                            self.JJ_params['indent'])*np.sin(self.angle)+(self.core/2+self.gap/2)
        else:
            self.JJ_params['y'] = self.Coaxmon1.center[1] + (self.Coaxmon1.R4 + self.JJ_params['indent']) * np.sin(self.angle)
        # print(self.angle)
        self.JJ = JJ4q.JJ_1(self.JJ_params['x'], self.JJ_params['y'],
                                self.JJ_params['a1'], self.JJ_params['a2'],
                                )
        result = self.JJ.generate_JJ()
        result = gdspy.boolean(result, result, 'or', layer=self.JJ_layer)
        angle = self.JJ_params['angle_JJ']
        result.rotate(angle, (self.JJ_params['x'], self.JJ_params['y']))
        indent = 1
        rect1 = gdspy.Rectangle((self.JJ_params['x'] - self.JJ.contact_pad_a / 2,
                                self.JJ_params['y'] +indent),
                               (self.JJ_params['x'] + self.JJ.contact_pad_a / 2,
                                self.JJ_params['y'] - self.JJ.contact_pad_b+indent), layer=6)
        rect2 = gdspy.Rectangle((self.JJ.x_end - self.JJ.contact_pad_a / 2,
                                self.JJ.y_end - 1),
                               (self.JJ.x_end + self.JJ.contact_pad_a /2 ,
                                self.JJ.y_end - self.JJ.contact_pad_b - indent), layer=6)
        if self.Coaxmon1.center[0] != self.Coaxmon2.center[0]:
            poly1 = gdspy.Polygon([(self.JJ_params['x'] - self.JJ.contact_pad_a / 2,
                                    self.JJ_params['y'] +indent),
                                   (self.JJ_params['x'] - self.JJ.contact_pad_a / 2,
                                    self.JJ_params['y'] + indent-self.JJ.contact_pad_b),
                                   (self.JJ_params['x'] - self.JJ.contact_pad_a-indent,self.Coaxmon1.center[1]-self.core/2),
                                   (self.JJ_params['x'] - self.JJ.contact_pad_a-indent,self.Coaxmon1.center[1]+self.core/2)
                                   ])
            poly2 = gdspy.Polygon([(self.JJ.x_end + self.JJ.contact_pad_a / 2,
                                    self.JJ.y_end -indent-self.JJ.contact_pad_b),
                                   (self.JJ.x_end + self.JJ.contact_pad_a / 2,
                                    self.JJ.y_end - indent ),
                                   (self.JJ.x_end + self.JJ.contact_pad_a + indent,
                                    self.Coaxmon1.center[1] + self.core / 2),
                                   (self.JJ.x_end + self.JJ.contact_pad_a + indent,
                                    self.Coaxmon1.center[1] - self.core / 2)
                                   ])
        else:
            poly1 = []
            poly2 = []
        rect = gdspy.boolean(rect1,[rect2,poly1,poly2], 'or', layer=6)
        rect.rotate(angle, (self.JJ_params['x'], self.JJ_params['y']))
        to_remove = gdspy.Polygon(self.JJ.points_to_remove,layer=self.layer_to_remove)
        to_remove.rotate(angle, (self.JJ_params['x'], self.JJ_params['y']))
        return result, rect , to_remove


    def generate_squid(self):
        # print(self.squid_params)
        self.squid = squid3JJ.JJ_2(self.squid_params['x'],
                                   self.squid_params['y'],
                self.squid_params['a1'], self.squid_params['a2'],
                self.squid_params['b1'], self.squid_params['b2'],
                self.squid_params['c1'], self.squid_params['c2'])
        squid = self.squid.generate_JJ()
        rect = gdspy.Rectangle((self.squid_params['x'] - self.squid.contact_pad_a_outer / 2,
                                self.squid_params['y'] + 0*self.squid.contact_pad_b_outer/2),
                               (self.squid_params['x'] + self.squid.contact_pad_a_outer / 2,
                                self.squid_params['y'] - self.squid.contact_pad_b_outer), layer=self.total_layer)

        if self.Coaxmon1.center[0] == self.Coaxmon2.center[0]:
            path1=gdspy.Polygon([(self.squid_params['x'], self.squid_params['y'] ),
                                 (self.Coaxmon1.center[0], self.squid_params['y'] ),
                                 (self.Coaxmon1.center[0], self.squid_params['y'] - self.squid.contact_pad_b_outer),
                                 (self.squid_params['x'], self.squid_params['y'] - self.squid.contact_pad_b_outer)])
            rect=gdspy.boolean(rect,path1,'or',layer=self.total_layer)

        # point1 =
        squid=gdspy.boolean(squid,squid,'or',layer=self.JJ_layer)
        squid.rotate(self.squid_params['angle'],(self.squid_params['x'],
                                   self.squid_params['y']))
        return squid ,rect

