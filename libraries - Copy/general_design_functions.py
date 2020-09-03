import gdspy
import numpy as np

#common clases
class coordinates:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class rotations:
    def __init__(self, phi1, phi2):
        self.phi1 = phi1
        self.phi2 = phi2





class Pads  :

    def __init__(self, TL_core, TL_gap, TL_ground, coordinate):
        self.coordinate = coordinate
        self.TL_core = TL_core
        self.TL_gap = TL_gap
        self.TL_ground = TL_ground

    @staticmethod
    def contact_pad(x, y, TL_core, TL_gap, TL_ground):
        #fundamental pad constants
        pad_core = 250 # to make pad with 50 Om impedance
        pad_vacuum = 146 # to make pad with 50 Om impedance
        pad_ground = TL_ground
        pad_length = 600
        pad_indent = 50
        edge_indent = 100
        narrowing = 160
        outer_pad_width = (pad_core + (pad_vacuum + pad_ground) * 2)
        inner_pad_width = (pad_core + pad_vacuum * 2)
        outer_TL_width = 2 * (TL_ground + TL_gap) + TL_core
        inner_TL_width = 2 * TL_gap + TL_core

        r1 = gdspy.Polygon([(x, y), (x + outer_TL_width, y),
                            (x + (outer_TL_width + outer_pad_width) / 2, y - narrowing),
                            (x + (outer_TL_width + outer_pad_width) / 2, y - (narrowing + pad_length)),
                            (x - (-outer_TL_width + outer_pad_width) / 2, y - (narrowing + pad_length)),
                            (x - (-outer_TL_width + outer_pad_width) / 2, y - narrowing)])
        x += TL_ground
        r2 = gdspy.Polygon([(x, y), (x + inner_TL_width, y),
                            (x + (inner_TL_width + inner_pad_width) / 2, y - narrowing),
                            (x + (inner_TL_width + inner_pad_width) / 2, y - (narrowing + pad_length - edge_indent)),
                            (x - (-inner_TL_width + inner_pad_width) / 2, y - (narrowing + pad_length - edge_indent)),
                            (x - (-inner_TL_width + inner_pad_width) / 2, y - narrowing)])
        x += TL_gap
        r3 = gdspy.Polygon([(x, y), (x + TL_core, y),
                            (x + (pad_core + TL_core) / 2, y - narrowing),
                            (x + (pad_core + TL_core) / 2, y - (narrowing + pad_length - pad_indent - edge_indent)),
                            (x - (pad_core - TL_core) / 2, y - (narrowing + pad_length - pad_indent - edge_indent)),
                            (x - (pad_core - TL_core) / 2, y - narrowing)])
        return gdspy.boolean(gdspy.boolean(r1, r2, 'not'), r3, 'or'), r1

    @staticmethod
    def generate_ground(Pads_array, sample_size_x, sample_size_y, total_layer, restricted_area_layer):
        edge = 600 #fundamental constant - edge length
        pad_total_length = 760+200 #fundamental constant - pad_length+narrowing+200, where 200 is an indent for an orientation checking
        r1 = gdspy.Rectangle((0, 0), (sample_size_x, sample_size_y))
        r2 = gdspy.Rectangle((edge, edge), (sample_size_x - edge, sample_size_y - edge))
        result = gdspy.boolean(r1, r2, 'not')
        restricted_area = result
        for one_pad in Pads_array:
            coord_init_x, coord_init_y = one_pad.coordinate
            TL_core = one_pad.TL_core
            TL_vacuum = one_pad.TL_gap
            TL_ground = one_pad.TL_ground
            outer_TL_width = TL_core + 2*(TL_ground + TL_vacuum)
            coord_x,coord_y = (coord_init_x-outer_TL_width/2,coord_init_y)
            pad, restricted_pad = one_pad.contact_pad(coord_x, coord_y, TL_core, TL_vacuum, TL_ground)
            if coord_x < pad_total_length:
                pad.rotate(-np.pi/2, [coord_init_x, coord_init_y])
                restricted_pad.rotate(-np.pi/2, [coord_init_x, coord_init_y])
            elif coord_x > sample_size_x - pad_total_length:
                pad.rotate(np.pi/2, [coord_init_x, coord_init_y])
                restricted_pad.rotate(np.pi/2, [coord_init_x, coord_init_y])
            elif coord_y > sample_size_y/2:
                pad.mirror([0,  coord_init_y], [sample_size_y,  coord_init_y])
                restricted_pad.mirror([0,  coord_init_y], [sample_size_y,  coord_init_y])
            to_bool = gdspy.Rectangle(pad.get_bounding_box()[0].tolist(), pad.get_bounding_box()[1].tolist())
            result = gdspy.boolean(gdspy.boolean(result, to_bool, 'not'), pad, 'or', layer=total_layer)
            restricted_area = gdspy.boolean(restricted_area, restricted_pad, 'or', layer=restricted_area_layer)
        return result, restricted_area

class Airbridge:
    def __init__(self, width, length, padsize, point, angle,line_type = None):
        self.x = point[0]
        self.y = point[1]
        self.angle = angle
        self.padsize = padsize
        self.width = width
        self.length = length
        self.restrictedArea = []
        self.line_type = line_type
        self.start = (None, None)
        self.end = (None, None)

    def generate_bridge(self, Padlayer, Bridgelayer):
        if self.line_type == 'line':
            self.x += (self.length/2 + self.padsize/2)*np.cos(self.angle)
            self.y += (self.length/2 + self.padsize/2)*np.sin(self.angle)

        #first the two contacts
        contact_1 = gdspy.Rectangle((self.x - self.length/2 - self.padsize/2, self.y-self.padsize/2),
                                    (self.x - self.length/2 + self.padsize/2, self.y + self.padsize/2))
        contact_2 = gdspy.Rectangle((self.x + self.length/2 - self.padsize/2, self.y-self.padsize/2),
                                    (self.x + self.length/2 + self.padsize/2, self.y + self.padsize/2))
        contacts = gdspy.boolean(contact_1, contact_2, 'or', layer=Padlayer)
        contacts.rotate(self.angle, (self.x, self.y))
        # add restricted area for holes
        self.restrictedArea.append(
            gdspy.Rectangle((self.x - self.length / 2 - self.padsize / 2, self.y - self.padsize / 2),
                            (self.x + self.length / 2 + self.padsize / 2, self.y + self.padsize / 2)))
        #now the bridge itself
        bridge = gdspy.Rectangle((self.x - self.length / 2, self.y - self.width / 2),
                                 (self.x + self.length / 2, self.y + self.width / 2),layer=Bridgelayer)
        bridge.rotate(self.angle, (self.x, self.y))

        return [contacts, bridge]




class Feedline:
    def __init__(self, points, core, gap, ground, nodes, total_layer, rectricted_area_layer,R):
        self.core = core
        self.ground = ground
        self.gap = gap
        self.points = points
        self.nodes = nodes
        self._R = R # fundamental constant
        self.rectricted_area = None
        self.total_layer = total_layer
        self.rectricted_area_layer = rectricted_area_layer
        self.end = self.points[-1]
        self.angle = None

    def generate_feedline(self, corners_type='round'):
        offset = self.gap + (self.core + self.ground) / 2

        self.R1 = self._R - (
                    self.core / 2 + self.gap + self.ground / 2)
        self.R2 = np.abs(
            self._R + self.core / 2 + self.gap + self.ground / 2)

        self.R1_new = self._R - (self.core / 2 + self.gap / 2)
        self.R2_new = np.abs(self._R + self.core / 2 + self.gap / 2)

        result = None
        result_ = None
        result_new = None

        if len(self.points)<3:
            self.points.insert(1,((self.points[0][0]+self.points[1][0])/2,
                                (self.points[0][1]+self.points[1][1])/2))
        new_points = self.points
        # new_points_restricted_line= self.points

        width_restricted_line = 2 * self.ground + 2 * self.gap + self.core
        width_new = self.gap
        offset_new = self.gap + self.core

        if self.R1 > 0:
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
                if corners_type is 'round':
                    vector1 = np.asarray(new_points[i + 1][:]) - new_points[i][:]
                    vector2 = np.asarray(new_points[i + 2][:]) - new_points[i + 1][:]
                    vector_prod = vector1[0] * vector2[1] - vector1[1] * vector2[0]
                    if vector_prod < 0:
                        line = gdspy.FlexPath([point1, point2, point3],
                                              [self.ground, self.core,
                                               self.ground], offset, ends=["flush", "flush", "flush"],
                                              corners=["circular bend", "circular bend", "circular bend"],
                                              bend_radius=[self.R1, self._R, self.R2], precision=0.001, layer=0)
                        rectricted_line = gdspy.FlexPath([point1, point2, point3], width_restricted_line, offset=0,
                                                         ends="flush", corners="circular bend", bend_radius=self._R,
                                                         precision=0.001, layer=1)
                        line_new = gdspy.FlexPath([point1, point2, point3], [width_new, width_new], offset=offset_new,
                                                  ends=["flush", "flush"], corners=["circular bend", "circular bend"],
                                                  bend_radius=[self.R1_new, self.R2_new], precision=0.001, layer=2)

                        # rectricted_line = gdspy.FlexPath([point1, point2, point3], width_restricted_line, offset=0,  ends="flush",  corners="circular bend",bend_radius= self._R, precision=0.001, layer=1)

                        # self.end = line.x
                    else:
                        line = gdspy.FlexPath([point1, point2, point3],
                                              [self.ground, self.core,
                                               self.ground], offset, ends=["flush", "flush", "flush"],
                                              corners=["circular bend", "circular bend", "circular bend"],
                                              bend_radius=[self.R2, self._R, self.R1], precision=0.001, layer=0)
                        rectricted_line = gdspy.FlexPath([point1, point2, point3], width_restricted_line, offset=0,
                                                         ends="flush", corners="circular bend", bend_radius=self._R,
                                                         precision=0.001, layer=1)
                        line_new = gdspy.FlexPath([point1, point2, point3], [width_new, width_new], offset=offset_new,
                                                  ends=["flush", "flush"], corners=["circular bend", "circular bend"],
                                                  bend_radius=[self.R2_new, self.R1_new], precision=0.001, layer=2)

                        # self.end = line.x
                    result = gdspy.boolean(line, result, 'or', layer=self.total_layer)
                    result_ = gdspy.boolean(rectricted_line, result_, 'or', layer=self.rectricted_area_layer)
                    result_new = gdspy.boolean(line_new, result_new, 'or', layer=2)
                    # self.rectricted_area = gdspy.boolean(rectricted_line, self.rectricted_area, 'or', layer = self.rectricted_area_layer)

                    # result= line, rectricted_line
                else:
                    line = gdspy.FlexPath([point1, point2, point3],
                                          [self.ground, self.core,
                                           self.ground], offset, ends=["flush", "flush", "flush"],
                                          precision=0.001, layer=0)
                    # self.end = line.x
                    rectricted_line = gdspy.FlexPath([point1, point2, point3],
                                                     width_restricted_line, offset=0, ends="flush",
                                                     precision=0.001, layer=1)

                    line_new = gdspy.FlexPath([point1, point2, point3],
                                              [width_new, width_new], offset=offset_new, ends=["flush", "flush"],
                                              precision=0.001, layer=2)

                    # result= line, rectricted_line
                    result = gdspy.boolean(line, result, 'or', layer=self.total_layer)
                    result_ = gdspy.boolean(rectricted_line, result_, 'or', layer=self.rectricted_area_layer)
                    result_new = gdspy.boolean(line_new, result_new, 'or', layer=2)

                    # self.rectricted_area = gdspy.boolean(rectricted_line, self.rectricted_area, 'or', layer = self.rectricted_area_layer)

        else:
            print('R small < 0')
            result = 0
            result_ = 0
            result_new = 0
        vector2_x = point3[0] - point2[0]
        vector2_y = point3[1] - point2[1]
        if vector2_x != 0 and vector2_x>=0:
            tang_alpha=vector2_y/vector2_x
            self.angle = np.arctan(tang_alpha)
        elif vector2_x != 0 and vector2_x<0:
            tang_alpha=vector2_y/vector2_x
            self.angle = np.arctan(tang_alpha)+np.pi
        elif vector2_x == 0 and vector2_y>0:
            self.angle = np.pi/2
        elif vector2_x == 0 and vector2_y<0:
            self.angle = -np.pi/2
        else:
            print("something is wrong in angle")
        return result, result_, result_new

    def generate_end(self, end):
        if end['type'] is 'open':
            return self.generate_open_end(end)
        if end['type'] is 'fluxline':
            return self.generate_fluxline_end(end)

    def generate_fluxline_end(self,end):
        JJ = end['JJ']
        length = end['length']
        width = end['width']
        point1 = JJ.rect1
        point2 = JJ.rect2
        result = None
        # rect_to_remove = gdspy.Rectangle((),
        #                                  ())
        for point in [point1,point2]:
            line = gdspy.Rectangle((point[0]-width/2, point[1]),
                                (point[0]+width/2, point[1]-length))
            result = gdspy.boolean(line, result, 'or', layer=6)
        # result = gdspy.boolean(line1,line2,'or',layer=self.total_layer)
        path1 = gdspy.Polygon([(point1[0]+width/2, point1[1]-length),(point1[0]-width/2, point1[1]-length),
                               (self.end[0]+(self.core/2+self.gap+width)*np.cos(self.angle+np.pi/2),
                                self.end[1]+(self.core/2+self.gap+width)*np.sin(self.angle+np.pi/2)),
                               (self.end[0] + (self.core / 2 + self.gap ) * np.cos(self.angle +np.pi/2),
                                self.end[1] + (self.core / 2 + self.gap ) * np.sin(self.angle +np.pi/2))])


        result = gdspy.boolean(path1,result, 'or', layer=6)

        path2 = gdspy.Polygon([(point2[0] + width / 2, point2[1] - length),(point2[0] - width / 2, point2[1] - length),
                               (self.end[0] +(self.core / 2)*np.cos(self.angle+np.pi/2),self.end[1]+( self.core / 2)*np.sin(self.angle+np.pi/2)),
                               (self.end[0] + self.core / 2 *np.cos(self.angle+3*np.pi/2),self.end[1]+( self.core / 2)*np.sin(self.angle+3*np.pi/2))])
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
                                        layer=self.rectricted_area_layer)
        # else:
        #
        return result,restricted_area,restricted_area
    def generate_open_end(self,end):
        end_gap = end['gap']
        end_ground_length = end['ground']
        x_begin = self.end[0]
        y_begin = self.end[1]

        restricted_area = gdspy.Rectangle((x_begin-self.core/2-self.gap-self.ground,y_begin),
                                         (x_begin+self.core/2+self.gap+self.ground,y_begin+end_gap+end_ground_length))
        rectangle_for_removing = gdspy.Rectangle((x_begin-self.core/2-self.gap,y_begin),
                                         (x_begin+self.core/2+self.gap, y_begin+end_gap))
        total = gdspy.boolean(restricted_area,rectangle_for_removing,'not')
        total.rotate(-np.pi/2+self.angle, self.end)
        restricted_area.rotate(-np.pi/2+self.angle, self.end)
        rectangle_for_removing.rotate(-np.pi/2+self.angle, self.end)
        return total, restricted_area , rectangle_for_removing

class Narrowing:

    def __init__(self, x, y, core1, gap1, ground1, core2, gap2, ground2, h, rotation):
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

        x_end=self._x_begin-self._h
        y_end=self._y_begin
        result=None

        points_for_poly1=[(self._x_begin,self._y_begin+self.first_core/2+self.first_gap+self.first_ground),
                  (self._x_begin,self._y_begin+self.first_core/2+self.first_gap),
                  (x_end,y_end+self.second_core/2+self.second_gap),
                  (x_end,y_end+self.second_core/2+self.second_gap+self.second_ground)]

        points_for_poly2=[(self._x_begin,self._y_begin+self.first_core/2),
                  (self._x_begin,self._y_begin-self.first_core/2),
                  (x_end,y_end-self.second_core/2),
                  (x_end,y_end+self.second_core/2)]

        points_for_poly3=[(self._x_begin,self._y_begin-(self.first_core/2+self.first_gap+self.first_ground)),
                  (self._x_begin,self._y_begin-(self.first_core/2+self.first_gap)),
                  (x_end,y_end-(self.second_core/2+self.second_gap)),
                  (x_end,y_end-(self.second_core/2+self.second_gap+self.second_ground))]

        points_for_restricted_area=[(self._x_begin, self._y_begin+ self.first_core/2 + self.first_gap + self.first_ground),
                                    (x_end,y_end+self.second_core/2+self.second_gap+self.second_ground),
                                    (x_end,y_end-(self.second_core/2+self.second_gap+self.second_ground)),
                                    (self._x_begin,self._y_begin-(self.first_core/2+self.first_gap+self.first_ground))]

        restricted_area = gdspy.Polygon(points_for_restricted_area)

        poly1 = gdspy.Polygon(points_for_poly1)
        poly2 = gdspy.Polygon(points_for_poly2)
        poly3 = gdspy.Polygon(points_for_poly3)



        if self._rotation==0:


            result=poly1, poly2, poly3

        else:

            poly1_=poly1.rotate(angle=self._rotation, center=(self._x_begin, self._y_begin))
            poly2_=poly2.rotate(angle=self._rotation, center=(self._x_begin, self._y_begin))
            poly3_=poly3.rotate(angle=self._rotation, center=(self._x_begin, self._y_begin))
            restricted_area.rotate(angle=self._rotation, center=(self._x_begin, self._y_begin))
            result=poly1_, poly2_, poly3_
            polygon_to_remove=gdspy.boolean(restricted_area,result,'not',layer=2)
        return result, restricted_area, polygon_to_remove





# class Resonator:
#
#     def __init__(self, frequency, initial_point, width_central, width_gap, width_ground, open_end_length, coupler_length, l3, l4, l5, h_end):
#         self._x = initial_point[0] + coupler_length/2
#         self._y = initial_point[1]-open_end_length
#         self._width_central = width_central
#         self._width_ground = width_ground
#         self._distance_between = width_gap
#         self._l1 = open_end_length
#         self._l2 = coupler_length
#         self._l3 = l3
#         self._l4 = l4
#         self._l5 = l5
#         self.f = frequency
#         self.c = 299792458
#         self.epsilon_eff = (11.45+1)/2
#         self._L = self.c/(4*np.sqrt(self.epsilon_eff)*frequency)*1e6 - self._l1-self._l2-self._l3-self._l4-self._l5
#         self._h_end = h_end
#
#     def Generate_resonator(self,angle):
#         const = self._width_ground + self._distance_between + self._width_central/2
#         offset=self._distance_between+(self._width_central+self._width_ground)/2
#
#         x1=self._l1+const
#         x2=self._l3+2*const
#         x3=self._l5-const
#
#         element1=x1-x2-x3-2*const
#         element2=self._x-(self._x-self._l2+self._l4)
#         element=element1+element2
#         D=element2
#
#         L_new=self._L-const
#         N = int(np.floor((L_new)/(element)))
#         tail=L_new-N*element
#
#         Number_of_points=[((self._x-self._l2+self._l4,self._y+self._l1-self._l3-self._l5))]
#         i=1
#         while i < N+1:
#
#             if i%2!=0:
#                 list1=[(self._x-self._l2+self._l4-(i-1)*D,self._y+self._width_ground+self._distance_between+self._width_central/2),
#                       (self._x-self._l2+self._l4-i*D,self._y+self._width_ground+self._distance_between+self._width_central/2)]
#
#                 Number_of_points.extend(list1)
#
#             else:
#                 list2=[(self._x-self._l2+self._l4-(i-1)*D,self._y+self._l1-self._l3-self._l5-(self._width_ground+self._distance_between+self._width_central/2)),
#                        (self._x-self._l2+self._l4-i*D,self._y+self._l1-self._l3-self._l5-(self._width_ground+self._distance_between+self._width_central/2))]
#                 Number_of_points.extend(list2)
#             i = i + 1
#
#         if (N)%2!=0:
#                 tail1=Number_of_points[2*N][1]+tail-const
#                 list_add_tail1=[(Number_of_points[2*N][0],tail1)]
#
#                 Number_of_points.extend(list_add_tail1)
#
#         else:
#                 tail2=Number_of_points[2*N][1]-tail-const
#                 list_add_tail2=[(Number_of_points[2*N][0],tail2) ]
#                 Number_of_points.extend(list_add_tail2)
#
#         if (N)%2!=0:
#
#
#             end1=(Number_of_points[2*N+1][0]+const, Number_of_points[2*N+1][1])
#             end2=(Number_of_points[2*N+1][0]+const, Number_of_points[2*N+1][1]+self._h_end)
#             end3=(Number_of_points[2*N+1][0]-const, Number_of_points[2*N+1][1]+self._h_end)
#             end4=(Number_of_points[2*N+1][0]-const, Number_of_points[2*N+1][1])
#
#         else:
#             end1=(Number_of_points[2*N+1][0]+const, Number_of_points[2*N+1][1])
#             end2=(Number_of_points[2*N+1][0]+const, Number_of_points[2*N+1][1]-self._h_end)
#             end3=(Number_of_points[2*N+1][0]-const, Number_of_points[2*N+1][1]-self._h_end)
#             end4=(Number_of_points[2*N+1][0]-const, Number_of_points[2*N+1][1])
#
#         line = gdspy.FlexPath([(self._x, self._y), (self._x, self._y+self._l1), (self._x-self._l2,self._y+self._l1),(self._x-self._l2, self._y+self._l1-self._l3),(self._x-self._l2+self._l4,self._y+self._l1-self._l3), (self._x-self._l2+self._l4,self._y+self._l1-self._l3-self._l5)],[self._width_ground, self._width_central, self._width_ground],offset,ends=["flush", "flush", "flush"],precision=0.1)
#         line1 = gdspy.FlexPath(Number_of_points,[self._width_ground, self._width_central, self._width_ground],offset,ends=["flush", "flush", "flush"])
#
#         end = gdspy.Polygon([end1,end2,end3,end4])
#
#         result = gdspy.boolean(line, line1, 'or')
#         result = gdspy.boolean(end, result, 'or')
#         return result.rotate(angle, (self._x,self._y))
