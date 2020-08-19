import numpy as np
import gdspy
import libraries.general_design_functions as gdf
from typing import NamedTuple, SupportsFloat, Any

Bridges_over_line_param = NamedTuple('Bridge_params',
                                     [('distance', SupportsFloat),
                                      ('offset', SupportsFloat),
                                      ('width', SupportsFloat),
                                      ('length', SupportsFloat),
                                      ('padsize', SupportsFloat),
                                      ('line_type', Any)])


class Sample:

    def __init__(self, name, layer_configurations):
        self.name = str(name)

        self.result = gdspy.Cell('result')
        self.total_cell = gdspy.Cell(self.name)
        self.restricted_area_cell = gdspy.Cell(self.name + 'resctricted area')
        self.label_cell = gdspy.Cell(self.name + ' labels')
        self.cell_to_remove = gdspy.Cell(self.name + ' remove')

        self.layer_configurations = layer_configurations
        self.total_layer = layer_configurations['total']
        self.restricted_area_layer = layer_configurations['restricted area']
        self.layer_to_remove = layer_configurations['for removing']
        self.JJ_layer = layer_configurations['JJs']
        self.AirbridgesLayer = layer_configurations['air bridges']
        self.AirbridgesPadLayer = layer_configurations['air bridge pads']
        self.gridline_x_layer = layer_configurations['vertical gridlines']
        self.gridline_y_layer = layer_configurations['horizontal gridlines']


        self.sample_vertical_size = None
        self.sample_horizontal_size = None

        self.pads = []
        self.qubits = []
        self.lines = []
        self.bridges = []
        self.couplers = []
        self.resonators = []
        self.purcells = []

    # General methods for all qubit classes
    def numerate(self, name, number, coordinate):
        size=70
        layer=51
        vtext = gdspy.Text(name + str(number), size, coordinate, horizontal=True, layer=layer)
        # label = gdspy.Label(name + str(number), coordinate, texttype=25)
        self.label_cell.add(vtext)
    def add_pad(self, TL_core, TL_gap, TL_ground, coordinate):
        self.pads.append(gdf.Pads(TL_core, TL_gap, TL_ground, coordinate))
        self.numerate("Pad", len(self.pads)-1,coordinate)

    def generate_sample_edges_and_pads(self):
        results_total, restricted_area = gdf.Pads.generate_ground(self.pads,
                                                              self.sample_horizontal_size, self.sample_vertical_size, self.total_layer,
                                                              self.restricted_area_layer)
        self.total_cell.add(results_total)
        self.restricted_area_cell.add(restricted_area)


    def generate_line(self, points, core, gap, ground, nodes=None, end=None, R=40, corner_type='round',bridge_params=None):
        """
                :param bridge_params default is None. In this way there won't be created any addidtional bridges.
                To create bridges crossing the line define "bridge_params" as a tuple with 5 elements in order:
                    distance, offset, width, length, padsize, line_type.
                'distance' is a minimal distance between bridges
                Not to get confused it's better to use Bridge_params. (Its just namedtuple.)
                The following lines are equivalent:
                    Bridges_over_line_param(distance=90, offset=40, width=15, length=90, padsize=30, line_type=None)
                    Bridges_over_line_param(90, 40, 15, length=90, padsize=30, line_type=None)
                    Bridges_over_line_param(90, 40, 15, 90, 30, None)
                    (90, 40, 15, 90, 30, None)
                """
        if bridge_params is not None:
            distance, offset, width, length, padsize, line_type = bridge_params
            for num_line in range(len(points) - 1):
                start, finish = points[num_line], points[num_line + 1]
                if finish[0] - start[0] == 0:
                    line_angle = np.pi / 2
                else:
                    line_angle = np.arctan((finish[1] - start[1]) / (finish[0] - start[0]))
                if finish[1] - start[1] < 0 or finish[1] - start[1] == 0 and finish[0] - start[0] < 0:
                    line_angle += np.pi
                line_length = np.sqrt((finish[0] - start[0]) ** 2 + (finish[1] - start[1]) ** 2)
                total_bridges = int((line_length - 2 * offset) / distance)
                offset = (line_length - total_bridges * float(distance)) / 2
                for num_bridge in range(int((line_length - 2 * offset) / distance) + 1):
                    bridge_center = (start[0] + np.cos(line_angle) * (offset + num_bridge * distance),
                                     start[1] + np.sin(line_angle) * (offset + num_bridge * distance))
                    self.generate_bridge(bridge_center, width, length, padsize, line_angle + np.pi / 2,
                                         line_type=line_type)

        self.lines.append(gdf.Feedline(points, core, gap, ground, nodes, self.total_layer, self.restricted_area_layer, R))
        line = self.lines[-1].generate_feedline(corner_type)
        if end is not None:
            end_line = self.lines[-1].generate_end(end)
            self.total_cell.add(end_line[0])
            self.restricted_area_cell.add(
                gdspy.boolean(end_line[1], end_line[1], 'or', layer=self.restricted_area_layer))
            self.cell_to_remove.add(gdspy.boolean(end_line[2], end_line[2], 'or', layer=self.layer_to_remove))
        self.total_cell.add(line[0])
        self.restricted_area_cell.add(line[1])
        self.cell_to_remove.add(line[2])



    def generate_bridge(self, point, width, length, padsize, angle, line_type=None):
        self.bridges.append(gdf.Airbridge(width, length, padsize, point, angle, line_type))
        bridge = self.bridges[-1].generate_bridge(self.AirbridgesPadLayer, self.AirbridgesLayer)
        self.total_cell.add(bridge)
        self.result.add(gdspy.boolean(bridge[0], bridge[0], 'or', layer=self.total_layer))
        self.restricted_area_cell.add(gdspy.boolean(bridge[0], bridge[0], 'or', layer=self.restricted_area_layer))

    def generate_narrowing_part(self, firstline, secondline):
        narrowing_length = 15
        narrowing1 = gdf.Narrowing(firstline.end[0], firstline.end[1],
                               firstline.core, firstline.gap, firstline.ground,
                               secondline[0], secondline[1], secondline[2],
                               narrowing_length, firstline.angle + np.pi)
        line1 = narrowing1.generate_narrowing()
        self.total_cell.add(line1[0])
        self.restricted_area_cell.add(gdspy.boolean(line1[1],line1[1],'or',layer=self.restricted_area_layer))
        self.cell_to_remove.add(line1[2])
        return (firstline.end[0] + narrowing_length * np.cos(firstline.angle),
                firstline.end[1] + narrowing_length * np.sin(firstline.angle))

    def generate_bridge_over_feedline(self, firstline, airbridge, secondline, distance_between_airbridges):
        narrowing_length = 15
        narrowing1 = gdf.Narrowing(firstline.end[0], firstline.end[1],
                               firstline.core, firstline.gap, firstline.ground,
                               airbridge[2], distance_between_airbridges, airbridge[2],
                               narrowing_length, np.pi + firstline.angle)
        narrowing2 = gdf.Narrowing(
            firstline.end[0] + np.cos(firstline.angle) * (narrowing_length + airbridge[0] * 2 + airbridge[1]),
            firstline.end[1] + np.sin(firstline.angle) * (narrowing_length + airbridge[0] * 2 + airbridge[1]),
            airbridge[2], distance_between_airbridges, airbridge[2],
            secondline[0], secondline[1], secondline[2],
            narrowing_length, np.pi + firstline.angle)
        self.generate_bridge((firstline.end[0] + np.cos(firstline.angle) * narrowing_length -
                              np.sin(firstline.angle) * (distance_between_airbridges + airbridge[2]),
                              firstline.end[1] + np.cos(firstline.angle) * (
                                          distance_between_airbridges + airbridge[2]) +
                              np.sin(firstline.angle) * narrowing_length),
                             airbridge[0], airbridge[1], airbridge[2], firstline.angle, 'line')
        self.generate_bridge((firstline.end[0] + np.cos(firstline.angle) * narrowing_length,
                              np.sin(firstline.angle) * narrowing_length + firstline.end[1]),
                             airbridge[0], airbridge[1], airbridge[2], firstline.angle, 'line')
        self.generate_bridge((firstline.end[0] + np.cos(firstline.angle) * narrowing_length +
                              np.sin(firstline.angle) * (distance_between_airbridges + airbridge[2]),
                              firstline.end[1] - np.cos(firstline.angle) * (
                                          distance_between_airbridges + airbridge[2]) +
                              np.sin(firstline.angle) * narrowing_length),
                             airbridge[0], airbridge[1], airbridge[2], firstline.angle, 'line')
        line1 = narrowing1.generate_narrowing()
        line2 = narrowing2.generate_narrowing()
        self.total_cell.add(line1[0])
        self.total_cell.add(line2[0])
        self.restricted_area_cell.add(line1[1])
        self.restricted_area_cell.add(line2[1])
        self.cell_to_remove.add(line1[2])
        self.cell_to_remove.add(line2[2])
        return (firstline.end[0] + 2 * narrowing_length + airbridge[0] * 2 + airbridge[1], firstline.end[1]), None

    def finish_him(self):
        # self.result.add(gdspy.boolean(self.total_cell.get_polygons(by_spec=True)[(self.total_layer, 0)],
        #                               self.total_cell.get_polygons(by_spec=True)[(self.total_layer, 0)], 'or',
        #                               layer=self.total_layer))
        self.result.add(gdspy.boolean(self.total_cell.get_polygons(by_spec=True)[(self.total_layer, 0)],
                                      self.cell_to_remove.get_polygons(by_spec=True)[(2, 0)], 'not',
                                      layer=self.total_layer))
        self.result.add(gdspy.boolean(self.total_cell.get_polygons(by_spec=True)[(self.AirbridgesLayer, 0)],
                                      self.total_cell.get_polygons(by_spec=True)[(self.AirbridgesLayer, 0)], 'or',
                                      layer=self.AirbridgesLayer))
        self.result.add(gdspy.boolean(self.total_cell.get_polygons(by_spec=True)[(self.AirbridgesPadLayer, 0)],
                                      self.total_cell.get_polygons(by_spec=True)[(self.AirbridgesPadLayer, 0)], 'or',
                                      layer=self.AirbridgesPadLayer))
        #
        # костыль

    #         self.result.add(gdspy.boolean(sample.total_cell.get_polygons(by_spec=True)[(self.total_layer,0)],
    #                                       sample.total_cell.get_polygons(by_spec=True)[(3,0)],'not',
    #                                       layer = self.total_layer))
    #         self.result.add(gdspy.boolean(sample.total_cell.get_polygons(by_spec=True)[(3,0)],
    #                                       sample.total_cell.get_polygons(by_spec=True)[(3,0)],'or',
    #                                       layer = 3))
    #         self.result.add(gdspy.boolean(sample.total_cell.get_polygons(by_spec=True)[(0,0)],
    #                                       sample.total_cell.get_polygons(by_spec=True)[(3,0)],'not',
    #                                       layer = 10))

    #         self.result.add(gdspy.boolean( sample.total_cell.get_polygons(by_spec=True)[(self.JJ_layer,0)],
    #                                        sample.total_cell.get_polygons(by_spec=True)[(self.JJ_layer,0)],'or',
    #                                        layer =self.JJ_layer))
    #         self.result.add(gdspy.boolean( sample.total_cell.get_polygons(by_spec=True)[(self.AirbridgesPadLayer,0)],
    #                                        sample.total_cell.get_polygons(by_spec=True)[(self.AirbridgesPadLayer,0)],'or',
    #                                        layer =self.AirbridgesPadLayer))
    #         self.result.add(gdspy.boolean( sample.total_cell.get_polygons(by_spec=True)[(self.AirbridgesLayer,0)],
    #                                        sample.total_cell.get_polygons(by_spec=True)[(self.AirbridgesLayer,0)],'or',
    #                                        layer =self.AirbridgesLayer))
    def create_grid(self, width, gap):
        """add rectangular grid to the structure:
        width = width of the lines
        gap = distance between two lines
        """
        result_x = None
        result_y = None
        i = 0
        while gap * i + width * (i + 1) < self.sample_horizontal_size:
            rect_x = gdspy.Rectangle((gap * i + width * i, 0), (gap * i + width * (i + 1), self.sample_vertical_size),
                                     layer=self.gridline_x_layer)
            i += 1
            result_x = gdspy.boolean(rect_x, result_x, 'or')
        i = 0
        while gap * i + width * (i + 1) < self.sample_vertical_size:
            rect_y = gdspy.Rectangle((0, gap * i + width * i), (self.sample_horizontal_size, gap * i + width * (i + 1)),
                                     layer=self.gridline_y_layer)
            i += 1
            result_y = gdspy.boolean(rect_y, result_y, 'or')
        result_x = gdspy.boolean(result_x, result_y, 'not', layer=self.gridline_y_layer)
        rest_area = self.restricted_area_cell.get_polygons(by_spec=True)[(self.restricted_area_layer, 0)]
        result_x = gdspy.boolean(result_x, rest_area, 'not', layer=self.gridline_x_layer)
        result_y = gdspy.boolean(result_y, rest_area, 'not', layer=self.gridline_y_layer)
        if result_x != None: self.result.add(result_x)
        if result_y != None: self.result.add(result_y)


    # Resonator + Purcell methods
    def generate_resonator_coupler(self, start, end, feedline_core, feedline_gap, feedline_ground, purcell_core,
                                   purcell_gap,
                                   purcell_ground, gap_feedline_purcell, rotation, coupler_type):
        import Coupler_feedline_purcell as coup
        coupler = coup.Coupler_feedline_and_purcell(start[0], start[1], end[0], end[1], feedline_core, feedline_gap,
                                                    feedline_ground,
                                                    purcell_core, purcell_gap,
                                                    purcell_ground, gap_feedline_purcell, rotation, coupler_type)
        line, connectors = coupler.Generate_coupler_feedline_and_purcell()
        self.total_cell.add(line)
        return connectors

    def generate_purcell(self, purcell_begin, purcell_end, restricted_area_points0, a_restricted_area,
                         b_restricted_area,
                         purcell_core, purcell_ground, purcell_gap, length_of_purcell_meander, begin_coupler=None,
                         type_mirror=None):
        import Purcell_idea1 as pur
        P = pur.Purcell(purcell_begin, purcell_end, restricted_area_points0, a_restricted_area, b_restricted_area,
                        purcell_core, purcell_ground, purcell_gap, length_of_purcell_meander)

        line = P.Generate_purcell()
        if begin_coupler != None:
            line1 = P.Generate_purcell_add_element(begin_coupler)
            self.total_cell.add(line1)
        #         self.total_cell.add(line[-1])
        if type_mirror == 'mirror':
            self.total_cell.add(line[0].mirror(purcell_end, (purcell_end[0], purcell_end[1] - 20)))
            self.total_cell.add(line[1].mirror(purcell_end, (purcell_end[0], purcell_end[1] - 20)))
            self.cell_to_remove.add(line[2].mirror(purcell_end, (purcell_end[0], purcell_end[1] - 20)))
            par_begin = (purcell_begin[0] + (purcell_end[0] - purcell_begin[0]), purcell_end[1])
            par_end = (purcell_end[0] + (purcell_end[0] - purcell_begin[0]), purcell_begin[1])
        else:
            #             self.total_cell.add(line1.mirror(purcell_begin,(purcell_begin[0],purcell_begin[1]-20)))
            self.total_cell.add(line[0])
            self.total_cell.add(line[1])
            self.cell_to_remove.add(line[2])
            par_end = purcell_end
            par_begin = purcell_begin
        return par_begin, par_end

    def generate_resonator(self, resonator_begin, resonator_end, restricted_area_points0,
                           a_restricted_area, b_restricted_area,
                           resonator_core, resonator_ground, resonator_gap, length_of_all_resonator, angle,
                           type_mirror=None):
        import Resonator_idea1 as res
        c = res.Resonator(resonator_begin, resonator_end, restricted_area_points0,
                          a_restricted_area, b_restricted_area,
                          resonator_core, resonator_ground, resonator_gap, length_of_all_resonator)
        line = c.Generate_resonator()

        if type_mirror == 'mirror':
            self.total_cell.add(line[0].mirror(resonator_end, (resonator_end[0], resonator_end[1] - 20)))
            self.total_cell.add(line[1].mirror(resonator_end, (resonator_end[0], resonator_end[1] - 20)))
            self.cell_to_remove.add(line[2].mirror(resonator_end, (resonator_end[0], resonator_end[1] - 20)))
            res_begin = (resonator_begin[0] + (resonator_end[0] - resonator_begin[0]), resonator_end[1])
            res_end = (resonator_end[0] + (resonator_end[0] - resonator_begin[0]), resonator_begin[1])
        else:
            self.total_cell.add(line)
            self.total_cell.add(line[1])
            self.cell_to_remove.add(line[2])
            res_end = resonator_end
            res_begin = resonator_begin

        return res_begin, res_end




    def generate_coupler_purcell_resonator(self, point1, point2, resonator_params, purcell_params, l, h, h_ground):
        import Coupler_purcell_resonator_idea1 as coup
        coupler = coup.Coupler_resonator_purcell(point1, point2,
                                                 resonator_params[0], resonator_params[1], resonator_params[2],
                                                 purcell_params[0], purcell_params[1], purcell_params[2], l, h,
                                                 h_ground)
        line = coupler.generate_coupler_resonator_purcell()
        self.total_cell.add(line)
        self.cell_to_remove.add(line[1])


    #Specific methods
    def add_qubit(self,Qubit,*args, **kwargs,):
        angle=kwargs['angle']

        self.qubits.append(gdf.Coaxmon(coordinate, r1, r2, r3, r4, outer_ground, self.total_layer,
                                         self.restricted_area_layer, self.JJ_layer, Couplers, JJ))
        qubit_total, restricted_area, JJ_total = self.qubits[-1].generate_qubit()
        self.total_cell.add(qubit_total.rotate(angle, args[0]))
        self.total_cell.add(JJ_total)  # .rotate(angle,(center_point.x,center_point.y))
        self.restricted_area_cell.add(restricted_area)
        self.numerate("Qb", len(self.qubits) - 1, args[0])

    def add_qubit_coupler(self, core, gap, ground, Coaxmon1, Coaxmon2, JJ, squid):
        coupler = gdf.IlyaCoupler(core, gap, ground, Coaxmon1, Coaxmon2, JJ, squid,
                              self.total_layer, self.restricted_area_layer, self.JJ_layer, self.layer_to_remove)
        self.couplers.append(coupler)
        line, JJ = coupler.generate_coupler()
        self.total_cell.add([line[0], JJ[0], JJ[1]])
        #         self.total_cell.add(line[1])
        self.restricted_area_cell.add(line[1])
        self.cell_to_remove.add([line[2], JJ[2]])
        self.numerate("IlCoup", len(self.couplers) - 1, ((Coaxmon1.center[0]+Coaxmon2.center[0])/2, (Coaxmon1.center[1]+Coaxmon2.center[1])/2))

    def generate_round_resonator(self, frequency, initial_point,
                               core, gap, ground,
                               open_end_length,open_end, coupler_length,
                               l1, l2, l3,l4,l5, h_end, corner_type='round',angle=0,
                               bridge_params=None):
        resonator = gdf.RoundResonator(frequency, initial_point,
                               core, gap, ground,
                               open_end_length,open_end, coupler_length,
                               l1, l2, l3,l4,l5, h_end,corner_type, self.total_layer, self.restricted_area_layer)
        self.resonators.append(resonator)
        line,line1,open_end, points = resonator.generate_resonator()
        if bridge_params is not None:
            distance, offset, width, length, padsize, line_type = bridge_params
            for num_line in range(len(points) - 1):
                start, finish = points[num_line], points[num_line + 1]
                if finish[0] - start[0] == 0:
                    line_angle = np.pi / 2
                else:
                    line_angle = np.arctan((finish[1] - start[1]) / (finish[0] - start[0]))
                if finish[1] - start[1] < 0 or finish[1] - start[1] == 0 and finish[0] - start[0] < 0:
                    line_angle += np.pi
                line_length = np.sqrt((finish[0] - start[0]) ** 2 + (finish[1] - start[1]) ** 2)
                total_bridges = int((line_length - 2 * offset) / distance)
                offset = (line_length - total_bridges * float(distance)) / 2
                for num_bridge in range(int((line_length - 2 * offset) / distance) + 1):
                    bridge_center = (start[0] + np.cos(line_angle) * (offset + num_bridge * distance),
                                     start[1] + np.sin(line_angle) * (offset + num_bridge * distance))
                    self.generate_bridge(bridge_center, width, length, padsize, line_angle + np.pi / 2,
                                         line_type=line_type)
        # self.total_cell.add(line)
        # self.total_cell.add(end.rotate(angle))
        # self.restricted_area_cell.add(end.rotate(angle))
        # self.cell_to_remove.add(end)


        for obj in [line, line1, open_end]:
            self.total_cell.add(obj[0].rotate(angle,initial_point))
            self.restricted_area_cell.add(obj[1].rotate(angle,initial_point))
            self.cell_to_remove.add(obj[2].rotate(angle,initial_point))
        # self.total_cell.add(line[1])
        # self.cell_to_remove.add(line[2])
        # res_end = resonator_end
        # res_begin = resonator_begin
        # return res_begin, res_end
    def generate_array_of_bridges(self,points,bridge_params):
        if bridge_params is not None:
            distance, offset, width, length, padsize, line_type = bridge_params
            for num_line in range(len(points) - 1):
                start, finish = points[num_line], points[num_line + 1]
                if finish[0] - start[0] == 0:
                    line_angle = np.pi / 2
                else:
                    line_angle = np.arctan((finish[1] - start[1]) / (finish[0] - start[0]))
                if finish[1] - start[1] < 0 or finish[1] - start[1] == 0 and finish[0] - start[0] < 0:
                    line_angle += np.pi
                line_length = np.sqrt((finish[0] - start[0]) ** 2 + (finish[1] - start[1]) ** 2)
                total_bridges = int((line_length - 2 * offset) / distance)
                offset = (line_length - total_bridges * float(distance)) / 2
                for num_bridge in range(int((line_length - 2 * offset) / distance) + 1):
                    bridge_center = (start[0] + np.cos(line_angle) * (offset + num_bridge * distance),
                                     start[1] + np.sin(line_angle) * (offset + num_bridge * distance))
                    self.generate_bridge(bridge_center, width, length, padsize,line_angle+np.pi/2,
                    line_type = line_type)
        return True
def calculate_total_length(points):
    i0, j0 = points[0]
    length = 0
    for i, j in points[1:]:
        length += np.sqrt((i - i0) ** 2 + (j - j0) ** 2)
    return length
