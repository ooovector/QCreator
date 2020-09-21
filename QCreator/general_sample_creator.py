import numpy as np
import gdspy
#import libraries.general_design_functions as gdf
from . import elements
from . import transmission_line_simulator as tlsim
from typing import NamedTuple, SupportsFloat, Any, Iterable, Tuple

Bridges_over_line_param = NamedTuple('Bridge_params',
                                     [('distance', SupportsFloat),
                                      ('offset', SupportsFloat),
                                      ('width', SupportsFloat),
                                      ('length', SupportsFloat),
                                      ('padsize', SupportsFloat),
                                      ('line_type', Any)])


class Sample:

    def __init__(self, name, configurations):
        self.layer_configuration = elements.LayerConfiguration(**configurations)
        self.chip_geometry = elements.ChipGeometry(**configurations)
        self.name = str(name)
        #self.total_cell = gdspy.Cell(self.name)
        #self.restricted_cell = gdspy.Cell(self.name + ' restricted')
        #self.
        #self.label_cell = gdspy.Cell(self.name + ' labels')
        #self.cell_to_remove = gdspy.Cell(self.name + ' remove')

        self.objects = []

        #self.pads = []
        self.qubits = []
        self.lines = []
        self.bridges = []
        self.couplers = []
        self.resonators = []
        self.purcells = []

        self.pads = elements.Pads(self.objects)
        self.connections = []

    @staticmethod
    def default_cpw_radius(w, s, g):
        return 2*(w+2*s+2*g)

    def add(self, object_):
        self.objects.append(object_)

    def draw_design(self):
        lib = gdspy.GdsLibrary()
        total_cell = lib.new_cell(self.name)
        restricted_cell = lib.new_cell(self.name + ' restricted')

        for object_ in self.objects:
            result = object_.get()
            if 'positive' in result:
                total_cell.add(result['positive'])
            if 'grid_x' in result:
                total_cell.add(result['grid_x'])
            if 'grid_y' in result:
                total_cell.add(result['grid_y'])
            if 'airbridges_pad_layer' in result:
                total_cell.add(result['airbridges_pad_layer'])
            if 'airbridges_layer' in result:
                total_cell.add(result['airbridges_layer'])
            if 'restrict' in result:
                restricted_cell.add(result['restrict'])

        return lib

    def ground(self, element: elements.DesignElement, port: str):
        self.connections.append(((element, port), ('gnd', 'gnd')))

    def fanout(self, o: elements.DesignElement, port: str, name: str, grouping: Tuple[int, int],
                down_s_right: float = None, center_s_left: float = None,
                 center_s_right: float = None, up_s_left: float = None):
        fanout = elements.RectFanout(name, o.get_terminals()[port], grouping, self.layer_configuration,
                                     down_s_right=down_s_right, center_s_left=center_s_left,
                                     center_s_right=center_s_right, up_s_left=up_s_left)
        self.add(fanout)
        for conductor_id in range(len(fanout.w)):
            self.connections.append(((o, port, conductor_id), (fanout, 'wide', conductor_id)))

        return fanout

    def connect_cpw(self, o1: elements.DesignElement, o2: elements.DesignElement, port1: str, port2: str, name: str,
                    points: list):
        """
        Create and return a CPW connecting two cpw-type ports, with point inbetween defined by point
        :param o1: first object
        :param o2: second object
        :param port1: first object's port name
        :param port2: second object's port name
        :param name: CPW name
        :param points: coordinates of the CPW's edges
        :return: TLCoupler object
        """
        if o1 not in self.objects:
            raise ValueError('Object o1 not in sample')
        if o2 not in self.objects:
            raise ValueError('Object o2 not in sample')
        t1 = o1.get_terminals()[port1]
        t2 = o2.get_terminals()[port2]
        assert t1.type == 'cpw'
        assert t2.type == 'cpw'

        connections_flat = [i for j in self.connections for i in j]
        assert (o1, port1) not in connections_flat
        assert (o2, port2) not in connections_flat

        w = t1.w
        s = t1.s
        g = t1.g

        assert w == t2.w
        assert s == t2.s
        assert g == t2.g

        orientation1 = t1.orientation + np.pi
        if orientation1 > np.pi:
            orientation1 -= 2*np.pi
        orientation2 = t2.orientation + np.pi
        if orientation2 > np.pi:
            orientation2 -= 2*np.pi

        points = [t1.position]+points+[t2.position]

        cpw = elements.CPW(name, points, w, s, g, self.layer_configuration, r=self.default_cpw_radius(w, s, g),
                           corner_type='round', orientation1=orientation1, orientation2=orientation2)
        self.add(cpw)
        self.connections.extend([((cpw, 'port1', 0), (o1, port1, 0)), ((cpw, 'port2', 0), (o2, port2, 0))])

        return cpw

    # TODO: Nice function for bridges over cpw, need to update
    def connect_bridged_cpw(self, name, points, core, gap, ground, nodes=None, end=None, R=40, corner_type='round', bridge_params=None):
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
                line_angle = np.arctan2(finish[1] - start[1], finish[0] - start[0])
                line_length = np.sqrt((finish[0] - start[0]) ** 2 + (finish[1] - start[1]) ** 2)
                total_bridges = int((line_length - 2 * offset) / distance) #TODO: rounding rules
                offset = (line_length - total_bridges * float(distance)) / 2
                for num_bridge in range(int((line_length - 2 * offset) / distance) + 1):
                    bridge_center = (start[0] + np.cos(line_angle) * (offset + num_bridge * distance),
                                     start[1] + np.sin(line_angle) * (offset + num_bridge * distance))
                    self.generate_bridge('noname', bridge_center, width, length, padsize, line_angle + np.pi / 2,
                                         line_type=line_type)

        self.lines.append(elements.Feedline(name, points, core, gap, ground, nodes, self.total_layer,
                                            self.restricted_area_layer, R))
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

    def get_tls(self):
        """
        Create a transmission line system of the design
        :return:
        """
        tls = tlsim.TLSystem()

        connections_flat = {}
        max_connection_id = 0 # g connection
        for connection in self.connections:
            max_connection_id += 1
            for terminal in connection:
                connections_flat[terminal] = max_connection_id

        element_assignments = {}

        for object_ in self.objects:
            terminal_node_assignments = {}
            for terminal_name, terminal in  object_.get_terminals().items():
                if hasattr(terminal.w, '__iter__'):
                    num_conductors = len(terminal.w)
                else:
                    num_conductors = 1

                for conductor_id in range(num_conductors):
                    if num_conductors == 1:
                        terminal_identifier = terminal_name
                    else:
                        terminal_identifier = (terminal_name, conductor_id)

                    if (object_, terminal_name, conductor_id) in connections_flat:
                        terminal_node_assignments[terminal_identifier] = connections_flat[(object_, terminal_name, conductor_id)]
                    else:
                        max_connection_id += 1
                        connections_flat[(object_, terminal_name, conductor_id)] = max_connection_id
                        terminal_node_assignments[terminal_identifier] = max_connection_id

            element_assignments[object_.name] = object_.add_to_tls(tls, terminal_node_assignments)
        return tls, connections_flat, element_assignments

    def get_s21(self, p1: str, p2: str, frequencies: Iterable[float]):
        """
        Use transmission line model to simulate S21(f) dependence
        :param p1: port1 name
        :param p2: port2 name
        :param frequencies: frequencies
        :return:
        """
        sys, connections, elements_ = self.get_tls()
        s = []
        for f_id, f in enumerate(frequencies):
            eq_vi, eq_dof = sys.get_element_dynamic_equations(elements_[p1][0])
            v2, i2, a2 = sys.get_element_dofs(elements_[p2][0])
            m = sys.create_boundary_problem_matrix(f * np.pi * 2)
            boundary = np.zeros(len(sys.dof_mapping))
            boundary[eq_vi] = 1
            s.append(np.linalg.lstsq(m, boundary)[0][a2[0]])

        return np.asarray(s)

    '''
    Deprecated stuff?
    # General methods for all qubit classes
    def numerate(self, name, number, coordinate):
        size = 70
        layer = 51
        vtext = gdspy.Text(name + str(number), size, coordinate, horizontal=True, layer=layer)
        # label = gdspy.Label(name + str(number), coordinate, texttype=25)
        self.label_cell.add(vtext)
    '''

    #TODO: the reason of the existence of this function is to connect two cpws with different w,s,g.
    # we don't really need it.
    '''
    def generate_narrowing_part(self, name, firstline, secondline):
        narrowing_length = 15
        narrowing1 = gdf.Narrowing(name, firstline.end[0], firstline.end[1],
                                   firstline.w, firstline.s, firstline.g,
                                   secondline[0], secondline[1], secondline[2],
                                   narrowing_length, firstline.angle + np.pi)
        line1 = narrowing1.generate_narrowing()
        self.total_cell.add(line1[0])
        self.restricted_area_cell.add(gdspy.boolean(line1[1], line1[1], 'or', layer=self.restricted_area_layer))
        self.cell_to_remove.add(line1[2])
        return (firstline.end[0] + narrowing_length * np.cos(firstline.angle),
                firstline.end[1] + narrowing_length * np.sin(firstline.angle))
    '''

    #TODO: what does this thing do? CPW-over-CPW element? Maybe we need an extra element for this
    # need to think of a an automatic way of determining the crossing position of two lines with each other
    def generate_bridge_over_feedline(self, name, firstline, airbridge, secondline, distance_between_airbridges):
        narrowing_length = 15
        narrowing1 = elements.Narrowing(firstline.end[0], firstline.end[1],
                                        firstline.core, firstline.gap, firstline.ground,
                                        airbridge[2], distance_between_airbridges, airbridge[2],
                                        narrowing_length, np.pi + firstline.angle)
        narrowing2 = elements.Narrowing(name,
                                        firstline.end[0] + np.cos(firstline.angle) * (narrowing_length + airbridge[0] * 2 + airbridge[1]),
                                        firstline.end[1] + np.sin(firstline.angle) * (narrowing_length + airbridge[0] * 2 + airbridge[1]),
                                        airbridge[2], distance_between_airbridges, airbridge[2],
                                        secondline[0], secondline[1], secondline[2],
                                        narrowing_length, np.pi + firstline.angle)
        self.generate_bridge(name, (firstline.end[0] + np.cos(firstline.angle) * narrowing_length -
                              np.sin(firstline.angle) * (distance_between_airbridges + airbridge[2]),
                              firstline.end[1] + np.cos(firstline.angle) * (
                                          distance_between_airbridges + airbridge[2]) +
                              np.sin(firstline.angle) * narrowing_length),
                             airbridge[0], airbridge[1], airbridge[2], firstline.angle, 'line')
        self.generate_bridge(name, (firstline.end[0] + np.cos(firstline.angle) * narrowing_length,
                              np.sin(firstline.angle) * narrowing_length + firstline.end[1]),
                             airbridge[0], airbridge[1], airbridge[2], firstline.angle, 'line')
        self.generate_bridge(name, (firstline.end[0] + np.cos(firstline.angle) * narrowing_length +
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

    """
    We are trying to get rid of the use cases of this function
    def finish_him(self):
        # self.result.add(gdspy.boolean(self.total_cell.get_polygons(by_spec=True)[(self.total_layer, 0)],
        #                               self.total_cell.get_polygons(by_spec=True)[(self.total_layer, 0)], 'or',
        #                               layer=self.total_layer))
        self.result.add(gdspy.boolean(self.total_cell.get_polygons(by_spec=True)[(self.total_layer, 0)],
                                      self.cell_to_remove.get_polygons(by_spec=True)[(2, 0)], 'not',
                                      layer=self.total_layer))
        self.result.add(gdspy.boolean(self.total_cell.get_polygons(by_spec=True)[(self.airbridges_layer, 0)],
                                      self.total_cell.get_polygons(by_spec=True)[(self.airbridges_layer, 0)], 'or',
                                      layer=self.airbridges_layer))
        self.result.add(gdspy.boolean(self.total_cell.get_polygons(by_spec=True)[(self.airbridges_pad_layer, 0)],
                                      self.total_cell.get_polygons(by_spec=True)[(self.airbridges_pad_layer, 0)], 'or',
                                      layer=self.airbridges_pad_layer))
        #
        # костыль
    """

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

'''
Beginning from here I have no idea what to this.



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
                                       self.restricted_area_layer, self.jj_layer, Couplers, JJ))
        qubit_total, restricted_area, JJ_total = self.qubits[-1].generate_qubit()
        self.total_cell.add(qubit_total.rotate(angle, args[0]))
        self.total_cell.add(JJ_total)  # .rotate(angle,(center_point.x,center_point.y))
        self.restricted_area_cell.add(restricted_area)
        self.numerate("Qb", len(self.qubits) - 1, args[0])

    def add_qubit_coupler(self, w, s, g, Coaxmon1, Coaxmon2, JJ, squid):
        coupler = gdf.IlyaCoupler(w, s, g, Coaxmon1, Coaxmon2, JJ, squid,
                                  self.total_layer, self.restricted_area_layer, self.jj_layer, self.layer_to_remove)
        self.couplers.append(coupler)
        line, JJ = coupler.generate_coupler()
        self.total_cell.add([line[0], JJ[0], JJ[1]])
        #         self.total_cell.add(line[1])
        self.restricted_area_cell.add(line[1])
        self.cell_to_remove.add([line[2], JJ[2]])
        self.numerate("IlCoup", len(self.couplers) - 1, ((Coaxmon1.center[0]+Coaxmon2.center[0])/2, (Coaxmon1.center[1]+Coaxmon2.center[1])/2))

    def generate_round_resonator(self, frequency, initial_point,
                               w, s, g,
                               open_end_length,open_end, coupler_length,
                               l1, l2, l3,l4,l5, h_end, corner_type='round',angle=0,
                               bridge_params=None):
        resonator = gdf.RoundResonator(frequency, initial_point,
                               w, s, g,
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
                    self.generate_bridge('noname', bridge_center, width, length, padsize, line_angle + np.pi / 2,
                    line_type = line_type)
        return True

'''
# TODO: might be useflu for elements/cpw.py to caluclate the line of the cpw line
def calculate_total_length(points):
    i0, j0 = points[0]
    length = 0
    for i, j in points[1:]:
        length += np.sqrt((i - i0) ** 2 + (j - j0) ** 2)
    return length
