import numpy as np
import gdspy
import os
from . import elements
from . import transmission_line_simulator as tlsim
from typing import NamedTuple, SupportsFloat, Any, Iterable, Tuple, List

from . import  meshing
from copy import deepcopy

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
        self.total_cell = gdspy.Cell(self.name)
        self.restricted_cell = gdspy.Cell(self.name + ' restricted')
        #for several additional features
        self.qubits_cells = []
        self.qubit_cap_cells = []
        self.caps_list = []

        self.objects = []

        # self.pads = []
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
        return 2 * (w + 2 * s + 2 * g)

    def add(self, object_):
        self.objects.append(object_)

    def draw_design(self):
        for object_ in self.objects:
            result = object_.get()
            if 'test' in result:
                self.total_cell.add(result['test'])
            if 'JJ' in result:
                self.total_cell.add(result['JJ'])
            if 'positive' in result:
                self.total_cell.add(result['positive'])
            if 'grid_x' in result:
                self.total_cell.add(result['grid_x'])
            if 'grid_y' in result:
                self.total_cell.add(result['grid_y'])
            if 'restricted' in result:
                self.restricted_cell.add(result['restricted'])

        self.fill_object_arrays()
    def draw_cap(self): # TODO: maybe we need to come up with a better way, but for this moment it's fine
        """
        This function creates new cells with specified qubits
        1) cells for full qubits
        2) cells for fastcap
        """
        qubit_cap_cell_counter = 0
        qubits_cell_counter = 0
        for object_ in self.objects:
            result = object_.get()
            if 'qubit_cap' in result:
                if result['qubit_cap'] is not None:
                    cap_cell = gdspy.Cell('qubit capacitance cell ' + str(qubit_cap_cell_counter))
                    cap_cell.add(result['qubit_cap'])
                    self.qubit_cap_cells.append(cap_cell)
                    qubit_cap_cell_counter = qubit_cap_cell_counter + 1
            if 'qubit' in result:
                if result['qubit'] is not None:
                    qubit_cell = gdspy.Cell('qubit cell ' + str(qubits_cell_counter))
                    qubit_cell.add(result['qubit'])
                    if 'JJ' in result:
                        qubit_cell.add(result['JJ'])
                    self.qubits_cells.append(qubit_cell)
                    qubits_cell_counter = qubits_cell_counter + 1

    def fill_object_arrays(self):
        self.qubits = [i for i in self.objects if i.type == 'qubit']
        self.couplers = [i for i in self.objects if i.type == 'qubit coupler']
    # def ground(self, element: elements.DesignElement, port: str):
    #     self.connections.append(((element, port), ('gnd', 'gnd')))

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

    def ground(self, o: elements.DesignElement, port: str, name: str, grounding_width: float,
               grounding_between: List[Tuple[int, int]]):
        if port == 'port1':
            reverse_type = 'Negative'
        else:
            reverse_type = 'Positive'

        closed_end = elements.RectGrounding(name, o.get_terminals()[port], grounding_width, grounding_between,
                                            self.layer_configuration, reverse_type)
        self.add(closed_end)

        conductor_in_narrow = 0

        for conductor_id in range(closed_end.initial_number_of_conductors):
            self.connections.append(((o, port, conductor_id), (closed_end, 'wide', conductor_id)))

        # if closed_end.final_number_of_conductors:
        #     for conductor_id in closed_end.free_core_conductors:
        #         self.connections.append(((closed_end, 'wide', conductor_id), (closed_end, 'narrow', conductor_in_narrow)))
        #         conductor_in_narrow += 1

        return closed_end

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
            orientation1 -= 2 * np.pi
        orientation2 = t2.orientation + np.pi
        if orientation2 > np.pi:
            orientation2 -= 2 * np.pi

        points = [tuple(t1.position)] + points + [tuple(t2.position)]
        cpw = elements.CPW(name, points, w, s, g, self.layer_configuration, r=self.default_cpw_radius(w, s, g),
                           corner_type='round', orientation1=orientation1, orientation2=orientation2)
        self.add(cpw)
        self.connections.extend([((cpw, 'port1', 0), (o1, port1, 0)), ((cpw, 'port2', 0), (o2, port2, 0))])

        return cpw

    def watch(self):
        gdspy.LayoutViewer(depth=0, pattern={'default': 8}, background='#FFFFFF')  # this opens a viewer

    def cpw_shift(self, element, port_name, length):
        return [(element.get_terminals()[port_name].position[0] + \
                 length * np.cos(element.get_terminals()[port_name].orientation),
                 element.get_terminals()[port_name].position[1] + \
                 length * np.sin(element.get_terminals()[port_name].orientation)),

                ]

    # functions to work and calculate capacitance
    def write_to_gds(self, name=None):
        if name is not None:
            gdspy.write_gds(name + '.gds', cells=None, name='library', unit=1e-06, precision=1e-09, timestamp=None,
                            binary_cells=None)
            self.path = os.getcwd() + '\\' + name + '.gds'
        else:
            gdspy.write_gds(self.name + '.gds', cells=None, name='library', unit=1e-06, precision=1e-09,
                            timestamp=None,
                            binary_cells=None)
            self.path = os.getcwd() + '\\' + self.name + '.gds'
        print("Gds file has been writen here: ", self.path)

    def calculate_qubit_capacitance(self, cell,qubit, mesh_volume, name=None):
        self.write_to_gds(name)
        mesh = meshing.Meshing(path=self.path,
                               cell_name=cell.name,
                               layers=list(cell.get_layers()))
        mesh.read_data_from_gds_file()
        mesh.prepare_for_meshing()
        mesh.run_meshing(mesh_volume=mesh_volume)
        mesh.write_into_file(os.getcwd() + '\\' + 'mesh_4k_data')
        mesh.run_fastcap(os.getcwd() + '\\' + 'mesh_4k_results')
        print("Capacitance results have been writen here: ", os.getcwd() + '\\' + 'mesh_4k_results')
        caps = np.round(mesh.get_capacitances(),1)
        self.fill_cap_matrix(qubit,caps) # TODO: can we improve this way?
        self.caps_list.append(caps)
        return caps
    def fill_cap_matrix(self, qubit, caps):
        qubit.C['qubit']=caps[1][1]
        i=2
        for id, coupler in enumerate(qubit.couplers):
            if coupler.coupler_type == 'coupler':
                qubit.C['coupler'+str(id)] = (caps[i][i], -caps[1][i])
                i=i+1
        return True
    # TODO: Nice function for bridges over cpw, need to update
    def connect_bridged_cpw(self, name, points, core, gap, ground, nodes=None, end=None, R=40, corner_type='round',
                            bridge_params=None):
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
                total_bridges = int((line_length - 2 * offset) / distance)  # TODO: rounding rules
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
        max_connection_id = 0  # g connection
        for connection in self.connections:
            max_connection_id += 1
            for terminal in connection:
                connections_flat[terminal] = max_connection_id

        element_assignments = {}

        for object_ in self.objects:
            terminal_node_assignments = {}
            for terminal_name, terminal in object_.get_terminals().items():
                num_conductors = 1
                if hasattr(terminal,'w'):
                    if hasattr(terminal.w, '__iter__'):
                        num_conductors = len(terminal.w)

                for conductor_id in range(num_conductors):
                    if num_conductors == 1:
                        terminal_identifier = terminal_name
                    else:
                        terminal_identifier = (terminal_name, conductor_id)

                    if (object_, terminal_name, conductor_id) in connections_flat:
                        terminal_node_assignments[terminal_identifier] = connections_flat[
                            (object_, terminal_name, conductor_id)]
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

    # TODO: the reason of the existence of this function is to connect two cpws with different w,s,g.
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

    # TODO: what does this thing do? CPW-over-CPW element? Maybe we need an extra element for this
    # need to think of a an automatic way of determining the crossing position of two lines with each other
    def generate_bridge_over_feedline(self, name, firstline, airbridge, secondline, distance_between_airbridges):
        narrowing_length = 15
        narrowing1 = elements.Narrowing(firstline.end[0], firstline.end[1],
                                        firstline.core, firstline.gap, firstline.ground,
                                        airbridge[2], distance_between_airbridges, airbridge[2],
                                        narrowing_length, np.pi + firstline.angle)
        narrowing2 = elements.Narrowing(name,
                                        firstline.end[0] + np.cos(firstline.angle) * (
                                                narrowing_length + airbridge[0] * 2 + airbridge[1]),
                                        firstline.end[1] + np.sin(firstline.angle) * (
                                                narrowing_length + airbridge[0] * 2 + airbridge[1]),
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

    def connect_meander(self, name: str, o1: elements.DesignElement, port1: str, meander_length: float,
                        restricted_scale: float, constant_scale: float = 0., o2: elements.DesignElement = None,
                        port2: str = None, radius: float = 0., connector: float = 10.):
        if not o2:
            t1 = o1.get_terminals()[port1]
            w = t1.w
            s = t1.s
            g = t1.g
            # w1 = t1.w
            # s1 = t1.s
            # g1 = t1.g
            # if type(w1) == list or type(s1) == list:
            #     if len(w1) == 1 and len(s1) == 2:
            #         w = w1[0]
            #         s = s1[0]
            #         g = g1
            #     else:
            #         raise ValueError('Unexpected size of CPW')
            # else:
            #     w = w1[0]
            #     s = s1[0]
            #     g = g1
            delta = g + s + w / 2
            connector_length = connector + 2 * delta
            angle = t1.orientation + np.pi
            initial_point = [(t1.position[0], t1.position[1])]
            meander = elements.CPWMeander(initial_point=initial_point, w=w, s=s, g=g, meander_length=meander_length,
                                          restricted_scale=restricted_scale, constant_scale=constant_scale,
                                          orientation=angle, connector_length=connector_length)
            points_for_creation = meander.points
            rendering_meander = elements.CPW(name=name, points=points_for_creation, w=w, s=s, g=g,
                                             layer_configuration=self.layer_configuration, r=radius)
            self.add(rendering_meander)
            self.connections.extend([((rendering_meander, 'port1', 0), (o1, port1, 0))])
        else:
            t1 = o1.get_terminals()[port1]
            t2 = o2.get_terminals()[port2]
            w1, s1, g1 = t1.w, t1.s, t1.g
            w2, s2, g2 = t2.w, t2.s, t2.g
            if w1 == w2 and s1 == s2 and g1 == g2:
                w, s, g = w1, s1, g1
                delta = g + s + w / 2
                connector_length = connector + 4 * delta
                distance = np.sqrt((t1.position[0] - t2.position[0]) ** 2 + (t1.position[1] - t2.position[1]) ** 2)

                angle = np.arctan((t1.position[1] - t2.position[1]) / (t1.position[0] - t2.position[0]))

                initial_point = [(t1.position[0], t1.position[1])]
                final_point = [(t2.position[0], t2.position[1])]

                meander = elements.CPWMeander(initial_point=initial_point, w=w, s=s, g=g, meander_length=meander_length,
                                              restricted_scale=restricted_scale, constant_scale=distance,
                                              orientation=angle, connector_length=connector_length)
                points_for_creation = meander.points
                # TODO: create a meander connection with different angles
                # TODO: create a meander connection using connector_cpw
                # points_for_creation = meander.points[1:len(meander.points) - 1]

                # angle1 = np.arctan(
                #     (t1.position[1] - points_for_creation[0][1]) / (t1.position[0] - points_for_creation[0][0]))
                # angle2 = np.arctan(
                #     (points_for_creation[-1][1] - t2.position[1]) / (points_for_creation[-1][0] - t2.position[0]))
                #
                # points_for_creation.insert(0, (t1.position[0] + connector_length * np.cos(angle1),
                #                                t1.position[1] + connector_length * np.sin(angle1)))
                #
                # points_for_creation.insert(len(points_for_creation),
                #                            (points_for_creation[-1][0] + connector_length * np.cos(angle2),
                #                             points_for_creation[-1][1] + connector_length * np.sin(angle2)))

                # points_for_creation.insert(0, (t1.position[0], t1.position[1]))
                #
                # points_for_creation.insert(len(points_for_creation), (t2.position[0], t2.position[1]))

                rendering_meander = elements.CPW(name=name, points=points_for_creation, w=w, s=s, g=g,
                                                 layer_configuration=self.layer_configuration, r=radius)
                self.add(rendering_meander)
                self.connections.extend([((rendering_meander, 'port1', 0), (o1, port1, 0)),
                                         ((rendering_meander, 'port2', 0), (o2, port2, 0))])
            else:
                raise ValueError('CPW parameters are not equal!')

        return rendering_meander


# TODO: might be useflu for elements/cpw.py to caluclate the line of the cpw line
def calculate_total_length(points):
    i0, j0 = points[0]
    length = 0
    for i, j in points[1:]:
        length += np.sqrt((i - i0) ** 2 + (j - j0) ** 2)
    return length
