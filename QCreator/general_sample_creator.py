import numpy as np
import gdspy
import os
from . import elements
from . import transmission_line_simulator as tlsim
from typing import NamedTuple, SupportsFloat, Any, Iterable, Tuple, List

from . import meshing
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

        self.lib = gdspy.GdsLibrary(unit=1e-06, precision=1e-09)

        self.total_cell = self.lib.new_cell(self.name, overwrite_duplicate=True, update_references=True)
        self.restricted_cell = self.lib.new_cell(self.name + ' restricted', overwrite_duplicate=True,
                                                 update_references=True)
        # Geometry must be placed in cells.

        # for several additional features
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
            if 'restrict' in result:
                self.restricted_cell.add(result['restrict'])
            if 'airbridges_pads' in result:
                self.total_cell.add(result['airbridges_pads'])
            if 'airbridges' in result:
                self.total_cell.add(result['airbridges'])

        self.fill_object_arrays()

    def draw_cap(self):  # TODO: maybe we need to come up with a better way, but for this moment it's fine
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
                    cap_cell = self.lib.new_cell('qubit capacitance cell ' + str(qubit_cap_cell_counter),
                                                 overwrite_duplicate=True, update_references=True)
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

    def connect(self, o1, p1, o2, p2):
        reverse = o1.get_terminals()[p1].order == o2.get_terminals()[p2].order
        try:
            for conductor_id in range(len(o1.get_terminals()[p1].w)):
                if reverse:
                    conductor_second = len(o1.get_terminals()[p1].w) - 1 - conductor_id
                else:
                    conductor_second = conductor_id
                self.connections.append(((o1, p1, conductor_id), (o2, p2, conductor_second)))
        except TypeError:
            self.connections.append(((o1, p1, 0), (o2, p2, 0)))

    def fanout(self, o: elements.DesignElement, port: str, name: str, grouping: Tuple[int, int],
               down_s_right: float = None, center_s_left: float = None,
               center_s_right: float = None, up_s_left: float = None):

        fanout = elements.RectFanout(name, o.get_terminals()[port], grouping, self.layer_configuration,
                                     down_s_right=down_s_right, center_s_left=center_s_left,
                                     center_s_right=center_s_right, up_s_left=up_s_left)
        self.add(fanout)
        self.connect(o, port, fanout, 'wide')
        return fanout

    def ground(self, o: elements.DesignElement, port: str, name: str, grounding_width: float,
               grounding_between: List[Tuple[int, int]]):
        t = o.get_terminals()[port]

        if type(t.w) and type(t.s) == list:
            w_ = t.w
            s_ = t.s

        elif type(t.w) == float or type(t.w) == int:
            w_ = [t.w]
            s_ = [t.s, t.s]
        else:
            raise ValueError('Unexpected types of CPW parameters')
        g_ = t.g

        # if port == 'port1':
        #     reverse_type = 'Negative'
        # else:
        #     reverse_type = 'Positive'

        position_ = t.position
        orientation_ = t.orientation

        closed_end = elements.RectGrounding(name, position_, orientation_, w_, s_, g_, grounding_width,
                                            grounding_between,
                                            self.layer_configuration)
        self.add(closed_end)

        conductor_in_narrow = 0

        # for conductor_id in range(closed_end.initial_number_of_conductors):
        #    self.connections.append(((o, port, conductor_id), (closed_end, 'wide', conductor_id)))
        self.connect(o, port, closed_end, 'wide')

        return closed_end

    def open_end(self, o: elements.DesignElement, port: str, name: str):
        position_ = o.get_terminals()[port].position
        orientation_ = o.get_terminals()[port].orientation

        if type(o.get_terminals()[port].w) and type(o.get_terminals()[port].s) == list:
            w_ = o.get_terminals()[port].w
            s_ = o.get_terminals()[port].s
            g_ = o.get_terminals()[port].g

        elif type(o.get_terminals()[port].w) == float or type(o.get_terminals()[port].w) == int:
            w_ = [o.get_terminals()[port].w]
            s_ = [o.get_terminals()[port].s, o.get_terminals()[port].s]
            g_ = o.get_terminals()[port].g

        else:
            raise ValueError('Unexpected data types of CPW parameters')

        open_end = elements.OpenEnd(name, position_, w_, s_, g_, orientation_, self.layer_configuration)
        # open_end = elements.OpenEnd(name, o.get_terminals()[port], self.layer_configuration)
        self.add(open_end)

        self.connect(o, port, open_end, 'wide')
        return open_end

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
        # self.connections.extend([((cpw, 'port1', 0), (o1, port1, 0)), ((cpw, 'port2', 0), (o2, port2, 0))])
        self.connect(cpw, 'port1', o1, port1)
        self.connect(cpw, 'port2', o2, port2)
        return cpw

    def watch(self):
        gdspy.LayoutViewer(depth=0, pattern={'default': 8}, background='#FFFFFF')  # this opens a viewer

    def cpw_shift(self, element, port_name, length):
        return [(element.get_terminals()[port_name].position[0] + \
                 length * np.cos(element.get_terminals()[port_name].orientation+np.pi),
                 element.get_terminals()[port_name].position[1] + \
                 length * np.sin(element.get_terminals()[port_name].orientation+np.pi)),]

    # functions to work and calculate capacitance
    def write_to_gds(self, name=None):
        if name is not None:
            self.lib.write_gds(name + '.gds', cells=None, timestamp=None,
                               binary_cells=None)
            self.path = os.getcwd() + '\\' + name + '.gds'
        else:
            self.lib.write_gds(self.name + '.gds', cells=None,
                               timestamp=None,
                               binary_cells=None)
            self.path = os.getcwd() + '\\' + self.name + '.gds'
        print("Gds file has been writen here: ", self.path)

    def calculate_qubit_capacitance(self, cell, qubit, mesh_volume, name=None):
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
        caps = np.round(mesh.get_capacitances(), 1)
        self.fill_cap_matrix_grounded(qubit, caps)  # TODO: can we improve this way?
        self.caps_list.append(caps)
        return caps

    def fill_cap_matrix(self, qubit, caps):
        qubit.C['qubit'] = caps[1][1]
        i = 2
        for id, coupler in enumerate(qubit.couplers):
            if coupler.coupler_type == 'coupler':
                qubit.C['coupler' + str(id)] = (caps[i][i], -caps[1][i])
                i = i + 1
        return True
    def fill_cap_matrix_grounded(self, qubit, caps):
        qubit.C['qubit'] = caps[1][1]
        print(caps)
        i = 2
        # print(qubit.C)
        for key,value in qubit.terminals.items():
            if value is not None and key is not 'flux':
                # print(key, value)
                qubit.C[key] = (caps[i][i], -caps[1][i])
                i = i + 1
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
        GND = tlsim.Short()
        tls.add_element(GND, [0])

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
                if hasattr(terminal, 'w'):
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

    def get_topology(self):
        sys, connections_, elements_ = self.get_tls()
        circuit_elements = sys.elements
        number_of_elem = len(circuit_elements )
        circuit_nodes = sys.terminal_node_mapping

        topology = []
        for elem in range(number_of_elem):
            topology.append([circuit_elements[elem], circuit_nodes[elem]])

        return topology




    def generate_bridge_over_cpw(self, name: str, o: elements.DesignElement, pads_geometry: Tuple[float, float],
                                 bridge_geometry: Tuple[float, float], distance_between_pads: float,
                                 min_spacing: float):
        """
        This method add air bridges on CPW line.
        :param name:
        :param o:
        :param pads_geometry:
        :param bridge_geometry:
        :param distance_between_pads:
        :param min_spacing: distance between air bridges
        """

        t = o.get_terminals()['port1']
        assert t.type == 'cpw'

        w = t.w
        s = t.s
        g = t.g
        radius = o.r

        segments_ = o.segments
        number_of_segments = len(segments_)

        # firstly we should sort all segments to understand which of them can be render with bridges
        for elem in range(number_of_segments):

            if segments_[elem]['type'] == 'segment':
                distance_between_points = np.sqrt(
                    (segments_[elem]['startpoint'][0] - segments_[elem]['endpoint'][0]) ** 2 + (
                            segments_[elem]['startpoint'][1] - segments_[elem]['endpoint'][1]) ** 2)
                if distance_between_points > pads_geometry[1] + 2 * min_spacing:
                    segments_[elem]['bridge'] = 'yes'
                else:
                    segments_[elem]['bridge'] = 'no'

            elif segments_[elem]['type'] == 'turn':
                segments_[elem]['startpoint'] = segments_[elem - 1]['endpoint']
                segments_[elem]['endpoint'] = segments_[elem + 1]['startpoint']

                segments_[elem]['orientation1'] = np.arctan2(
                    -segments_[elem - 1]['startpoint'][1] + segments_[elem - 1]['endpoint'][1],
                    -segments_[elem - 1]['startpoint'][0] + segments_[elem - 1]['endpoint'][0])
                segments_[elem]['orientation2'] = np.arctan2(
                    segments_[elem + 1]['startpoint'][1] - segments_[elem + 1]['endpoint'][1],
                    segments_[elem + 1]['startpoint'][0] - segments_[elem + 1]['endpoint'][0])

            else:
                segments_[elem]['bridge'] = 'no'

        real_segments = o.segments[1: len(o.segments) + 1]

        total_cpw = []
        all_bridges = []

        for elem in range(len(real_segments)):
            if real_segments[elem]['type'] == 'turn':
                x0, y0, x1, y1 = segment_points(real_segments[elem])
                points_ = [(x0, y0), (x1, y1)]
                line = elements.CPW(name=name + str(len(total_cpw)),
                                    points=points_, w=w, s=s, g=g,
                                    layer_configuration=self.layer_configuration, r=self.default_cpw_radius(w, s, g),
                                    orientation1=real_segments[elem]['orientation1'],
                                    orientation2=real_segments[elem]['orientation2'])
                self.add(line)
                total_cpw.append(line)
                if elem > 0:
                    self.connect(total_cpw[elem - 1], 'port2', total_cpw[elem], 'port1')

            elif real_segments[elem]['type'] == 'segment':
                if real_segments[elem]['bridge'] == 'yes':
                    x0, y0, x1, y1 = segment_points(real_segments[elem])

                    orientation_ = np.arctan2(y1 - y0, x1 - x0)

                    segment_length = np.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)
                    number_of_bridges = int((segment_length - min_spacing) // (pads_geometry[1] + min_spacing))

                    t_list = []  # list of parameters for lines
                    bridges_t_list = []  # list of parameters for bridges

                    for i in range(number_of_bridges):
                        bridges_t_list.append(((i + 1) * (pads_geometry[1] / 2 + min_spacing) + i * pads_geometry[
                            1] / 2) / segment_length)
                        delta1 = ((i + 1) * min_spacing + i * pads_geometry[1]) / segment_length
                        delta2 = ((min_spacing + pads_geometry[1]) * (i + 1)) / segment_length
                        t_list.append(delta1)
                        t_list.append(delta2)

                    subsegments_points = [(x0, y0)]
                    bridges_points = []

                    for t in t_list:
                        x, y = parametric_equation_of_line(x0, y0, x1, y1, t)
                        subsegments_points.append((x, y))

                    for t in bridges_t_list:
                        x, y = parametric_equation_of_line(x0, y0, x1, y1, t)
                        bridges_points.append((x, y))

                    subsegments_points.append((x1, y1))

                    bridges = []
                    sub_cpw = []

                    for bridge_elem in range(number_of_bridges + 1):
                        line = elements.CPW(name=name + str(len(total_cpw)),
                                            points=subsegments_points[2 * bridge_elem:2 * bridge_elem + 2], w=w, s=s,
                                            g=g,
                                            layer_configuration=self.layer_configuration, r=radius)
                        self.add(line)
                        sub_cpw.append(line)
                        total_cpw.append(line)

                    for bridge_elem in range(number_of_bridges):
                        bridge_elem = elements.AirbridgeOverCPW(name=name + str(len(all_bridges)),
                                                                position=bridges_points[bridge_elem],
                                                                orientation=orientation_, w=w, s=s, g=g,
                                                                pads_geometry=pads_geometry,
                                                                bridge_geometry=bridge_geometry,
                                                                layer_configuration=self.layer_configuration,
                                                                distance_between_pads=distance_between_pads)
                        bridges.append(bridge_elem)
                        all_bridges.append(bridge_elem)
                        self.add(bridge_elem)

                    if elem > 0:
                        self.connect(total_cpw[elem - 1], 'port2', sub_cpw[0], 'port1')

                    for i in range(number_of_bridges):
                        self.connect(sub_cpw[i], 'port2', bridges[i], 'port1')
                        self.connect(bridges[i], 'port2', sub_cpw[i + 1], 'port1')

                else:
                    x0, y0, x1, y1 = segment_points(real_segments[elem])
                    points_ = ([x0, y0], [x1, y1])
                    line = elements.CPW(name=name + str(len(total_cpw)), points=points_, w=w, s=s, g=g,
                                        layer_configuration=self.layer_configuration, r=0)
                    self.add(line)
                    total_cpw.append(line)

                    if elem > 0:
                        self.connect(total_cpw[elem - 1], 'port2', total_cpw[elem], 'port1')
            else:
                continue
        return total_cpw, all_bridges

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
                        length_left: float, length_right: float, first_step_orientation: float,
                        meander_orientation: float,
                        end_point=None, end_orientation=None, meander_type='round'):
        # make small indent from the starting point
        bend_radius = 40
        # print('Meander')
        t1 = o1.get_terminals()[port1]
        points = [tuple(t1.position)]
        (w, s, g) = (t1.w, t1.s, t1.g)
        indent_length = 50  # (w+2*s)*2
        points.append((t1.position[0] + np.cos(t1.orientation + np.pi) * indent_length,
                       t1.position[1] + np.sin(t1.orientation + np.pi) * indent_length))
        indent_first = length_left if first_step_orientation == 'left' else length_right
        points.append((points[-1][0] + np.cos(meander_orientation) * indent_first,
                       points[-1][1] + np.sin(meander_orientation) * indent_first))
        if end_point is not None:
            end_point_indent = [(end_point[0] + np.cos(end_orientation + np.pi) * indent_length,
                                 end_point[1] + np.sin(end_orientation + np.pi) * indent_length)]
        else:
            end_point_indent = []
        # check that we have enough distance
        rendering_meander = elements.CPW(name=name, points=deepcopy(points + end_point_indent), w=w, s=s, g=g,
                                         layer_configuration=self.layer_configuration, r=bend_radius,
                                         corner_type=meander_type)
        if rendering_meander.length > meander_length:
            print('error, length is too small')
            return 1
        # lets fill the whole rectangular
        default_bend_diameter = bend_radius * 2 + 5
        if meander_type == 'flush':
            meander_step = length_left + length_right + default_bend_diameter
        elif meander_type == 'round':
            meander_step = length_left - bend_radius + length_right - bend_radius + \
                           (default_bend_diameter - 2 * bend_radius) + np.pi * bend_radius
        # print(meander_length-rendering_meander.length)
        # meander_length+=length_left
        N = int((meander_length - rendering_meander.length) // meander_step)
        # print("N",N)
        # print(meander_step)
        if N == 0:
            print("meander is too small, N=0")
        # subtract one to connect to the end point, here i assume that the distance
        # from the end of the meander is less than the meander step
        if end_orientation is not None:
            N = N - 1
        # make a meander
        i = 1
        while i <= N:
            list = [(points[-1][0] + np.sin(meander_orientation) * default_bend_diameter,
                     points[-1][1] - np.cos(meander_orientation) * default_bend_diameter)]
            points.extend(list)
            list = [(points[-1][0] + np.sin(meander_orientation - np.pi / 2 + np.pi * ((i - 1) % 2)) * (
                    length_left + length_right),
                     points[-1][1] - np.cos(meander_orientation - np.pi / 2 + np.pi * ((i - 1) % 2)) * (
                             length_left + length_right))]
            points.extend(list)

            i = i + 1
        # print(points)
        # print(calculate_total_length(points))
        if end_orientation is not None:
            i = 0
        else:
            rendering_meander = elements.CPW(name=name, points=deepcopy(points), w=w, s=s, g=g,
                                             layer_configuration=self.layer_configuration, r=bend_radius,
                                             corner_type=meander_type)
            # print(rendering_meander.length)
            # print(meander_length)
            tail = np.abs(np.floor(rendering_meander.length - meander_length))
            # print(tail)
            # print((default_bend_diameter - 2 * bend_radius) + np.pi * bend_radius)
            if tail < np.pi * bend_radius / 2:
                list = [(points[-1][0] + np.sin(meander_orientation - np.pi / 2 + np.pi * ((i) % 2)) * tail,
                         points[-1][1] - np.cos(meander_orientation - np.pi / 2 + np.pi * ((i) % 2)) * tail)]
                points.extend(list)
                rendering_meander = elements.CPW(name=name, points=deepcopy(points), w=w, s=s, g=g,
                                                 layer_configuration=self.layer_configuration, r=bend_radius,
                                                 corner_type=meander_type)
            elif tail < (default_bend_diameter - 2 * bend_radius) + np.pi * bend_radius + 1:
                list = [(points[-1][0] + np.sin(meander_orientation) * (bend_radius + 1),
                         points[-1][1] - np.cos(meander_orientation) * (bend_radius + 1))]
                points.extend(list)
                # print(points)
                rendering_meander = elements.CPW(name=name, points=deepcopy(points), w=w, s=s, g=g,
                                                 layer_configuration=self.layer_configuration, r=bend_radius,
                                                 corner_type=meander_type)
                tail = np.abs(np.floor(rendering_meander.length - meander_length))
                # print(tail)
                list = [(points[-1][0] + np.sin(meander_orientation) * tail,
                         points[-1][1] - np.cos(meander_orientation) * tail)]
                points.extend(list)
                rendering_meander = elements.CPW(name=name, points=deepcopy(points), w=w, s=s, g=g,
                                                 layer_configuration=self.layer_configuration, r=bend_radius,
                                                 corner_type=meander_type)
            else:
                # print("I'm here")
                list = [(points[-1][0] + np.sin(meander_orientation) * default_bend_diameter,
                         points[-1][1] - np.cos(meander_orientation) * default_bend_diameter)]
                points.extend(list)
                rendering_meander = elements.CPW(name=name, points=deepcopy(points), w=w, s=s, g=g,
                                                 layer_configuration=self.layer_configuration, r=bend_radius,
                                                 corner_type=meander_type)
                # print(rendering_meander.length)
                error = np.abs(rendering_meander.length - meander_length)
                # print(error)

                list = [(points[-1][0] + np.sin(meander_orientation - np.pi / 2 + np.pi * ((i - 1) % 2)) *
                         error,
                         points[-1][1] - np.cos(meander_orientation - np.pi / 2 + np.pi * ((i - 1) % 2)) *
                         error)]
                points.extend(list)
                rendering_meander = elements.CPW(name=name, points=deepcopy(points), w=w, s=s, g=g,
                                                 layer_configuration=self.layer_configuration, r=bend_radius,
                                                 corner_type=meander_type)
                # print(points)
                # points.pop(-1)
                error = np.abs(rendering_meander.length - meander_length)
                list = [(points[-1][0] + np.sin(meander_orientation - np.pi / 2 + np.pi * ((i - 1) % 2)) *
                         (error),
                         points[-1][1] - np.cos(meander_orientation - np.pi / 2 + np.pi * ((i - 1) % 2)) *
                         (error))]
                points.extend(list)
                rendering_meander = elements.CPW(name=name, points=deepcopy(points), w=w, s=s, g=g,
                                                 layer_configuration=self.layer_configuration, r=bend_radius,
                                                 corner_type=meander_type)
        # print(points)
        # print(rendering_meander.length)
        self.connections.extend([((rendering_meander, 'port1', 0), (o1, port1, 0))])
        self.add(rendering_meander)
        return rendering_meander
        # check the distance to the end


#         calculate_total_length()
#
#         #write some points to connect
#         #make an indent from the open end
#
#
#
#
#         if not o2:
#             t1 = o1.get_terminals()[port1]
#             w = t1.w
#             s = t1.s
#             g = t1.g
#             # w1 = t1.w
#             # s1 = t1.s
#             # g1 = t1.g
#             # if type(w1) == list or type(s1) == list:
#             #     if len(w1) == 1 and len(s1) == 2:
#             #         w = w1[0]
#             #         s = s1[0]
#             #         g = g1
#             #     else:
#             #         raise ValueError('Unexpected size of CPW')
#             # else:
#             #     w = w1[0]
#             #     s = s1[0]
#             #     g = g1
#             delta = g + s + w / 2
#             connector_length = connector + 2 * delta
#             angle = t1.orientation + np.pi
#             initial_point = [(t1.position[0], t1.position[1])]
#             meander = elements.CPWMeander(initial_point=initial_point, w=w, s=s, g=g, meander_length=meander_length,
#                                           restricted_scale=restricted_scale, constant_scale=constant_scale,
#                                           orientation=angle, connector_length=connector_length)
#             points_for_creation = meander.points
#             rendering_meander = elements.CPW(name=name, points=points_for_creation, w=w, s=s, g=g,
#                                              layer_configuration=self.layer_configuration, r=radius)
#             self.add(rendering_meander)
#             self.connections.extend([((rendering_meander, 'port1', 0), (o1, port1, 0))])
#         else:
#             t1 = o1.get_terminals()[port1]
#             t2 = o2.get_terminals()[port2]
#             w1, s1, g1 = t1.w, t1.s, t1.g
#             w2, s2, g2 = t2.w, t2.s, t2.g
#             if w1 == w2 and s1 == s2 and g1 == g2:
#                 w, s, g = w1, s1, g1
#                 delta = g + s + w / 2
#                 connector_length = connector + 4 * delta
#                 distance = np.sqrt((t1.position[0] - t2.position[0]) ** 2 + (t1.position[1] - t2.position[1]) ** 2)
#
#                 angle = np.arctan((t1.position[1] - t2.position[1]) / (t1.position[0] - t2.position[0]))
#
#                 initial_point = [(t1.position[0], t1.position[1])]
#                 final_point = [(t2.position[0], t2.position[1])]
#
#                 meander = elements.CPWMeander(initial_point=initial_point, w=w, s=s, g=g, meander_length=meander_length,
#                                               restricted_scale=restricted_scale, constant_scale=distance,
#                                               orientation=angle, connector_length=connector_length)
#                 points_for_creation = meander.points
#                 # TODO: create a meander connection with different angles
#                 # TODO: create a meander connection using connector_cpw
#                 # points_for_creation = meander.points[1:len(meander.points) - 1]
#
#                 # angle1 = np.arctan(
#                 #     (t1.position[1] - points_for_creation[0][1]) / (t1.position[0] - points_for_creation[0][0]))
#                 # angle2 = np.arctan(
#                 #     (points_for_creation[-1][1] - t2.position[1]) / (points_for_creation[-1][0] - t2.position[0]))
#                 #
#                 # points_for_creation.insert(0, (t1.position[0] + connector_length * np.cos(angle1),
#                 #                                t1.position[1] + connector_length * np.sin(angle1)))
#                 #
#                 # points_for_creation.insert(len(points_for_creation),
#                 #                            (points_for_creation[-1][0] + connector_length * np.cos(angle2),
#                 #                             points_for_creation[-1][1] + connector_length * np.sin(angle2)))
#
#                 # points_for_creation.insert(0, (t1.position[0], t1.position[1]))
#                 #
#                 # points_for_creation.insert(len(points_for_creation), (t2.position[0], t2.position[1]))
#
#                 rendering_meander = elements.CPW(name=name, points=points_for_creation, w=w, s=s, g=g,
#                                                  layer_configuration=self.layer_configuration, r=radius)
#                 self.add(rendering_meander)
# #                self.connections.extend([((rendering_meander, 'port1', 0), (o1, port1, 0)),
# #                                         ((rendering_meander, 'port2', 0), (o2, port2, 0))])
#                 self.connect(rendering_meander, 'port1', o1, port1)
#                 self.connect(rendering_meander, 'port2', o2, port2)
#             else:
#                 raise ValueError('CPW parameters are not equal!')


# TODO: might be useflu for elements/cpw.py to caluclate the line of the cpw line
def calculate_total_length(points):
    i0, j0 = points[0]
    length = 0
    for i, j in points[1:]:
        length += np.sqrt((i - i0) ** 2 + (j - j0) ** 2)
    return length


def parametric_equation_of_line(x0, y0, x1, y1, t):
    ax = x1 - x0
    ay = y1 - y0

    x = x0 + ax * t
    y = y0 + ay * t

    return x, y


def segment_points(segment):
    x0 = segment['startpoint'][0]
    y0 = segment['startpoint'][1]
    x1 = segment['endpoint'][0]
    y1 = segment['endpoint'][1]

    return x0, y0, x1, y1
