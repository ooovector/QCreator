import numpy as np
import gdspy
import os
from . import elements
from . import transmission_line_simulator as tlsim
from typing import NamedTuple, SupportsFloat, Any, Iterable, Tuple, List

from . import meshing
from copy import deepcopy


class Sample:

    def __init__(self, name, configurations, epsilon=11.45):
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

        self.epsilon = epsilon
        self.negative_layer_polygons = []

    @staticmethod
    def default_cpw_radius(w, s, g):
        return 2 * (w + 2 * s + 2 * g)

    def add(self, object_):
        self.objects.append(object_)

    def draw_design(self,PP_qubits=False):
        for object_ in self.objects:
            object_.resource = None
        if PP_qubits ==True:
            None
        else:
            self.total_cell.remove_polygons(lambda pts, layer, datatype: True)
            self.restricted_cell.remove_polygons(lambda pts, layer, datatype: True)
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
            if 'airbridges_sm_pads' in result:
                self.total_cell.add(result['airbridges_sm_pads'])
            if 'airbridges' in result:
                self.total_cell.add(result['airbridges'])
            if 'inverted' in result:
                self.total_cell.add(result['inverted'])
            #elif 'positive' in result and 'restrict' in result:
            #    inverted = gdspy.boolean(result['restrict'], result['positive'],
            #                             'not', layer=self.layer_configuration.inverted)
            #    if inverted is not None:
            #        for polygon in inverted.polygons:
            #            self.negative_layer_polygons.append(polygon)
            #        self.total_cell.add(inverted)
            if 'bandages' in result:
                self.total_cell.add(result['bandages'])

        self.fill_object_arrays()

    def render_negative(self, positive_layers, negative_layer, slices = 20, apply_restrict = True):
        box = self.total_cell.get_bounding_box()

        slices_x = np.linspace(box[0][0], box[1][0], slices, endpoint=False)[1:].tolist()
        slices_y = np.linspace(box[0][1], box[1][1], slices, endpoint=False)[1:].tolist()

        polygons = {layer_id: gdspy.slice(self.total_cell.get_polygons((layer_id, 0)), slices_x, 0)
                    for layer_id in positive_layers}
        if apply_restrict:
            polygons_restrict = gdspy.slice(self.restricted_cell.get_polygons((self.layer_configuration.restricted_area_layer, 0)), slices_x, 0)

        for i in range(slices):
            polygons_slice = {layer_id: gdspy.slice(polygons[layer_id][i], slices_y, 1)
                    for layer_id in positive_layers}
            if apply_restrict:
                polygons_restrict_slice = gdspy.slice(polygons_restrict[i], slices_y, 1)
            for j in range(slices):
                if apply_restrict:
                    negative = polygons_restrict_slice[j]
                else:
                    negative = gdspy.Rectangle((box[0][0] + (box[1][0] - box[0][0]) * i / slices,
                                                box[0][1] + (box[1][0] - box[0][0]) * j / slices),
                                               (box[0][0] + (box[1][0] - box[0][0]) * (i + 1) / slices,
                                                box[0][1] + (box[1][0] - box[0][0]) * (j + 1) / slices))
                for layer_id, polygons_slice_y in polygons_slice.items():
                    negative = gdspy.boolean(negative, polygons_slice_y[j], 'not', layer = negative_layer)
                negative_filtered = []
                if negative is not None:
                    for poly in negative.polygons:
                        if gdspy.Polygon(poly).area() > 0.05:
                            negative_filtered.append(poly)
                if negative_filtered is not None:
                    self.total_cell.add(gdspy.PolygonSet(negative_filtered, layer = negative_layer))

    def layer_expansion(self, shift, old_layer, new_layer, N_shifts = 8):
        total = None
        polygonset = gdspy.PolygonSet(self.total_cell.get_polygons((old_layer, 0)), layer=new_layer)
        for i in np.linspace(-np.pi, np.pi, N_shifts, endpoint=False):
            dx, dy = shift * np.cos(i), shift * np.sin(i)
            new_polygonset = gdspy.copy(polygonset, dx, dy)
            #total = gdspy.boolean(total, new_polygonset, 'or', layer=new_layer)
            self.total_cell.add(new_polygonset)

    def draw_terminals(self):
        #draws all terminals as an arrow
        for object_ in self.objects:
            if hasattr(object_,'terminals'):
                for terminal_ in object_.terminals:
                    if object_.terminals[terminal_] != None:
                        ter = object_.terminals[terminal_]
                        T = gdspy.Rectangle((-10,0),(10,4))
                        T = gdspy.boolean(T,gdspy.Rectangle((-1,0),(1,30)),'or',layer=13)
                        T = T.rotate(ter.orientation+np.pi/2)
                        T.translate(ter.position[0],ter.position[1])
                        self.total_cell.add(T)





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
               center_s_right: float = None, up_s_left: float = None, kinetic_inductance=None):

        fanout = elements.RectFanout(name, o.get_terminals()[port], grouping, self.layer_configuration,
                                     down_s_right=down_s_right, center_s_left=center_s_left,
                                     center_s_right=center_s_right, up_s_left=up_s_left,
                                     kinetic_inductance=kinetic_inductance)
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

        open_end = elements.OpenEnd(name, position_, w_, s_, g_, orientation_+np.pi, self.layer_configuration)
        # open_end = elements.OpenEnd(name, o.get_terminals()[port], self.layer_configuration)
        self.add(open_end)

        self.connect(o, port, open_end, 'wide')
        return open_end

    def airbridge(self, o: elements.DesignElement, port: str, name: str, geometry: elements.AirBridgeGeometry):
        terminal = o.get_terminals()[port]
        airbridge_position = self.cpw_shift(o, port, geometry.pad_width / 2)[0]
        bridge = elements.airbridge.AirbridgeOverCPW(
            name=name,
            position=airbridge_position,
            orientation=terminal.orientation, w=terminal.w,
            s=terminal.s, g=terminal.g,
            geometry=geometry)
        self.add(bridge)
        self.connect(bridge, 'port2', o, port)
        return bridge, 'port1'

    def connect_cpw(self, o1: elements.DesignElement, o2: elements.DesignElement, port1: str, port2: str, name: str,
                    points: list, airbridge: elements.AirBridgeGeometry = None, min_spacing: float = None, r=None):
        """
        Create and return a CPW connecting two cpw-type ports, with point inbetween defined by point
        :param o1: first object
        :param o2: second object
        :param port1: first object's port name
        :param port2: second object's port name
        :param name: CPW name
        :param points: coordinates of the CPW's edges
        :param airbridge:
        :param min_spacing: spacing between two airbridges
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

        if r is None:
            r = self.default_cpw_radius(w, s, g)

        points = [tuple(t1.position)] + points + [tuple(t2.position)]
        cpw = elements.CPW(name, points, w, s, g, self.layer_configuration, r=r,
                           corner_type='round', orientation1=orientation1, orientation2=orientation2)

        if airbridge:
            cpw_with_bridges = self.generate_bridge_over_cpw(name=name, o=cpw,
                                                             geometry=airbridge,
                                                             min_spacing=min_spacing)
            self.connect(cpw_with_bridges[0], 'port1', o1, port1)
            self.connect(cpw_with_bridges[-1], 'port2', o2, port2)
            return cpw_with_bridges
        else:
            self.add(cpw)
            self.connect(cpw, 'port1', o1, port1)
            self.connect(cpw, 'port2', o2, port2)
            return [cpw]

    def watch(self,dark = False):
        #Gerhards color scheme here :D, dark and smooth colours
        bkg = '#2C2A4A'

        colour = {}
        c_list = ['#DABFFF','#D5A021','#95B2B0','#7FDEFF','#D3BDB0','#89937C','#EAE2B7','#EDA4BD','#A0ACAD','#C8D6AF','#95B2B0','#BA274A','#2191FB','#FCB0B3','#CE7DA5','#CFFFB0','#907AD6','#A44200']
        for i in range(18):
            colour[(i,0)] = c_list[i]
        if dark:
            gdspy.LayoutViewer(depth=0, pattern={'default': 8}, background=bkg,color=colour)  # this opens a viewer
        else:
            gdspy.LayoutViewer(depth=0, pattern={'default': 8}, background='#FFFFFF')


    def cpw_shift(self, element, port_name, length):
        return [(element.get_terminals()[port_name].position[0] + \
                 length * np.cos(element.get_terminals()[port_name].orientation + np.pi),
                 element.get_terminals()[port_name].position[1] + \
                 length * np.sin(element.get_terminals()[port_name].orientation + np.pi)), ]

    # functions to work and calculate capacitance
    def write_to_gds(self, name=None):
        if name is not None:
            self.lib.write_gds(name + '.gds', cells=None, timestamp=None,
                               binary_cells=None)
            self.path = os.getcwd() + '/' + name + '.gds'
        else:
            self.lib.write_gds(self.name + '.gds', cells=None,
                               timestamp=None,
                               binary_cells=None)
            self.path = os.getcwd() + '/' + self.name + '.gds'
        print("Gds file has been writen here: ", self.path)

    def calculate_qubit_capacitance(self, cell, qubit, mesh_volume, name=None):
        self.write_to_gds(name)
        mesh = meshing.Meshing(path=self.path,
                               cell_name=cell.name,
                               layers=list(cell.get_layers()))
        mesh.read_data_from_gds_file()
        mesh.prepare_for_meshing()
        mesh.run_meshing(mesh_volume=mesh_volume)
        mesh.write_into_file(os.getcwd() + '/' + 'mesh_4k_data')
        mesh.run_fastcap(os.getcwd() + '/' + 'mesh_4k_results')
        print("Capacitance results have been writen here: ", os.getcwd() + '/' + 'mesh_4k_results')
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
        for key, value in qubit.terminals.items():
            if value is not None and key is not 'flux':
                # print(key, value)
                qubit.C[key] = (caps[i][i]+caps[1][i], -caps[1][i])
                qubit.C['qubit'] += caps[1][i] # remove from qubit-ground
                i = i + 1
        return True

    def get_tls(self, cutoff=np.inf):
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
            element_assignments[object_.name] = object_.add_to_tls(tls, terminal_node_assignments,
                                                                   cutoff=cutoff, epsilon=self.epsilon)
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

    def get_topology(self, cutoff=np.inf):
        sys, connections_, elements_ = self.get_tls(cutoff)
        circuit_elements = sys.elements
        number_of_elem = len(circuit_elements)
        circuit_nodes = sys.terminal_node_mapping

        topology = []
        for elem in range(number_of_elem):
            topology.append([circuit_elements[elem], circuit_nodes[elem]])

        return topology

    def generate_bridge_over_cpw(self, name: str, o: elements.DesignElement, geometry: elements.AirBridgeGeometry,
                                 min_spacing: float, bridge_part_decimation=1):
        """
        This method add air bridges on CPW line.
        :param name:
        :param o:
        :param geometry:
        :param min_spacing: distance between air bridges
        """

        t = o.get_terminals()['port1']
        assert t.type == 'cpw'

        w = t.w
        s = t.s
        g = t.g
        radius = o.r

        cpws = []
        airbridges = []
        last_port = None
        current_cpw_points = []
        bridge_counter = -1
        for segment in o.segments:
            if segment['type'] == 'turn':
                current_cpw_points.append(segment['instead_point'])
                continue
            if len(current_cpw_points) == 0 and segment['type'] == 'endpoint':
                current_cpw_points.append(segment['endpoint'])
                continue

            num_bridges = int(np.floor(segment['length'] / (geometry.pad_width + min_spacing)))
            if num_bridges < 1:
                current_cpw_points.append(segment['endpoint'])
                continue

            spacing = (segment['length'] - num_bridges * geometry.pad_width) / num_bridges
            direction = segment['endpoint'] - segment['startpoint']
            direction = direction/np.sqrt(np.sum(direction**2))
            orientation = np.arctan2(direction[1], direction[0])
            begin = segment['startpoint']

            for bridge_id in range(num_bridges):
                bridge_counter += 1
                midpoint = begin + direction * (spacing * (bridge_id + 0.5) + geometry.pad_width * bridge_id)
                current_cpw_points.append(midpoint)

                cpw = elements.CPW(name=name + 's{}'.format(len(cpws)),
                                   points=current_cpw_points, w=w, s=s, g=g,
                                   layer_configuration=self.layer_configuration, r=radius)

                self.add(cpw)

                if bridge_counter % bridge_part_decimation == 0:
                    airbridge = elements.AirbridgeOverCPW(name=name + 'b{}'.format(len(cpws)),
                                                          position=midpoint+direction * geometry.pad_width/2,
                                                          orientation=orientation, w=w, s=s, g=g,
                                                          geometry=geometry)
                    self.add(airbridge)
                    self.connect(cpw, 'port2', airbridge, 'port1')
                    airbridges.append(airbridge)
                else:
                    bridge_cpw_points = []
                    bridge_cpw_points.append(midpoint)
                    bridge_cpw_points.append(midpoint+direction * geometry.pad_width)
                    bridge_cpw = elements.CPW(name=name + 's{}'.format(len(cpws)),
                                       points=bridge_cpw_points, w=w, s=s, g=g,
                                       layer_configuration=self.layer_configuration, r=radius)
                    self.add(bridge_cpw)
                    self.connect(cpw, 'port2', bridge_cpw, 'port1')
                    cpws.append(bridge_cpw)

                if last_port is not None:
                    self.connect(last_port, 'port2', cpw, 'port1')
                if bridge_counter % bridge_part_decimation == 0:
                    last_port = airbridge
                else:
                    last_port=bridge_cpw
                cpws.append(cpw)

                current_cpw_points = [midpoint + direction * geometry.pad_width]



        if current_cpw_points[-1][0] != o.segments[-1]['endpoint'][0] or current_cpw_points[-1][1] != \
                o.segments[-1]['endpoint'][1]:
            current_cpw_points.append(o.segments[-1]['endpoint'])
        cpw = elements.CPW(name=name + 's{}'.format(len(cpws)),
                           points=current_cpw_points, w=w, s=s, g=g,
                           layer_configuration=self.layer_configuration, r=radius)
        if last_port is not None:
            self.connect(last_port, 'port2', cpw, 'port1')
        self.add(cpw)

        cpws.append(cpw)
        return cpws

    def connect_meander(self, name: str, o1: elements.DesignElement, port1: str, meander_length: float,
                        length_left: float, length_right: float, first_step_orientation: float,
                        meander_orientation: float,
                        end_point=None, end_orientation=None, meander_type='round',
                        airbridge: elements.AirBridgeGeometry = None, min_spacing: float = None,
                        r = None, bridge_part_decimation=1
                        ):

        t1 = o1.get_terminals()[port1]
        (w_, s_, g_) = (t1.w, t1.s, t1.g)
        initial_position_ = t1.position
        orientation_ = t1.orientation

        meander = elements.meander_creation(name=name, initial_position=initial_position_, w=w_, s=s_, g=g_,
                                            orientation=orientation_, meander_length=meander_length,
                                            length_left=length_left, length_right=length_right,
                                            first_step_orientation=first_step_orientation,
                                            meander_orientation=meander_orientation,
                                            end_point=end_point, end_orientation=end_orientation,
                                            layer_configuration=self.layer_configuration,
                                            meander_type=meander_type, r=r)

        if airbridge is not None:
            meander_with_bridges = self.generate_bridge_over_cpw(name=name, o=meander,
                                                                 geometry=airbridge,
                                                                 min_spacing=min_spacing,
                                                                 bridge_part_decimation=bridge_part_decimation)

            self.connect(meander_with_bridges[0], 'port1', o1, port1)

            return meander_with_bridges

        else:
            self.add(meander)
            self.connect(meander, 'port1', o1, port1)

            return [meander]
