import numpy as np
import gdspy
import os
from . import elements
from . import transmission_line_simulator as tlsim
from typing import NamedTuple, SupportsFloat, Any, Iterable, Tuple, List, Mapping

from . import meshing
from copy import deepcopy


class Sample:
    def __init__(self, name: str, configurations: Mapping[str, float], epsilon=11.45):
        """
        Design class in QCreator
        :param name: name of sample which will be used for saving gds file
        :param configurations: dictionary of parameters of chip geometry and layers
        :param epsilon: dielectric permittivity of substrate
        """
        self.layer_configuration = elements.LayerConfiguration(**configurations)
        self.chip_geometry = elements.ChipGeometry(**configurations)
        self.name = str(name)

        self.lib = gdspy.GdsLibrary(unit=1e-06, precision=1e-10)

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
    def default_cpw_radius(w: float, s: float, g: float):
        """
        Default CPW radius for given center conductor width, conductor-groud gap and grounding conductors
        formula: 2(w+2*s+2*g)
        :param w:
        :param s:
        :param g:
        """
        return 2 * (w + 2 * s + 2 * g)

    def add(self, object_: elements.DesignElement):
        """
        Add object_ to design.
        """
        self.objects.append(object_)

    def draw_design(self, PP_qubits: bool = False, debug = False):
        """
        Renders the design, calling render() if required.
        :param PP_qubits: if True, polygons are not removed from the design prior to drawing
        """
        for object_ in self.objects:
            object_.resource = None
        if PP_qubits:
            pass
        else:
            self.total_cell.remove_polygons(lambda pts, layer, datatype: True)
            self.restricted_cell.remove_polygons(lambda pts, layer, datatype: True)
        for object_ in self.objects:
            if debug:
                print(object_.name)
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
            if 'dielectric' in result:
                self.total_cell.add(result['dielectric'])
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

    def render_negative(self, positive_layers: Iterable[int], negative_layer: Iterable[int],
                        slices: int = 20, apply_restrict = True):
        """
        Renders the difference between the positive layers and the negative layers.
        :param positive_layers: layers to substract from
        :param negative_layer: layer to substract
        :param slices: amount of slices that are used to fracture the positive layers. The algorithm works
        slowly if all polygons are in a single slice; if the function is slow, increase this parameter
        :param apply_restrict: something that should be True otherwise everything will die
        """
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

    def layer_expansion(self, shift: float, old_layer: int, new_layer: int, n_shifts: int = 8):
        """
        Shift the layer in all directions and sum the results up.

        :param shift: the distance to shift everything by
        :param old_layer: layer id to expand
        :param new_layer: layer id to save the expanded polygons to
        :param n_shifts: amount of different directions to shift to
        """
        total = None
        polygonset = gdspy.PolygonSet(self.total_cell.get_polygons((old_layer, 0)), layer=new_layer)
        for i in np.linspace(-np.pi, np.pi, n_shifts, endpoint=False):
            dx, dy = shift * np.cos(i), shift * np.sin(i)
            new_polygonset = gdspy.copy(polygonset, dx, dy)
            #total = gdspy.boolean(total, new_polygonset, 'or', layer=new_layer)
            self.total_cell.add(new_polygonset)

    def draw_terminals(self, layer: int = 13):
        """
        Draw arrows pointing into the terminals in the design to layer 13. Used for debugging.\
        :param layer_id: layer to draw the arrows to
        """
        #draws all terminals as an arrow
        for object_ in self.objects:
            if hasattr(object_, 'terminals'):
                for terminal_ in object_.terminals:
                    if object_.terminals[terminal_] is not None:
                        ter = object_.terminals[terminal_]
                        T = gdspy.Rectangle((-10, 0), (10, 4))
                        T = gdspy.boolean(T, gdspy.Rectangle((-1, 0), (1, 30)), 'or', layer=layer)
                        T = T.rotate(ter.orientation+np.pi/2)
                        T.translate(ter.position[0], ter.position[1])
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
                    try:
                        self.lib.remove('qubit capacitance cell ' + str(qubit_cap_cell_counter))
                    except ValueError:
                        pass
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
        """
        TODO: what is this and dow it exist
        """
        self.qubits = [i for i in self.objects if i.type == 'qubit']
        self.couplers = [i for i in self.objects if i.type == 'qubit coupler']

    # def ground(self, element: elements.DesignElement, port: str):
    #     self.connections.append(((element, port), ('gnd', 'gnd')))

    def find_wires_coordinates(self, o: elements.DesignElement, p: str):
        """
        Выдает отстройки и ширины проводов для определенного порта объекта
        :param o: object
        :param p: port
        :return: offsets and widths of wires of port p of object o
        """
        terminal = o.get_terminals()[p]
        w = terminal.w
        s = terminal.s
        g = terminal.g
        if type(terminal.w) is not list:
            w = [terminal.w]
        if type(terminal.s) is not list:
            s = [terminal.s, terminal.s]
        if type(terminal.g) is not list:
            g = [terminal.g, terminal.g]
        line_o = elements.straight_CPW_with_different_g(name='object_line',
                                                        points=[(0, 0), (1, 0)],
                                                        w=w,
                                                        s=s,
                                                        g=g,
                                                        layer_configuration=self.layer_configuration)
        offsets = np.asarray(line_o.offsets[1:-1])
        widths = np.asarray(line_o.widths[1:-1])
        return (offsets, widths)

    def connect_all(self, eps=1e-10):
        """
        Соединяет провода (кроме земли элементов) друг с другом для всех объектов и портов
        eps - точность соприкосновения проводов (если они лежат на одной прямой и расстояние между их центрами меньше этого,
        то считается, что они соприкасаются и фукнция их соединит)
        Возвращает сомнительно соединенные элементы и их порты (те элементы и те их порты, что имеют различное количество проводов
        и соединений) и само количество соединений порта этого элемента"""
        number_of_connections = [] # список количества соединений для всех портов всех объектов
        for i in range(0, len(self.objects)):
            number_of_connections.append([0] * len(self.objects[i].get_terminals().keys()))
        possible_not_correctly_connected_objects = []
        for i in range(1, len(self.objects)):
            terminal_i = self.objects[i].get_terminals()
            for (k_i, port_i) in enumerate(terminal_i.keys()):
                for j in range(0, i):
                    terminal_j = self.objects[j].get_terminals()
                    for (k_j, port_j) in enumerate(terminal_j.keys()):
                        # print(self.objects[i], self.objects[i].get_terminals()[port_i], self.objects[j],
                        #       self.objects[j].get_terminals()[port_j])
                        connects = self.connect(self.objects[i], port_i, self.objects[j], port_j,
                                                raise_errors=False, eps=eps)
                        # print(connects)
                        number_of_connections[i][k_i] += connects
                        number_of_connections[j][k_j] += connects
        for i in range(0, len(self.objects)):
            terminal_i = self.objects[i].get_terminals()
            for (k_i, port_i) in enumerate(terminal_i.keys()):
                # print(self.objects[i], port_i, number_of_connections[i][k_i])
                if type(terminal_i[port_i].w) is list:
                    if number_of_connections[i][k_i] != len(terminal_i[port_i].w):
                        possible_not_correctly_connected_objects.append(
                            (self.objects[i], port_i, number_of_connections[i][k_i]))
                else:
                    if number_of_connections[i][k_i] != 1:
                        possible_not_correctly_connected_objects.append(
                            (self.objects[i], port_i, number_of_connections[i][k_i]))
        return possible_not_correctly_connected_objects

    def connect(self, o1, p1, o2, p2, raise_errors=True, eps=1e-10):
        """
        Connects the wires of two objects with given ports. The ports must be at the same location
        :param eps: threshold for the distance between ports

        (Соединяет провода двух объектов с заданными портами основываясь на том, что эти порты должны быть в одном месте
        eps - точность соприкосновения проводов (если они лежат на одной прямой и расстояние между их центрами меньше этого,
        то считается, что они соприкасаются и фукнция их соединит)
        raise_errors если True, то ошибки возникают, в обратном случае не возникают
        Возвращает сколько проводов соединено (если ничего не соединилось, но флаг поднятия ошибок не выбран, то возвращает ноль),
        возвращаемое значение используется для удобства поиска ошибок соединений (как в модели, так и на картинке),
        особенно удобно, при использовании функции connect_all.)
        """
        if (abs((o1.get_terminals()[p1].orientation) % (2 * np.pi) - (o2.get_terminals()[p2].orientation + np.pi) % (
                (2 * np.pi)))%(2*np.pi) > eps and
            abs((o1.get_terminals()[p1].orientation) % (2 * np.pi) - (o2.get_terminals()[p2].orientation) % (
                    (2 * np.pi)))%(2*np.pi) > eps):
            if raise_errors:
                print(o1.name,o1.get_terminals()[p1])
                print(o2.name,o2.get_terminals()[p2])
                raise ValueError("Connecting parts do not fill one line, check orientations")
            # print('ORIENTATION', o1, p1, o2, p2, 0)
            # print(o1.get_terminals()[p1].orientation,o2.get_terminals()[p2].orientation)
            # print((abs((o1.get_terminals()[p1].orientation) % (2 * np.pi) - (o2.get_terminals()[p2].orientation + np.pi) % (
            #     (2 * np.pi))),
            # abs((o1.get_terminals()[p1].orientation) % (2 * np.pi) - (o2.get_terminals()[p2].orientation) % (
            #         (2 * np.pi)))))
            return 0
        (offsets1, widths1) = self.find_wires_coordinates(o1, p1)
        (offsets2, widths2) = self.find_wires_coordinates(o2, p2)
        # print(offsets1,offsets2)
        if o1.get_terminals()[p1].order is False:
            offsets1 = - offsets1
        if o2.get_terminals()[p2].order:
            offsets2 =  - offsets2
        # Поворачиваем объекты так, чтобы линия соединения была горизонтальной и при этом соединение было слева направо
        angle = o1.get_terminals()[p1].orientation + np.pi
        # то есть на угол -angle
        #     print(angle/np.pi)
        delta = (np.asarray([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]) @
                 (np.asarray(o2.get_terminals()[p2].position) - np.asarray(o1.get_terminals()[p1].position)))
        # print(delta)
        if abs(delta[0]) > eps:
            if raise_errors:
                raise ValueError("ERROR: The ports have different positions")
            # print('DELTA', o1, p1, o2, p2, 0)
            return 0
        positions1 = offsets1
        positions2 = offsets2 + delta[1]
        # print(positions1,positions2)
        error = 0
        for conductor_id_1 in range(0, len(offsets1)):
            for conductor_id_2 in range(0, len(offsets2)):
                if abs(positions1[conductor_id_1] - positions2[conductor_id_2]) < eps:
                    self.connections.append(((o1, p1, conductor_id_1), (o2, p2, conductor_id_2)))
                    error += 1
        if raise_errors:
            if error == 0:
                raise ValueError('There is no ports to connect')
        # print('FINAL',o1,p1,o2,p2,error)
        return error

    def fanout(self, o: elements.DesignElement, port: str, name: str, grouping: Tuple[int, int],
               down_s_right: float = None, center_s_left: float = None,
               center_s_right: float = None, up_s_left: float = None, kinetic_inductance: float = None):
        """
        Append Fanout DesignElement to port `port` of object `o`. The port will be connected to the `wide` port
        of the fanout
        :param o: object to append to
        :param port: port of `o` to append to
        :param name: name of the resulting Fanout element
        :param grouping: Two numbers that dissect the wires of the multiconductor CPW in port. Conductors with id
        less than grouping[0] will be routed to the `down` port, conductors with id larger or equal to grouping[1] will
        be routed the `up` port; those inbetween will be routed into the `center` port
        :param down_s_right: gap with between newly-created ground electrode and the last conductor in the down group
        :param center_s_left: gap with between newly-created ground electrode and the first conductor in the center group
        :param center_s_right: gap with between newly-created ground electrode and the last conductor in the center group
        :param up_s_left: gap with between newly-created ground electrode and the first conductor in the up group
        :param kinetic_inductance: kinetic inductance per unit length #TODO: how does this work?
        """

        fanout = elements.RectFanout(name, o.get_terminals()[port], grouping, self.layer_configuration,
                                     down_s_right=down_s_right, center_s_left=center_s_left,
                                     center_s_right=center_s_right, up_s_left=up_s_left,
                                     kinetic_inductance=kinetic_inductance)
        self.add(fanout)
        self.connect(o, port, fanout, 'wide')
        return fanout

    def ground(self, o: elements.DesignElement, port: str, name: str, grounding_width: float,
               grounding_between: List[Tuple[int, int]]) -> elements.RectGrounding:
        """
        Append RectGrounding element to a port of an element. See documentation of :ref:`RectGrounding` for details
        :param o: element (object) to append to
        :param port: port of `o` to append to
        :param name: name of resulting RectGrounding element
        :param grounding_width: width of the ground strip
        :param grounding_between: conductors to add a short between; ids are calculated from the ground electrode
        :return: RectGrounding element
        """
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

    def open_end(self, o: elements.DesignElement, port: str, name: str, h1=20, h2=20) -> elements.OpenEnd:
        """
        Append OpenEnd element to a port of an element. See documentation of :ref:`OpenEnd` for details
        :param o: element (object) to append to
        :param port: port of `o` to append to
        :param name: name of resulting RectGrounding element
        :param h1: #TODO:
        :param h2: #TODO:
        :return: OpenEnd element
        """
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

        open_end = elements.OpenEnd(name, position_, w_, s_, g_, orientation_+np.pi, self.layer_configuration,
                                    h1=h1, h2=h2)
        # open_end = elements.OpenEnd(name, o.get_terminals()[port], self.layer_configuration)
        self.add(open_end)

        self.connect(o, port, open_end, 'wide')
        return open_end

    def airbridge(self, o: elements.DesignElement, port: str, name: str, geometry: elements.AirBridgeGeometry):
        """
        Append Airbridge element to a port of an element. See documentation of :ref:`Airbridge` for details
        :param o: element (object) to append to
        :param port: port of `o` to append to
        :param name: name of resulting Airbridge element
        :param geometry: geometry of the airbridge
        :return: Airbridge element
        """
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

    def watch(self, dark=False):
        """
        Launch gdspy's default interactive viewer and show the design.
        """
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

    def cpw_shift(self, o: elements.DesignElement, port: str, length: float) -> Tuple[float, float]:
        """
        :param o: object to calculate shift from
        :param port: port name to calculate the shift from
        :param length: distance to shift
        :return: point length away from port `port` of object `o`
        """
        t = o.get_terminals()[port]

        shift = [(t.position[0] + length * np.cos(t.orientation + np.pi),
                  t.position[1] + length * np.sin(t.orientation + np.pi)), ]
        return shift

    # functions to work and calculate capacitance
    def write_to_gds(self, name=None):
        """
        Write a GDS file to the current directory
        :param name: name of the file, default is the design name
        :return: none
        """
        if name is not None:
            self.lib.write_gds(name + '.gds', cells=None, timestamp=None,
                               binary_cells=None)
            self.path = os.getcwd() + '/' + name + '.gds'
        else:
            self.lib.write_gds(self.name + '.gds', cells=None,
                               timestamp=None,
                               binary_cells=None)
            #TODO fix this
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
        caps = np.round(mesh.get_capacitances(self.epsilon), 1)
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

    #TODO: qubit-speicific code in sample class.
    # This function should go into qubit classes
    # specifically into add_to_tls
    def fill_cap_matrix_grounded(self, qubit, caps):
        qubit.C['qubit'] = caps[1][1]
        print(caps)
        i = 2
        # print(qubit.C)
        for key, value in qubit.terminals.items():
            if value is not None and key != 'flux':
                # print(key, value)
                qubit.C[key] = (caps[i][i]+caps[1][i], -caps[1][i])
                qubit.C['qubit'] += caps[1][i] # remove from qubit-ground
                i = i + 1
        return True

    def get_tls(self, cutoff=np.inf, num_modes = 2):
        """
        Create a transmission line system of the design
        :return: tls, connections_flat, element_assignments
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
            try:
                element_assignments[object_.name] = object_.add_to_tls(tls, terminal_node_assignments,
                                                                   cutoff=cutoff, epsilon=self.epsilon, num_modes = num_modes)
            except:
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
                        length_left: float, length_right: float, first_step_orientation: str,
                        meander_orientation: float,
                        end_point=None, end_orientation=None, meander_type='round',
                        airbridge: elements.AirBridgeGeometry = None, min_spacing: float = None,
                        r: float = None, bridge_part_decimation: int = 1) -> List[elements.CPW]:
        """
        Creates meander-shaped single-conductor CPW beginning from a given start point with given length
        and constrained by given bounding box.
        :param name: name of elements
        :param o1: element to connect meander to
        :param port1: name of port in o1 to connect to
        :param meander_length: length of the meander
        :param length_left: distance from beginning of meander to the meander bounding box to the left
        :param length_right: distance from beginning of meander to the meander bounding box to the right
        :param first_step_orientation: direction of first turn: 'left' or 'right'
        :param meander_orientation: orientation of meander #TODO: why does this even exist? it should be constrained by port
        :param end_point: additional final point which probably breaks the length of the meander #TODO: why do we need this?
        :param end_orientation: orientation of final point position #TODO: end_point is broken
        :param meander_type: type of meander: 'round' or 'flush'
        :param airbridge: airbridge geometry to place over meander
        :param min_spacing: minimum spacing between airbridges
        :param r: radius of meander turns
        :param bridge_part_decimation: place each n-th airbridge (used for alternating airbridges when radius is small)
        """

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

    def graphviz(self, draw=True):
        """
        Draw a graphviz schematic of the connections in the sample.
        :param draw: if True, calls render and creates a pdf in the current directory.
        if False, returns a graphviz.Graph object

        :return: graphviz.Graph or None
        """
        import graphviz
        dot = graphviz.Graph(self.name + '-connections', comment=self.name)

        clusters = {}
        cluster_names = {}

        for object_id, o in enumerate(self.objects):
            name = '#' + str(object_id) + ' ' + o.name
            # name = 'cluster object #' + str(object_id) + ' ' + o.name
            # name='cluster'+str(object_id)
            clusters[(o,)] = graphviz.Graph(name)
            cluster_names[(o,)] = name
            dot.subgraph(clusters[(o,)])

        for connection in sqd.sample.connections:
            o1 = connection[0][0]
            o2 = connection[1][0]
            p1 = connection[0][1]
            p2 = connection[1][1]
            w1 = connection[0][2]
            w2 = connection[1][2]

            l1 = cluster_names[(o1,)]  # +' '+p1+' '+str(w1)
            l2 = cluster_names[(o2,)]  # +' '+p2+' '+str(w2)

            # clusters[(o1,)].node(l1)
            # clusters[(o2,)].node(l2)
            dot.edge(l1, l2)

        if not draw:
            return dot

        dot.render(view=True)




