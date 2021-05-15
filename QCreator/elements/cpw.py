from QCreator.elements import DesignTerminal
from .core import DesignElement, DesignTerminal, LayerConfiguration
import numpy as np
import gdspy
from .. import conformal_mapping as cm
from .. import transmission_line_simulator as tlsim
from typing import List, Tuple, Mapping, Union, Iterable, Dict
from copy import  deepcopy


class CPWCoupler(DesignElement):
    def __init__(self, name: str, points: List[Tuple[float, float]], w: List[float], s: List[float], g: float,
                 layer_configuration: LayerConfiguration, r: float, corner_type: str = 'round',
                 orientation1: float = None, orientation2: float = None, kinetic_inductance = None):
        """
        Create a coplanar waveguide (CPW) through points.
        :param name: element identifier
        :param points: points which the CPW traverses
        :param w: CPW signal conductor
        :param s: CPW signal-g s
        :param g:CPW finite g width
        :param layer_configuration:
        :param r: bend radius
        :param corner_type: 'round' for circular arcs instead of sharp corners, anything else for sharp corners
        :param kinetic_inductance: additional kinetic inductance (added direclty to wire inductance) per square in Henries
        """
        super().__init__('mc-cpw', name)
        self.w = w
        self.g = g
        self.s = s
        self.points = [np.asarray(p) for p in points]
        self.r = r
        self.restricted_area = None
        self.layer_configuration = layer_configuration
        self.end = self.points[-1]
        self.angle = None
        self.corner_type = corner_type
        self.length = None
        self.tls_cache = []
        self.first_segment_orientation = np.arctan2(self.points[1][1] - self.points[0][1],
                                                    self.points[1][0] - self.points[0][0])
        self.last_segment_orientation = np.arctan2(self.points[-2][1] - self.points[-1][1],
                                                   self.points[-2][0] - self.points[-1][0])
        if kinetic_inductance is not None:
            self.kinetic_inductance = kinetic_inductance
        else:
            self.kinetic_inductance = [0 for i in range(len(w))]

        if orientation1 is None:
            orientation1 = self.first_segment_orientation
        if orientation2 is None:
            orientation2 = self.last_segment_orientation

        self.terminals = {
            'port1': DesignTerminal(self.points[0], orientation1, g=g, s=s, w=w, type='mc-cpw', order=False),
            'port2': DesignTerminal(self.points[-1], orientation2, g=g, s=s, w=w, type='mc-cpw')}

        self.width_total, self.widths, self.offsets = widths_offsets(self.w, self.s, self.g)

        self.segments = []
        self.finalize_points()

    def finalize_points(self):
        eps = 1e-7
        orientation1 = np.asarray([np.cos(self.terminals['port1'].orientation),
                                   np.sin(self.terminals['port1'].orientation)])

        orientation2 = np.asarray([np.cos(self.terminals['port2'].orientation),
                                   np.sin(self.terminals['port2'].orientation)])

        adapter_length = self.width_total + self.r
        adapted_points = [p for p in self.points]
        orientation1_delta = np.abs(self.terminals['port1'].orientation - self.first_segment_orientation)
        if orientation1_delta > np.pi:
            orientation1_delta -= 2 * np.pi
            orientation1_delta = np.abs(orientation1_delta)
        if orientation1_delta > eps:
            second_point = adapted_points[0] + adapter_length * orientation1 * np.tan(orientation1_delta / 2)
            adapted_points = [adapted_points[0], second_point] + adapted_points[1:]

        orientation2_delta = np.abs(self.terminals['port2'].orientation - self.last_segment_orientation)
        if orientation2_delta > np.pi:
            orientation2_delta -= 2 * np.pi
            orientation2_delta = np.abs(orientation2_delta)
        if orientation2_delta > eps:
            blast_point = self.points[-1] + adapter_length * orientation2 * np.tan(orientation2_delta / 2 + 0.001)
            adapted_points = adapted_points[:-1] + [blast_point, adapted_points[-1]]

        # remove duplicates
        adapted_points2 = [adapted_points[0]]

        for p in adapted_points[1:]:
            if np.sqrt(np.sum((p - adapted_points2[-1]) ** 2)) > eps:
                adapted_points2.append(p)
        adapted_points = adapted_points2

        self.segments = []
        self.length = 0
        last_point = None

        # if we are not in the endpoints, morph points
        for point_id, point in enumerate(adapted_points):
            if point_id == 0 or point_id == len(adapted_points) - 1:
                segment = {'type': 'endpoint', 'endpoint': point, 'length': 0}
                if last_point is not None:
                    self.length += np.sqrt(np.sum((point - last_point) ** 2))
                    segment['length'] = np.sqrt(np.sum((point - last_point) ** 2))
                    segment['startpoint'] = last_point
                # first_point = self.segments[0]['endpoint']
                self.segments.append(segment)
                last_point = point
                continue
            current_corner_type = 'pointy'
            if self.corner_type == 'round':
                current_corner_type = 'round'
                next_point = adapted_points[point_id + 1]
                #last_point = adapted_points[point_id - 1]

                length1 = np.sqrt(np.sum((point - last_point) ** 2))
                length2 = np.sqrt(np.sum((point - next_point) ** 2))
                direction1 = (point - last_point) / length1
                direction2 = (point - next_point) / length2

                # determine turn angle of next section wrt current section
                turn = np.arctan2(next_point[1] - point[1], next_point[0] - point[0]) - \
                       np.arctan2(point[1] - last_point[1], point[0] - last_point[0])
                # in [-pi, +pi] range
                if turn > np.pi:
                    turn -= 2 * np.pi
                if turn < -np.pi:
                    turn += 2 * np.pi

                if np.abs(turn) < 0.001:
                    current_corner_type = 'pointy'
            if current_corner_type == 'round':
                replaced_length = np.abs(np.tan(turn / 2) * self.r)
                replaced_point1 = point - direction1 * replaced_length
                replaced_point2 = point - direction2 * replaced_length

                assert np.all(np.isfinite(replaced_point2)) and np.all(np.isfinite(replaced_point1))

                if replaced_length > length1+eps or replaced_length > length2:
                    raise ValueError('Too short segment in line to round corner with given radius')

                if length1 - replaced_length > eps:
                    self.segments.append({'type': 'segment', 'startpoint': last_point, 'endpoint': replaced_point1,
                                          'length': np.sqrt(np.sum((last_point - replaced_point1) ** 2))})
                else:
                    replaced_point1 = last_point
                self.segments.append({'type': 'turn', 'turn': turn, 'startpoint': replaced_point1,
                                      'endpoint': replaced_point2, 'length': np.abs(turn)*self.r,
                                      'instead_point': point})

                self.length += (np.sqrt(np.sum((replaced_point1 - last_point) ** 2)) + np.abs(turn) * self.r)
                last_point = replaced_point2
            else:
                self.segments.append({'type': 'segment', 'startpoint': last_point, 'endpoint': point,
                                      'length': np.sqrt(np.sum((point - last_point) ** 2))})
                self.length += np.sqrt(np.sum((point - last_point) ** 2))
                last_point = point

        self.segments = [s for s in self.segments if s['type'] != 'segment' or s['length'] != 0]

        assert (np.isfinite(self.length))

    def render(self):
        bend_radius = self.g
        precision = 0.001
        p1 = gdspy.FlexPath([self.segments[0]['endpoint']], width=self.widths, offset=self.offsets, ends='flush',
                                 corners='natural', bend_radius=bend_radius, precision=precision,
                                 layer=self.layer_configuration.total_layer)
        p2 = gdspy.FlexPath([self.segments[0]['endpoint']], width=self.width_total, offset=0, ends='flush',
                                 corners='natural', bend_radius=self.g, precision=precision,
                                 layer=self.layer_configuration.restricted_area_layer)
        for segment in self.segments[1:]:
            if segment['type'] == 'turn':
                p1.turn(self.r, angle=segment['turn'])
                p2.turn(self.r, angle=segment['turn'])
            else:
                p1.segment(segment['endpoint'])
                p2.segment(segment['endpoint'])

        return {'positive': p1.to_polygonset(), 'restrict': p2.to_polygonset()}

    def get_terminals(self):
        return self.terminals

    def cm(self, epsilon):
        cross_section = [self.s[0]]
        for c in range(len(self.w)):
            cross_section.append(self.w[c])
            cross_section.append(self.s[c + 1])

        cl, ll = cm.ConformalMapping(cross_section, epsilon=epsilon).cl_and_Ll()

        if not self.terminals['port1'].order:
            ll, cl = ll[::-1, ::-1], cl[::-1, ::-1]

        return cl, ll

    def add_to_tls(self, tls_instance: tlsim.TLSystem, terminal_mapping: Mapping[str, int], track_changes: bool = True,
                   cutoff: float = np.inf, epsilon=11.45) -> list:
        cl, ll = self.cm(epsilon)
        lk = np.asarray(self.kinetic_inductance) / np.asarray(self.w)
        ll = ll + np.diag(lk)
        line = tlsim.TLCoupler(n=len(self.w),
                               l=self.length,  # TODO: get length
                               cl=cl,
                               ll=ll,
                               rl=np.zeros((len(self.w), len(self.w))),
                               gl=np.zeros((len(self.w), len(self.w))),
                               name=self.name,
                               cutoff=cutoff)

        if track_changes:
            self.tls_cache.append([line])

        if len(self.w) == 1:
            if 'port1' in terminal_mapping:
                p1 = terminal_mapping['port1']
            elif ('port1', 0) in terminal_mapping:
                p1 = terminal_mapping[('port1', 0)]
            else:
                raise ValueError('Neither (port1, 0) or port1 found in terminal_mapping')

            if 'port2' in terminal_mapping:
                p2 = terminal_mapping['port2']
            elif ('port2', 0) in terminal_mapping:
                p2 = terminal_mapping[('port2', 0)]
            else:
                raise ValueError('Neither (port2, 0) or port2 found in terminal_mapping')

            tls_instance.add_element(line, [p1, p2])
        else:
            mapping = [terminal_mapping[('port1', i)] for i in range(len(self.w))] + \
                      [terminal_mapping[('port2', i)] for i in range(len(self.w))]
            tls_instance.add_element(line, mapping)
        return [line]

    def __repr__(self):
        return 'CPWCoupler "{}", n={}, l={:4.3f}'.format(self.name, len(self.w), np.round(self.length, 3))


class CPW(CPWCoupler):
    def __init__(self, name: str, points: List[Tuple[float, float]], w: float, s: float, g: float,
                 layer_configuration: LayerConfiguration, r: float, corner_type: str = 'round',
                 orientation1: float = None, orientation2: float = None):
        super().__init__(name, points, [w], [s, s], g, layer_configuration, r, corner_type, orientation1, orientation2)

        self.terminals['port1'].type = 'cpw'
        self.terminals['port1'].w = w
        self.terminals['port1'].s = s
        self.terminals['port1'].g = g

        self.terminals['port2'].type = 'cpw'
        self.terminals['port2'].w = w
        self.terminals['port2'].s = s
        self.terminals['port2'].g = g

    def __repr__(self):
        return 'CPW "{}", l={:4.3f}'.format(self.name, np.round(self.length, 3))


# TODO: make compatible with DesignElement and implement add_to_tls
class Narrowing(DesignElement):
    def __init__(self, name: str, position: Tuple[float, float], orientation: float, w1: float, s1: float, g1: float,
                 w2: float, s2: float, g2: float, layer_configuration: LayerConfiguration, length: float):
        # c: float=0, l: float=0):
        """
        Isosceles trapezoid-form adapter from one CPW to another.
        :param name: Element name
        :param position: position of center
        :param orientation: orientation in radians
        :param w1: signal conductor width of port 1
        :param s1: signal-ground gap of port 1
        :param g1: finite ground width of port 1
        :param w2: signal conductor width of port 2
        :param s2: signal-ground gap of port 2
        :param g2: finite ground width of port 2
        :param layer_configuration:
        :param length: height of trapezoid
        :param c: signal-to-ground capacitance
        :param l: port 1 to port 2 inductance
        """
        super().__init__('narrowing', name)
        self.position = position
        self.orientation = orientation

        self.w1 = w1
        self.s1 = s1
        self.g1 = g1

        self.w2 = w2
        self.s2 = s2
        self.g2 = g2

        self.layer_configuration = layer_configuration
        self.length = length

        """self.c = c
        self.l = l"""

        x_begin = self.position[0] - self.length / 2 * np.cos(self.orientation)
        x_end = self.position[0] + self.length / 2 * np.cos(self.orientation)
        y_begin = self.position[1] - self.length / 2 * np.sin(self.orientation)
        y_end = self.position[1] + self.length / 2 * np.sin(self.orientation)

        self.terminals = {'port1': DesignTerminal((x_begin, y_begin), self.orientation, w=w1, s=s1, g=g1, type='cpw'),
                          'port2': DesignTerminal((x_end, y_end), self.orientation + np.pi, w=w2, s=s2, g=g2,
                                                  type='cpw')}

        self.tls_cache = []

    def render(self):
        x_begin = self.position[0] - self.length / 2
        x_end = self.position[0] + self.length / 2
        y_begin = self.position[1]
        y_end = self.position[1]
        # x_end = self._x_begin-self.length
        # y_end = self._y_begin

        points_for_poly1 = [(x_begin, y_begin + self.w1 / 2 + self.s1 + self.g1),
                            (x_begin, y_begin + self.w1 / 2 + self.s1),
                            (x_end, y_end + self.w2 / 2 + self.s2),
                            (x_end, y_end + self.w2 / 2 + self.s2 + self.g2)]

        points_for_poly2 = [(x_begin, y_begin + self.w1 / 2),
                            (x_begin, y_begin - self.w1 / 2),
                            (x_end, y_end - self.w2 / 2),
                            (x_end, y_end + self.w2 / 2)]

        points_for_poly3 = [(x_begin, y_begin - (self.w1 / 2 + self.s1 + self.g1)),
                            (x_begin, y_begin - (self.w1 / 2 + self.s1)),
                            (x_end, y_end - (self.w2 / 2 + self.s2)),
                            (x_end, y_end - (self.w2 / 2 + self.s2 + self.g2))]

        points_for_restricted_area = [(x_begin, y_begin + self.w1 / 2 + self.s1 + self.g1),
                                      (x_end, y_end + self.w2 / 2 + self.s2 + self.g2),
                                      (x_end, y_end - (self.w2 / 2 + self.s2 + self.g2)),
                                      (x_begin, y_begin - (self.w1 / 2 + self.s1 + self.g1))]

        restricted_area = gdspy.Polygon(points_for_restricted_area,
                                        layer=self.layer_configuration.restricted_area_layer)

        poly1 = gdspy.Polygon(points_for_poly1)
        poly2 = gdspy.Polygon(points_for_poly2)
        poly3 = gdspy.Polygon(points_for_poly3)

        if self.orientation == 0:
            result = poly1, poly2, poly3
        else:
            poly1_ = poly1.rotate(angle=self.orientation, center=self.position)
            poly2_ = poly2.rotate(angle=self.orientation, center=self.position)
            poly3_ = poly3.rotate(angle=self.orientation, center=self.position)
            restricted_area.rotate(angle=self.orientation, center=self.position)
            result = poly1_, poly2_, poly3_

        polygon_to_remove = gdspy.boolean(restricted_area, result, 'not',
                                          layer=self.layer_configuration.layer_to_remove)

        return {'positive': result, 'restrict': [restricted_area], 'remove': polygon_to_remove}

    def get_terminals(self):
        return self.terminals

    def add_to_tls(self, tls_instance: tlsim.TLSystem, terminal_mapping: Mapping[str, int], track_changes: bool = True,
                   cutoff: float = np.inf, epsilon=11.45) -> list:

        cl1, ll1 = cm.ConformalMapping([self.s1, self.w1, self.s1], epsilon=epsilon).cl_and_Ll()
        cl2, ll2 = cm.ConformalMapping([self.s2, self.w2, self.s2], epsilon=epsilon).cl_and_Ll()

        l = tlsim.Inductor(l=(ll1 + ll2) / 2 * self.length)
        c1 = tlsim.Capacitor(c=cl1 / 2 * self.length)
        c2 = tlsim.Capacitor(c=cl2 / 2 * self.length)

        if track_changes:
            self.tls_cache.append([l, c1, c2])

        tls_instance.add_element(l, [terminal_mapping['port1'], terminal_mapping['port2']])
        tls_instance.add_element(c1, [terminal_mapping['port1'], 0])
        tls_instance.add_element(c2, [terminal_mapping['port2'], 0])

        return [l, c1, c2]


def widths_offsets(w, s, g):
    width_total = g * 2 + sum(s) + sum(w)
    widths = [g] + w + [g]
    offsets = [-(width_total - g) / 2]
    for c in range(len(widths) - 1):
        offsets.append(offsets[-1] + widths[c] / 2 + s[c] + widths[c + 1] / 2)

    return width_total, widths, offsets


class RectGrounding(DesignElement):
    terminals: Dict[str, DesignTerminal]

    def __init__(self, name: str, position: Tuple[float, float], orientation: float, w: List[float], s: List[float],
                 g: float, grounding_width: float,
                 grounding_between: List[Tuple[int, int]],
                 layer_configuration: LayerConfiguration, reverse_type=None):
        """
        Create ground element for  CPWs
        :param name: element identifier
        :param grounding_width: width of grounding wire
        :param grounding_between: tuple shows which conductors should be shorted
        :param layer_configuration:
        """
        super().__init__('rect-grounding', name)
        self.position = position
        self.orientation = orientation

        self.grounding_width = grounding_width
        self.layer_configuration = layer_configuration
        self.grounding_between = grounding_between
        self.tls_cache = []

        if type(w) and type(s) != list:  # create lists of w and s
            self.w = [w]
            self.s = [s, s]
            self.initial_number_of_conductors = 1
        else:
            self.w = w
            self.s = s
            self.initial_number_of_conductors = len(self.w)
        self.g = g
        # create a list of all widths of CPW including widths of conductors, gaps and ground
        widths_of_cpw = [self.g]
        for i in range(len(self.w)):
            width1 = self.s[i]
            widths_of_cpw.append(width1)
            width2 = self.w[i]
            widths_of_cpw.append(width2)
        widths_of_cpw.append(self.s[len(self.w)])
        widths_of_cpw.append(self.g)
        self.widths_of_cpw = widths_of_cpw

        end_points = (self.position[0] - self.grounding_width * np.cos(self.orientation),
                      self.position[1] - self.grounding_width * np.sin(self.orientation))
        self.end_points = end_points

        widths_of_cpw_new = []

        tail1 = widths_of_cpw[:2 * grounding_between[0][0]]  # if no conductor then tail1 = []
        tail2 = widths_of_cpw[2 * grounding_between[len(grounding_between) - 1][1] + 1:]

        short_list = []
        delta_list = [tail1]

        for i in range(len(grounding_between) - 1):
            delta = widths_of_cpw[2 * grounding_between[i][1] + 1:2 * grounding_between[i + 1][0]]
            delta_list.extend([delta])
        for i in range(len(grounding_between)):
            short = widths_of_cpw[2 * grounding_between[i][0]:2 * grounding_between[i][1] + 1]
            short_list.extend([sum(short)])
        delta_list.extend([tail2])

        for i in range(len(short_list)):
            width1 = delta_list[i]
            widths_of_cpw_new.extend(width1)

            width2 = short_list[i]
            widths_of_cpw_new.extend([width2])
        if len(delta_list) > len(short_list):
            widths_of_cpw_new.extend(delta_list[len(delta_list) - 1])

        #if reverse_type == 'Negative':
        #    widths_of_cpw_new.reverse()

        list_of_conductors = []
        list_of_gaps = []
        for i in range(len(widths_of_cpw_new)):
            if i % 2 == 0:
                list_of_conductors.extend([widths_of_cpw_new[i]])
            else:
                list_of_gaps.extend([widths_of_cpw_new[i]])

        delta_width = ((list_of_conductors[0] - self.g) - (
                list_of_conductors[len(list_of_conductors) - 1] - self.g))
        # TODO: check new_end_points
        new_end_points = (self.end_points[0] - (delta_width / 2) * np.sin(self.orientation),
                          self.end_points[1] - (delta_width / 2) * np.cos(self.orientation))

        self.narrow_port_s = list_of_gaps
        self.narrow_port_w = list_of_conductors[1:len(list_of_conductors) - 1]
        # list_of_conductors: widths of all conductors including ground
        self.final_number_of_conductors = len(self.narrow_port_w)
        if self.narrow_port_s and self.narrow_port_w:
            if len(self.narrow_port_s) == 2 and len(self.narrow_port_w) == 1 and self.narrow_port_s[0] == \
                    self.narrow_port_s[1]:

                self.terminals = {
                    'wide': DesignTerminal(position=self.position, orientation=self.orientation,
                                           g=self.g, s=self.s,
                                           w=self.w, type='cpw', order=False),
                    'narrow': DesignTerminal(position=new_end_points, orientation=self.orientation, g=self.g,
                                             s=self.narrow_port_s[0],
                                             w=self.narrow_port_w[0], type='cpw')}

            else:
                self.terminals = {
                    'wide': DesignTerminal(position=self.position, orientation=self.orientation,
                                           g=self.g, s=self.s,
                                           w=self.w, type='cpw', order=False),
                    'narrow': DesignTerminal(position=new_end_points, orientation=self.orientation, g=self.g,
                                             s=self.narrow_port_s,
                                             w=self.narrow_port_w, type='cpw')}

        else:
            self.terminals = {
                'wide': DesignTerminal(position=self.position, orientation=self.orientation,
                                       g=self.g, s=self.s[0],
                                       w=self.w[0], type='cpw', order=False)}

        self.widths_ground, self.offsets_ground = widths_offsets_for_ground(list_of_conductors, list_of_gaps)
        self.widths_of_cpw_new = widths_of_cpw_new

        real_number_of_conductors = len(self.w) + 2  # number of cores plus two parts of ground
        real_conductors, closed_conductors, free_conductors = sort_grounding_conductors(self.grounding_between,
                                                                                        real_number_of_conductors)
        free_core_conductors = []
        for elem in free_conductors:
            if elem != 0 and elem != real_number_of_conductors - 1:
                free_core_conductors.append(elem - 1)
        self.free_core_conductors = free_core_conductors

    def render(self):
        bend_radius = self.g
        precision = 0.001

        positive_total = None
        restrict_total = None

        ground = gdspy.FlexPath([self.position, self.end_points], width=self.widths_ground,
                                offset=self.offsets_ground, ends='flush',
                                corners='natural', bend_radius=bend_radius, precision=precision,
                                layer=self.layer_configuration.total_layer)

        ground_restricted = gdspy.FlexPath([self.position, self.end_points], width=sum(self.widths_of_cpw_new),
                                           offset=0, ends='flush',
                                           corners='natural', bend_radius=bend_radius, precision=precision,
                                           layer=self.layer_configuration.restricted_area_layer)

        positive_total = ground
        restrict_total = ground_restricted
        return {'positive': positive_total, 'restrict': restrict_total}

    def get_terminals(self):
        return self.terminals

    def cm(self, epsilon):
        cross_section_narrow = [self.narrow_port_s[0]]
        for c in range(len(self.narrow_port_w)):
            cross_section_narrow.append(self.narrow_port_w[c])
            cross_section_narrow.append(self.narrow_port_s[c + 1])

        return cm.ConformalMapping(cross_section_narrow, epsilon=epsilon).cl_and_Ll()

    def add_to_tls(self, tls_instance: tlsim.TLSystem, terminal_mapping: Mapping[str, int], track_changes: bool = True,
                   cutoff: float = np.inf, epsilon=11.45) -> list:

        cache = []

        if len(self.terminals.keys()) == 1:
            if len(self.w) > 1:
                for conductor_id in range(len(self.w)):  # loop over all conductors
                    g = tlsim.Short()
                    tls_instance.add_element(g, [
                        terminal_mapping[('wide', conductor_id)]])  # tlsim.TLSystem.add_element(name, nodes)
                    cache.append(g)
            else:
                g = tlsim.Short()
                cache.append(g)
                tls_instance.add_element(g, [terminal_mapping['wide']])  # tlsim.TLSystem.add_element(name, nodes)

            if track_changes:
                self.tls_cache.append(cache)
            return cache

        elif len(self.terminals.keys()) == 2:

            real_number_of_conductors = len(self.w) + 2  # number of cores plus two parts of ground
            real_conductors, closed_conductors, free_conductors = sort_grounding_conductors(self.grounding_between,
                                                                                            real_number_of_conductors)
            if 0 in free_conductors:
                free_conductors.pop(0)
            if real_number_of_conductors - 1 in free_conductors:
                free_conductors.pop(-1)

            conductor_in_narrow = 0
            mapping_ = []
            for conductor in real_conductors:
                if conductor in closed_conductors:
                    if conductor == 0:
                        zero_resistor = tlsim.Resistor(r=0, name=self.name + str(len(cache)))
                        mapping = [terminal_mapping[('wide', conductor - 1 + 1)], 0]
                        tls_instance.add_element(zero_resistor, mapping)
                        cache.append(zero_resistor)
                    elif conductor == real_number_of_conductors - 2:
                        zero_resistor = tlsim.Resistor(r=0, name=self.name + str(len(cache)))
                        mapping = [terminal_mapping[('wide', conductor - 1)], 0]
                        tls_instance.add_element(zero_resistor, mapping)
                        cache.append(zero_resistor)
                    else:
                        zero_resistor = tlsim.Resistor(r=0, name=self.name + str(len(cache)))
                        mapping = [terminal_mapping[('wide', conductor - 1)]] + [
                            terminal_mapping[('wide', conductor - 1 + 1)]]
                        tls_instance.add_element(zero_resistor, mapping)
                        cache.append(zero_resistor)
                elif conductor in free_conductors:
                    mapping_ += [terminal_mapping[('wide', conductor - 1)]]

            for narrow_conductor in range(len(self.narrow_port_w)):
                if ('narrow') in terminal_mapping:
                    mapping_ += [terminal_mapping['narrow']]

                elif ('narrow', 0) in terminal_mapping:
                    mapping_ += [terminal_mapping[('narrow', narrow_conductor)]]
            cl, ll = self.cm(epsilon)
            continued_line = tlsim.TLCoupler(n=len(self.narrow_port_w),
                                             l=self.grounding_width,
                                             cl=cl,
                                             ll=ll,
                                             rl=np.zeros((len(self.narrow_port_w), len(self.narrow_port_w))),
                                             gl=np.zeros((len(self.narrow_port_w), len(self.narrow_port_w))),
                                             name=self.name,
                                             cutoff=cutoff)
            tls_instance.add_element(continued_line, mapping_)
            cache.append(continued_line)

            if track_changes:
                self.tls_cache = cache
                return [cache]

    def __repr__(self):
        return "RectGrounding {}".format(self.name)


def widths_offsets_for_ground(list_of_conductors, list_of_gaps):
    widths = list_of_conductors
    if len(list_of_gaps) == 0:
        width_total = sum(list_of_conductors)
        offsets = 0
    else:
        width_total = sum(list_of_conductors) + sum(list_of_gaps)
        offsets = [-(width_total - widths[0]) / 2]
        for c in range(len(widths) - 1):
            offsets.append(offsets[-1] + widths[c] / 2 + list_of_gaps[c] + widths[c + 1] / 2)

    return widths, offsets


def sort_grounding_conductors(grounded_conductors, total_conductors):
    real_conductors = []
    for i in range(total_conductors):
        real_conductors.append(i)
    free = []
    closed = []
    grounded = []

    for i in range(len(grounded_conductors)):
        left_ind = grounded_conductors[i][0]
        right_ind = grounded_conductors[i][1]
        for j in range(int(left_ind), int(right_ind), 1):
            closed.append(j)
            grounded.append(j)
        grounded.append(right_ind)
    for conductor in real_conductors:
        if not conductor in grounded:
            free.append(conductor)

    return real_conductors, closed, free


class RectFanout(DesignElement):
    terminals: Dict[str, DesignTerminal]

    def __init__(self, name: str, port: DesignTerminal, grouping: Tuple[int, int],
                 layer_configuration: LayerConfiguration, down_s_right: float = None, center_s_left: float = None,
                 center_s_right: float = None, up_s_left: float = None, kinetic_inductance = None):
        """
        Create fanout element for coupled CPWs. Ground electrodes are added between the groups.
        :param name: element identifier
        :param port: port of CPWCoupler to attach to
        :param groups: tuple of conductor number, fanout angle, conductor-ground width and finite ground width
        :param layer_configuration:
        :param down_s_new:
        """
        super().__init__('rect-fanout', name)
        self.port = port
        self.grouping = grouping
        self.layer_configuration = layer_configuration
        self.tls_cache = []

        e = np.asarray([np.cos(port.orientation + np.pi), np.sin(port.orientation + np.pi)])
        e_ = np.asarray([np.cos(port.orientation + np.pi), np.sin(port.orientation + np.pi)])
        e_down = np.asarray([np.cos(port.orientation + np.pi / 2), np.sin(port.orientation + np.pi / 2)])
        e_up = np.asarray([np.cos(port.orientation - np.pi / 2), np.sin(port.orientation - np.pi / 2)])

        self.terminals = {'wide': DesignTerminal(position=self.port.position, orientation=self.port.orientation + np.pi,
                                                 type='mc-cpw', w=self.port.w, s=self.port.s, g=self.port.g,
                                                 disconnected='short')}

        if not self.port.order:
            self.w = self.port.w[::-1]
            self.s = self.port.s[::-1]
            self.g = self.port.g
            self.grouping = [len(self.w) - self.grouping[1], len(self.w) - self.grouping[0]]
        else:
            self.w = self.port.w
            self.s = self.port.s
            self.g = self.port.g

        if kinetic_inductance is not None:
            self.kinetic_inductance = kinetic_inductance
        else:
            self.kinetic_inductance = [0 for i in range(len(self.w))]

        self.width_total, self.widths, self.offsets = widths_offsets(self.w, self.s, self.g)

        if down_s_right is None:
            down_s_right = self.s[0]
        if center_s_left is None:
            center_s_left = self.s[self.grouping[0]]
        if center_s_right is None:
            center_s_right = self.s[self.grouping[1]]
        if up_s_left is None:
            up_s_left = self.s[-1]

        # list of booleans identifing if down, center and up conductor groups exist [down, center, up]
        self.group_names = ['down', 'center', 'up']
        self.group_orientations = [np.pi / 2, 0, -np.pi / 2]
        self.groups_exist = [bool(self.grouping[0]), self.grouping[0] != self.grouping[1],
                             self.grouping[1] != len(self.w)]
        self.groups_s = [self.s[:self.grouping[0]] + [down_s_right],
                         [center_s_left] + self.s[(self.grouping[0] + 1):self.grouping[1]] + [center_s_right],
                         [up_s_left] + self.s[(self.grouping[1] + 1):]]
        self.groups_w = [self.w[:self.grouping[0]], self.w[self.grouping[0]:self.grouping[1]],
                         self.w[self.grouping[1]:]]

        self.groups_kinetic_inductance = [self.kinetic_inductance[:self.grouping[0]],
                                          self.kinetic_inductance[self.grouping[0]:self.grouping[1]],
                                          self.kinetic_inductance[self.grouping[1]:]]

        self.groups_widths_total = []
        self.groups_widths = []
        self.groups_offsets = []
        self.groups_global_offsets = []
        self.groups_first_conductor = [0, self.grouping[0], self.grouping[1]]
        self.groups_last_conductor = [self.grouping[0], self.grouping[1], len(self.w)]
        for group_exists, group_w, group_s, first_conductor in zip(self.groups_exist, self.groups_w, self.groups_s,
                                                                   self.groups_first_conductor):
            if group_exists:
                group_width_total, group_widths, group_offsets = widths_offsets(group_w, group_s, self.g)
                group_global_offset = self.offsets[first_conductor + 1] - group_offsets[1]
            else:
                group_width_total, group_widths, group_offsets = 0, [], []
                group_global_offset = 0
            self.groups_widths_total.append(group_width_total)
            self.groups_widths.append(group_widths)
            self.groups_offsets.append(group_offsets)
            self.groups_global_offsets.append(group_global_offset)

        self.length = max([self.groups_widths_total[0], self.groups_widths_total[2]]) - self.g  # length of element

        # length_down = offsets[grouping[0] + 1] - offsets[1]
        # length_center = offsets[grouping[1] + 1] - offssets[grouping[0] + 1]
        # length_up = offsets[-2] - offsets[grouping[1] + 1]

        points_down = [port.position, port.position + e * (
                self.length - self.groups_widths_total[0] / 2 + np.abs(self.groups_global_offsets[0])),
                       port.position + e * (self.length - self.groups_widths_total[0] / 2 + np.abs(
                           self.groups_global_offsets[0])) + e_down * self.width_total / 2]
        points_center = [port.position, port.position + e * (self.length - self.g)]
        points_up = [port.position, port.position + e * (
                self.length - self.groups_widths_total[2] / 2 + np.abs(self.groups_global_offsets[2])),
                     port.position + e * (self.length - self.groups_widths_total[2] / 2 + np.abs(
                         self.groups_global_offsets[2])) + e_up * self.width_total / 2]

        # self.dict_of_lengths = {'down': total_length_of_line(points_down),
        #                         'center': total_length_of_line(points_center), 'up': total_length_of_line(points_up)}

        self.groups_points = [points_down, points_center, points_up]

        # TODO: add position of middle port
        self.terminals['middle'] = DesignTerminal(position=(0, 0),
                                                  orientation=self.get_terminals()['wide'].orientation,
                                                  w=self.get_terminals()['wide'].w, s=self.get_terminals()['wide'].s,
                                                  g=self.get_terminals()['wide'].g, type='mc-cpw')

        for name, exists, points, orientation, w, s, global_offset in zip(
                self.group_names, self.groups_exist, self.groups_points, self.group_orientations, self.groups_w,
                self.groups_s, self.groups_global_offsets):
            if exists:
                if len(w) == 1 and s[0] == s[1]:
                    type_, w, s = 'cpw', w[0], s[0]
                else:
                    type_ = 'mc-cpw'

                if name in ['up', 'down']:
                    position_correction = - e * np.abs(global_offset)
                else:
                    position_correction = e_up * global_offset

                group_orientation = -orientation + self.port.orientation
                if group_orientation > 2 * np.pi:
                    group_orientation -= 2 * np.pi
                self.terminals[name] = DesignTerminal(position=points[-1] + position_correction,
                                                      orientation=group_orientation, type=type_,
                                                      w=w, s=s, g=self.g, disconnected='short')

        # correction of points
        # it is a dictionary with creation points for every type of points
        self.dict_of_points = {'down': points_down, 'center': points_center, 'up': points_up}

        delta_correction = {'down': self.groups_global_offsets[0], 'center': self.groups_global_offsets[1],
                            'up': self.groups_global_offsets[2]}

        for group in self.dict_of_points.keys():
            if group in ['down', 'up']:
                corrected_points = [(0, 0), (0, 0), (0, 0)]
                global_offset = delta_correction[group]

                if group == 'down':
                    corrected_points[0] = self.dict_of_points[group][0] + e_down * np.abs(global_offset)
                elif group == 'up':
                    corrected_points[0] = self.dict_of_points[group][0] + e_up * np.abs(global_offset)

                corrected_points[2] = self.dict_of_points[group][2] - e * np.abs(global_offset)

                corrected_points[1] = (corrected_points[2][0], corrected_points[0][1])

                self.dict_of_points[group] = corrected_points

    def finilize_points_for_tls(self):
        """
        This method creates a structure of fanout element and returns it.
        """
        dict_of_parts = {'center': [], 'down': [], 'up': []}

        for group in self.terminals.keys():
            if group == 'up' or group == 'down':
                first_part_length = total_length_of_line(self.dict_of_points[group][:2])
                second_part_length = total_length_of_line(self.dict_of_points[group][1:])
                dict_of_parts[group] = [first_part_length, second_part_length]

            elif group == 'center':
                part_length = total_length_of_line(self.dict_of_points[group])
                dict_of_parts[group] = [0, part_length]
            else:
                continue

        terminals_keys = list(self.get_terminals().keys())
        terminals_keys.sort()
        terminals_keys.remove('wide')
        terminals_keys.remove('middle')
        # keys of new created terminals

        if len(terminals_keys) > 1:
            if ('down' in terminals_keys) and ('up' in terminals_keys):
                coupling_length = min([dict_of_parts['down'][0]] + [dict_of_parts['up'][0]])
            elif 'down' in terminals_keys:
                coupling_length = dict_of_parts['down'][0]
            elif 'up' in terminals_keys:
                coupling_length = dict_of_parts['up'][0]
            else:
                coupling_length = 0

        elif len(terminals_keys) == 1:
            coupling_length = 0
        else:
            raise ValueError('Unexpected topology of fanout ports!')

        dict_of_lengths = dict.fromkeys(['coupler'] + terminals_keys)

        dict_of_lengths['coupler'] = coupling_length

        for group in terminals_keys:
            independent_length = dict_of_parts[group][0] - coupling_length + dict_of_parts[group][1]
            dict_of_lengths[group] = independent_length

        structure_of_fanout = dict.fromkeys(['coupler'] + terminals_keys)

        for group in (['coupler'] + terminals_keys):
            if group == 'coupler':
                port_type = 'wide'
            else:
                port_type = group
            # param l: length of fanout substructure
            # param w:
            # param s:
            # param g:
            # param noc: number of conductors
            if type(self.get_terminals()[port_type].w) == list:
                number_of_conductors = len(self.get_terminals()[port_type].w)
                w = self.get_terminals()[port_type].w
                s = self.get_terminals()[port_type].s
            else:
                number_of_conductors = 1
                w = [self.get_terminals()[port_type].w]
                s = [self.get_terminals()[port_type].s, self.get_terminals()[port_type].s]

            if port_type == 'down':
                kinetic_inductance = self.groups_kinetic_inductance[0]
            elif port_type == 'center':
                kinetic_inductance = self.groups_kinetic_inductance[1]
            elif port_type == 'up':
                kinetic_inductance = self.groups_kinetic_inductance[2]
            else:
                kinetic_inductance = self.kinetic_inductance

            structure_of_fanout[group] = {'l': dict_of_lengths[group],
                                          'w': w,
                                          's': s,
                                          'g': self.get_terminals()[port_type].g,
                                          'kinetic_inductance': kinetic_inductance,
                                          'noc': number_of_conductors}

        return structure_of_fanout

    def render(self):
        precision = 0.001
        # ground = self.width_total - (self.offsets[self.grouping[0]] - self.widths[:self.grouping[0]]/2)
        restrict_total = None
        lines, protect = [], []

        for name, exists, points, orientation, widths, offsets, global_offset, width_total in zip(
                self.group_names, self.groups_exist, self.groups_points, self.group_orientations, self.groups_widths,
                self.groups_offsets, self.groups_global_offsets, self.groups_widths_total):
            if exists:
                global_offsets = [offset + global_offset for offset in offsets]
                lines.append(
                    gdspy.FlexPath(points, width=widths, offset=global_offsets, ends='flush', corners='natural',
                                   bend_radius=0, precision=precision, layer=self.layer_configuration.total_layer))

                restrict = gdspy.FlexPath(points, width=width_total, offset=global_offset, ends='flush',
                                          corners='natural', bend_radius=0, precision=precision,
                                          layer=self.layer_configuration.restricted_area_layer)

                protect.append(
                    gdspy.FlexPath(points, width=width_total - 2 * self.g, offset=global_offset, ends='extended',
                                   corners='natural', bend_radius=0, precision=precision,
                                   layer=self.layer_configuration.restricted_area_layer))

                restrict_total = gdspy.boolean(restrict_total, restrict.to_polygonset(), 'or',
                                               layer=self.layer_configuration.restricted_area_layer)

        positive_total = None
        for line_id in range(len(lines)):
            protect_total = None
            for other_line_id in range(len(lines)):
                if other_line_id != line_id:
                    protect_total = gdspy.boolean(protect_total, protect[other_line_id].to_polygonset(),
                                                  'or', layer=self.layer_configuration.total_layer)

            positive = gdspy.boolean(lines[line_id].to_polygonset(), protect_total, 'not',
                                     layer=self.layer_configuration.total_layer)
            positive_total = gdspy.boolean(positive_total, positive, 'or', layer=self.layer_configuration.total_layer)

        return {'positive': positive_total, 'restrict': restrict_total}

    def cm(self, epsilon):
        structure_of_fanout = self.finilize_points_for_tls()
        structure_for_tls = dict.fromkeys(structure_of_fanout.keys())

        for group in structure_of_fanout.keys():
            group_cl, group_ll = cm.ConformalMapping(
                cross_section(structure_of_fanout[group]['w'], structure_of_fanout[group]['s']), epsilon).cl_and_Ll()

            group_ll = group_ll + np.asarray(structure_of_fanout[group]['kinetic_inductance'])/np.asarray(structure_of_fanout[group]['w'])

            structure_for_tls[group] = {'n': structure_of_fanout[group]['noc'],
                                        'l': structure_of_fanout[group]['l'],
                                        'Cl': group_cl,
                                        'Ll': group_ll}

        return structure_for_tls

    def add_to_tls(self, tls_instance: tlsim.TLSystem, terminal_mapping: Mapping[str, int], track_changes: bool = True,
                   cutoff: float = np.inf, epsilon=11.45) -> list:
        structure_for_tls = self.cm(epsilon)

        cache = []

        # for elem in list(structure_for_tls.keys()):
        for elem in (list(self.get_terminals().keys()) + ['coupler']):
            mapping_ = []
            if elem in ['coupler', 'down', 'center', 'up']:
                number_of_conductors = structure_for_tls[elem]['n']

            if elem == 'coupler':
                coupled_line = tlsim.TLCoupler(n=number_of_conductors,
                                               l=structure_for_tls[elem]['l'],
                                               cl=structure_for_tls[elem]['Cl'],
                                               ll=structure_for_tls[elem]['Ll'],
                                               rl=np.zeros_like(structure_for_tls[elem]['Cl']),
                                               gl=np.zeros_like(structure_for_tls[elem]['Cl']),
                                               name=self.name + '_coupled_line_' + str(len(cache)),
                                               cutoff=cutoff)

                for conductor_id in range(number_of_conductors):
                    mapping_ += [terminal_mapping[('wide', conductor_id)]]
                for conductor_id in range(number_of_conductors):
                    mapping_ += [terminal_mapping[('middle', conductor_id)]]

                tls_instance.add_element(coupled_line, mapping_)
                cache.append(coupled_line)

                if track_changes:
                    self.tls_cache.append(coupled_line)

            elif elem in ['down', 'center', 'up']:
                line = tlsim.TLCoupler(n=number_of_conductors,
                                       l=structure_for_tls[elem]['l'],
                                       cl=structure_for_tls[elem]['Cl'],
                                       ll=structure_for_tls[elem]['Ll'],
                                       rl=np.zeros_like(structure_for_tls[elem]['Cl']),
                                       gl=np.zeros_like(structure_for_tls[elem]['Cl']),
                                       name=self.name + '_line_' + str(len(cache)),
                                       cutoff=cutoff)

                conductor_bounds = [0, self.grouping[0], self.grouping[1], len(self.w)]
                if elem == 'down':
                    begin_conductor = conductor_bounds[0]
                    end_conductor = conductor_bounds[1]
                elif elem == 'center':
                    begin_conductor = conductor_bounds[1]
                    end_conductor = conductor_bounds[2]
                elif elem == 'up':
                    begin_conductor = conductor_bounds[2]
                    end_conductor = conductor_bounds[3]

                for conductor_id in range(begin_conductor, end_conductor):

                    mapping_ += [terminal_mapping[('middle', conductor_id)]]

                if elem in terminal_mapping:
                    mapping_ += [terminal_mapping[elem]]

                elif (elem, 0) in terminal_mapping:

                    for conductor_id in range(number_of_conductors):
                        mapping_ += [terminal_mapping[(elem, conductor_id)]]

                else:
                    raise ValueError('Neither ({}, 0) or port2 found in terminal_mapping'.format(elem))

                tls_instance.add_element(line, mapping_)
                cache.append(line)

                if track_changes:
                    self.tls_cache.append(line)

            else:
                continue

        return cache

    # def cm(self):
    #     cross_section = [self.s[0]]
    #     for c in range(len(self.w)):
    #         cross_section.append(self.w[c])
    #         cross_section.append(self.s[c + 1])
    #
    #     wide_cl, wide_ll = cm.ConformalMapping(cross_section).cl_and_Ll()
    #
    #     groups_cl = []
    #     groups_ll = []
    #     for group_id in range(3):
    #         if not self.groups_exist[group_id]:
    #             cl = [[]]
    #             ll = [[]]
    #         else:
    #             cross_section = [self.groups_s[group_id][0]]
    #             for c in range(len(self.groups_w[group_id])):
    #                 cross_section.append(self.groups_w[group_id][c])
    #                 cross_section.append(self.groups_s[group_id][c + 1])
    #
    #             cl, ll = cm.ConformalMapping(cross_section).cl_and_Ll()
    #         groups_cl.append(cl)
    #         groups_ll.append(ll)
    #
    #     return wide_cl, wide_ll, groups_cl, groups_ll
    #
    # def add_to_tls(self, tls_instance: tlsim.TLSystem,
    #                terminal_mapping: Mapping[str, int], track_changes: bool = True) -> list:
    #     wide_cl, wide_ll, groups_cl, groups_ll = self.cm()
    #     '''
    #     coupled_lengths = []
    #     if self.groups_exist[0]:
    #         coupled_lengths.append(self.groups_widths_total[0])
    #     if self.groups_exist[2]:
    #         coupled_lengths.append(self.groups_widths_total[2])
    #
    #     coupled_length = self.length - np.max(coupled_lengths)  / 2
    #
    #     full_coupled_line = tlsim.TLCoupler(n=len(self.w),
    #                                         l=coupled_length,  # TODO: get length
    #                                         cl=wide_cl,
    #                                         ll=wide_ll,
    #                                         rl=np.zeros((len(self.w), len(self.w))),
    #                                         gl=np.zeros((len(self.w), len(self.w))))
    #     '''
    #
    #     group_lines = []
    #     for group_id in range(3):
    #         if self.groups_exist[group_id]:
    #             line = tlsim.TLCoupler(n=len(self.groups_w[group_id]),
    #                                    l=self.length,
    #                                    cl=groups_cl[group_id],
    #                                    ll=groups_ll[group_id],
    #                                    rl=np.zeros_like(groups_cl[group_id]),
    #                                    gl=np.zeros_like(groups_cl[group_id]),
    #                                    name=self.name + '_group' + str(group_id))
    #
    #             if self.port.order:
    #                 mapping = [terminal_mapping[('wide', len(self.w) - i - 1)]
    #                            for i in
    #                            range(self.groups_first_conductor[group_id], self.groups_last_conductor[group_id])]
    #             else:
    #                 mapping = [terminal_mapping[('wide', i)]
    #                            for i in
    #                            range(self.groups_first_conductor[group_id], self.groups_last_conductor[group_id])]
    #
    #             if self.groups_last_conductor[group_id] - self.groups_first_conductor[group_id] == 1 and \
    #                     self.group_names[group_id] in terminal_mapping:
    #                 mapping = mapping + [terminal_mapping[self.group_names[group_id]]]
    #             else:
    #                 mapping = mapping + [terminal_mapping[(self.group_names[group_id], i)]
    #                                      for i in range(0, self.groups_last_conductor[group_id] -
    #                                                     self.groups_first_conductor[group_id])]
    #             tls_instance.add_element(line, mapping)
    #             group_lines.append(line)
    #     #            else:
    #     #                group_lines.append([])
    #
    #     if track_changes:
    #         self.tls_cache.append(group_lines)
    #
    #     return group_lines
    def get_terminals(self):
        return self.terminals

    def __repr__(self):
        return "RectFanout {}, n={}, grouping=({}, {})".format(self.name, len(self.w), *self.grouping)


def total_length_of_line(points):
    length = 0
    i0, j0 = points[0]
    for i, j in points[1:]:
        length += np.sqrt((i - i0) ** 2 + (j - j0) ** 2)
        i0, j0 = i, j
    return length


def cross_section(w, s):
    section = [s[0]]
    for c in range(len(w)):
        section.append(w[c])
        section.append(s[c + 1])
    return section

    # TODO: modify two classes with position and orientation and port parameters use  in general_sample_creator


class OpenEnd(DesignElement):
    terminals: Dict[str, DesignTerminal]

    def __init__(self, name: str, position: Tuple[float, float], w: List[float], s: List[float], g: float,
                 orientation: float,
                 layer_configuration: LayerConfiguration, h1: float = 20., h2: float = 20):
        """
        Create open end (or connect two grounds of CPW).
        :param name: element identifier
        :param position: initial position of an element
        :param w: widths of conductors
        :param s: widths of gaps between conductors
        :param g: ground width
        :param h1: first length of open end
        :param h2: second length of open end
        :param layer_configuration
        """
        super().__init__('open_end', name)

        self.position = position
        self.orientation = orientation
        self.h1 = h1
        self.h2 = h2
        self.layer_configuration = layer_configuration

        self.tls_cache = []

        self.w = w
        self.s = s
        self.g = g

        self.number_of_conductors = len(self.w)

        width_of_line = sum(self.w) + sum(self.s) + 2 * self.g
        angle = self.orientation - np.pi

        self.gap = width_of_line - 2 * self.g

        self.initial_points = self.position
        self.final_points = (
            self.position[0] - self.h1 * np.cos(angle), self.position[1] - self.h1 * np.sin(angle))
        self.final_points_ = (
            self.final_points[0] - self.h2 * np.cos(angle), self.final_points[1] - self.h2 * np.sin(angle))

        if len(self.w) == 1 and len(self.s) == 2 and self.s[0] == self.s[0]:
            self.terminals = {'wide': DesignTerminal(position=self.position, orientation=self.orientation,
                                                     g=self.g, s=self.s[0],
                                                     w=self.w[0], type='cpw')}

        else:
            self.terminals = {'wide': DesignTerminal(position=self.position, orientation=self.orientation,
                                                     g=self.g, s=self.s,
                                                     w=self.w, type='cpw')}

    def render(self):
        positive_total = None
        restrict_total = None

        continue_ground = gdspy.FlexPath(points=[self.initial_points, self.final_points],
                                         width=[self.g, self.g],
                                         offset=[-self.gap / 2 - self.g / 2, self.gap / 2 + self.g / 2],
                                         corners='natural', ends='flush',
                                         layer=self.layer_configuration.total_layer)

        add_connection = gdspy.FlexPath(points=[self.final_points, self.final_points_],
                                        width=self.gap + 2 * self.g,
                                        corners='natural', ends='flush', layer=self.layer_configuration.total_layer)

        restrict_total = gdspy.FlexPath(points=[self.initial_points, self.final_points_],
                                        width=self.gap + 2 * self.g,
                                        corners='natural', ends='flush',
                                        layer=self.layer_configuration.restricted_area_layer)

        positive_total = gdspy.boolean(operand1=continue_ground, operand2=add_connection, operation='or')

        return {'positive': positive_total, 'restrict': restrict_total}

    def get_terminals(self) -> Mapping[str, DesignTerminal]:
        return self.terminals

    def add_to_tls(self, tls_instance: tlsim.TLSystem, terminal_mapping: Mapping[str, int], track_changes: bool = True,
                   cutoff: float = np.inf, epsilon=11.45) -> list:
        if self.number_of_conductors == 1:
            cache = []
            capacitance_value = 1e-15*20*0
            capacitor = tlsim.Capacitor(capacitance_value, 'open_end')
            cache.append(capacitor)
            tls_instance.add_element(capacitor,
                                     [terminal_mapping['wide'], 0])  # tlsim.TLSystem.add_element(name, nodes)

            return cache

        elif self.number_of_conductors > 1:
            cache = []
            for conductor_id in range(self.number_of_conductors):  # loop over all conductors
                capacitance_value = 20e-15
                capacitor = tlsim.Capacitor(capacitance_value, 'open_end')
                tls_instance.add_element(capacitor, [terminal_mapping[('wide', conductor_id)],
                                                     0])  # tlsim.TLSystem.add_element(name, nodes)
                cache.append(capacitor)
            return cache

        if track_changes:
            self.tls_cache.append(cache)

    def __repr__(self):
        return "OpenEnd {}".format(self.name)

'''
class RectFanout(DesignElement):
    def __init__(self, name: str, coupler: CPWCoupler, port: DesignTerminal,
                 groups: Iterable[Tuple[int, float, float, float]], r:Union[float, type(None)] = None):
        """
        Create fanout element for coupled CPWs. Ground electrodes are added between the groups.
        :param name: element identifier
        :param coupler: CPWCoupler element to attach to
        :param port: port of CPWCoupler to attach to
        :param groups: tuple of conductor number, fanout angle, conductor-ground width and finite ground width
        :param r: minimal bend radius. If 0 or None, use edgy corners
        :param length: fanout length along the direction of the CPWcoupler. If zero or None
        """
        self.name = name
        self.coupler = coupler
        self.port = port
        self.groups = groups
        self.wire_groups = []
        if not self.r:
            self.r = 0
        else:
            self.r = r
        last_angle = -np.pi/2
        last_wire = 0
        max_length = 0
        group_offsets = []
        for group_size, angle, s, g in groups:
            #self.wire_groups.append(last_wire, last_wire+group_size)
            last_wire = last_wire + group_size
            if angle < last_angle:
                raise ValueError ('Fanout angles are not monotone')
            if angle < 0:
                min_length = np.sin(angle) * (np.min(self.coupler.offsets[last_wire:last_wire + group_size]) - np.min(
                    self.coupler.offsets) + self.r) - g - s
                group_offsets.append(min_length - max_length)
                max_length = np.sin(angle) * (np.max(self.coupler.offsets[last_wire:last_wire + group_size]) - np.min(
                    self.coupler.offsets) + self.r) + g + s
        max_length = 0
        for group_size, angle, s, g in groups[::-1]:
            # self.wire_groups.append(last_wire, last_wire+group_size)
            last_wire = last_wire + group_size
            if angle < last_angle:
                raise ValueError('Fanout angles are not monotone')
            if angle > 0:
                min_length = np.sin(angle) * (np.max(self.coupler.offsets) - np.max(
                    self.coupler.offsets[last_wire:last_wire + group_size]) + self.r)
                max_length = np.sin(angle) * (np.max(self.coupler.offsets) - np.max(
                    self.coupler.offsets[last_wire:last_wire + group_size]) + self.r)
'''
# class RectGrounding(DesignElement):
#     terminals: Dict[str, DesignTerminal]
#
#     def __init__(self, name: str, port: DesignTerminal, grounding_width: float,
#                  grounding_between: List[Tuple[int, int]],
#                  layer_configuration: LayerConfiguration, reverse_type):
#         """
#         Create ground element for  CPWs.
#         :param name: element identifier
#         :param port: port of CPWCoupler to attach to
#         :param grounding_width: width of grounding wire
#         :param grounding_between: tuple shows which conductors should be shorted
#         :param layer_configuration:
#         """
#         super().__init__('rect-grounding', name)
#         self.port = port
#         self.grounding_width = grounding_width
#         self.layer_configuration = layer_configuration
#         self.grounding_between = grounding_between
#         self.tls_cache = []
#
#         if type(port.w) and type(port.s) != list:  # create lists of w and s
#             self.port.w = [port.w]
#             self.port.s = [port.s, port.s]
#             self.initial_number_of_conductors = 1
#         else:
#             self.port.w = port.w
#             self.port.s = port.s
#             self.initial_number_of_conductors = len(self.port.w)
#
#         # create a list of all widths of CPW including widths of conductors, gaps and ground
#         widths_of_cpw = [self.port.g]
#         for i in range(len(self.port.w)):
#             width1 = self.port.s[i]
#             widths_of_cpw.append(width1)
#             width2 = self.port.w[i]
#             widths_of_cpw.append(width2)
#         widths_of_cpw.append(self.port.s[len(self.port.w)])
#         widths_of_cpw.append(self.port.g)
#         self.widths_of_cpw = widths_of_cpw
#
#         end_points = (self.port.position[0] - self.grounding_width * np.cos(self.port.orientation),
#                       self.port.position[1] - self.grounding_width * np.sin(self.port.orientation))
#         self.end_points = end_points
#
#         widths_of_cpw_new = []
#
#         tail1 = widths_of_cpw[:2 * grounding_between[0][0]]  # if no conductor then tail1 = []
#         tail2 = widths_of_cpw[2 * grounding_between[len(grounding_between) - 1][1] + 1:]
#
#         short_list = []
#         delta_list = [tail1]
#
#         for i in range(len(grounding_between) - 1):
#             delta = widths_of_cpw[2 * grounding_between[i][1] + 1:2 * grounding_between[i + 1][0]]
#             delta_list.extend([delta])
#         for i in range(len(grounding_between)):
#             short = widths_of_cpw[2 * grounding_between[i][0]:2 * grounding_between[i][1] + 1]
#             short_list.extend([sum(short)])
#         delta_list.extend([tail2])
#
#         for i in range(len(short_list)):
#             width1 = delta_list[i]
#             widths_of_cpw_new.extend(width1)
#
#             width2 = short_list[i]
#             widths_of_cpw_new.extend([width2])
#         if len(delta_list) > len(short_list):
#             widths_of_cpw_new.extend(delta_list[len(delta_list) - 1])
#
#         if reverse_type == 'Negative':
#             widths_of_cpw_new.reverse()
#
#         list_of_conductors = []
#         list_of_gaps = []
#         for i in range(len(widths_of_cpw_new)):
#             if i % 2 == 0:
#                 list_of_conductors.extend([widths_of_cpw_new[i]])
#             else:
#                 list_of_gaps.extend([widths_of_cpw_new[i]])
#
#         delta_width = ((list_of_conductors[0] - self.port.g) - (
#                 list_of_conductors[len(list_of_conductors) - 1] - self.port.g))
#         # TODO: check new_end_points
#         new_end_points = (self.end_points[0] - (delta_width / 2) * np.sin(self.port.orientation),
#                           self.end_points[1] - (delta_width / 2) * np.cos(self.port.orientation))
#
#         self.narrow_port_s = list_of_gaps
#         self.narrow_port_w = list_of_conductors[
#                              1:len(
#                                  list_of_conductors) - 1]  # list_of_conductors: widths of all conductors including ground
#         self.final_number_of_conductors = len(self.narrow_port_w)
#         if self.narrow_port_s and self.narrow_port_w:
#             if len(self.narrow_port_s) == 2 and len(self.narrow_port_w) == 1 and self.narrow_port_s[0] == \
#                     self.narrow_port_s[1]:
#
#                 self.terminals = {
#                     'wide': DesignTerminal(position=self.port.position, orientation=self.port.orientation - np.pi,
#                                            g=self.port.g, s=self.port.s,
#                                            w=self.port.w, type='mc-cpw'),
#                     'narrow': DesignTerminal(position=new_end_points, orientation=self.port.orientation, g=self.port.g,
#                                              s=self.narrow_port_s[0],
#                                              w=self.narrow_port_w[0], type='cpw')}
#             else:
#                 self.terminals = {
#                     'wide': DesignTerminal(position=self.port.position, orientation=self.port.orientation - np.pi,
#                                            g=self.port.g, s=self.port.s,
#                                            w=self.port.w, type='mc-cpw'),
#                     'narrow': DesignTerminal(position=new_end_points, orientation=self.port.orientation, g=self.port.g,
#                                              s=self.narrow_port_s,
#                                              w=self.narrow_port_w, type='mc-cpw')}
#
#         else:
#             self.terminals = {
#                 'wide': DesignTerminal(position=self.port.position, orientation=self.port.orientation - np.pi,
#                                        g=self.port.g, s=self.port.s,
#                                        w=self.port.w, type='mc-cpw')}
#
#         self.widths_ground, self.offsets_ground = widths_offsets_for_ground(list_of_conductors, list_of_gaps)
#         self.widths_of_cpw_new = widths_of_cpw_new
#
#         real_number_of_conductors = len(self.port.w) + 2  # number of cores plus two parts of ground
#         real_conductors, closed_conductors, free_conductors = sort_grounding_conductors(self.grounding_between,
#                                                                                         real_number_of_conductors)
#         free_core_conductors = []
#         for elem in free_conductors:
#             if elem != 0 and elem != real_number_of_conductors - 1:
#                 free_core_conductors.append(elem - 1)
#         self.free_core_conductors = free_core_conductors
#
#     def render(self):
#         bend_radius = self.port.g
#         precision = 0.001
#
#         positive_total = None
#         restrict_total = None
#
#         ground = gdspy.FlexPath([self.port.position, self.end_points], width=self.widths_ground,
#                                 offset=self.offsets_ground, ends='flush',
#                                 corners='natural', bend_radius=bend_radius, precision=precision,
#                                 layer=self.layer_configuration.total_layer)
#
#         ground_restricted = gdspy.FlexPath([self.port.position, self.end_points], width=sum(self.widths_of_cpw_new),
#                                            offset=0, ends='flush',
#                                            corners='natural', bend_radius=bend_radius, precision=precision,
#                                            layer=self.layer_configuration.restricted_area_layer)
#
#         positive_total = ground
#         restrict_total = ground_restricted
#         return {'positive': positive_total, 'restrict': restrict_total}
#
#     def get_terminals(self):
#         return self.terminals
#
#     def cm(self):
#         cross_section_narrow = [self.narrow_port_s[0]]
#         for c in range(len(self.narrow_port_w)):
#             cross_section_narrow.append(self.narrow_port_w[c])
#             cross_section_narrow.append(self.narrow_port_s[c + 1])
#
#         return cm.ConformalMapping(cross_section_narrow).cl_and_Ll()
#
#     def add_to_tls(self, tls_instance: tlsim.TLSystem, terminal_mapping: Mapping[str, int],
#                    track_changes: bool = True) -> list:
#
#         cache = []
#
#         if len(self.terminals.keys()) == 1:
#             if len(self.port.w) > 1:
#                 for conductor_id in range(len(self.port.w)):  # loop over all conductors
#                     g = tlsim.Short()
#                     tls_instance.add_element(g, [
#                         terminal_mapping[('wide', conductor_id)]])  # tlsim.TLSystem.add_element(name, nodes)
#                     cache.append(g)
#             else:
#                 g = tlsim.Short()
#                 cache.append(g)
#                 tls_instance.add_element(g, [terminal_mapping['wide']])  # tlsim.TLSystem.add_element(name, nodes)
#
#             if track_changes:
#                 self.tls_cache.append(cache)
#             return cache
#
#         elif len(self.terminals.keys()) == 2:
#
#             real_number_of_conductors = len(self.port.w) + 2  # number of cores plus two parts of ground
#             real_conductors, closed_conductors, free_conductors = sort_grounding_conductors(self.grounding_between,
#                                                                                             real_number_of_conductors)
#             if 0 in free_conductors:
#                 free_conductors.pop(0)
#             if real_number_of_conductors - 1 in free_conductors:
#                 free_conductors.pop(-1)
#
#             conductor_in_narrow = 0
#             mapping_ = []
#             for conductor in real_conductors:
#                 if conductor in closed_conductors:
#                     if conductor == 0:
#                         zero_resistor = tlsim.Resistor(r=0, name=self.name + str(len(cache)))
#                         mapping = [terminal_mapping[('wide', conductor - 1 + 1)], 0]
#                         tls_instance.add_element(zero_resistor, mapping)
#                         cache.append(zero_resistor)
#                     elif conductor == real_number_of_conductors - 2:
#                         zero_resistor = tlsim.Resistor(r=0, name=self.name + str(len(cache)))
#                         mapping = [terminal_mapping[('wide', conductor - 1)], 0]
#                         tls_instance.add_element(zero_resistor, mapping)
#                         cache.append(zero_resistor)
#                     else:
#                         zero_resistor = tlsim.Resistor(r=0, name=self.name + str(len(cache)))
#                         mapping = [terminal_mapping[('wide', conductor - 1)]] + [
#                             terminal_mapping[('wide', conductor - 1 + 1)]]
#                         tls_instance.add_element(zero_resistor, mapping)
#                         cache.append(zero_resistor)
#                 elif conductor in free_conductors:
#                     mapping_ += [terminal_mapping[('wide', conductor - 1)]]
#
#             for narrow_conductor in range(len(self.narrow_port_w)):
#                 if ('narrow') in terminal_mapping:
#                     mapping_ += [terminal_mapping['narrow']]
#
#                 elif ('narrow', 0) in terminal_mapping:
#                     mapping_ += [terminal_mapping[('narrow', narrow_conductor)]]
#
#                 # if len(self.narrow_port_w) == 1:
#                 #     mapping_ += [terminal_mapping['narrow']]
#                 # else:
#                 #     mapping_ += [terminal_mapping[('narrow', narrow_conductor)]]
#             cl, ll = self.cm()
#             continued_line = tlsim.TLCoupler(n=len(self.narrow_port_w),
#                                              l=self.grounding_width,
#                                              cl=cl,
#                                              ll=ll,
#                                              rl=np.zeros((len(self.narrow_port_w), len(self.narrow_port_w))),
#                                              gl=np.zeros((len(self.narrow_port_w), len(self.narrow_port_w))),
#                                              name=self.name)
#             tls_instance.add_element(continued_line, mapping_)
#             cache.append(continued_line)
#
#             if track_changes:
#                 self.tls_cache = cache
#                 return [cache]
#
#     def __repr__(self):
#         return "RectGrounding {}".format(self.name)