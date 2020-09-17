from .core import DesignElement, DesignTerminal, LayerConfiguration
import numpy as np
import gdspy
from .. import conformal_mapping as cm
from .. import transmission_line_simulator as tlsim
from typing import List, Tuple, Mapping, Union, Iterable


class CPWCoupler(DesignElement):
    def __init__(self, name: str, points: List[Tuple[float, float]], w: List[float], s: List[float], g: float,
                 layer_configuration: LayerConfiguration, r: float, corner_type: str = 'round',
                 orientation1: float = None, orientation2: float = None):
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

        if orientation1 is None:
            orientation1 = self.first_segment_orientation
        if orientation2 is None:
            orientation2 = self.last_segment_orientation

        self.terminals = {'port1': DesignTerminal(self.points[0], orientation1, g=g, s=s, w=w, type='mc-cpw'),
                          'port2': DesignTerminal(self.points[-1], orientation2, g=g, s=s, w=w, type='mc-cpw')}

        self.segments = []
        self.finalize_points()

        self.width_total = self.g * 2 + sum(self.s) + sum(self.w)

        self.widths = [self.g] + self.w + [self.g]
        self.offsets = [-(self.width_total-self.g)/2]
        for c in range(len(self.widths)-1):
            self.offsets.append(self.offsets[-1]+self.widths[c]/2+self.s[c]+self.widths[c+1]/2)

    def get_width(self):
        return sum(self.w) + sum(self.s) + 2 * self.g

    def finalize_points(self):
        orientation1 = np.asarray([np.cos(self.terminals['port1'].orientation),
                                   np.sin(self.terminals['port1'].orientation)])

        orientation2 = np.asarray([np.cos(self.terminals['port2'].orientation),
                                   np.sin(self.terminals['port2'].orientation)])

        orientation1_delta = np.abs(self.terminals['port1'].orientation - self.first_segment_orientation)
        if orientation1_delta > np.pi:
            orientation1_delta -= 2 * np.pi
            orientation1_delta = np.abs(orientation1_delta)
        orientation2_delta = np.abs(self.terminals['port2'].orientation - self.last_segment_orientation)
        if orientation2_delta > np.pi:
            orientation2_delta -= 2 * np.pi
            orientation2_delta = np.abs(orientation2_delta)

        adapter_length = self.get_width() + self.r
        first_point = self.points[0]
        second_point = self.points[0] + adapter_length * orientation1 * np.tan(orientation1_delta / 2 + 0.001)
        last_point = self.points[-1]
        blast_point = self.points[-1] + adapter_length * orientation2 * np.tan(orientation2_delta / 2 + 0.001)

        adapted_points = [first_point, second_point] + self.points[1:-1] + [blast_point, last_point]

        self.segments = []
        self.length = 0

        # if we are not in the endpoints, morph points
        for point_id, point in enumerate(adapted_points):
            if point_id == 0 or point_id == len(adapted_points) - 1:
                self.segments.append({'type': 'endpoint', 'endpoint': point})
                continue
            if self.corner_type == 'round':
                next_point = adapted_points[point_id + 1]
                last_point = adapted_points[point_id - 1]

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

                replaced_length = np.abs(np.tan(turn / 2) * self.r)
                replaced_point1 = point - direction1 * replaced_length
                replaced_point2 = point - direction2 * replaced_length

                if replaced_length > length1 or replaced_length > length2:
                    raise ValueError('Too short segment in line to round corner with given radius')

                self.segments.append({'type': 'segment', 'endpoint': replaced_point1})
                self.segments.append({'type': 'turn', 'turn': turn})

                self.length += (np.sqrt(np.sum((replaced_point1 - last_point) ** 2)) + turn * self.r)
            else:
                self.segments.append({'type': 'segment', 'endpoint': point})
                self.length += np.sqrt(np.sum((point - last_point) ** 2))

    def render(self):
        ends = ["flush"]*len(self.widths)
        corners = ["natural"]*len(self.widths)
        bend_radius = self.g
        precision = 0.001

        p1 = gdspy.FlexPath([self.segments[0]['endpoint']], width=self.widths, offset=self.offsets, ends=ends,
                            corners=corners, bend_radius=bend_radius, precision=precision,
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

    def cm(self):
        cross_section = [self.s[0]]
        for c in range(len(self.w)):
            cross_section.append(self.w[c])
            cross_section.append(self.s[c+1])

        return cm.ConformalMapping(cross_section).cl_and_Ll()

    def add_to_tls(self, tls_instance: tlsim.TLSystem,
                   terminal_mapping: Mapping[str, int], track_changes: bool = True) -> list:
        cl, ll = self.cm()
        line = tlsim.TLCoupler(n=len(self.w),
                               l=self.length,  # TODO: get length
                               cl=cl,
                               ll=ll,
                               rl=np.zeros((len(self.w), len(self.w))),
                               gl=np.zeros((len(self.w), len(self.w))))

        if track_changes:
            self.tls_cache.append([line])

        tls_instance.add_element(line, [terminal_mapping['port1'], terminal_mapping['port2']])
        return [line]


class CPW(CPWCoupler):
    def __init__(self, name: str, points: List[Tuple[float, float]], w: float, s: float, g: float,
                 layer_configuration: LayerConfiguration, r: float, corner_type: str = 'round',
                 orientation1: float = None, orientation2: float = None):
        super().__init__(name, points, [w], [s, s], g, layer_configuration, r, corner_type, orientation1, orientation2)

        self.terminals = {'port1': DesignTerminal(self.points[0], orientation1, g=g, s=s, w=w, type='cpw'),
                          'port2': DesignTerminal(self.points[-1], orientation2, g=g, s=s, w=w, type='cpw')}


#TODO: make compatible with DesignElement and implement add_to_tls
class Narrowing(DesignElement):
    def __init__(self, name: str, position: Tuple[float, float], orientation: float, w1: float, s1: float, g1: float,
                 w2: float, s2: float, g2: float, layer_configuration: LayerConfiguration, length: float):
                 #c: float=0, l: float=0):
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

        x_begin = self.position[0] - self.length/2*np.cos(self.orientation)
        x_end = self.position[0] + self.length/2*np.cos(self.orientation)
        y_begin = self.position[1] - self.length/2*np.sin(self.orientation)
        y_end = self.position[1] + self.length/2*np.sin(self.orientation)

        self.terminals = {'port1': DesignTerminal((x_begin, y_begin), self.orientation, w=w1, s=s1, g=g1, type='cpw'),
                          'port2': DesignTerminal((x_end, y_end), self.orientation+np.pi, w=w2, s=s2, g=g2, type='cpw')}

        self.tls_cache = []

    def render(self):
        x_begin = self.position[0] - self.length/2
        x_end = self.position[0] + self.length/2
        y_begin = self.position[1]
        y_end = self.position[1]
        #x_end = self._x_begin-self.length
        #y_end = self._y_begin

        points_for_poly1 = [(x_begin, y_begin + self.w1 / 2 + self.s1 + self.g1),
                            (x_begin, y_begin + self.w1 / 2 + self.s1),
                            (x_end, y_end + self.w2 / 2 + self.s2),
                            (x_end, y_end + self.w2 / 2 + self.s2 + self.g2)]

        points_for_poly2 = [(x_begin, y_begin + self.w1 / 2),
                            (x_begin, y_begin - self.w1 / 2),
                            (x_end, y_end - self.w2 / 2),
                            (x_end, y_end + self.w2 / 2)]

        points_for_poly3 = [(x_begin, y_begin-(self.w1 / 2 + self.s1 + self.g1)),
                            (x_begin, y_begin-(self.w1 / 2 + self.s1)),
                            (x_end, y_end-(self.w2 / 2 + self.s2)),
                            (x_end, y_end-(self.w2 / 2 + self.s2 + self.g2))]

        points_for_restricted_area = [(x_begin, y_begin + self.w1 / 2 + self.s1 + self.g1),
                                      (x_end, y_end + self.w2 / 2 + self.s2 + self.g2),
                                      (x_end, y_end-(self.w2 / 2 + self.s2 + self.g2)),
                                      (x_begin, y_begin-(self.w1 / 2 + self.s1 + self.g1))]

        restricted_area = gdspy.Polygon(points_for_restricted_area, layer=self.layer_configuration.restricted_area_layer)

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

    def add_to_tls(self, tls_instance: tlsim.TLSystem,
                   terminal_mapping: Mapping[str, int], track_changes: bool = True) -> list:

        cl1, ll1 = cm.ConformalMapping([self.s1, self.w1, self.s1]).cl_and_Ll()
        cl2, ll2 = cm.ConformalMapping([self.s2, self.w2, self.s2]).cl_and_Ll()

        l = tlsim.Inductor(l=(ll1+ll2)/2*self.length)
        c1 = tlsim.Capacitor(c=cl1 / 2*self.length)
        c2 = tlsim.Capacitor(c=cl2 / 2*self.length)

        if track_changes:
            self.tls_cache.append([l, c1, c2])

        tls_instance.add_element(l, [terminal_mapping['port1'], terminal_mapping['port2']])
        tls_instance.add_element(c1, [terminal_mapping['port1'], 0])
        tls_instance.add_element(c2, [terminal_mapping['port2'], 0])

        return [l, c1, c2]


class RectFanout(DesignElement):
    def __init__(self, name: str, port: DesignTerminal, grouping: Tuple[int, int],
                 layer_configuration: LayerConfiguration):
        """
        Create fanout element for coupled CPWs. Ground electrodes are added between the groups.
        :param name: element identifier
        :param port: port of CPWCoupler to attach to
        :param groups: tuple of conductor number, fanout angle, conductor-ground width and finite ground width
        :param layer_configuration:
        """
        super().__init__('rect-fanout', name)
        self.port = port
        self.grouping = grouping
        self.layer_configuration = layer_configuration

        e = np.asarray([np.cos(port.orientation), np.sin(port.orientation)])
        e_down = np.asarray([np.cos(port.orientation + np.pi / 2), np.sin(port.orientation + np.pi / 2)])
        e_up = np.asarray([np.cos(port.orientation - np.pi / 2), np.sin(port.orientation - np.pi / 2)])

        self.width_total = self.port.g * 2 + sum(self.port.s) + sum(self.port.w)

        self.widths = [self.port.g] + self.port.w + [self.port.g]
        self.offsets = [-(self.width_total-self.port.g)/2]
        for c in range(len(self.widths)-1):
            self.offsets.append(self.offsets[-1]+self.widths[c]/2+self.port.s[c]+self.widths[c+1]/2)

        #length_down = offsets[grouping[0] + 1] - offsets[1]
        #length_center = offsets[grouping[1] + 1] - offsets[grouping[0] + 1]
        #length_up = offsets[-2] - offsets[grouping[1] + 1]

        self.points_down = [port.position, port.position + e * self.width_total ,
                            port.position + (e+e_down) * self.width_total ]
        self.points_center = [port.position, port.position + e * self.width_total + port.g]
        self.points_up = [port.position, port.position + e * self.width_total ,
                            port.position + (e + e_up) * self.width_total ]

        self.terminals = {'wide': DesignTerminal(position=self.port.position, orientation=self.port.orientation+np.pi,
                            type='mc-cpw', w=self.port.w, s=self.port.s, g=self.port.g, disconnected='short'),
                          'down': DesignTerminal(position=self.points_down[-1], orientation=self.port.orientation+np.pi/2,
                            type='mc-cpw', w=self.port.w, s=self.port.s, g=self.port.g, disconnected='short'),}

    def render(self):
        precision = 0.001
        #ground = self.width_total - (self.offsets[self.grouping[0]] - self.widths[:self.grouping[0]]/2)

        p_down = gdspy.FlexPath(self.points_down, width=self.widths[:self.grouping[0]+1],
                            offset=self.offsets[:self.grouping[0]+1], ends="flush",
                            corners="natural", bend_radius=0, precision=precision,
                            layer=self.layer_configuration.total_layer)

        p_center = gdspy.FlexPath(self.points_center, width=self.widths[self.grouping[0]+1:self.grouping[1]+1],
                            offset=self.offsets[self.grouping[0]+1:self.grouping[1]+1], ends="flush",
                            corners="natural", bend_radius=0, precision=precision,
                            layer=self.layer_configuration.total_layer)

        p_up = gdspy.FlexPath(self.points_center, width=self.widths[self.grouping[1]+1:],
                            offset=self.offsets[self.grouping[1]+1:], ends="flush",
                            corners="natural", bend_radius=0, precision=precision,
                            layer=self.layer_configuration.total_layer)

        p2 = gdspy.FlexPath(self.points_center, width=self.width_total, offset=0, ends='flush',
                            corners='natural', bend_radius=0, precision=precision,
                            layer=self.layer_configuration.restricted_area_layer)

        positive = gdspy.boolean(p_down.to_polygonset(), p_center.to_polygonset(), 'or',
                                 layer=self.layer_configuration.total_layer)
        positive = gdspy.boolean(positive.to_polygonset(), p_up.to_polygonset(), 'or',
                                 layer=self.layer_configuration.total_layer)

        return {'positive': positive, 'restrict': p2.to_polygonset()}

    def add_to_tls(self, tls_instance: tlsim.TLSystem,
                   terminal_mapping: Mapping[str, int], track_changes: bool = True) -> list:
        cl, ll = self.cm()
        line = tlsim.TLCoupler(n=len(self.w),
                               l=self.width_total,  # TODO: get length
                               cl=cl,
                               ll=ll,
                               rl=np.zeros((len(self.w), len(self.w))),
                               gl=np.zeros((len(self.w), len(self.w))))

        if track_changes:
            self.tls_cache.append([line])

        tls_instance.add_element(line, [terminal_mapping['port1'], terminal_mapping['port2']])
        return [line]


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

