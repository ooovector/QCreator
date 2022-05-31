from QCreator.elements import DesignTerminal
from .core import DesignElement, DesignTerminal, LayerConfiguration
import numpy as np
import gdspy
from .. import conformal_mapping as cm
from .. import transmission_line_simulator as tlsim
from typing import List, Tuple, Mapping, Union, Iterable, Dict
from QCreator.elements.cpw import CPWCoupler

class Turn(DesignElement):
    """
    Create cocentric arc circles
    """

    def __init__(self, name: str, position: Tuple[float, float], angle1: float, angle2: float,
                 r: float, w: List, s: List, g: float, layer_configuration: LayerConfiguration,
                 remove_ground = None, restrict_rectangular = None, sharp_corner_rectangular = None):
        """
        Contact pad for bonding the chip to the PCB
        :param name: Design element name
        :param r: radius
        :param w: widths of wires
        :param s: distance between wires
        :param g: width of ground
        :param position: Position of centre on the chip
        :param angle: start and finish angle
        :param remove_ground: determine what part should be without ground can be None, 'both', 'smallest', 'largest' (smallest and largest related to the radius of the removed ground)
        :param restrict_rectangular: for turns that should be restricted (for example, connecting to transmissional line)
        :param sharp_corner_rectangular: for turns that should be expansed by added rectangular (for example, connecting to transmissional line)
        """
        super().__init__('turn', name)
        self.layer_configuration = layer_configuration
        self.tls_cache = []
        radius_for_length=(sum(w)+sum(s))/2+r
        if remove_ground == None or remove_ground == 'largest':
            radius_for_length += g
        position1=np.asarray(position)+np.asarray((np.cos(angle1),np.sin(angle1)))*radius_for_length
        position2=np.asarray(position)+np.asarray((np.cos(angle2),np.sin(angle2)))*radius_for_length
        if angle1<angle2:
            znak=1
        else:
            znak=-1
        orientation1=(angle1+znak*np.pi/2)%(2*np.pi)
        orientation2 = (angle2 - znak * np.pi / 2) % (2 * np.pi)
        self.terminals={'port1': DesignTerminal(position=tuple(position1), orientation=orientation1, type='cpw', w=w, s=s,
                                        g=g, disconnected='short', order = False),
                        'port2': DesignTerminal(position=tuple(position2), orientation=orientation2, type='cpw', w=w  ,s=s,
                                        g=g, disconnected='short')
                        }
        # if len(w)==1 and s[0]==s[1]:
        #     self.terminals['wide1']=DesignTerminal(position=tuple(position1), orientation=orientation1, type='cpw', w=w[0], s=s[0],
        #                                 g=g, disconnected='short', order = False)
        #     self.terminals['wide2'] = DesignTerminal(position=tuple(position2), orientation=orientation2, type='cpw',
        #                                              w=w[0], s=s[0],
        #                                              g=g, disconnected='short')
        self.r = r
        self.w = w
        self.s = s
        self.g = g
        self.position = position
        self.angle1 = angle1
        self.angle2 = angle2
        self.remove_ground = remove_ground
        if restrict_rectangular is not None:
            self.restrict_rectangular = (np.asarray(restrict_rectangular)+np.asarray(position)).tolist()
        else:
            self.restrict_rectangular = None
        if sharp_corner_rectangular is not None:
            self.sharp_corner_rectangular = (np.asarray(sharp_corner_rectangular) + np.asarray(position)).tolist()
        else:
            self.sharp_corner_rectangular = None
        self.length=abs(angle2-angle1)*radius_for_length

    def render(self):
        positive_total=[]
        inverted_total=[]
        radius=self.r
        if self.remove_ground == None or self.remove_ground == 'largest':
            positive_total.append(gdspy.Round(self.position,
                                              radius+self.g,
                                              inner_radius=radius,
                                              initial_angle=self.angle1,
                                              final_angle=self.angle2,
                                              tolerance=0.001,
                                              layer=self.layer_configuration.total_layer
                                              ))
            radius+=self.g
        for i in range(0,len(self.s)):
            inverted_total.append(gdspy.Round(self.position,
                                              self.s[i] + radius,
                                              inner_radius=radius,
                                              initial_angle=self.angle1,
                                              final_angle=self.angle2,
                                              tolerance=0.001,
                                              layer=self.layer_configuration.inverted
                                              ))
            radius+=self.s[i]
            if (i<len(self.w)):
                positive_total.append( gdspy.Round(self.position,
                    self.w[i]+radius,
                    inner_radius=radius,
                    initial_angle=self.angle1,
                    final_angle=self.angle2,
                    tolerance=0.001,
                    layer=self.layer_configuration.total_layer
                ))
                radius += self.w[i]
        if self.remove_ground == None or self.remove_ground == 'smallest':
            positive_total.append(gdspy.Round(self.position,
                                              radius+self.g,
                                              inner_radius=radius,
                                              initial_angle=self.angle1,
                                              final_angle=self.angle2,
                                              tolerance=0.001,
                                              layer=self.layer_configuration.total_layer
                                              ))
            radius+=self.g
        restrict_total = (gdspy.Round(self.position,
                  radius,
                  inner_radius=self.r,
                  initial_angle=self.angle1,
                  final_angle=self.angle2,
                  tolerance=0.001,
                  layer=self.layer_configuration.restricted_area_layer
                  ))
        if self.sharp_corner_rectangular is not None:
            sharp_corner = gdspy.Rectangle(self.sharp_corner_rectangular[0],self.sharp_corner_rectangular[1], layer=self.layer_configuration.restricted_area_layer)
            for i in range(0,len(positive_total)):
                sharp_corner = gdspy.boolean(sharp_corner, positive_total[i] , 'not',
                                         layer=self.layer_configuration.total_layer)
            for i in range(0,len(inverted_total)):
                sharp_corner = gdspy.boolean(sharp_corner, inverted_total[i] , 'not',
                                         layer=self.layer_configuration.inverted)
            positive_total.append(sharp_corner)
        if self.restrict_rectangular is not None:
            restriction = gdspy.Rectangle(self.restrict_rectangular[0],self.restrict_rectangular[1], layer=self.layer_configuration.restricted_area_layer)
            for i in range(0,len(positive_total)):
                positive_total[i] = gdspy.boolean(restriction, positive_total[i] , 'and',
                                         layer=self.layer_configuration.total_layer)
            for i in range(0,len(inverted_total)):
                inverted_total[i] = gdspy.boolean(restriction, inverted_total[i] , 'and',
                                         layer=self.layer_configuration.inverted)
            restrict_total = gdspy.boolean(restriction, restrict_total , 'and',
                                         layer=self.layer_configuration.restricted_area_layer)
        return {'positive': positive_total, 'restrict': restrict_total, 'inverted': inverted_total}

    def cm(self, epsilon):
        cross_section = [self.s[0]]
        for c in range(len(self.w)):
            cross_section.append(self.w[c])
            cross_section.append(self.s[c + 1])

        cl, ll = cm.ConformalMapping(cross_section, epsilon=epsilon).cl_and_Ll()

        if not self.terminals['port1'].order:
            ll, cl = ll[::-1, ::-1], cl[::-1, ::-1]

        return cl, ll

    def get_terminals(self):
        return self.terminals

    def add_to_tls(self, tls_instance: tlsim.TLSystem, terminal_mapping: Mapping[str, int], track_changes: bool = True,
                   cutoff: float = np.inf, epsilon=11.45) -> list:
        cl, ll = self.cm(epsilon)
        #lk = np.asarray(self.kinetic_inductance) / np.asarray(self.w)
        #ll = ll + np.diag(lk)
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
        return "Turn {}".format(self.name)

class straight_CPW_with_different_g(DesignElement):
    """
    Create straight coplanar with different grounds
    """

    def __init__(self, name: str, points: List[Tuple[float, float]],
                 w: List, s: List, g: List, layer_configuration: LayerConfiguration):
        """
        Create straight coplanar with different grounds
        :param name: Design element name
        :param points: start and end points (of line center)
        :param w: widths of wires
        :param s: distance between wires
        :param g: widths of ground
         """
        super().__init__('straight_CPW_with_different_g', name)
        self.layer_configuration = layer_configuration
        self.tls_cache = []
        alpha=np.angle((points[1][0]-points[0][0])+1j*(points[1][1]-points[0][1]))
        self.terminals={'port1': DesignTerminal(position=tuple(points[0]), orientation=alpha, type='cpw', w=w, s=s,
                                        g=g, disconnected='short', order = False),
                        'port2': DesignTerminal(position=tuple(points[1]), orientation=(alpha+np.pi)%(2*np.pi), type='cpw', w=w  ,s=s,
                                        g=g, disconnected='short')
                        }
        self.w = w
        self.s = s
        self.g = g
        self.points = points
        (self.width_total, self.widths, self.offsets, self.holes, self.holes_offsets)=self.widths_offsets()
        self.length= sum((np.asarray(points[1])-np.asarray(points[0]))**2)**0.5

    def widths_offsets(self):
        g=self.g
        w=self.w
        s=self.s
        width_total = sum(g) + sum(s) + sum(w)
        if g[0]==0 and g[1]==0:
            widths = w
        elif g[0]==0:
            widths =w + [g[1]]
        elif g[1]==0:
            widths = [g[0]] + w
        else:
            widths = [g[0]] + w + [g[1]]
        holes = s
        holes_offsets = [-(width_total - 2 * g[0] - s[0]) / 2]
        offsets = [-(width_total - g[0]) / 2]
        for c in range(len(widths) - 1):
            offsets.append(offsets[-1] + widths[c] / 2 + s[c] + widths[c + 1] / 2)
        for c in range(len(w)):
            holes_offsets.append(holes_offsets[-1] + holes[c] / 2 + w[c] + holes[c + 1] / 2)

        return width_total, widths, offsets, holes, holes_offsets

    def render(self):
        start=self.points[0]
        end = self.points[1]
        precision = 0.001
        p1 = gdspy.FlexPath([start, end], width=self.widths, offset=self.offsets,
                            ends='flush',
                            corners='natural', precision=precision,
                            layer=self.layer_configuration.total_layer)
        p2 = gdspy.FlexPath([start, end], width=self.width_total, offset=0, ends='flush',
                            corners='natural', precision=precision,
                            layer=self.layer_configuration.restricted_area_layer)

        p3 = gdspy.FlexPath([start, end], width=self.holes, offset=self.holes_offsets,
                            ends='flush',
                            corners='natural', precision=precision,
                            layer=self.layer_configuration.inverted)

        return {'positive': p1, 'restrict': p2, 'inverted': p3}

    def cm(self, epsilon):
        cross_section = [self.s[0]]
        for c in range(len(self.w)):
            cross_section.append(self.w[c])
            cross_section.append(self.s[c + 1])

        cl, ll = cm.ConformalMapping(cross_section, epsilon=epsilon).cl_and_Ll()

        if not self.terminals['port1'].order:
            ll, cl = ll[::-1, ::-1], cl[::-1, ::-1]

        return cl, ll

    def get_terminals(self):
        return self.terminals

    def add_to_tls(self, tls_instance: tlsim.TLSystem, terminal_mapping: Mapping[str, int], track_changes: bool = True,
                   cutoff: float = np.inf, epsilon=11.45) -> list:
        cl, ll = self.cm(epsilon)
        #lk = np.asarray(self.kinetic_inductance) / np.asarray(self.w)
        #ll = ll + np.diag(lk)
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
        return "straight_CPW_with_different_g {}".format(self.name)

class MultiOpenEnd(DesignElement):
    """
    Делает штуку, которая на часть линий ставит открытый конец, а часть линий продолжает
    """

    def __init__(self, name: str, position: Tuple, w: List, s: List, g: Union[List, float],
                 orientation: float, continue_lines: List,
                 layer_configuration: LayerConfiguration, h1: float = 20., h2: float = 20, ):
        """
        Create straight coplanar with different grounds
        :param name: Design element name
        :param position: initial position of an element
        :param w: widths of wires
        :param s: distance between wires
        :param g: widths of ground
        :param orientation: orientation of an element
        :param continue_lines: which lines should be continued
        :param h1: first length of open end
        :param h2: second length of open end
         """
        super().__init__('MultiOpenEnd', name)
        self.layer_configuration = layer_configuration
        self.tls_cache = []
        self.w = w
        self.s = s
        if type(g) == float:
            self.g = [g, g]
        else:
            self.g = g
        self.position = position
        self.orientation = orientation
        self.continue_lines = continue_lines
        self.h1 = h1
        self.h2 = h2
        # я буду рисовать две прямые копланарные линии с h1 и h2. С h1 - это продолжение линий, с h2 -  "закрывание" открытого конца.
        # здесь я определяю w,s,g для этих линий исходя из того, какие должны быть продолжены.
        self.w1 = list(np.asarray(w)[continue_lines])
        self.s1 = []
        self.w2 = []
        self.s2 = []
        w2_wire = self.g[0]
        s1 = 0
        for i in range(0,len(w)):
            if i in continue_lines:
                s1 += s[i]
                self.s1.append(s1)
                s1 = 0
                if w2_wire!=0:
                    self.w2.append(w2_wire)
                w2_wire = w[i]
                self.w2.append(w2_wire)
                w2_wire = 0
                self.s2.append(s[i])
                last_line_existence=True
            else:
                s1 += w[i] + s[i]
                if i-1 in continue_lines:
                    w2_wire += w[i]
                    self.s2.append(s[i])
                else:
                    w2_wire += s[i] + w[i]
                last_line_existence=False
        if last_line_existence:
            self.s1.append(s[-1])
            self.s2.append(s[-1])
            self.w2.append(self.g[1])
        else:
            s1+=s[-1]
            self.s1.append(s1)
            w2_wire+=s[-1]+self.g[1]
            self.w2.append(w2_wire)

        r = (self.h2 + self.h1) / 2
        start = np.asarray(self.position) - r * np.asarray([np.cos(self.orientation), np.sin(self.orientation)])
        end = np.asarray(self.position) + r * np.asarray([np.cos(self.orientation), np.sin(self.orientation)])
        offsets = self.offsets()
        self.terminals = {'port1': DesignTerminal(position=tuple(start), orientation=self.orientation, type='cpw', w=self.w, s=self.s,
                                                  g=self.g, disconnected='short', order=False)
                          }
        for i in range(0,len(self.w1)):
            terminal_position=tuple(end+offsets[i]*np.asarray([np.cos(self.orientation+np.pi/2),np.sin(self.orientation+np.pi/2)]))
            self.terminals['port2_'+str(i)]=DesignTerminal(position=terminal_position, orientation=(self.orientation + np.pi) % (2 * np.pi),
                                                          type='cpw', w=[self.w1[i]], s=[self.s1[i],self.s1[i+1]],
                                                          g=[self.w2[i],self.w2[i+1]], disconnected='short',)


    def offsets(self):
        g=self.g
        w=self.w1
        s=self.s1
        width_total = sum(g) + sum(s) + sum(w)
        offsets = [-(width_total - w[0]) / 2]
        for c in range(len(w) - 1):
            offsets.append(offsets[-1] + w[c] / 2 + s[c] + w[c + 1] / 2)
        return offsets

    def render(self):
        r=(self.h2+self.h1)/2
        rc=(self.h1-self.h2)/2
        start=np.asarray(self.position) - r * np.asarray([np.cos(self.orientation), np.sin(self.orientation)])
        end = np.asarray(self.position) + r * np.asarray([np.cos(self.orientation), np.sin(self.orientation)])
        middle = np.asarray(self.position) + rc * np.asarray([np.cos(self.orientation), np.sin(self.orientation)])
        line_1 = straight_CPW_with_different_g(name='line_1',
                                      points=[tuple(start), tuple(middle)],
                                      w=self.w1,
                                      s=self.s1,
                                      g=self.g,
                                      layer_configuration=self.layer_configuration)
        line_2 = straight_CPW_with_different_g(name='line_2',
                                               points=[tuple(middle), tuple(end)],
                                               w=self.w2[1:-1],
                                               s=self.s2,
                                               g=[self.w2[0],self.w2[-1]],
                                               layer_configuration=self.layer_configuration)
        area_1 = line_1.render()
        area_2 = line_2.render()
        positive = gdspy.boolean(area_1['positive'], area_2['positive'], 'or', layer=self.layer_configuration.total_layer)
        restrict = gdspy.boolean(area_1['restrict'], area_2['restrict'], 'or',layer=self.layer_configuration.restricted_area_layer)
        inverted = gdspy.boolean(area_1['inverted'], area_2['inverted'], 'or',layer=self.layer_configuration.inverted)
        return {'positive': positive, 'restrict': restrict, 'inverted': inverted}

    def __repr__(self):
        return "MultiOpenEnd {}".format(self.name)

    def get_terminals(self):
        return self.terminals


def add_to_tls(self, tls_instance: tlsim.TLSystem, terminal_mapping: Mapping[str, int], track_changes: bool = True,
               cutoff: float = np.inf, epsilon=11.45) -> list:
    if len(self.w) == 1:
        cache = []
        capacitance_value = 1e-15 * 20 * 0
        capacitor = tlsim.Capacitor(capacitance_value, 'open_end')
        cache.append(capacitor)
        tls_instance.add_element(capacitor,
                                 [terminal_mapping['port1'], 0],[terminal_mapping['port2_0'], 0])  # tlsim.TLSystem.add_element(name, nodes)

        return cache

    elif len(self.w1) > 1:
        cache = []
        for conductor_id in range(self.w1):  # loop over all conductors
            capacitance_value = 20e-15*0
            capacitor = tlsim.Capacitor(capacitance_value, 'open_end')
            p = [terminal_mapping[('port1', conductor_id)]]
            for i in range(0, len(self.w1)):
                p.append(terminal_mapping[('port2_'+str(i), 0)])
            tls_instance.add_element(capacitor, p)  # tlsim.TLSystem.add_element(name, nodes)
            cache.append(capacitor)
        return cache

    if track_changes:
        self.tls_cache.append(cache)

class Short(DesignElement):
    """
    делает закоротку на один провод, ничего не рисуя
    """

    def __init__(self, name: str, position: Tuple, orientation: float, g: float, layer_configuration: LayerConfiguration):
        """
        Create straight coplanar with different grounds
        :param name: Design element name
        :param position: position
        :param angle: start and finish angle start line has no more wires than finish
        :param r: radius
        :param w: widths of wires
        :param s: distance between wires
        :param g: widths of ground
         """
        super().__init__('Short', name)
        self.layer_configuration = layer_configuration
        self.tls_cache = []
        self.terminals={'port': DesignTerminal(position=position, orientation=orientation, type='cpw', w=0, s=0,
                                        g=g, disconnected='short', order = False),
                        }
        self.g = g
        self.position = position
        self.orientation = orientation

    def render(self):
        return {}


    def get_terminals(self):
        return self.terminals

    def add_to_tls(self, tls_instance: tlsim.TLSystem, terminal_mapping: Mapping[str, int], track_changes: bool = True,
                   cutoff: float = np.inf, epsilon=11.45) -> list:
        cache = []
        zero_resistor = tlsim.Resistor(r=0, name=self.name)
        tls_instance.add_element(zero_resistor, [0, terminal_mapping[('port')]])
        cache.append(zero_resistor)
        if track_changes:
            self.tls_cache.append(cache)
        return cache
    def __repr__(self):
        return "Short {}".format(self.name)