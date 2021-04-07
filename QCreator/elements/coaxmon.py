from .core import DesignElement, LayerConfiguration, DesignTerminal
from .. import conformal_mapping as cm
from .. import transmission_line_simulator as tlsim
import gdspy
import numpy as np
from typing import List, Tuple, Mapping, Dict
from . import squid3JJ #TODO make this qubit class suitable for any squid types
from copy import deepcopy

class Coaxmon(DesignElement):
    """
    Coaxmon consists of several parts:
    1) Central part - central circuit
    params: center = center of the circle/qubit, center_radius = radius of the center part=qubit
    2) Couplers - 5 couplers, 4 are used to connect two-qubit couplers. 1 for a flux line or a microwave line.
    They are arcs, you can read more about them in CoaxmonCoupler description
    params: inner_couplers_radius,outer_couplers_radius
    3) Ground = grounded 6th coupler which is used for a flux line and a microwave line
    params: inner_ground_radius,outer_ground_radius
    4)layer_configuration
    5)Couplers - coupler classes
    6) jj_params - parameters of the SQUID which here is 3JJ SQUID.#TODO add more information
    """
    def __init__(self, name: str, center: Tuple[float, float],
                 center_radius: float, inner_couplers_radius: float,
                 outer_couplers_radius: float, inner_ground_radius: float, outer_ground_radius: float,
                 layer_configuration: LayerConfiguration, Couplers, jj_params: Dict, transformations: Dict,
                 calculate_capacitance: False, JJ_type=None):
        super().__init__(type='qubit', name=name)
        self.JJ_type=JJ_type
        #qubit parameters
        self.transformations = transformations# to mirror the structure
        self.center = center
        self.R1 = center_radius
        self.R2 = inner_couplers_radius
        self.R3 = outer_couplers_radius
        self.R4 = inner_ground_radius
        self.outer_ground = outer_ground_radius
        #layers
        self.layer_configuration = layer_configuration
        #couplers
        self.couplers = Couplers
        # there is one special coupler - for the fluxline
        for coupler in self.couplers:
            if coupler.coupler_type == 'grounded':
                self.grounded = coupler
        # JJs and fluxline
        self.JJ_params = jj_params
        self.JJ = None
        self.core = self.grounded.w
        self.gap = self.grounded.g
        self.ground = self.grounded.w
        # qubit terminals
        self.terminals = {#'coupler0': None,
                          #'coupler1': None,
                          #'coupler2': None,
                          #'coupler3': None,
                          #'coupler4': None,
                          #'flux': None,
                          'qubit': None}
        # model evaluation
        self.calculate_capacitance = calculate_capacitance
        self.tls_cache = []
        self.L1 = 60e-9#20nHr
        self.L2 = 20e-9
        self.M = 12e-12
        self.C = {'coupler0': None,
                  'coupler1': None,
                  'coupler2': None,
                  'coupler3': None,
                  'coupler4': None,
                  'qubit': None}
        self.layers = []

    def render(self):
        """
        This function draw everything: core circle, couplers, JJs
        """
        qubit_cap_parts=[]
        ground = gdspy.Round(self.center, self.outer_ground, self.R4, initial_angle=0, final_angle=2 * np.pi)
        # restricted area for a future grid lines
        result_restricted = gdspy.Round(self.center, self.outer_ground,
                                        layer=self.layer_configuration.restricted_area_layer)
        core = gdspy.Round(self.center, self.R1, inner_radius=0, initial_angle=0, final_angle=2 * np.pi)
        qubit_cap_parts.append(gdspy.boolean(core,core,'or',layer=9))  #TODO: fix this fundamental constant for capacitance layers
        self.layers.append(9)
        result = gdspy.boolean(ground, core, 'or', layer=self.layer_configuration.total_layer)
        # add couplers
        last_step_cap = [core] # to get a correct structure for capacitances
        self.layers.append(self.layer_configuration.total_layer)
        if len(self.couplers) != 0:
            for id, coupler in enumerate(self.couplers):
                coupler_parts = coupler.render(self.center, self.R2, self.R3, self.R4, self.outer_ground)
                if 'remove' in coupler_parts:
                    result = gdspy.boolean(result, coupler_parts['remove'], 'not',
                                           layer=self.layer_configuration.total_layer)
                result = gdspy.boolean(coupler_parts['positive'], result, 'or',
                                       layer=self.layer_configuration.total_layer)
                if coupler.coupler_type == 'coupler':
                    qubit_cap_parts.append(gdspy.boolean(coupler.result_coupler, coupler.result_coupler, 'or',layer=10+id))
                    self.layers.append(10+id)
                    last_step_cap.append(coupler.result_coupler)
        qubit_cap_parts.append(gdspy.boolean(result,last_step_cap,'not'))
        # add JJs
        if self.JJ_params is not None:
            self.JJ_coordinates = (self.center[0] + self.R1 * np.cos(self.JJ_params['angle_qubit']),
                                   self.center[1] + self.R1 * np.sin(self.JJ_params['angle_qubit']))
            JJ, rect = self.generate_JJ() #TODO change it in a new manner, probably one day
            result = gdspy.boolean(result, rect, 'or')
            # add flux line
            flux_line = self.connection_to_ground(self.JJ_params['length'], self.JJ_params['width'])
            result = gdspy.boolean(result, flux_line['remove'], 'not')
            result = gdspy.boolean(result, flux_line['positive'], 'or', layer=self.layer_configuration.total_layer)

        # set terminals for couplers
        self.set_terminals()
        qubit=deepcopy(result)
        if self.calculate_capacitance is False:
            qubit_cap_parts = None
            qubit = None
        if 'mirror' in self.transformations:
            return {'positive': result.mirror(self.transformations['mirror'][0], self.transformations['mirror'][1]),
                    'restrict': result_restricted,
                    'qubit': qubit.mirror(self.transformations['mirror'][0], self.transformations['mirror'][1]) if qubit is not None else None,
                    'qubit_cap': qubit_cap_parts,
                    'JJ': JJ.mirror(self.transformations['mirror'][0], self.transformations['mirror'][1]),
                    }
        if 'rotate' in self.transformations:
            return {'positive': result.rotate(self.transformations['rotate'][0], self.transformations['rotate'][1]),
                    'restrict': result_restricted,
                    'qubit': qubit.rotate(self.transformations['rotate'][0], self.transformations['rotate'][1]) if qubit is not None else None,
                    'qubit_cap': qubit_cap_parts,
                    'JJ': JJ.rotate(self.transformations['rotate'][0], self.transformations['rotate'][1]),
                    }
        elif self.transformations == {}:
            return {'positive': result,
                    'restrict': result_restricted,
                    'qubit': qubit,
                    'qubit_cap': qubit_cap_parts,
                    'JJ': JJ,
                    }

    def set_terminals(self):
        for id, coupler in enumerate(self.couplers):
            if 'mirror' in self.transformations:
                if coupler.connection is not None:
                    coupler_connection = mirror_point(coupler.connection, self.transformations['mirror'][0], self.transformations['mirror'][1])
                    qubit_center = mirror_point(deepcopy(self.center), self.transformations['mirror'][0], self.transformations['mirror'][1])
                    coupler_phi = np.arctan2(coupler_connection[1]-qubit_center[1], coupler_connection[0]-qubit_center[0])+np.pi
            if 'rotate' in self.transformations:
                if coupler.connection is not None:
                    coupler_connection = rotate_point(coupler.connection, self.transformations['rotate'][0], self.transformations['rotate'][1])
                    qubit_center = rotate_point(deepcopy(self.center), self.transformations['rotate'][0], self.transformations['rotate'][1])
                    coupler_phi = np.arctan2(coupler_connection[1]-qubit_center[1], coupler_connection[0]-qubit_center[0])+ np.pi
            if self.transformations == {}:
                coupler_connection = coupler.connection
                coupler_phi = coupler.phi*np.pi + np.pi
            if coupler.connection is not None:
                self.terminals['coupler'+str(id)] = DesignTerminal(tuple(coupler_connection),
                                                                   coupler_phi, g=coupler.g, s=coupler.s,
                                                                w=coupler.w, type='cpw')
        return True
    def get_terminals(self):
        return self.terminals

    def generate_JJ(self):
        if self.JJ_type == 'JJ_2':
            self.JJ = squid3JJ.JJ_2(self.JJ_coordinates[0], self.JJ_coordinates[1],
                                    self.JJ_params['a1'], self.JJ_params['a2'],
                                    self.JJ_params['b1'], self.JJ_params['b2'],
                                    self.JJ_params['c1'], self.JJ_params['c2'])
        else:
            self.JJ = squid3JJ.JJ_2(self.JJ_coordinates[0], self.JJ_coordinates[1],
                                    self.JJ_params['a1'], self.JJ_params['a2'],
                                    self.JJ_params['b1'], self.JJ_params['b2'],
                                    self.JJ_params['c1'], self.JJ_params['c2'], add_JJ=True)
        result = self.JJ.generate_jj()
        result = gdspy.boolean(result, result, 'or', layer=self.layer_configuration.jj_layer)
        angle = self.JJ_params['angle_JJ']
        result.rotate(angle, (self.JJ_coordinates[0], self.JJ_coordinates[1]))
        rect = gdspy.Rectangle((self.JJ_coordinates[0] - self.JJ.contact_pad_a_outer / 2,
                                self.JJ_coordinates[1] + self.JJ.contact_pad_b_outer),
                               (self.JJ_coordinates[0] + self.JJ.contact_pad_a_outer / 2,
                                self.JJ_coordinates[1] - self.JJ.contact_pad_b_outer),
                               layer=self.layer_configuration.total_layer)
        rect.rotate(angle, (self.JJ_coordinates[0], self.JJ_coordinates[1]))
        return result, rect

    def connection_to_ground(self, length, width):
        """
        This function generate a connection from JJ rectangulars to a flux line output. Should be changed if you want
        to use another type of JJ or a flux line
        """
        result = None
        for point in [self.JJ.rect1, self.JJ.rect2]:
            orientation = np.arctan2(-(self.center[1] - (point[1]-length)), -(self.center[0] - point[0]))
            points =[point, (point[0], point[1] - length),
                 (self.center[0]+self.R2*np.cos(orientation), self.center[1]+self.R2*np.sin(orientation))]
            path = gdspy.FlexPath(deepcopy(points), width, offset=0, layer=self.layer_configuration.total_layer)
            result = gdspy.boolean(path, result, 'or', layer=self.layer_configuration.total_layer)
        orientation = np.arctan2(-(self.center[1] - (self.JJ.rect1[1] - length)), -(self.center[0] - self.JJ.rect1[0]))
        #to fix rounding bug
        bug=5
        connection = (self.center[0]+(self.R2-bug)*np.cos(orientation),self.center[1]+(self.R2-bug)*np.sin(orientation))
        # add cpw from
        flux_line_output=(connection[0]+(self.outer_ground-self.R2+bug)*np.cos(orientation),
                              connection[1]+(self.outer_ground-self.R2+bug)*np.sin(orientation))
        # to fix rounding bug
        bug = 1
        flux_line_output_connection = (flux_line_output[0]+bug*np.cos(np.pi+orientation),
                                       flux_line_output[1]+bug*np.sin(np.pi+orientation))
        remove = gdspy.FlexPath(deepcopy([connection,flux_line_output]), [self.gap, self.gap], offset=[-self.core/2-self.gap/2,self.core/2+self.gap/2])
        if 'mirror' in self.transformations:
            flux_line_output_connection = mirror_point(flux_line_output_connection, self.transformations['mirror'][0],
                                                  self.transformations['mirror'][1])
            qubit_center = mirror_point(deepcopy(self.center), self.transformations['mirror'][0],
                                        self.transformations['mirror'][1])

            orientation = np.arctan2(flux_line_output_connection[1] - qubit_center[1],
                                     flux_line_output_connection[0] - qubit_center[0])+np.pi
        if 'rotate' in self.transformations:
            flux_line_output_connection = rotate_point(flux_line_output_connection, self.transformations['rotate'][0],
                                                  self.transformations['rotate'][1])
            qubit_center = rotate_point(deepcopy(self.center), self.transformations['rotate'][0],
                                        self.transformations['rotate'][1])
            orientation = np.arctan2(flux_line_output_connection[1] - qubit_center[1],
                                     flux_line_output_connection[0] - qubit_center[0])+np.pi
        if self.transformations == {}:
            orientation=orientation+np.pi
        self.terminals['flux'] = DesignTerminal(flux_line_output_connection, orientation, g=self.grounded.w, s=self.grounded.g,
                                                     w=self.grounded.w, type='cpw')
        return {'positive': result,
                'remove': remove,
                }
    def add_to_tls(self, tls_instance: tlsim.TLSystem, terminal_mapping: dict,
                   track_changes: bool = True) -> list:
        #scaling factor for C
        scal_C = 1e-15
        JJ1 = tlsim.Inductor(self.L1)
        JJ2 = tlsim.Inductor(self.L2)
        M = tlsim.Inductor(self.M)
        C = tlsim.Capacitor(c=self.C['qubit']*scal_C, name=self.name+' qubit-ground')
        tls_instance.add_element(JJ1, [0, terminal_mapping['qubit']])
        tls_instance.add_element(JJ2, [terminal_mapping['flux'], terminal_mapping['qubit']])
        tls_instance.add_element(M, [0, terminal_mapping['flux']])
        tls_instance.add_element(C, [0, terminal_mapping['qubit']])
        mut_cap = []
        cap_g = []
        for id, coupler in enumerate(self.couplers):
            if coupler.coupler_type == 'coupler':
                c0 = tlsim.Capacitor(c=self.C['coupler'+str(id)][1]*scal_C, name=self.name+' qubit-coupler'+str(id))
                c0g = tlsim.Capacitor(c=self.C['coupler'+str(id)][0]*scal_C, name=self.name+' coupler'+str(id)+'-ground')
                tls_instance.add_element(c0, [terminal_mapping['qubit'], terminal_mapping['coupler'+str(id)]])
                tls_instance.add_element(c0g, [terminal_mapping['coupler'+str(id)], 0])
                mut_cap.append(c0)
                cap_g.append(c0g)
            # elif coupler.coupler_type =='grounded':
            #     tls_instance.add_element(tlsim.Short(), [terminal_mapping['flux line'], 0])

        if track_changes:
            self.tls_cache.append([JJ1, JJ2, C]+mut_cap+cap_g)
        return [JJ1, JJ2, C]+mut_cap+cap_g

class CoaxmonCoupler:
    """
    This class represents a coupler for a coaxmon qubit.
    There are several parameters:
    1) arc_start - the starting angle of the coupler arc in terms of pi
    2) arc_finish - the ending angle of the coupler arc in terms of pi
    3) phi - the angle of the coupler's rectangular connector to other structures in terms of pi
    4) coupler_type - it shows whether the coupler is used for fluxline and should be "grounded"
    or it is used as a "coupler", or "None" if it should be not connected to other structures
    5) w - the width of the core of the coupler's rectangular connector to other structures
    6) g - the gap of the coupler's rectangular connector to other structures
    """
    def __init__(self, arc_start, arc_finish, phi, coupler_type=None, w=None, g=None,s=None):
        self.arc_start = arc_start
        self.arc_finish = arc_finish
        self.phi = phi
        self.w = w
        self.g = g
        self.s = s
        self.coupler_type = coupler_type
        self.connection = None
        self.result_coupler = None

    def render(self, center, r_init, r_final, rect_end, outer_ground):
        remove=None
        if self.coupler_type is None:
            arc = gdspy.Round(center, r_init, r_final,
                              initial_angle=self.arc_start * np.pi, final_angle=self.arc_finish * np.pi)
            bug = 5# to fix intersection bug with the circle
            rect = gdspy.Rectangle((center[0] + r_final - bug, center[1] - self.w / 2),
                                   (center[0] + rect_end + bug, center[1] + self.w / 2))
            rect.rotate(self.phi * np.pi, center)
            result = gdspy.boolean(arc, rect, 'or')
        elif self.coupler_type == 'grounded':
            result = gdspy.Round(center, r_init, outer_ground,
                                 initial_angle=self.arc_start * np.pi, final_angle=self.arc_finish * np.pi)
        elif self.coupler_type == 'coupler':
            arc = gdspy.Round(center, r_init, r_final,
                              initial_angle=self.arc_start * np.pi, final_angle=self.arc_finish * np.pi)
            rect = gdspy.Rectangle((center[0] + r_final-1, center[1] - self.w / 2),# 1 to fix rounding bug
                                   (center[0] + rect_end, center[1] + self.w / 2))
            rect.rotate(self.phi * np.pi, center)
            self.connection = (center[0] + rect_end * np.cos(self.phi * np.pi),
                               center[1] + rect_end * np.sin(self.phi * np.pi))
            part_to_remove = gdspy.Rectangle((center[0] + r_final, center[1] - self.w - self.g/2),
                                           (center[0] + outer_ground, center[1] + self.w  + self.g/2))
            remove = part_to_remove.rotate(self.phi * np.pi, center)
            result = gdspy.boolean(arc, rect, 'or')
            self.result_coupler = result
            return {
                'positive': result,
                'remove': remove
            }
        return {
            'positive': result,
        }


def mirror_point(point,ref1,ref2):
    """
       Mirror a point by a given line specified by 2 points ref1 and ref2.
    """
    [x1, y1] =ref1
    [x2, y2] =ref2

    dx = x2-x1
    dy = y2-y1
    a = (dx * dx - dy * dy) / (dx * dx + dy * dy)
    b = 2 * dx * dy / (dx * dx + dy * dy)
    x2 = round(a * (point[0] - x1) + b * (point[1] - y1) + x1)
    y2 = round(b * (point[0] - x1) - a * (point[1] - y1) + y1)
    return x2, y2


def rotate_point(point, angle, origin):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point
    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return qx, qy