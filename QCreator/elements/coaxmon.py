from .core import DesignElement, LayerConfiguration, DesignTerminal
from .. import conformal_mapping as cm
from .. import transmission_line_simulator as tlsim
import gdspy
import numpy as np
from typing import List, Tuple, Mapping, Dict
from . import squid3JJ#TODO make this qubit class suitable for any squid types
from copy import deepcopy
from .functions import find_normal_point

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
                 calculate_capacitance: False, third_JJ=False, hole_in_squid_pad=True,
                 JJ_pad_connection_shift=False, draw_bandages=False, coil_type='old', import_jj=False, file_jj=None, cell_jj=None):
        super().__init__(type='qubit', name=name)
        self.third_JJ = third_JJ
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
        self.gap = self.grounded.s
        self.ground = self.grounded.g
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
        self.C = {'coupler0': None,
                  'coupler1': None,
                  'coupler2': None,
                  'coupler3': None,
                  'coupler4': None,
                  'qubit': None}
        self.layers = []
        self.hole_in_squid_pad = hole_in_squid_pad
        self.JJ_pad_connection_shift = JJ_pad_connection_shift
        self.draw_bandages = draw_bandages
        self.coil_type = coil_type
        self.import_jj = import_jj
        self.file_jj = file_jj
        self.cell_jj = cell_jj

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

            # TODO change it in a new manner, probably one day
            if self.import_jj:
                JJ, rect = self.import_JJ(third_JJ=self.third_JJ, file_name=self.file_jj, cell_name=self.cell_jj)
            else:
                JJ, rect = self.generate_JJ()
            result = gdspy.boolean(result, rect, 'or')
            # add flux line
            flux_line = self.connection_to_ground(self.JJ_params['length'], self.JJ_params['width'],
                                                  JJ_pad_connection_shift=self.JJ_pad_connection_shift, coil=self.coil_type)
            result = gdspy.boolean(result, flux_line['remove'], 'not')
            result = gdspy.boolean(result, flux_line['positive'], 'or', layer=self.layer_configuration.total_layer)
        if self.draw_bandages:
            bandages = self.add_bandages()
        else:
            bandages = None
        # set terminals for couplers
        self.set_terminals()
        qubit=deepcopy(result)
        if self.calculate_capacitance is False:
            qubit_cap_parts = None
            qubit = None
        if 'mirror' in self.transformations:
            render_result = {'positive': result.mirror(self.transformations['mirror'][0], self.transformations['mirror'][1]),
                    'restrict': result_restricted.mirror(self.transformations['mirror'][0], self.transformations['mirror'][1]),
                    'qubit': qubit.mirror(self.transformations['mirror'][0], self.transformations['mirror'][1]) if qubit is not None else None,
                    'qubit_cap': qubit_cap_parts,
                    'JJ': JJ.mirror(self.transformations['mirror'][0], self.transformations['mirror'][1])}
            if self.draw_bandages:
                render_result.update({'bandages':bandages.mirror(self.transformations['mirror'][0], self.transformations['mirror'][1])})
            return render_result

        if 'rotate' in self.transformations:
            render_result =  {'positive': result.rotate(self.transformations['rotate'][0], self.transformations['rotate'][1]),
                    'restrict': result_restricted.rotate(self.transformations['rotate'][0], self.transformations['rotate'][1]),
                    'qubit': qubit.rotate(self.transformations['rotate'][0], self.transformations['rotate'][1]) if qubit is not None else None,
                    'qubit_cap': qubit_cap_parts,
                    'JJ': JJ.rotate(self.transformations['rotate'][0], self.transformations['rotate'][1])}
            if self.draw_bandages:
                render_result.update(
                    {'bandages': bandages.rotate(self.transformations['rotate'][0], self.transformations['rotate'][1])})
            return render_result
        elif self.transformations == {}:
            render_result =  {'positive': result,
                    'restrict': result_restricted,
                    'qubit': qubit,
                    'qubit_cap': qubit_cap_parts,
                    'JJ': JJ}
            if self.draw_bandages:
                render_result.update({'bandages':bandages})
            return render_result

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

        if self.third_JJ == True:
            self.terminals['squid_intermediate'] = None
        return True
    def get_terminals(self):
        return self.terminals

    def generate_JJ(self):
        # polygonset
        self.JJ = squid3JJ.JJ_2(self.JJ_coordinates[0], self.JJ_coordinates[1],
                                self.JJ_params['a1'],
                                self.JJ_params['jj1_width'], self.JJ_params['jj1_height'],
                                self.JJ_params['jj2_width'], self.JJ_params['jj2_height'],
                                self.JJ_params['jj3_width'], self.JJ_params['jj3_height'],
                                self.JJ_params['c2'], add_JJ=self.third_JJ,
                                hole_in_squid_pad=self.hole_in_squid_pad)
        result = self.JJ.generate_jj()
        result = gdspy.boolean(result, result, 'or', layer=self.layer_configuration.jj_layer)
        angle = self.JJ_params['angle_JJ']
        if self.JJ_pad_connection_shift:
            connection_shift = self.JJ.contact_pad_b_outer/2
        else:
            connection_shift = 0
        result.rotate(angle, (self.JJ_coordinates[0], self.JJ_coordinates[1]))
        rect = gdspy.Rectangle((self.JJ_coordinates[0] - self.JJ.contact_pad_a_outer / 2,
                                self.JJ_coordinates[1]+connection_shift + self.JJ.contact_pad_b_outer),
                               (self.JJ_coordinates[0] + self.JJ.contact_pad_a_outer / 2,
                                self.JJ_coordinates[1]+connection_shift - self.JJ.contact_pad_b_outer),
                               layer=self.layer_configuration.total_layer)
        rect.rotate(angle, (self.JJ_coordinates[0], self.JJ_coordinates[1]))
        return result, rect

    def import_JJ(self, third_JJ, file_name, cell_name):
        """
        Import SQUID topology for the qubit from GDS file. Transmission line model it defined be user itself.
        """
        # TODO: how to make it better? It is temporary solution
        self.JJ = squid3JJ.JJ_2(self.JJ_coordinates[0], self.JJ_coordinates[1],
                                self.JJ_params['a1'],
                                self.JJ_params['jj1_width'], self.JJ_params['jj1_height'],
                                self.JJ_params['jj2_width'], self.JJ_params['jj2_height'],
                                self.JJ_params['jj3_width'], self.JJ_params['jj3_height'],
                                self.JJ_params['c2'], add_JJ=self.third_JJ,
                                hole_in_squid_pad=self.hole_in_squid_pad)
        self.JJ.generate_jj()

        import os
        self.third_JJ = third_JJ
        path = os.getcwd()
        path_for_file = path[:path.rindex('QCreator')] + 'QCreator\QCreator\elements\junctions' + file_name
        # import cell
        squid = gdspy.GdsLibrary().read_gds(infile=path_for_file).cells[cell_name].remove_polygons(lambda pts, layer,
                                                                                                          datatype: layer not in [
            self.layer_configuration.jj_layer])
        # convert to polygonset
        squid_polygons = []
        squid_layers = []
        for p_id, p in enumerate(squid.polygons):
            points = p.polygons[0]
            l = p.layers[0]
            squid_polygons.append(points)
            squid_layers.append(l)
        squid_polygonset = gdspy.PolygonSet(squid_polygons, layer=self.layer_configuration.jj_layer)

        # TODO: how to shift this squid?
        squid_polygonset.translate(self.JJ_coordinates[0], self.JJ_coordinates[1])

        angle = self.JJ_params['angle_JJ']
        squid_polygonset.rotate(angle, (self.JJ_coordinates[0], self.JJ_coordinates[1]))
        connection_shift = 0
        rect = None
        rect = gdspy.Rectangle((self.JJ_coordinates[0] - self.JJ.contact_pad_a_outer / 2,
                                self.JJ_coordinates[1] + connection_shift + self.JJ.contact_pad_b_outer),
                               (self.JJ_coordinates[0] + self.JJ.contact_pad_a_outer / 2,
                                self.JJ_coordinates[1] + connection_shift - self.JJ.contact_pad_b_outer),
                               layer=self.layer_configuration.total_layer)
        rect.rotate(angle, (self.JJ_coordinates[0], self.JJ_coordinates[1]))
        return squid_polygonset, rect

    def connection_to_ground(self, length, width, JJ_pad_connection_shift=False, coil='old'):
        """
        This function generate a connection from JJ rectangulars to a flux line output. Should be changed if you want
        to use another type of JJ or a flux line
        """
        result = None
        remove = None

        for point in [self.JJ.rect1, self.JJ.rect2]:
            orientation = np.arctan2(-(self.center[1] - (point[1] - length)), -(self.center[0] - point[0]))
            if JJ_pad_connection_shift:
                connection_shift = self.JJ.rect_size_b / 2
            else:
                connection_shift = 0
            points = [(point[0], point[1] - connection_shift), (point[0], point[1] - length),
                      (self.center[0] + self.R2 * np.cos(orientation), self.center[1] + self.R2 * np.sin(orientation))]
            path = gdspy.FlexPath(deepcopy(points), width, offset=0, layer=self.layer_configuration.total_layer)
            result = gdspy.boolean(path, result, 'or', layer=self.layer_configuration.total_layer)
        orientation = np.arctan2(-(self.center[1] - (self.JJ.rect1[1] - length)), -(self.center[0] - self.JJ.rect1[0]))
        if coil == 'old':
            bug = 5
            coil_shift = 17
            connection = (self.center[0] + (self.R2 - bug + coil_shift) * np.cos(orientation),
                          self.center[1] + (self.R2 - bug + coil_shift) * np.sin(orientation))
            # add cpw from
            flux_line_output = (connection[0] + (self.outer_ground - self.R2 - coil_shift + bug) * np.cos(orientation),
                                connection[1] + (self.outer_ground - self.R2 - coil_shift + bug) * np.sin(orientation))
            # to fix rounding bug
            bug = 1
            connection_0 = find_normal_point(connection, flux_line_output, 20)
            flux_line_output_connection = (flux_line_output[0] + bug * np.cos(np.pi + orientation),
                                           flux_line_output[1] + bug * np.sin(np.pi + orientation))
            remove = gdspy.FlexPath(deepcopy([connection_0, connection, flux_line_output]), [self.gap, self.gap],
                                    offset=[-self.core / 2 - self.gap / 2, self.core / 2 + self.gap / 2])
            remove_extra = gdspy.FlexPath(deepcopy([find_normal_point(connection, flux_line_output, 25),
                                                    find_normal_point(connection, flux_line_output, 15, reverse=True)]),
                                          [self.gap], offset=[-self.core / 2 - self.gap / 2])
            remove = gdspy.boolean(remove_extra, remove, 'or', layer=self.layer_configuration.total_layer)

        elif coil == 'tzar-coil':
            middle_shift = 30
            bug = 5
            connection = (self.center[0] + (self.R2 - bug) * np.cos(orientation),
                          self.center[1] + (self.R2 - bug) * np.sin(orientation))

            flux_line_output = (connection[0] + (self.outer_ground - self.R2 + bug) * np.cos(orientation),
                                connection[1] + (self.outer_ground - self.R2 + bug) * np.sin(orientation))

            flux_line_output_connection = (flux_line_output[0] + bug * np.cos(np.pi + orientation),
                                           flux_line_output[1] + bug * np.sin(np.pi + orientation))

            path1 = gdspy.FlexPath(deepcopy([connection, flux_line_output]), [self.gap],
                                    offset=[-self.core / 2 - self.gap / 2])
            remove = gdspy.boolean(path1, remove, 'or', layer=self.layer_configuration.total_layer)

            middle_point = (connection[0] + middle_shift * np.cos(orientation),
                      connection[1] + middle_shift * np.sin(orientation))

            cut = gdspy.FlexPath(deepcopy([connection, middle_point]), [connection_shift + length],
                                     offset=[self.core / 2 + (connection_shift + length) / 2])
            remove = gdspy.boolean(cut, remove, 'or', layer=self.layer_configuration.total_layer)

            middle_point_ = (connection[0] + (middle_shift + self.gap / 2) * np.cos(orientation),
                            connection[1] + (middle_shift + self.gap / 2) * np.sin(orientation))

            middle_point__ = find_normal_point(middle_point_, flux_line_output, connection_shift + length + self.core / 2)

            path2 = gdspy.FlexPath(deepcopy([middle_point__, middle_point_, flux_line_output]), [self.gap],
                                    offset=[self.core / 2 + self.gap / 2])

            remove = gdspy.boolean(path2, remove, 'or', layer=self.layer_configuration.total_layer)

        elif coil == 'new':
            bug = 5
            connection = (self.center[0] + (self.R2 - bug) * np.cos(orientation),
                          self.center[1] + (self.R2 - bug) * np.sin(orientation))
            # add cpw from
            flux_line_output = (connection[0] + (self.outer_ground - self.R2 + bug) * np.cos(orientation),
                                connection[1] + (self.outer_ground - self.R2 + bug) * np.sin(orientation))
            # to fix rounding bug
            bug = 1
            flux_line_output_connection = (flux_line_output[0] + bug * np.cos(np.pi + orientation),
                                           flux_line_output[1] + bug * np.sin(np.pi + orientation))
            remove = gdspy.FlexPath(deepcopy([connection, flux_line_output]), [self.gap, self.gap],
                                    offset=[-self.core / 2 - self.gap / 2, self.core / 2 + self.gap / 2])

        else:
            raise ValueError('Coil type of flux line is not defined!')

        if 'mirror' in self.transformations:
            flux_line_output_connection = mirror_point(flux_line_output_connection, self.transformations['mirror'][0],
                                                       self.transformations['mirror'][1])
            qubit_center = mirror_point(deepcopy(self.center), self.transformations['mirror'][0],
                                        self.transformations['mirror'][1])
            orientation = np.arctan2(flux_line_output_connection[1] - qubit_center[1],
                                     flux_line_output_connection[0] - qubit_center[0]) + np.pi
        if 'rotate' in self.transformations:
            flux_line_output_connection = rotate_point(flux_line_output_connection, self.transformations['rotate'][0],
                                                       self.transformations['rotate'][1])
            qubit_center = rotate_point(deepcopy(self.center), self.transformations['rotate'][0],
                                        self.transformations['rotate'][1])

            orientation = np.arctan2(flux_line_output_connection[1] - qubit_center[1],
                                     flux_line_output_connection[0] - qubit_center[0]) + np.pi
        if self.transformations == {}:
            orientation = orientation + np.pi
        self.terminals['flux'] = DesignTerminal(flux_line_output_connection, orientation, g=self.grounded.g,
                                                s=self.grounded.s,
                                                w=self.grounded.w, type='cpw')

        return {'positive': result,
                'remove': remove,
                }

    def add_bandages(self):
        bandage_to_island = gdspy.Rectangle((self.JJ_coordinates[0] - self.JJ.contact_pad_a_outer / 4,
                                self.JJ_coordinates[1] + self.JJ.contact_pad_b_outer/2),
                               (self.JJ_coordinates[0] + self.JJ.contact_pad_a_outer / 4,
                                self.JJ_coordinates[1] - 3*self.JJ.contact_pad_b_outer/4),
                               layer=self.layer_configuration.bandages_layer)
        bandage_to_ground = gdspy.Rectangle((self.JJ.rect2[0] - self.JJ.rect_size_a/4,
                                               self.JJ.rect2[1] - self.JJ.rect_size_b/4),
                                              (self.JJ.rect2[0] + self.JJ.rect_size_a / 4,
                                               self.JJ.rect2[1] - 5*self.JJ.rect_size_b/4),
                               layer=self.layer_configuration.bandages_layer)

        bandage_to_fluxline = gdspy.Rectangle((self.JJ.rect1[0] - self.JJ.rect_size_a/4,
                                               self.JJ.rect1[1] - self.JJ.rect_size_b/4),
                                              (self.JJ.rect1[0] + self.JJ.rect_size_a / 4,
                                               self.JJ.rect1[1] - 5*self.JJ.rect_size_b/4),
                               layer=self.layer_configuration.bandages_layer)
        bandages = gdspy.boolean(bandage_to_island, [bandage_to_fluxline, bandage_to_ground], 'or',
                                 layer=self.layer_configuration.bandages_layer)
        return bandages

    def add_to_tls(self, tls_instance: tlsim.TLSystem, terminal_mapping: dict, track_changes: bool = True,
                   cutoff: float = np.inf, epsilon: float = 11.45) -> list:
        # scaling factor for C
        from scipy.constants import hbar, e
        scal_C = 1e-15
        jj1 = tlsim.JosephsonJunction(self.JJ_params['ic1'] * hbar / (2 * e), name=self.name + ' jj1')
        jj2 = tlsim.JosephsonJunction(self.JJ_params['ic2'] * hbar / (2 * e), name=self.name + ' jj2')
        m = tlsim.Inductor(self.JJ_params['lm'], name=self.name + ' flux-wire')
        c = tlsim.Capacitor(c=self.C['qubit'] * scal_C, name=self.name + ' qubit-ground')
        cache = [jj1, jj2, m, c]
        if self.third_JJ == False:
            squid_top = terminal_mapping['qubit']
        else:
            squid_top = terminal_mapping['squid_intermediate']
            jj3 = tlsim.JosephsonJunction(self.JJ_params['ic3'] * hbar / (2 * e), name=self.name + ' jj3')
            tls_instance.add_element(jj3, [squid_top, terminal_mapping['qubit']])
            cache.append(jj3)

        tls_instance.add_element(jj1, [0, squid_top])
        tls_instance.add_element(jj2, [terminal_mapping['flux'], squid_top])
        tls_instance.add_element(m, [0, terminal_mapping['flux']])
        tls_instance.add_element(c, [0, terminal_mapping['qubit']])
        mut_cap = []
        cap_g = []
        for id, coupler in enumerate(self.couplers):
            if coupler.coupler_type == 'coupler':
                c0 = tlsim.Capacitor(c=self.C['coupler' + str(id)][1] * scal_C,
                                     name=self.name + ' qubit-coupler' + str(id))
                c0g = tlsim.Capacitor(c=self.C['coupler' + str(id)][0] * scal_C,
                                      name=self.name + ' coupler' + str(id) + '-ground')
                tls_instance.add_element(c0, [terminal_mapping['qubit'], terminal_mapping['coupler' + str(id)]])
                tls_instance.add_element(c0g, [terminal_mapping['coupler' + str(id)], 0])
                mut_cap.append(c0)
                cap_g.append(c0g)
            # elif coupler.coupler_type =='grounded':
            #     tls_instance.add_element(tlsim.Short(), [terminal_mapping['flux line'], 0])

        if track_changes:
            self.tls_cache.append(cache + mut_cap + cap_g)
        return cache + mut_cap + cap_g

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


def mirror_point(point, ref1, ref2):
    """
       Mirror a point by a given line specified by 2 points ref1 and ref2.
    """
    [x1, y1] =ref1
    [x2, y2] =ref2

    dx = x2-x1
    dy = y2-y1
    a = (dx * dx - dy * dy) / (dx * dx + dy * dy)
    b = 2 * dx * dy / (dx * dx + dy * dy)
    # x2 = round(a * (point[0] - x1) + b * (point[1] - y1) + x1)
    # y2 = round(b * (point[0] - x1) - a * (point[1] - y1) + y1)
    x2 = a * (point[0] - x1) + b * (point[1] - y1) + x1
    y2 = b * (point[0] - x1) - a * (point[1] - y1) + y1
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



