import gdspy
from .core import DesignElement, LayerConfiguration, DesignTerminal
import numpy as np
from . import squid3JJ
from typing import List, Tuple, Mapping, Dict, AnyStr
from copy import deepcopy
from .. import conformal_mapping as cm
from .. import transmission_line_simulator as tlsim


class GFluxControllableSquid(DesignElement):
    def __init__(self, name: str,  position: tuple, w: float, s: float, g: float,
                 layer_configuration: LayerConfiguration, squid_params: Dict,
                 flux_w: float, flux_s: float, flux_g: float,
                 invert_x = False, invert_y = False):
        super().__init__(type='squid in line', name=name)
        self.w = w
        self.s = s
        self.g = g
        self.position = np.asarray(position)
        self.squid_params = self.default_squid_params()
        self.squid_params.update(squid_params)
        print (squid_params)

        self.flux_w = flux_w
        self.flux_s = flux_s
        self.flux_g = flux_g

        self.layer_configuration = layer_configuration
        # coupler terminals
        self.tls_cache = []

        rectangles, bandage_rectangles, optical_litho_rectangles, \
        flux_port_position_raw, port1_position_raw, port2_position_raw = self.squid_rectangles()

        if squid_params['squid_orientation'] == 'horizontal':
            orientation_cpw = 0
            orientation_fluxline = np.pi/2
        elif squid_params['squid_orientation'] == 'vertical':
            orientation_cpw = np.pi/2
            orientation_fluxline = 0

        if invert_x:
            if squid_params['squid_orientation'] == 'horizontal':
                orientation_fluxline -= np.pi
            rectangles[:, :, 0] = -rectangles[:, :, 0]
            bandage_rectangles[:, :, 0] = -bandage_rectangles[:, :, 0]
            optical_litho_rectangles[:, :, 0] = -optical_litho_rectangles[:, :, 0]
            flux_port_position_raw[0] = -flux_port_position_raw[0]
            port1_position_raw[0] = -port1_position_raw[0]
            port2_position_raw[0] = -port2_position_raw[0]
        if invert_y:
            if squid_params['squid_orientation'] == 'horizontal':
                orientation_fluxline -= np.pi
            rectangles[:, :, 1] = -rectangles[:, :, 1]
            bandage_rectangles[:, :, 1] = -bandage_rectangles[:, :, 1]
            optical_litho_rectangles[:, :, 1] = -optical_litho_rectangles[:, :, 1]
            flux_port_position_raw[1] = -flux_port_position_raw[1]
            port1_position_raw[1] = -port1_position_raw[1]
            port2_position_raw[1] = -port2_position_raw[1]

        self.rectangles = rectangles + self.position
        self.bandage_rectangles = bandage_rectangles + self.position
        self.optical_litho_rectangles = optical_litho_rectangles + self.position

        flux_port_position_raw += self.position
        port1_position_raw += self.position
        port2_position_raw += self.position

        self.terminals = dict()
        self.terminals['port1'] = DesignTerminal(position=port1_position_raw, orientation=orientation_cpw,
                                                 type='cpw', w=self.w, s=self.s, g=self.g,
                                                 disconnected='short')
        self.terminals['port2'] = DesignTerminal(position=port2_position_raw, orientation=orientation_cpw+np.pi,
                                                 type='cpw', w=self.w, s=self.s, g=self.g,
                                                 disconnected='short')
        self.terminals['flux'] = DesignTerminal(position=flux_port_position_raw, orientation=orientation_fluxline,
                                                 type='cpw', w=self.flux_w, s=self.flux_s, g=self.flux_g,
                                                 disconnected='short')

    def default_squid_params(self):
        squid_params = {}
        squid_params['shadow_orientation'] = 0
        squid_params['shadow_offset'] = 0.5

        # squid_orientation = 'horizontal'
        squid_params['squid_orientation'] = 'vertical'

        squid_params['scaffold_length'] = 5
        squid_params['scaffold_extra_length'] = 0.2
        squid_params['scaffold_height'] = 5
        squid_params['invert_scaffold'] = False

        squid_params['jjs_distance'] = 0.1
        squid_params['jjs_height'] = 0.3
        squid_params['jjs_width'] = 0.3

        squid_params['jj_lead_width'] = 0.5

        squid_params['chain_width'] = 0.5
        squid_params['chains_distance'] = 5
        squid_params['chain_top_offset'] = 5
        squid_params['chain_junctions'] = 1
        squid_params['chain_jj_distance'] = 0.1

        squid_params['bottom_lead_width'] = 0.5

        squid_params['flux_line_outer_extension'] = 5
        squid_params['flux_line_scaffold_width'] = 5
        squid_params['flux_line_scaffold_height'] = 0

        squid_params['flux_line_return_wire_distance'] = 10
        squid_params['flux_line_extent'] = 10
        squid_params['vertical_orientation_detour'] = 5

        squid_params['bottom_layer_overlap'] = 2
        squid_params['bandage_offset'] = 1

        squid_params['squid_contact_extension'] = 10

        return squid_params

    def squid_rectangles(self):
        shadow_orientation = self.squid_params['shadow_orientation']
        shadow_offset = self.squid_params['shadow_offset']

        # squid_orientation = 'horizontal'
        squid_orientation = self.squid_params['squid_orientation']

        scaffold_length = self.squid_params['scaffold_length']
        scaffold_extra_length = self.squid_params['scaffold_extra_length']
        scaffold_height = self.squid_params['scaffold_height']
        invert_scaffold = self.squid_params['invert_scaffold']

        jjs_distance = self.squid_params['jjs_distance']
        jjs_height = self.squid_params['jjs_height']
        jjs_width = self.squid_params['jjs_width']

        jj_lead_width = self.squid_params['jj_lead_width']

        chain_width = self.squid_params['chain_width']
        chains_distance = self.squid_params['chains_distance']
        chain_top_offset = self.squid_params['chain_top_offset']
        chain_junctions = self.squid_params['chain_junctions']
        chain_jj_distance = self.squid_params['chain_jj_distance']

        bottom_lead_width = self.squid_params['bottom_lead_width']

        flux_line_outer_extension = self.squid_params['flux_line_outer_extension']
        flux_line_scaffold_width = self.squid_params['flux_line_scaffold_width']
        flux_line_scaffold_height = self.squid_params['flux_line_scaffold_height']

        flux_line_return_wire_distance = self.squid_params['flux_line_return_wire_distance']
        flux_line_extent = self.squid_params['flux_line_extent']
        vertical_orientation_detour = self.squid_params['vertical_orientation_detour']

        bottom_layer_overlap = self.squid_params['bottom_layer_overlap']
        bandage_offset = self.squid_params['bandage_offset']

        squid_contact_extension = self.squid_params['squid_contact_extension']

        chain_jj_height = shadow_offset - chain_jj_distance
        chain_rectangle_height = 2 * shadow_offset - chain_jj_distance

        assert (chain_jj_height > 0)

        if np.abs(np.sin(shadow_orientation)) > 0.01:
            raise ValueError('SQUID works only for vertically oriented shadow-evaporated junctions')

        scaffold_rectangles = [[[-scaffold_length / 2, -jj_lead_width / 2],
                                [-scaffold_length / 2 + jj_lead_width, scaffold_height]],
                               [[-scaffold_length / 2 + jj_lead_width, scaffold_height],
                                [scaffold_length / 2 + scaffold_extra_length, scaffold_height - jjs_height]],
                               [[-jjs_width + scaffold_length / 2, scaffold_height - jjs_height - jjs_distance],
                                [scaffold_length / 2, -jj_lead_width / 2]]]

        scaffold_rectangles = np.asarray(scaffold_rectangles)
        if invert_scaffold:
            scaffold_rectangles[:, :, 1] = -scaffold_rectangles[:, :, 1]

        # drawing left chain only (right part by reflection)
        outer_chain_position = -chains_distance / 2 - chain_width
        chain_rectangles = [[[-scaffold_length / 2, -jj_lead_width / 2],
                             [outer_chain_position - squid_contact_extension, jj_lead_width / 2]]]

        if chain_top_offset > 0:
            chain_rectangles.append([[outer_chain_position, -jj_lead_width / 2],
                                     [outer_chain_position + chain_width, -jj_lead_width / 2 - chain_top_offset]])

        current_position = -jj_lead_width / 2 - chain_top_offset - chain_jj_distance

        if not chain_junctions % 2:
            raise ValueError('Can only draw odd number of junctions in chain')

        for rectangle_id in range((chain_junctions - 1) // 2):
            chain_rectangles.append([[outer_chain_position, current_position],
                                     [outer_chain_position - chain_width,
                                      current_position - chain_rectangle_height]])
            current_position -= chain_rectangle_height + chain_jj_distance

        # drawing bottom lead that connects chain with flux line inductance
        if flux_line_scaffold_height > 0:
            chain_rectangles.append([[outer_chain_position - flux_line_outer_extension, current_position],
                                     [-flux_line_scaffold_width / 2, current_position - bottom_lead_width]])

            # flux line scaffold
            chain_rectangles.append([[-flux_line_scaffold_width / 2, current_position],
                                     [-flux_line_scaffold_width / 2 + jj_lead_width,
                                      current_position - flux_line_scaffold_height]])
            chain_rectangles.append([[-flux_line_scaffold_width / 2 + jj_lead_width,
                                      current_position - flux_line_scaffold_height],
                                     [0, current_position - flux_line_scaffold_height + jj_lead_width]])
        else:
            chain_rectangles.append([[outer_chain_position - flux_line_outer_extension, current_position],
                                     [0, current_position - flux_line_scaffold_height - bottom_lead_width]])

        chain_rectangles = np.asarray(chain_rectangles)
        inverted_chain = chain_rectangles.copy()
        inverted_chain[:, :, 0] = -inverted_chain[:, :, 0]
        chain_rectangles = np.vstack([chain_rectangles, inverted_chain])

        flux_line_return_rectangles = [
            [[outer_chain_position - flux_line_outer_extension, current_position - bottom_lead_width],
             [outer_chain_position - flux_line_outer_extension + jj_lead_width,
              current_position - flux_line_return_wire_distance]],
            [[outer_chain_position - flux_line_outer_extension + jj_lead_width,
              current_position - flux_line_return_wire_distance],
             [-jj_lead_width / 2, current_position - flux_line_return_wire_distance + jj_lead_width]]]

        extra_leads = []

        if squid_orientation == 'horizontal':
            flux_line_return_rectangles.extend([
                [[-jj_lead_width / 2, current_position - flux_line_return_wire_distance + jj_lead_width],
                 [jj_lead_width / 2, current_position - flux_line_return_wire_distance - flux_line_extent]],
                [[- outer_chain_position + flux_line_outer_extension, current_position - bottom_lead_width],
                 [- outer_chain_position + flux_line_outer_extension - jj_lead_width,
                  current_position - flux_line_return_wire_distance - flux_line_extent]],
                [[-bottom_layer_overlap / 2 - bandage_offset, current_position - flux_line_return_wire_distance - flux_line_extent],
                 [bottom_layer_overlap / 2 + bandage_offset, current_position - flux_line_return_wire_distance - flux_line_extent - bottom_layer_overlap - 2 * bandage_offset]],
                [[-outer_chain_position + flux_line_outer_extension - jj_lead_width/2 - bottom_layer_overlap / 2 - bandage_offset,
                  current_position - flux_line_return_wire_distance - flux_line_extent],
                 [-outer_chain_position + flux_line_outer_extension - jj_lead_width/2 + bottom_layer_overlap / 2 + bandage_offset,
                  current_position - flux_line_return_wire_distance - flux_line_extent - bottom_layer_overlap - 2 * bandage_offset]]])
            extra_leads.extend([
                [[outer_chain_position - squid_contact_extension, -bottom_layer_overlap / 2 - bandage_offset],
                 [outer_chain_position - squid_contact_extension - bottom_layer_overlap - 2 * bandage_offset,
                  bottom_layer_overlap / 2 + bandage_offset]],
                [[- outer_chain_position + squid_contact_extension, -bottom_layer_overlap / 2 - bandage_offset],
                 [- outer_chain_position + squid_contact_extension + bottom_layer_overlap + 2 * bandage_offset,
                  bottom_layer_overlap / 2 + bandage_offset]]
                ])
        elif squid_orientation == 'vertical':
            flux_line_return_rectangles.extend([
                [[-jj_lead_width / 2, current_position - flux_line_return_wire_distance + jj_lead_width],
                 [-outer_chain_position + flux_line_outer_extension + flux_line_extent,
                  current_position - flux_line_return_wire_distance]],
                [[-outer_chain_position + flux_line_outer_extension, current_position],
                 [-outer_chain_position + flux_line_outer_extension + flux_line_extent,
                  current_position - jj_lead_width]],
                [[-outer_chain_position + flux_line_outer_extension + flux_line_extent,
                  current_position - flux_line_return_wire_distance - bottom_layer_overlap / 2 - bandage_offset],
                 [-outer_chain_position + flux_line_outer_extension + flux_line_extent + bottom_layer_overlap + 2 * bandage_offset,
                  current_position - flux_line_return_wire_distance + bottom_layer_overlap / 2 + bandage_offset]],
                [[-outer_chain_position + flux_line_outer_extension + flux_line_extent,
                  current_position - bottom_layer_overlap / 2 - bandage_offset],
                 [-outer_chain_position + flux_line_outer_extension + flux_line_extent + bottom_layer_overlap + 2 * bandage_offset,
                  current_position + bottom_layer_overlap / 2 + bandage_offset]]
            ])
            # left lead
            extra_leads.extend([
                 [[outer_chain_position - squid_contact_extension, -jj_lead_width / 2],
                  [outer_chain_position - squid_contact_extension + jj_lead_width,
                   current_position - flux_line_return_wire_distance - vertical_orientation_detour]],
                 [[outer_chain_position - squid_contact_extension + jj_lead_width,
                   current_position - flux_line_return_wire_distance - vertical_orientation_detour],
                  [-jj_lead_width / 2,
                   current_position - flux_line_return_wire_distance - vertical_orientation_detour + jj_lead_width]],
                 [[-jj_lead_width / 2,
                   current_position - flux_line_return_wire_distance - vertical_orientation_detour + jj_lead_width],
                  [jj_lead_width / 2,
                  current_position - flux_line_return_wire_distance - vertical_orientation_detour + jj_lead_width - flux_line_extent]],
                 [[-bottom_layer_overlap / 2-bandage_offset,
                   current_position - flux_line_return_wire_distance - vertical_orientation_detour + jj_lead_width - flux_line_extent],
                  [bottom_layer_overlap / 2 + bandage_offset,
                   current_position - flux_line_return_wire_distance - vertical_orientation_detour + jj_lead_width - flux_line_extent - bottom_layer_overlap - 2 * bandage_offset]]
                 ])

            # right lead
            extra_leads.extend([
                 [[- outer_chain_position + squid_contact_extension, jj_lead_width / 2],
                  [- outer_chain_position + squid_contact_extension - jj_lead_width,
                     scaffold_height + vertical_orientation_detour]],
                 [[- outer_chain_position + squid_contact_extension - jj_lead_width,
                  scaffold_height + vertical_orientation_detour],
                  [jj_lead_width / 2, scaffold_height + vertical_orientation_detour - jj_lead_width]],
                 [[jj_lead_width / 2, scaffold_height + vertical_orientation_detour - jj_lead_width],
                  [-jj_lead_width / 2, scaffold_height + vertical_orientation_detour + squid_contact_extension]],
                 [[-bottom_layer_overlap / 2 - bandage_offset, scaffold_height + vertical_orientation_detour + squid_contact_extension],
                  [bottom_layer_overlap / 2 + bandage_offset, scaffold_height + vertical_orientation_detour + squid_contact_extension + bottom_layer_overlap + 2 * bandage_offset]]
                 ])

        rectangles = np.vstack([scaffold_rectangles, chain_rectangles, flux_line_return_rectangles, extra_leads])

        # draw bandages
        if self.squid_params['squid_orientation'] == 'horizontal':
            bandage_rectangles = [
                [[-bottom_layer_overlap / 2,  # flux line
                  current_position - flux_line_return_wire_distance - flux_line_extent - bandage_offset],
                 [bottom_layer_overlap / 2,
                  current_position - flux_line_return_wire_distance - flux_line_extent - bottom_layer_overlap - 3 * bandage_offset]],
                [[-outer_chain_position + flux_line_outer_extension - jj_lead_width / 2 - bottom_layer_overlap / 2,  # flux line
                  current_position - flux_line_return_wire_distance - flux_line_extent - bandage_offset],
                 [-outer_chain_position + flux_line_outer_extension - jj_lead_width / 2 + bottom_layer_overlap / 2,
                  current_position - flux_line_return_wire_distance - flux_line_extent - bottom_layer_overlap - 3 * bandage_offset]],
                [[outer_chain_position - squid_contact_extension - bandage_offset,  # left lead
                  -bottom_layer_overlap / 2],
                 [outer_chain_position - squid_contact_extension - bottom_layer_overlap - 3 * bandage_offset,
                  bottom_layer_overlap / 2]],
                [[- outer_chain_position + squid_contact_extension + bandage_offset,  # right lead
                  -bottom_layer_overlap / 2],
                 [- outer_chain_position + squid_contact_extension + bottom_layer_overlap + 3 * bandage_offset,
                  bottom_layer_overlap / 2]]
            ]
        elif self.squid_params['squid_orientation'] == 'vertical':
            bandage_rectangles = [
                [[-bottom_layer_overlap / 2,  # left lead
                  current_position - flux_line_return_wire_distance - vertical_orientation_detour + jj_lead_width - flux_line_extent - bandage_offset],
                 [bottom_layer_overlap / 2,
                  current_position - flux_line_return_wire_distance - vertical_orientation_detour + jj_lead_width - flux_line_extent - bottom_layer_overlap - 3 * bandage_offset]],
                [[-bottom_layer_overlap / 2,  # right lead
                  scaffold_height + vertical_orientation_detour + squid_contact_extension + bandage_offset],
                 [bottom_layer_overlap / 2,
                  scaffold_height + vertical_orientation_detour + squid_contact_extension + bottom_layer_overlap + 3 * bandage_offset]],
                [[-outer_chain_position + flux_line_outer_extension + flux_line_extent + bandage_offset,  # flux line
                  current_position - flux_line_return_wire_distance - bottom_layer_overlap / 2],
                 [-outer_chain_position + flux_line_outer_extension + flux_line_extent + bottom_layer_overlap + 3 * bandage_offset,
                  current_position - flux_line_return_wire_distance + bottom_layer_overlap / 2]],
                [[-outer_chain_position + flux_line_outer_extension + flux_line_extent + bandage_offset,  # flux line
                  current_position - bottom_layer_overlap / 2],
                 [-outer_chain_position + flux_line_outer_extension + flux_line_extent + bottom_layer_overlap + 3 * bandage_offset,
                  current_position + bottom_layer_overlap / 2]]
                ]

        # draw CPW port for flux line
        if self.squid_params['squid_orientation'] == 'horizontal':
            extent_horizontal = - outer_chain_position + squid_contact_extension + bottom_layer_overlap + 4 * bandage_offset
            optical_litho_rectangles = [
                [[-self.flux_w / 2, # flux line center
                  current_position - flux_line_return_wire_distance - flux_line_extent - 2 * bandage_offset],
                 [self.flux_w / 2,
                  -self.g - self.s - self.w/2]],
                [[-self.flux_w / 2 - self.flux_s,
                  current_position - flux_line_return_wire_distance - flux_line_extent - 2 * bandage_offset],
                 [-extent_horizontal,
                  -self.g - self.s - self.w / 2]],
                [[self.flux_w / 2 + self.flux_s,
                  current_position - flux_line_return_wire_distance - flux_line_extent - 2 * bandage_offset],
                 [extent_horizontal,
                  -self.g - self.s - self.w / 2]], # TODO: assert y in range for CPW
                # draw CPW line central conductor

                [[-extent_horizontal+bottom_layer_overlap + 2 * bandage_offset,
                  -self.w / 2],
                 [-extent_horizontal,
                  self.w / 2]],
                [[extent_horizontal - bottom_layer_overlap - 2 * bandage_offset,
                  -self.w / 2],
                 [extent_horizontal,
                  self.w / 2]],
                # draw CPW continuous ground
                [[-extent_horizontal,
                  self.w / 2 + self.s],
                 [extent_horizontal,
                  self.w / 2 + self.s + self.g]]
            ]
            flux_port_position_raw = [0, -self.w / 2 - self.s - self.g]
            port1_position_raw = [-extent_horizontal, 0]
            port2_position_raw = [extent_horizontal, 0]

        elif self.squid_params['squid_orientation'] == 'vertical':
            bottom_position = current_position - flux_line_return_wire_distance - vertical_orientation_detour + jj_lead_width - flux_line_extent - bottom_layer_overlap - 4 * bandage_offset
            top_position = scaffold_height + vertical_orientation_detour + squid_contact_extension + bottom_layer_overlap + 4 * bandage_offset

            optical_litho_rectangles = [
                [[-self.w / 2,
                  top_position - bottom_layer_overlap - 2 * bandage_offset],
                 [self.w / 2,
                  top_position]],
                [[-self.w / 2,
                  bottom_position + bottom_layer_overlap + 2 * bandage_offset],
                 [self.w / 2,
                  bottom_position]],
                [[- self.w / 2 - self.s - self.g,
                  bottom_position],
                 [- self.w / 2 - self.s,
                  top_position]], # TODO: assert y in range for CPW
                # draw CPW line central conductor

                [[-outer_chain_position + flux_line_outer_extension + flux_line_extent + 2 * bandage_offset,
                  current_position - self.flux_w / 2 - self.flux_s],
                 [self.w / 2 + self.s + self.g,
                  bottom_position]],
                [[-outer_chain_position + flux_line_outer_extension + flux_line_extent + 2 * bandage_offset,
                  current_position + self.flux_w / 2 + self.flux_s],
                 [self.w / 2 + self.s + self.g,
                  top_position]],
                [[-outer_chain_position + flux_line_outer_extension + flux_line_extent + 2 * bandage_offset,
                  current_position - self.flux_w / 2],
                 [self.w / 2 + self.s + self.g,
                  current_position + self.flux_w / 2]],
            ]

            flux_port_position_raw = [self.w / 2 + self.s + self.g, current_position]
            port1_position_raw = [0, bottom_position]
            port2_position_raw = [0, top_position]

        return rectangles, np.asarray(bandage_rectangles), np.asarray(optical_litho_rectangles), \
               flux_port_position_raw, port1_position_raw, port2_position_raw

    def render(self):
        jj = []
        bandages = []
        positive = []
        for rectangle in self.rectangles:
            jj.append(gdspy.Rectangle(*rectangle.tolist(), layer=self.layer_configuration.jj_layer))
        for rectangle in self.bandage_rectangles:
            bandages.append(gdspy.Rectangle(*rectangle.tolist(), layer=self.layer_configuration.bandages_layer))
        for rectangle in self.optical_litho_rectangles:
            positive.append(gdspy.Rectangle(*rectangle.tolist(), layer=self.layer_configuration.total_layer))

        min_x = np.min(self.optical_litho_rectangles[:, :, 0])
        max_x = np.max(self.optical_litho_rectangles[:, :, 0])
        min_y = np.min(self.optical_litho_rectangles[:, :, 1])
        max_y = np.max(self.optical_litho_rectangles[:, :, 1])
        restrict = gdspy.Rectangle((min_x, min_y), (max_x, max_y), layer=self.layer_configuration.restricted_area_layer)

        return {'positive': positive,
                'restrict': restrict,
                'JJ': jj,
                'bandages': bandages}

    def get_terminals(self):
        return self.terminals

    def add_to_tls(self, tls_instance: tlsim.TLSystem, terminal_mapping: dict, track_changes: bool = True,
                       cutoff: float = np.inf, epsilon: float = 11.45) -> list:
        from scipy.constants import hbar, e
        jj_small = tlsim.JosephsonJunction(self.squid_params['ics'] * hbar / (2 * e), name=self.name + ' jj small')
        jj_left = tlsim.JosephsonJunction(self.squid_params['icb'] * hbar / (2 * e),
                                          name=self.name + ' jj big left',
                                          n_junctions=self.squid_params['chain_junctions'])
        jj_right = tlsim.JosephsonJunction(self.squid_params['icb'] * hbar / (2 * e),
                                           name=self.name + ' jj big right',
                                           n_junctions=self.squid_params['chain_junctions'])

        m = tlsim.Inductor(self.squid_params['lm'], name=self.name + ' flux-wire')

        #cl, ll = self.cm(epsilon)
        #c = cl[0, 0] * self.line_length
        #c1 = tlsim.Capacitor(c=c/2, name=self.name + '_c1')
        #c2 = tlsim.Capacitor(c=c/2, name=self.name + '_c2')

        cache = [jj_small, jj_left, jj_right, m]#, c1, c2]

        tls_instance.add_element(jj_small, [terminal_mapping['port1'], terminal_mapping['port2']])
        tls_instance.add_element(jj_left, [0, terminal_mapping['port1']])
        tls_instance.add_element(jj_right, [terminal_mapping['flux'], terminal_mapping['port2']])
        tls_instance.add_element(m, [0, terminal_mapping['flux']])

        #tls_instance.add_element(c1, [terminal_mapping['port1'], 0])
        #tls_instance.add_element(c2, [terminal_mapping['port2'], 0])

        if track_changes:
            self.tls_cache.append(cache)
        return cache
