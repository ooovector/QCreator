import gdspy
from .core import DesignElement, LayerConfiguration, DesignTerminal
import numpy as np
import copy
from . import squid3JJ
from typing import List, Tuple, Mapping, Dict, AnyStr
from copy import deepcopy
from .. import conformal_mapping as cm
from .. import transmission_line_simulator as tlsim
from QCreator.elements.cross_lines import CrossLinesViaAirbridges
from QCreator.elements.cpw import CPWCoupler


class Fluxonium(DesignElement):
    def __init__(self, name: str, position: tuple, w: float, s: float, g: float,
                 layer_configuration: LayerConfiguration, squid_params: Dict,
                 flux_w: float, flux_s: float, flux_g: float,
                 short_flux_g: List = [0,0],
                 cross_lines : List = [],
                 invert_x=False, invert_y=False, invert_flux_line=False):
        """
        short_flux_g:  shifts of edge grounds
        cross_lines: List of added cross lines over thin line, here you need all information about CrossLinesViaAirbridges
            except name. Position should be same: dictionary(port,alpha), where alpha - coordinate on thin line from 0 to 1,
            orientation is special: it can be 0 or 1 (0 for vertical qubit, 1 for horizontal)
        """
        # invert_flux_line works only for squid_params['capacity_angle'] = True
        super().__init__(type='squid in line', name=name)
        self.w = w
        self.s = s
        self.g = g
        self.position = np.asarray(position)
        self.squid_params = self.default_squid_params()
        # print(squid_params)
        self.squid_params.update(squid_params)



        self.flux_w = flux_w
        self.flux_s = flux_s
        self.flux_g = flux_g

        self.s_ground = self.squid_params['s_ground']
        self.g_ground = self.squid_params['g_ground']
        self.s_ground_turning = self.squid_params['s_ground_turning']
        # Эта штука нужна для того, чтобы резонаторы не перекрывались с землей (remove crossing between resonator and flux_ground)
        self.short_flux_g = short_flux_g
        # add cross lines to over thin line
        self.cross_lines = []

        # for capacitive matrix
        self.C = {'C_1': None,
                  'C_j': None,
                  'C_2': None,
                  }

        self.invert_flux_line = invert_flux_line

        self.layer_configuration = layer_configuration
        # coupler terminals
        self.tls_cache = []

        rectangles, bandage_rectangles, optical_litho_rectangles, \
        flux_port_position_raw, port1_position_raw, port2_position_raw, i_extra, i_chain,i_inverted_chain = self.squid_rectangles()

        if squid_params['squid_orientation'] == 'horizontal':
            if squid_params['capacity_angle'] == True:
                orientation_cpw1 = -np.pi / 2
            else:
                orientation_cpw1 = 0
            orientation_cpw2 = np.pi
            orientation_fluxline = np.pi / 2
        elif squid_params['squid_orientation'] == 'vertical':
            orientation_cpw1 = np.pi / 2
            orientation_cpw2 = -np.pi / 2
            orientation_fluxline = np.pi

        if invert_x:
            if squid_params['squid_orientation'] == 'vertical':
                orientation_fluxline -= np.pi
            if squid_params['squid_orientation'] == 'horizontal' and squid_params['capacity_angle'] == True:
                orientation_cpw2 -= np.pi
            rectangles[:, :, 0] = -rectangles[:, :, 0]
            bandage_rectangles[:, :, 0] = -bandage_rectangles[:, :, 0]
            optical_litho_rectangles[:, :, 0] = -optical_litho_rectangles[:, :, 0]
            flux_port_position_raw[0] = -flux_port_position_raw[0]
            port1_position_raw[0] = -port1_position_raw[0]
            port2_position_raw[0] = -port2_position_raw[0]
        if invert_y:
            if squid_params['squid_orientation'] == 'horizontal':
                if squid_params['capacity_angle'] == True:
                    orientation_cpw1 += np.pi
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
        self.i_extra = i_extra
        self.i_chain = i_chain
        self.i_inverted_chain = i_inverted_chain

        flux_port_position_raw += self.position
        port1_position_raw += self.position
        port2_position_raw += self.position

        self.terminals = dict()
        self.terminals['port1'] = DesignTerminal(position=port1_position_raw, orientation=orientation_cpw1,
                                                 type='cpw', w=self.w, s=self.s, g=self.g,
                                                 disconnected='short')
        self.terminals['port2'] = DesignTerminal(position=port2_position_raw, orientation=orientation_cpw2,
                                                 type='cpw', w=self.w, s=self.s, g=self.g,
                                                 disconnected='short')
        self.terminals['flux'] = DesignTerminal(position=flux_port_position_raw, orientation=orientation_fluxline,
                                                type='cpw', w=self.flux_w, s=self.flux_s, g=self.flux_g,
                                                disconnected='short')
        for i in range(len(cross_lines)):
            if cross_lines[i]['port'] == 'port1':
                position_cross = copy.copy(np.asarray(self.terminals['port1'].position))
                if squid_params['capacity_angle'] and cross_lines[i]['orientation'] == 1:
                    position_cross[1] = self.position[1]
            else:
                position_cross = copy.copy(np.asarray(self.terminals['port2'].position))
            position_cross[(cross_lines[i]['orientation']+1)%2] *= cross_lines[i]['alpha']
            position_cross[(cross_lines[i]['orientation'] + 1) % 2] += ((1-cross_lines[i]['alpha'])
                                                                       *self.position[(cross_lines[i]['orientation'] + 1) % 2])
            self.cross_lines.append(CrossLinesViaAirbridges('cross_line'+str(i),position = position_cross,
                orientation = cross_lines[i]['orientation']*np.pi/2,
                top_w = cross_lines[i]['top_w'], top_s = cross_lines[i]['top_s'], top_g = cross_lines[i]['top_g'],
                bot_w = self.squid_params['jj_lead_width'], bot_s = 0, bot_g = 0,
                narrowing_length =  cross_lines[i]['narrowing_length'], geometry = cross_lines[i]['geometry'],
                line_w = cross_lines[i]['line_w'], line_s = cross_lines[i]['line_s'], line_g = cross_lines[i]['line_g'],
                                                                                                with_ground=False))
            self.cross_lines[i].render()
            self.terminals[str(i) + 'top_1'] = self.cross_lines[i].get_terminals()['top_1']
            self.terminals[str(i) + 'top_2'] = self.cross_lines[i].get_terminals()['top_2']

    def default_squid_params(self):
        squid_params = {}
        squid_params['shadow_orientation'] = 0
        squid_params['shadow_offset'] = 0.5

        # squid_orientation = 'horizontal'
        squid_params['squid_orientation'] = 'vertical'
        squid_params['capacity_angle'] = False  # If True the angle between capacities is 90. Work only for horizontal squid_orientation

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
        squid_params['vertical_orientation_extent'] = 0 # extention for vertical oriaentation

        squid_params['bottom_layer_overlap'] = 2
        squid_params['bandage_offset'] = 1

        squid_params['squid_contact_extension'] = 10

        # Параметры земли над тонкой линией (for ground near thin line)
        squid_params['s_ground'] = 20
        squid_params['g_ground'] = 20
        squid_params['s_ground_turning'] = 0 # adding distance for ground for turning line if squid_params['capacity_angle']==True to make C_left==C_right
        # print(squid_params)
        return squid_params

    def squid_rectangles(self):
        shadow_orientation = self.squid_params['shadow_orientation']
        shadow_offset = self.squid_params['shadow_offset']

        # squid_orientation = 'horizontal'
        squid_orientation = self.squid_params['squid_orientation']
        capacity_angle = self.squid_params['capacity_angle']

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
        vertical_orientation_extent = self.squid_params['vertical_orientation_extent']

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

        if self.invert_flux_line and self.squid_params['capacity_angle']:
            scaffold_rectangles[:, :, 1] = -scaffold_rectangles[:, :, 1]

        # drawing left chain only (right part by reflection)
        outer_chain_position = -chains_distance / 2 - chain_width
        if capacity_angle:
            chain_rectangles = [[[-scaffold_length / 2, -jj_lead_width / 2],
                                 [outer_chain_position, jj_lead_width / 2]]]
        else:
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
                                     [outer_chain_position + chain_width,
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
            if self.squid_params['capacity_angle']:
                chain_rectangles.append([[outer_chain_position - flux_line_outer_extension*0, current_position],
                                     [0, current_position - flux_line_scaffold_height - bottom_lead_width]])
            else:
                chain_rectangles.append([[outer_chain_position - flux_line_outer_extension, current_position],
                                         [0, current_position - flux_line_scaffold_height - bottom_lead_width]])

        chain_rectangles = np.asarray(chain_rectangles)
        inverted_chain = chain_rectangles.copy()
        inverted_chain[:, :, 0] = -inverted_chain[:, :, 0]
        chain_rectangles = np.vstack([chain_rectangles, inverted_chain])

        if self.invert_flux_line and self.squid_params['capacity_angle']:
            chain_rectangles[:, :, 1] = -chain_rectangles[:, :, 1]

        if self.squid_params['capacity_angle']:
            flux_line_return_rectangles = [
                [[outer_chain_position, current_position - bottom_lead_width],
                 [outer_chain_position + jj_lead_width,
                  current_position - flux_line_return_wire_distance]],

                [[outer_chain_position + jj_lead_width,
                  current_position - flux_line_return_wire_distance],
                 [-jj_lead_width / 2  - self.flux_w/2 - self.flux_s - 2 * bottom_layer_overlap + flux_line_outer_extension,
                  current_position - flux_line_return_wire_distance + jj_lead_width]]
              ]
        else:
            flux_line_return_rectangles = [
                [[outer_chain_position - flux_line_outer_extension, current_position - bottom_lead_width],
                 [outer_chain_position - flux_line_outer_extension + jj_lead_width,
                  current_position - flux_line_return_wire_distance]],
                [[outer_chain_position - flux_line_outer_extension + jj_lead_width,
                  current_position - flux_line_return_wire_distance],
                 [-jj_lead_width / 2, current_position - flux_line_return_wire_distance + jj_lead_width]]]

        extra_leads = []

        if squid_orientation == 'horizontal':
            if self.squid_params['capacity_angle']:
                flux_line_return_rectangles.extend([
                    [[-jj_lead_width / 2  - self.flux_w/2 - self.flux_s - 2 * bottom_layer_overlap + flux_line_outer_extension,
                      current_position - flux_line_return_wire_distance + jj_lead_width],
                     [jj_lead_width / 2  - self.flux_w/2 - self.flux_s - 2 * bottom_layer_overlap + flux_line_outer_extension,
                      current_position - flux_line_return_wire_distance - flux_line_extent]],

                    [[-outer_chain_position,
                      current_position - bottom_lead_width + jj_lead_width],
                     [jj_lead_width / 2 + flux_line_outer_extension,
                      current_position - bottom_lead_width]],

                    [[-jj_lead_width / 2 + flux_line_outer_extension,
                      current_position - bottom_lead_width],
                     [jj_lead_width / 2 + flux_line_outer_extension,
                      current_position - flux_line_return_wire_distance - flux_line_extent]],

                    [[-bottom_layer_overlap / 2 - bandage_offset + flux_line_outer_extension,
                      current_position - flux_line_return_wire_distance - flux_line_extent],
                     [bottom_layer_overlap / 2 + bandage_offset + flux_line_outer_extension,
                      current_position - flux_line_return_wire_distance - flux_line_extent - bottom_layer_overlap - 2 * bandage_offset]],

                    [[ - bottom_layer_overlap / 2 - bandage_offset  + flux_line_outer_extension -
                       self.flux_w/2 - self.flux_s - 2 * bottom_layer_overlap,
                        current_position - flux_line_return_wire_distance - flux_line_extent],
                        [bottom_layer_overlap / 2 + bandage_offset + flux_line_outer_extension -
                         self.flux_w/2 - self.flux_s - 2 * bottom_layer_overlap,
                            current_position - flux_line_return_wire_distance - flux_line_extent - bottom_layer_overlap - 2 * bandage_offset]]
                ])
            else:
                flux_line_return_rectangles.extend([
                    [[-jj_lead_width / 2, current_position - flux_line_return_wire_distance + jj_lead_width],
                     [jj_lead_width / 2, current_position - flux_line_return_wire_distance - flux_line_extent]],
                    [[- outer_chain_position + flux_line_outer_extension, current_position - bottom_lead_width],
                     [- outer_chain_position + flux_line_outer_extension - jj_lead_width,
                      current_position - flux_line_return_wire_distance - flux_line_extent]],
                    [[-bottom_layer_overlap / 2 - bandage_offset,
                      current_position - flux_line_return_wire_distance - flux_line_extent],
                     [bottom_layer_overlap / 2 + bandage_offset,
                      current_position - flux_line_return_wire_distance - flux_line_extent - bottom_layer_overlap - 2 * bandage_offset]],
                    [[
                         -outer_chain_position + flux_line_outer_extension - jj_lead_width / 2 - bottom_layer_overlap / 2 - bandage_offset,
                         current_position - flux_line_return_wire_distance - flux_line_extent],
                     [
                         -outer_chain_position + flux_line_outer_extension - jj_lead_width / 2 + bottom_layer_overlap / 2 + bandage_offset,
                         current_position - flux_line_return_wire_distance - flux_line_extent - bottom_layer_overlap - 2 * bandage_offset]]])
            if capacity_angle:
                extra_leads.extend([
                    [[outer_chain_position, -jj_lead_width / 2],
                     [outer_chain_position*3, jj_lead_width / 2]],

                    [[outer_chain_position*3, -jj_lead_width / 2],
                     [outer_chain_position*3 - jj_lead_width,
                      jj_lead_width / 2 - outer_chain_position + vertical_orientation_extent]],

                    [[outer_chain_position*3 + bottom_layer_overlap / 2 + bandage_offset - jj_lead_width / 2,
                         jj_lead_width / 2 - outer_chain_position + vertical_orientation_extent - bottom_layer_overlap / 2 - bandage_offset],
                     [outer_chain_position*3 - bottom_layer_overlap / 2 - bandage_offset - jj_lead_width / 2,
                         jj_lead_width / 2 - outer_chain_position + vertical_orientation_extent + bottom_layer_overlap / 2 + bandage_offset]],

                    [[-outer_chain_position, -jj_lead_width / 2],
                     [-outer_chain_position + squid_contact_extension, jj_lead_width / 2]],

                    [[- outer_chain_position + squid_contact_extension, -bottom_layer_overlap / 2 - bandage_offset],
                     [- outer_chain_position + squid_contact_extension + bottom_layer_overlap + 2 * bandage_offset,
                      bottom_layer_overlap / 2 + bandage_offset]],
                ])
            else:
                extra_leads.extend([
                    [[outer_chain_position - squid_contact_extension, -bottom_layer_overlap / 2 - bandage_offset],
                     [outer_chain_position - squid_contact_extension - bottom_layer_overlap - 2 * bandage_offset,
                      bottom_layer_overlap / 2 + bandage_offset]],
                    [[- outer_chain_position + squid_contact_extension, -bottom_layer_overlap / 2 - bandage_offset],
                     [- outer_chain_position + squid_contact_extension + bottom_layer_overlap + 2 * bandage_offset,
                      bottom_layer_overlap / 2 + bandage_offset]],
                ])
        elif squid_orientation == 'vertical':
            flux_line_return_rectangles.extend([
                [[-jj_lead_width / 2, current_position - flux_line_return_wire_distance + jj_lead_width],
                 [self.w / 2,
                  current_position - flux_line_return_wire_distance]],
                [[self.w / 2, current_position - flux_line_return_wire_distance + jj_lead_width],
                 [self.w / 2 + jj_lead_width,
                  current_position - self.flux_w / 2 - self.flux_s - bottom_layer_overlap
                  - 2 * bandage_offset - 1 / 2 * jj_lead_width]],
                [[self.w / 2 + jj_lead_width,
                  current_position - self.flux_w / 2 - self.flux_s - bottom_layer_overlap
                  - 2 * bandage_offset - 1 / 2 * jj_lead_width
                  ],
                 [-outer_chain_position + flux_line_outer_extension + flux_line_extent,
                  current_position - self.flux_w / 2 - self.flux_s - bottom_layer_overlap
                  - 2 * bandage_offset + 1 / 2 * jj_lead_width]],
                [[-outer_chain_position + flux_line_outer_extension, current_position],
                 [-outer_chain_position + flux_line_outer_extension + flux_line_extent,
                  current_position - jj_lead_width]],
                [[-outer_chain_position + flux_line_outer_extension + flux_line_extent,
                  current_position - self.flux_w / 2 - self.flux_s - 3 / 2 * bottom_layer_overlap - 3 * bandage_offset],
                 [-outer_chain_position + flux_line_outer_extension + flux_line_extent + bottom_layer_overlap + 2 * bandage_offset,
                     current_position - self.flux_w / 2 - self.flux_s - 1 / 2 * bottom_layer_overlap - 1 * bandage_offset]],

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
                  current_position - vertical_orientation_detour - vertical_orientation_extent]],

                [[-bottom_layer_overlap / 2 - bandage_offset,
                  current_position - vertical_orientation_detour - vertical_orientation_extent],
                 [bottom_layer_overlap / 2 + bandage_offset,
                  current_position - vertical_orientation_detour -
                  vertical_orientation_extent - bottom_layer_overlap - 2 * bandage_offset]]
            ])

            # right lead
            extra_leads.extend([
                [[- outer_chain_position + squid_contact_extension, jj_lead_width / 2],
                 [- outer_chain_position + squid_contact_extension - jj_lead_width,
                  - current_position + flux_line_return_wire_distance + vertical_orientation_detour]],

                [[- outer_chain_position + squid_contact_extension - jj_lead_width,
                  - current_position + flux_line_return_wire_distance + vertical_orientation_detour],
                 [jj_lead_width / 2, - current_position + flux_line_return_wire_distance + vertical_orientation_detour - jj_lead_width]],

                [[jj_lead_width / 2, - current_position + flux_line_return_wire_distance + vertical_orientation_detour - jj_lead_width],
                 [-jj_lead_width / 2, - current_position + vertical_orientation_detour + vertical_orientation_extent]],

                [[-bottom_layer_overlap / 2 - bandage_offset,
                  - current_position + vertical_orientation_detour + vertical_orientation_extent],
                 [bottom_layer_overlap / 2 + bandage_offset,
                  - current_position + vertical_orientation_detour + vertical_orientation_extent + bottom_layer_overlap + 2 * bandage_offset]]
            ])

        if self.invert_flux_line and self.squid_params['capacity_angle']:
            flux_line_return_rectangles = np.asarray(flux_line_return_rectangles)
            flux_line_return_rectangles[:, :, 1] = -flux_line_return_rectangles[:, :, 1]

        rectangles = np.vstack([scaffold_rectangles, chain_rectangles, flux_line_return_rectangles, extra_leads])
        i_extra = len(scaffold_rectangles)+len(chain_rectangles)+len(flux_line_return_rectangles)
        i_chain = len(scaffold_rectangles)
        i_inverted_chain = len(scaffold_rectangles) + len(inverted_chain)
        # draw bandages
        if self.squid_params['squid_orientation'] == 'horizontal':
            if capacity_angle:
                bandage_rectangles = [
                    [[-bottom_layer_overlap / 2 + flux_line_outer_extension -
                      self.flux_w/2 - self.flux_s - 2 * bottom_layer_overlap,  # flux line
                      current_position - flux_line_return_wire_distance - flux_line_extent - bandage_offset],
                     [bottom_layer_overlap / 2 + flux_line_outer_extension -
                      self.flux_w/2 - self.flux_s - 2 * bottom_layer_overlap,
                      current_position - flux_line_return_wire_distance - flux_line_extent - bottom_layer_overlap - 3 * bandage_offset]],
                    [[- bottom_layer_overlap / 2 + flux_line_outer_extension,
                      # flux line
                      current_position - flux_line_return_wire_distance - flux_line_extent - bandage_offset],
                     [bottom_layer_overlap / 2 + flux_line_outer_extension,
                      current_position - flux_line_return_wire_distance - flux_line_extent - bottom_layer_overlap - 3 * bandage_offset]], ]
            else:
                bandage_rectangles = [
                    [[-bottom_layer_overlap / 2,  # flux line
                      current_position - flux_line_return_wire_distance - flux_line_extent - bandage_offset],
                     [bottom_layer_overlap / 2,
                      current_position - flux_line_return_wire_distance - flux_line_extent - bottom_layer_overlap - 3 * bandage_offset]],
                    [[-outer_chain_position + flux_line_outer_extension - jj_lead_width / 2 - bottom_layer_overlap / 2,
                      # flux line
                      current_position - flux_line_return_wire_distance - flux_line_extent - bandage_offset],
                     [-outer_chain_position + flux_line_outer_extension - jj_lead_width / 2 + bottom_layer_overlap / 2,
                      current_position - flux_line_return_wire_distance - flux_line_extent - bottom_layer_overlap - 3 * bandage_offset]], ]

            if self.invert_flux_line and self.squid_params['capacity_angle']:
                bandage_rectangles = np.asarray(bandage_rectangles)
                bandage_rectangles[:, :, 1] = -bandage_rectangles[:, :, 1]
                bandage_rectangles = list(bandage_rectangles)

            if capacity_angle:
                bandage_rectangles.extend([[
                    # left lead
                    [outer_chain_position*3 + bottom_layer_overlap / 2 - jj_lead_width / 2,
                     jj_lead_width / 2 - outer_chain_position + vertical_orientation_extent - bottom_layer_overlap / 2],
                    [outer_chain_position*3 - bottom_layer_overlap / 2 - jj_lead_width / 2,
                     jj_lead_width / 2 - outer_chain_position + vertical_orientation_extent + bottom_layer_overlap / 2 + 2 * bandage_offset]],
                    # right lead
                    [[- outer_chain_position + squid_contact_extension + bandage_offset,
                      -bottom_layer_overlap / 2],
                     [- outer_chain_position + squid_contact_extension + bottom_layer_overlap + 3 * bandage_offset,
                      bottom_layer_overlap / 2]]
                ])

            else:
                bandage_rectangles.extend([[
                    # left lead
                    [outer_chain_position - squid_contact_extension - bandage_offset,
                     -bottom_layer_overlap / 2],
                    [outer_chain_position - squid_contact_extension - bottom_layer_overlap - 3 * bandage_offset,
                     bottom_layer_overlap / 2]],
                    # right lead
                    [[- outer_chain_position + squid_contact_extension + bandage_offset,
                      -bottom_layer_overlap / 2],
                     [- outer_chain_position + squid_contact_extension + bottom_layer_overlap + 3 * bandage_offset,
                      bottom_layer_overlap / 2]]
                ])




        elif self.squid_params['squid_orientation'] == 'vertical':
            bandage_rectangles = [
                [[-bottom_layer_overlap / 2,  # left lead
                  current_position - vertical_orientation_detour -
                  vertical_orientation_extent - bandage_offset],
                 [bottom_layer_overlap / 2,
                  current_position - vertical_orientation_detour -
                  vertical_orientation_extent - bottom_layer_overlap - 3 * bandage_offset]],

                [[-bottom_layer_overlap / 2,  # right lead
                  - current_position + vertical_orientation_detour + vertical_orientation_extent + bandage_offset],
                 [bottom_layer_overlap / 2,
                  - current_position + vertical_orientation_detour + vertical_orientation_extent + bottom_layer_overlap + 3 * bandage_offset]],

                [[-outer_chain_position + flux_line_outer_extension + flux_line_extent + bandage_offset,  # flux line
                  current_position - self.flux_w / 2 - self.flux_s
                  - 3 / 2 * bottom_layer_overlap - 2 * bandage_offset],
                 [-outer_chain_position + flux_line_outer_extension + flux_line_extent + bottom_layer_overlap + 3 * bandage_offset,
                     current_position - self.flux_w / 2 - self.flux_s
                     - 1 / 2 * bottom_layer_overlap - 2 * bandage_offset]],

                [[-outer_chain_position + flux_line_outer_extension + flux_line_extent + bandage_offset,  # flux line
                  current_position - bottom_layer_overlap / 2],
                 [
                     -outer_chain_position + flux_line_outer_extension + flux_line_extent + bottom_layer_overlap + 3 * bandage_offset,
                     current_position + bottom_layer_overlap / 2]]
            ]

        # draw CPW port for flux line
        if self.squid_params['squid_orientation'] == 'horizontal':
            extent_horizontal = - outer_chain_position + squid_contact_extension + bottom_layer_overlap + 4 * bandage_offset
            if self.squid_params['capacity_angle']:
                if self.invert_flux_line:
                    optical_litho_rectangles = [
                        [[-self.flux_w / 2 + flux_line_outer_extension,  # flux line center
                          current_position - flux_line_return_wire_distance - flux_line_extent - 2 * bandage_offset],
                         [self.flux_w / 2 + flux_line_outer_extension,
                          -self.g + current_position - flux_line_return_wire_distance - flux_line_extent - 2 * bandage_offset]],

                    [[-self.flux_w / 2 - self.flux_s  + flux_line_outer_extension,
                      current_position - flux_line_return_wire_distance - flux_line_extent - 2 * bandage_offset-self.s_ground_turning],
                     [outer_chain_position*3 - jj_lead_width / 2 -
                      (current_position - flux_line_return_wire_distance - flux_line_extent - 2 * bandage_offset) + self.s_ground_turning,
                      -self.g + current_position - flux_line_return_wire_distance - flux_line_extent - 2 * bandage_offset]],

                     [[self.flux_w / 2 + self.flux_s + flux_line_outer_extension,
                       current_position - flux_line_return_wire_distance - flux_line_extent - 2 * bandage_offset],
                      [extent_horizontal - self.short_flux_g[0],
                       -self.g + current_position - flux_line_return_wire_distance - flux_line_extent - 2 * bandage_offset]],

                     [[-self.flux_w / 2 - self.flux_s  + flux_line_outer_extension,
                       #+ self.s_ground + self.g_ground,
                       -self.g + current_position - flux_line_return_wire_distance - flux_line_extent - 2 * bandage_offset],
                      [outer_chain_position*3 - jj_lead_width / 2 -
                      (current_position - flux_line_return_wire_distance - flux_line_extent - 2 * bandage_offset)+ self.s_ground_turning,
                       #+ self.s_ground,
                           -jj_lead_width / 2 + outer_chain_position - vertical_orientation_extent
                           - bottom_layer_overlap / 2 - 3 * bandage_offset + self.short_flux_g[1]]],
                     ]
                else:
                    optical_litho_rectangles = [
                        [[-self.flux_w / 2 + flux_line_outer_extension,  # flux line center
                          current_position - flux_line_return_wire_distance - flux_line_extent - 2 * bandage_offset],
                         [self.flux_w / 2 + flux_line_outer_extension,
                          -self.g + current_position - flux_line_return_wire_distance - flux_line_extent - 2 * bandage_offset]],

                        [[-self.flux_w / 2 - self.flux_s + flux_line_outer_extension,
                          current_position - flux_line_return_wire_distance - flux_line_extent - 2 * bandage_offset],
                         [outer_chain_position*3 - jj_lead_width / 2 +
                          (current_position - flux_line_return_wire_distance - flux_line_extent - 2 * bandage_offset),
                          -self.g + current_position - flux_line_return_wire_distance - flux_line_extent - 2 * bandage_offset]],

                        [[self.flux_w / 2 + self.flux_s + flux_line_outer_extension,
                          current_position - flux_line_return_wire_distance - flux_line_extent - 2 * bandage_offset],
                         [extent_horizontal - self.short_flux_g[0],
                          -self.g + current_position - flux_line_return_wire_distance - flux_line_extent - 2 * bandage_offset]],

                        [[outer_chain_position*3 - jj_lead_width / 2 +
                          (current_position - flux_line_return_wire_distance - flux_line_extent - 2 * bandage_offset),
                          -self.g + current_position - flux_line_return_wire_distance - flux_line_extent - 2 * bandage_offset],
                         [outer_chain_position*3 - jj_lead_width / 2 +
                          (current_position - flux_line_return_wire_distance - flux_line_extent - 2 * bandage_offset)-self.g,
                          jj_lead_width / 2 - outer_chain_position + vertical_orientation_extent
                          + bottom_layer_overlap / 2 + 3 * bandage_offset - + self.short_flux_g[1]]],
                    ]
            else:
                optical_litho_rectangles = [
                        [[-self.flux_w / 2,  # flux line center
                          current_position - flux_line_return_wire_distance - flux_line_extent - 2 * bandage_offset],
                         [self.flux_w / 2,
                          -self.g + current_position - flux_line_return_wire_distance - flux_line_extent - 2 * bandage_offset]],
                        [[-self.flux_w / 2 - self.flux_s,
                      current_position - flux_line_return_wire_distance - flux_line_extent - 2 * bandage_offset],
                     [-extent_horizontal + self.short_flux_g[0],
                      -self.g + current_position - flux_line_return_wire_distance - flux_line_extent - 2 * bandage_offset]],
                    [[self.flux_w / 2 + self.flux_s,
                      current_position - flux_line_return_wire_distance - flux_line_extent - 2 * bandage_offset],
                     [extent_horizontal - self.short_flux_g[1],
                      -self.g + current_position - flux_line_return_wire_distance - flux_line_extent - 2 * bandage_offset]],
                     ]

            if self.invert_flux_line and self.squid_params['capacity_angle']:
                optical_litho_rectangles = np.asarray(optical_litho_rectangles)
                optical_litho_rectangles[:, :, 1] = -optical_litho_rectangles[:, :, 1]
                optical_litho_rectangles = list(optical_litho_rectangles)

            if capacity_angle:
                optical_litho_rectangles.extend(
                    # draw CPW line central conductor

                    [
                        [[outer_chain_position*3 - jj_lead_width / 2 - self.w / 2,
                       jj_lead_width / 2 - outer_chain_position + vertical_orientation_extent
                       - bottom_layer_overlap / 2 + bandage_offset],
                      [outer_chain_position*3 - jj_lead_width / 2 + self.w / 2,
                       jj_lead_width / 2 - outer_chain_position + vertical_orientation_extent
                       + bottom_layer_overlap / 2 + 3 * bandage_offset]],

                     [[extent_horizontal - bottom_layer_overlap - 2 * bandage_offset,
                       -self.w / 2],
                      [extent_horizontal,
                       self.w / 2]],
                    ])
                # draw CPW continuous ground
                if self.invert_flux_line:
                    optical_litho_rectangles.extend([
                        [[outer_chain_position*3 - jj_lead_width / 2 - self.w / 2 - self.s_ground-self.s_ground_turning,
                             -self.w / 2 - self.s_ground-self.s_ground_turning],
                         [0,
                          - self.w / 2 - self.s_ground - self.g_ground]],

                        [[0,
                             -self.w / 2 - self.s_ground],
                         [extent_horizontal,
                          - self.w / 2 - self.s_ground - self.g_ground]],

                        [[outer_chain_position*3 - jj_lead_width / 2 - self.w / 2 - self.s_ground-self.s_ground_turning,
                             - self.w / 2 - self.s_ground - self.g_ground],
                         [outer_chain_position*3 - jj_lead_width / 2 - self.w / 2 - self.s_ground - self.g_ground,
                             jj_lead_width / 2 - outer_chain_position + vertical_orientation_extent
                             + bottom_layer_overlap / 2 + 3 * bandage_offset]],
                    ],
                    )
                else:
                    optical_litho_rectangles.extend([
                        [[outer_chain_position*3 - jj_lead_width / 2 + self.w / 2 + self.s_ground,
                          self.w / 2 + self.s_ground],
                         [extent_horizontal,
                          + self.w / 2 + self.s_ground + self.g_ground]],
                        [[outer_chain_position*3 - jj_lead_width / 2 + self.w / 2 + self.s_ground,
                          + self.w / 2 + self.s_ground + self.g_ground],
                         [outer_chain_position*3 - jj_lead_width / 2 + self.w / 2 + self.s_ground + self.g_ground,
                          jj_lead_width / 2 - outer_chain_position + vertical_orientation_extent
                          + bottom_layer_overlap / 2 + 3 * bandage_offset
                          ]],
                    ],
                    )

                if self.invert_flux_line:
                    flux_port_position_raw = [flux_line_outer_extension, self.g - current_position + flux_line_return_wire_distance
                                              + flux_line_extent + 2 * bandage_offset]
                else:
                    flux_port_position_raw = [flux_line_outer_extension, -self.g + current_position - flux_line_return_wire_distance
                                              - flux_line_extent - 2 * bandage_offset]
                port1_position_raw = [outer_chain_position*3 - jj_lead_width / 2,
                                      jj_lead_width / 2 - outer_chain_position + vertical_orientation_extent
                                      + bottom_layer_overlap / 2 + 3 * bandage_offset]
                port2_position_raw = [extent_horizontal, 0]
            else:
                optical_litho_rectangles.extend(
                    # draw CPW line central conductor

                    [[[-extent_horizontal + bottom_layer_overlap + 2 * bandage_offset,
                       -self.w / 2],
                      [-extent_horizontal,
                       self.w / 2]],
                     [[extent_horizontal - bottom_layer_overlap - 2 * bandage_offset,
                       -self.w / 2],
                      [extent_horizontal,
                       self.w / 2]],
                     # draw CPW continuous ground
                     [[-extent_horizontal,
                      self.w / 2 + self.s_ground],
                     [extent_horizontal,
                      self.w / 2 + self.s_ground + self.g_ground]]
                     ])
                # flux_port_position_raw = [0, -self.w / 2 - self.s - self.g] # check
                flux_port_position_raw = [0, -self.g + current_position - flux_line_return_wire_distance
                                          - flux_line_extent - 2 * bandage_offset]
                port1_position_raw = [-extent_horizontal, 0]
                port2_position_raw = [extent_horizontal, 0]

        elif self.squid_params['squid_orientation'] == 'vertical':
            bottom_position = current_position - vertical_orientation_detour - vertical_orientation_extent - bottom_layer_overlap - 4 * bandage_offset
            top_position =  - current_position + vertical_orientation_detour + vertical_orientation_extent + bottom_layer_overlap + 4 * bandage_offset

            optical_litho_rectangles = [
                [[-self.w / 2,
                  top_position - bottom_layer_overlap - 2 * bandage_offset],
                 [self.w / 2,
                  top_position]],
                [[-self.w / 2,
                  bottom_position + bottom_layer_overlap + 2 * bandage_offset],
                 [self.w / 2,
                  bottom_position]],
                [[- self.w / 2 - self.s_ground - self.g_ground,
                 bottom_position],
                [- self.w / 2 - self.s_ground,
                 top_position]], # TODO: assert y in range for CPW

                # draw CPW line central conductor

                [[-outer_chain_position + flux_line_outer_extension + flux_line_extent + 2 * bandage_offset,
                  current_position - self.flux_w / 2 - self.flux_s],
                 [self.w / 2 + self.s + self.g,
                  #current_position - self.flux_w / 2 - self.flux_s - self.flux_g]],
                  bottom_position + self.short_flux_g[0]]],
                [[-outer_chain_position + flux_line_outer_extension + flux_line_extent + 2 * bandage_offset,
                  current_position + self.flux_w / 2 + self.flux_s],
                 [self.w / 2 + self.s + self.g,
                  #current_position + self.flux_w / 2 + self.flux_s + self.flux_g]],
                  top_position - self.short_flux_g[1]]],
                [[-outer_chain_position + flux_line_outer_extension + flux_line_extent + 2 * bandage_offset,
                  current_position - self.flux_w / 2],
                 [self.w / 2 + self.s + self.g,
                  current_position + self.flux_w / 2]],
            ]

            flux_port_position_raw = [self.w / 2 + self.s + self.g, current_position]  # check
            port1_position_raw = [0, bottom_position]
            port2_position_raw = [0, top_position]

        return rectangles, np.asarray(bandage_rectangles), np.asarray(optical_litho_rectangles), \
               flux_port_position_raw, port1_position_raw, port2_position_raw, i_extra, i_chain,i_inverted_chain

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
        # print(len(self.optical_litho_rectangles))
        if self.squid_params['squid_orientation'] == 'horizontal':
            if (self.squid_params['capacity_angle']):
                if self.invert_flux_line:
                    ground = list(np.asarray(self.optical_litho_rectangles)[[1,2,3,6,7,8]])
                else:
                    ground = list(np.asarray(self.optical_litho_rectangles)[[1,2,3,6,7]])
            else:
                ground = list(np.asarray(self.optical_litho_rectangles)[[1,2,5]])
        else:
            ground = list(np.asarray(self.optical_litho_rectangles)[[2,3,4]])

        electrode1 = list(self.rectangles[self.i_chain:self.i_chain+1])
        electrode2 = list(self.rectangles[self.i_inverted_chain:self.i_inverted_chain+1])
        if self.squid_params['squid_orientation'] == 'horizontal':
            if (self.squid_params['capacity_angle']):
                electrode1 += list(np.asarray(self.rectangles)[[self.i_extra, self.i_extra + 1,self.i_extra + 2]]) + list(
                    self.optical_litho_rectangles[4:5])
                electrode2 += list(np.asarray(self.rectangles)[[self.i_extra + 3,self.i_extra + 4]]) + list(
                    self.optical_litho_rectangles[5:6])
            else:
                electrode1 += list(np.asarray(self.rectangles)[[self.i_extra]]) + list(self.optical_litho_rectangles[3:4])
                electrode2 += list(np.asarray(self.rectangles)[[self.i_extra+1]]) + list(self.optical_litho_rectangles[4:5])
        else:
            electrode1 += list(np.asarray(self.rectangles)[self.i_extra:self.i_extra + 4]) + list(self.optical_litho_rectangles[1:2])
            electrode2 += list(np.asarray(self.rectangles)[self.i_extra+4:self.i_extra+8]) + list(self.optical_litho_rectangles[0:1])
        qubit_cap_parts = []
        for rectangle in electrode1:
            qubit_cap_parts.append(gdspy.Rectangle(*rectangle.tolist(), layer=9))
        for rectangle in electrode2:
            qubit_cap_parts.append(gdspy.Rectangle(*rectangle.tolist(), layer=10))
        for rectangle in ground:
            qubit_cap_parts.append(gdspy.Rectangle(*rectangle.tolist(), layer=0))

        contacts = []
        contacts_sm = []
        bridges = []
        for i in range(len(self.cross_lines)):
            draw_cross = self.cross_lines[i].render()
            contacts.append(draw_cross['airbridges_pads'])
            contacts_sm.append(draw_cross['airbridges_sm_pads'])
            restrict_cross = draw_cross['restrict']
            positive_cross = draw_cross['positive']
            bridges.append(draw_cross['airbridges'])
            (top_w, top_s, position, orientation) = (self.cross_lines[i].top_w,self.cross_lines[i].top_s,
                                        self.cross_lines[i].position,self.cross_lines[i].orientation)
            restrict_line_1 = gdspy.Rectangle((position[0]-500, position[1]-top_w/2-top_s),
                                            (position[0]+500, position[1]-top_w/2))
            restrict_line_1.rotate(orientation, position)
            restrict_line_2 = gdspy.Rectangle((position[0] - 500, position[1] + top_w / 2 + top_s),
                                              (position[0] + 500, position[1] + top_w / 2))
            restrict_line_2.rotate(orientation, position)
            positive_cross = gdspy.boolean(positive_cross, jj, 'not')
            positive = gdspy.boolean(positive, restrict_cross, 'not')
            positive = gdspy.boolean(positive, restrict_line_1, 'not')
            positive = gdspy.boolean(positive, restrict_line_2, 'not')
            positive = gdspy.boolean(positive, positive_cross, 'or')
            restrict = gdspy.boolean(restrict, restrict_cross, 'or')

        return {'positive': positive ,
                'restrict': restrict,
                'JJ': jj,
                'bandages': bandages,
                'airbridges_pads': contacts,
                'airbridges_sm_pads': contacts_sm,
                'airbridges': bridges,
                'qubit_cap': qubit_cap_parts,
                }

    def fill_capacitance_matrix(self, matrix):
        if matrix.shape != (3, 3):
            raise ValueError("Incorrect capacitance matrix shape ")
        else:
            self.capacitance_matrix = matrix

        self.C = {'C_left': self.capacitance_matrix[1,1],
                  'C_right': self.capacitance_matrix[2,2],
                  'C_j': -self.capacitance_matrix[1,2]}

    def get_terminals(self):
        return self.terminals

    def add_to_tls(self, tls_instance: tlsim.TLSystem, terminal_mapping: dict, track_changes: bool = True,
                   cutoff: float = np.inf, epsilon: float = 11.45) -> list:
        from scipy.constants import hbar, e
        jj_small = tlsim.JosephsonJunction(self.squid_params['ics'] * hbar / (2 * e), name=self.name + ' jj small')
        jj_left = tlsim.JosephsonJunctionChain(self.squid_params['icb'] * hbar / (2 * e),
                                               name=self.name + ' jj big left',
                                               n_junctions=self.squid_params['chain_junctions'])
        jj_right = tlsim.JosephsonJunctionChain(self.squid_params['icb'] * hbar / (2 * e),
                                                name=self.name + ' jj big right',
                                                n_junctions=self.squid_params['chain_junctions'])
        # My changes
        L_left = tlsim.Inductor(l=jj_left.L_lin(), name=self.name + ' left')
        L_right = tlsim.Inductor(l=jj_right.L_lin(), name=self.name + ' right')
        C_left = tlsim.Capacitor(c=self.C['C_left'], name=self.name + '_C_left')
        C_right = tlsim.Capacitor(c=self.C['C_right'], name=self.name + '_C_right')
        C_j = tlsim.Capacitor(c=self.C['C_j'], name=self.name + '_C_j')
        # end

        m = tlsim.Inductor(self.squid_params['lm'], name=self.name + ' flux-wire')

        # cl, ll = self.cm(epsilon)
        # c = cl[0, 0] * self.line_length
        # c1 = tlsim.Capacitor(c=c/2, name=self.name + '_c1')
        # c2 = tlsim.Capacitor(c=c/2, name=self.name + '_c2')

        #cache = [jj_small, jj_left, jj_right, m]  # , c1, c2]
        cache = [jj_small, L_left, L_right, m, C_left, C_right, C_j]

        tls_instance.add_element(jj_small, [terminal_mapping['port1'], terminal_mapping['port2']])
        # My changes
        # tls_instance.add_element(jj_left, [0, terminal_mapping['port1']])
        # tls_instance.add_element(jj_right, [terminal_mapping['flux'], terminal_mapping['port2']])
        tls_instance.add_element(L_left, [0, terminal_mapping['port1']])
        tls_instance.add_element(L_right, [terminal_mapping['flux'], terminal_mapping['port2']])
        tls_instance.add_element(C_left, [0, terminal_mapping['port1']])
        tls_instance.add_element(C_right, [0, terminal_mapping['port2']])
        tls_instance.add_element(C_j, [terminal_mapping['port1'], terminal_mapping['port2']])
        # end
        tls_instance.add_element(m, [0, terminal_mapping['flux']])

        # tls_instance.add_element(c1, [terminal_mapping['port1'], 0])
        # tls_instance.add_element(c2, [terminal_mapping['port2'], 0])

        for i in range(len(self.cross_lines)):
            line_i = CPWCoupler('0',
                points = [self.cross_lines[i].get_terminals()[str(i) + 'top_1'].position,
                          self.cross_lines[i].get_terminals()[str(i) + 'top_2'].position],
                            w = self.cross_lines[i].line_w, s = self.cross_lines[i].line_s, g = self.cross_lines[i].line_g,
                 layer_configuration =self.layer_configuration, r=1000)
            cl,ll = line_i.cm(epsilon)
            line = tlsim.TLCoupler(n=len(line_i.w),
                                   l=line_i.length,  # TODO: get length
                                   cl=cl,
                                   ll=ll,
                                   rl=np.zeros((len(line_i.w), len(line_i.w))),
                                   gl=np.zeros((len(line_i.w), len(line_i.w))),
                                   name=line_i.name,
                                   cutoff=cutoff)
            cache.extend([line])

            if str(i)+'top_1' in terminal_mapping:
                p1 = terminal_mapping[str(i)+'top_1']
            elif (str(i)+'top_1', 0) in terminal_mapping:
                p1 = terminal_mapping[(str(i)+'top_1', 0)]
            else:
                raise ValueError('Neither (port1, 0) or port1 found in terminal_mapping')

            if str(i)+'top_2' in terminal_mapping:
                p2 = terminal_mapping[str(i)+'top_2']
            elif (str(i)+'top_2', 0) in terminal_mapping:
                p2 = terminal_mapping[(str(i)+'top_2', 0)]
            else:
                raise ValueError('Neither (port2, 0) or port2 found in terminal_mapping')

            tls_instance.add_element(line, [p1, p2])

        if track_changes:
            self.tls_cache.append(cache)
        return cache