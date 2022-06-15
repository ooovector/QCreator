import unittest

import numpy as np
from copy import deepcopy
from QCreator import elements
from QCreator import general_sample_creator as creator
from QCreator import meshing
import QCreator.auxiliary_functions as auxfuncs


class ResonatorsTest(unittest.TestCase):
    def test_res(self):
        tl_core = 11.
        tl_gap = 6.
        tl_ground = 20.

        pad_offset = 1100
        pad_element_offset = 400

        layers_configuration = {
            'total': 0,
            'restricted area': 10,
            'for removing': 100,
            'JJs': 3,
            'air bridges': 2,
            'air bridge pads': 1,
            'air bridge sm pads': 4,
            'vertical gridlines': 15,
            'horizontal gridlines': 16,
            'inverted': 17,
            'bandages': 101
        }

        sample = creator.Sample('Res-test-16-bridges', layers_configuration, epsilon=11.45)
        airbridge = elements.AirBridgeGeometry(pad_width=30,
                                               pad_length=30,
                                               narrow_width=10,
                                               narrow_length=56,
                                               pad_distance=36,
                                               sm_pad_length=10,
                                               sm_pad_distance=56,
                                               layer_configuration=sample.layer_configuration,
                                               bridge_style='misis')

        # specify sample vertical and horizontal lengths
        sample.chip_geometry.sample_vertical_size=4.7e3
        sample.chip_geometry.sample_horizontal_size=9.7e3
        central_line_y = sample.chip_geometry.sample_vertical_size/2
        chip_edge_ground = elements.ChipEdgeGround(sample.chip_geometry, sample.layer_configuration, sample.pads, 800)
        sample.add(chip_edge_ground)

        # 1. Create contact pads:
        pads_left = []
        pads_right = []
        for pad_side_id in range(1):
            pad = elements.Pad('pad-left-' + str(pad_side_id),
                               (pad_offset, sample.chip_geometry.sample_vertical_size / 2), np.pi, tl_core,
                               tl_gap, tl_ground,
                               layer_configuration=sample.layer_configuration, chip_geometry=sample.chip_geometry,
                               **elements.default_pad_geometry())
            pads_left.append(pad)
            sample.add(pad)
            pad = elements.Pad('pad-right-' + str(pad_side_id),
                               (sample.chip_geometry.sample_horizontal_size - pad_offset,
                                sample.chip_geometry.sample_vertical_size / 2), 0, tl_core,
                               tl_gap, tl_ground,
                               layer_configuration=sample.layer_configuration, chip_geometry=sample.chip_geometry,
                               **elements.default_pad_geometry())
            pads_right.append(pad)
            sample.add(pad)

        pad0 = pads_left[0]
        pad1 = pads_right[0]


        # resonator position parameters
        initial_spacing = 200

        coupler_lengths = [280, 300, 330, 365, 390, 420, 460, 520]
        coupler_starts = [[pad_offset + initial_spacing,
                           sample.chip_geometry.sample_vertical_size / 2]]
        for i in range(1, 8):
            coupler_starts.append([pad_offset + i * (sample.chip_geometry.sample_horizontal_size - 2 * pad_offset) / 8,
                                   sample.chip_geometry.sample_vertical_size / 2])

        closed_end_meander_lengths1 = [2900, 3110, 3400, 3680, 4000, 4450, 5000, 5600]
        closed_end_meander_lengths2 = [3000, 3250, 3600, 3850, 4250, 4730, 5300, 6000]
        length_lefts1 = [150, 200, 200, 200, 200, 200, 300, 400]
        length_rights1 = [150, 200, 200, 200, 200, 350, 150, 150]
        length_lefts2 = [150, 200, 200, 200, 200, 350, 150, 150]
        length_rights2 = [150, 200, 200, 200, 200, 200, 300, 400]

        object_left = pad0
        port_left = 'port'

        # draw resonators
        for resonator_pair_id in range(8):
            g1, g2, par1, par2, res12 = auxfuncs.draw_double_resonator(
                sample,
                coupler_starts[resonator_pair_id][0],
                coupler_starts[resonator_pair_id][1],
                coupler_lengths[resonator_pair_id],
                resonator_core=7,
                resonator_gap=8,
                resonator_ground=20,
                tl_core=11,
                tl_gap=6,
                tl_ground=20,
                grounding_width=10,
                closed_end_meander_length1=closed_end_meander_lengths1[resonator_pair_id],
                length_left1=length_lefts1[resonator_pair_id],
                length_right1=length_rights1[resonator_pair_id],
                closed_end_meander_length2=closed_end_meander_lengths2[resonator_pair_id],
                length_left2=length_lefts2[resonator_pair_id],
                length_right2=length_rights2[resonator_pair_id],
                open_end_length1=500,
                open_end_length2=500,
                port_orientation='left',
                airbridge=airbridge,
                min_bridge_spacing_closed_end=100,
                min_bridge_spacing_open_end=150,
                meander_r=55)

            sample.connect_cpw(object_left, g1, port1=port_left, port2='narrow', name='right TL {}'.format(resonator_pair_id),
                               points=[], airbridge=airbridge, min_spacing=150)
            object_left = g2
            port_left = 'narrow'

        sample.connect_cpw(object_left, pad1, port1=port_left, port2='port', name='right TL', points=[],
                           airbridge=airbridge, min_spacing=150)

        # Draw design
        sample.draw_design()

        # Get tlsim model
        sys, connections, elements_ = sample.get_tls(cutoff=2e11)

        # Calculate eigenfrequencies
        f, delta, modes = sys.get_modes()


if __name__ == '__main__':
    unittest.main()
