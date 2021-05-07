import sys
sys.path.append('..\\..')
import gdspy
import numpy as np
from importlib import reload
from copy import deepcopy
from QCreator import elements
from QCreator import general_sample_creator as creator
from QCreator import meshing
reload(gdspy)

### to have 50 Oms impedance with eps=11.75
tl_core = 20.
tl_gap = 12.
tl_ground = 10.

pad_offset = 800

layers_configuration = {
    'total':0,
    'restricted area':10,
    'for removing':100,
    'JJs':1,
    'air bridges': 2,
    'air bridge pads': 3,
    'vertical gridlines':15,
    'horizontal gridlines':16,
    'inverted':17
}

sample = creator.Sample('pcb',layers_configuration)

#specify sample vertical and horizontal lengths
sample.chip_geometry.sample_vertical_size=4.3e3
sample.chip_geometry.sample_horizontal_size=7e3

chip_edge_ground = elements.ChipEdgeGround(sample.chip_geometry, sample.layer_configuration, sample.pads)
sample.add(chip_edge_ground)

# # 1. Create contact pads for 7*4.3 pcb WMI/ETH:
# pads_left = []
# pads_right = []
# for pad_side_id in range(1):
#     pad = elements.Pad('pad-left-' + str(pad_side_id),
#                        (pad_offset, sample.chip_geometry.sample_vertical_size / 2), np.pi, tl_core,
#                        tl_gap, tl_ground,
#                        layer_configuration=sample.layer_configuration, chip_geometry=sample.chip_geometry,
#                        **elements.default_pad_geometry())
#     pads_left.append(pad)
#     sample.add(pad)
#     pad = elements.Pad('pad-right-' + str(pad_side_id),
#                        (sample.chip_geometry.sample_horizontal_size - pad_offset,
#                         sample.chip_geometry.sample_vertical_size / 2), 0, tl_core,
#                        tl_gap, tl_ground,
#                        layer_configuration=sample.layer_configuration, chip_geometry=sample.chip_geometry,
#                        **elements.default_pad_geometry())
#     pads_right.append(pad)
#     sample.add(pad)
#
# pads_top = []
# pads_bottom = []
# eth_offset=2470
# for pad_side_id in range(3):
#     pad = elements.Pad('pad-bottom-' + str(pad_side_id),
#                        (sample.chip_geometry.sample_horizontal_size / 2+ eth_offset * (pad_side_id - 1), pad_offset),
#                        -np.pi / 2, tl_core, tl_gap, tl_ground,
#                        layer_configuration=sample.layer_configuration, chip_geometry=sample.chip_geometry,
#                        **elements.default_pad_geometry())
#     pads_bottom.append(pad)
#     sample.add(pad)
#     pad = elements.Pad('pad-top-' + str(pad_side_id),
#                        (sample.chip_geometry.sample_horizontal_size / 2 + eth_offset * (pad_side_id - 1),
#                         sample.chip_geometry.sample_vertical_size - pad_offset),
#                        np.pi / 2, tl_core, tl_gap, tl_ground,
#                        layer_configuration=sample.layer_configuration, chip_geometry=sample.chip_geometry,
#                        **elements.default_pad_geometry())
#     pads_top.append(pad)
#     sample.add(pad)
#
# p1 = pads_left[0]
# p2 = pads_right[0]

# # 1. Create contact pads for 7*6.6 pcb WMI/ETH:
# pads_left = []
# pads_right = []
# for pad_side_id in range(1):
#     pad = elements.Pad('pad-left-' + str(pad_side_id),
#                        (pad_offset, sample.chip_geometry.sample_vertical_size / 2), np.pi, tl_core,
#                        tl_gap, tl_ground,
#                        layer_configuration=sample.layer_configuration, chip_geometry=sample.chip_geometry,
#                        **elements.default_pad_geometry())
#     pads_left.append(pad)
#     sample.add(pad)
#     pad = elements.Pad('pad-right-' + str(pad_side_id),
#                        (sample.chip_geometry.sample_horizontal_size - pad_offset,
#                         sample.chip_geometry.sample_vertical_size / 2), 0, tl_core,
#                        tl_gap, tl_ground,
#                        layer_configuration=sample.layer_configuration, chip_geometry=sample.chip_geometry,
#                        **elements.default_pad_geometry())
#     pads_right.append(pad)
#     sample.add(pad)
#
# pads_top = []
# pads_bottom = []
# eth_offset=2470
# for pad_side_id in range(3):
#     pad = elements.Pad('pad-bottom-' + str(pad_side_id),
#                        (sample.chip_geometry.sample_horizontal_size / 2+ eth_offset * (pad_side_id - 1), pad_offset),
#                        -np.pi / 2, tl_core, tl_gap, tl_ground,
#                        layer_configuration=sample.layer_configuration, chip_geometry=sample.chip_geometry,
#                        **elements.default_pad_geometry())
#     pads_bottom.append(pad)
#     sample.add(pad)
#     pad = elements.Pad('pad-top-' + str(pad_side_id),
#                        (sample.chip_geometry.sample_horizontal_size / 2 + eth_offset * (pad_side_id - 1),
#                         sample.chip_geometry.sample_vertical_size - pad_offset),
#                        np.pi / 2, tl_core, tl_gap, tl_ground,
#                        layer_configuration=sample.layer_configuration, chip_geometry=sample.chip_geometry,
#                        **elements.default_pad_geometry())
#     pads_top.append(pad)
#     sample.add(pad)
#
# p1 = pads_left[0]
# p2 = pads_right[0]


# # 1. Create contact pads for 6*10 pcb WMI from Huns:
# pads_left = []
# pads_right = []
# for pad_side_id in range(1):
#     pad = elements.Pad('pad-left-' + str(pad_side_id),
#                        (pad_offset, sample.chip_geometry.sample_vertical_size / 2), np.pi, tl_core,
#                        tl_gap, tl_ground,
#                        layer_configuration=sample.layer_configuration, chip_geometry=sample.chip_geometry,
#                        **elements.default_pad_geometry())
#     pads_left.append(pad)
#     sample.add(pad)
#     pad = elements.Pad('pad-right-' + str(pad_side_id),
#                        (sample.chip_geometry.sample_horizontal_size - pad_offset,
#                         sample.chip_geometry.sample_vertical_size / 2), 0, tl_core,
#                        tl_gap, tl_ground,
#                        layer_configuration=sample.layer_configuration, chip_geometry=sample.chip_geometry,
#                        **elements.default_pad_geometry())
#     pads_right.append(pad)
#     sample.add(pad)
#
# pads_top = []
# pads_bottom = []
# huns_offset=2470
# pad_bottom_1 = elements.Pad('pad-bottom-' + str(1),
#                    (sample.chip_geometry.sample_horizontal_size / 2+ huns_offset *(- 1), pad_offset),
#                    -np.pi / 2, tl_core, tl_gap, tl_ground,
#                    layer_configuration=sample.layer_configuration, chip_geometry=sample.chip_geometry,
#                    **elements.default_pad_geometry())
# pad_bottom_2 = elements.Pad('pad-bottom-' + str(2),
#                    (sample.chip_geometry.sample_horizontal_size / 2+ huns_offset, pad_offset),
#                    -np.pi / 2, tl_core, tl_gap, tl_ground,
#                    layer_configuration=sample.layer_configuration, chip_geometry=sample.chip_geometry,
#                    **elements.default_pad_geometry())
# pads_bottom.append(pad_bottom_1)
# pads_bottom.append(pad_bottom_2)
# sample.add(pad_bottom_1)
# sample.add(pad_bottom_2)
# pad = elements.Pad('pad-top-' + str(pad_side_id),
#                    (sample.chip_geometry.sample_horizontal_size / 2,
#                     sample.chip_geometry.sample_vertical_size - pad_offset),
#                    np.pi / 2, tl_core, tl_gap, tl_ground,
#                    layer_configuration=sample.layer_configuration, chip_geometry=sample.chip_geometry,
#                    **elements.default_pad_geometry())
# pads_top.append(pad)
# sample.add(pad)
#
# p1 = pads_left[0]
# p2 = pads_top[0]