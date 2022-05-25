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

# 0. Define geometry
# define transmission line parameters
# to have 50 Oms impedance with eps=11.75
tl_core = 10
tl_gap = 6
tl_ground = 20
# define coplanar resonator parameters
resonator_core = 10
resonator_gap = 6
resonator_ground = 20
# define pads parameters
pad_offset = 550
pad_element_offset = 400
pad_geometry = {'pad_w': 250, 'pad_s': 146, 'pad_g': 8, 'pad_length': 250, 'narrowing_length': 100, 'stub_length': 100,
                'z0': 50}
                
jc = 0.5e-6  # 1e-6 # uA/um^2
layers_configuration = {
    'total': 0,
    'restricted area': 10,
    'for removing': 100,
    'JJs': 1,
    'air bridges': 2,
    'air bridge pads': 3,
    'air bridge sm pads': 4,
    'vertical gridlines': 0,
    'horizontal gridlines': 0,
    'inverted': 101,
    'bandages': 20
}

sample = creator.Sample('TCCF-2', layers_configuration)

# specify sample vertical and horizontal lengths
# substrate size is 5*10 mm
sample.chip_geometry.sample_vertical_size = 9.7e3
sample.chip_geometry.sample_horizontal_size = 9.7e3
central_line_y = sample.chip_geometry.sample_vertical_size / 2
chip_edge_ground = elements.ChipEdgeGround(sample.chip_geometry, sample.layer_configuration, sample.pads, 350)
sample.add(chip_edge_ground)

# 1. Create contact pads:
nu_pads_side = 8
pads_left = []
pads_top = []
pads_right = []
pads_bottom = []

for side_id, side_list in enumerate([pads_left, pads_top, pads_right, pads_bottom]):
    for pad_side_id in range(nu_pads_side):
        position = (-sample.chip_geometry.sample_horizontal_size/2 + pad_offset, 
                    sample.chip_geometry.sample_vertical_size / (nu_pads_side + 1) * (-nu_pads_side/2 + pad_side_id + 1/2))
        position = -((-1j)**side_id)*(position[0] + 1j * position[1])
        position = (np.real(position) + sample.chip_geometry.sample_horizontal_size/2, 
                    np.imag(position) + sample.chip_geometry.sample_vertical_size/2 )
        
        pad = elements.Pad('pad-{}-side-{}'.format(pad_side_id, side_id),
                           position,
                           np.angle(((-1j)**side_id)),
#                            tl_core,
#                            tl_gap, tl_ground,
                           14,7,30,
                           layer_configuration=sample.layer_configuration, chip_geometry=sample.chip_geometry,
                           **pad_geometry)
        side_list.append(pad)
        sample.add(pad)

# qubit params

coupler_s = 26
flux_w = 14
flux_s = 7
flux_g = 30

squid_coupler_horizontal =  {'jjs_height': 0.1,
                             'jjs_width': 0.1,
                             'jj_lead_width':0.5,
                             'chain_width':0.5,
                             'ics': 0.1*0.1*jc,
                             'icb': 0.5*0.5*jc,
                             'chain_junctions': 45,
                             'lm': 3.3e-12,
                             'squid_orientation': 'horizontal',
                             'jjs_distance': 0.16,
                             'chain_jj_distance': 0.1,
                             'chain_top_offset': 1.0,
                             'flux_line_outer_extension': 11}
squid_coupler_vertical =    {'jjs_height': 0.1,
                             'jjs_width': 0.1,
                             'jj_lead_width':0.5,
                             'chain_width':0.5,
                             'ics': 0.1*0.1*jc,
                             'icb': 0.5*0.5*jc,
                             'chain_junctions': 45,
                             'lm': 3.3e-12,
                             'squid_orientation': 'vertical',
                             'jjs_distance': 0.16,
                             'chain_jj_distance': 0.1,
                             'chain_top_offset': 1.0,
                             'flux_line_outer_extension': 11}