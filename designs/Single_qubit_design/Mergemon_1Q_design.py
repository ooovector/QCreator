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
tl_core = 10.
tl_gap = 6.
tl_ground = 20.

resonator_core = 10#8
resonator_gap = 6#7
resonator_ground = 20

pad_offset = 800
pad_element_offset = 400
qubit_position_offset = 800

coupler_start = pad_offset + pad_element_offset
coupler_delta = 500
coupler_length = 320
num_couplers = 1

jc = 1e-8#uA/um^2

layers_configuration = {
    'total':0,
    'restricted area':10,
    'for removing':100,
    'JJs':1,
    'air bridges':2,
    'air bridge pads':3,
    'air bridge sm pads':4,
    'vertical gridlines':15,
    'horizontal gridlines':16,
    'inverted':17,
    'bandages': 20
}

sample = creator.Sample('1Q_test',layers_configuration)

#specify sample vertical and horizontal lengths
sample.chip_geometry.sample_vertical_size=4.7e3
sample.chip_geometry.sample_horizontal_size=4.7e3
central_line_y = sample.chip_geometry.sample_vertical_size/2
chip_edge_ground = elements.ChipEdgeGround(sample.chip_geometry, sample.layer_configuration, sample.pads)
chip_edge_ground.render_direct = True
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

pads_top = []
pads_bottom = []
for pad_side_id in range(3):
    pad = elements.Pad('pad-bottom-' + str(pad_side_id),
                       (sample.chip_geometry.sample_horizontal_size / 4 * (pad_side_id + 1), pad_offset),
                       -np.pi / 2, tl_core, tl_gap, tl_ground,
                       layer_configuration=sample.layer_configuration, chip_geometry=sample.chip_geometry,
                       **elements.default_pad_geometry())
    pads_bottom.append(pad)
    sample.add(pad)
    pad = elements.Pad('pad-top-' + str(pad_side_id),
                       (sample.chip_geometry.sample_horizontal_size / 4 * (pad_side_id + 1),
                        sample.chip_geometry.sample_vertical_size - pad_offset),
                       np.pi / 2, tl_core, tl_gap, tl_ground,
                       layer_configuration=sample.layer_configuration, chip_geometry=sample.chip_geometry,
                       **elements.default_pad_geometry())
    pads_top.append(pad)
    sample.add(pad)

p1 = pads_left[0]
p2 = pads_right[0]

##################### Coaxmons
shift=-1/10
phi1=1/5


Couplers_qubit_mergemon=[elements.coaxmon.CoaxmonCoupler(arc_start=-1/6-3/100+shift,arc_finish=-3.5/6-1/100+shift,phi=-1/2,
                                          coupler_type= 'grounded', w =7,s=4, g=20),
          elements.coaxmon.CoaxmonCoupler(arc_start = 1/100 + shift + 1/20,arc_finish = 2/3 - 1/100 + shift - 1/10,
                                          phi=phi1,
                                          coupler_type='distinguishable', w =10,g=20,s=6),# for resonator
]





# jj_coaxmon_2JJ_mergemon = {'a1':10,
#                    'jj1_width':0.2,  #0.16,
#                    'jj1_height':0.3,  #0.2,
#                   'jj2_width': 0.1,  # 0.45,
#                    'jj2_height':0.4,  #0.486,
#                   'jj3_height': None,  # 0.2,
#                   'jj3_width': None,  # 0.486,
#                    'c2':10,
#                    'angle_qubit':-np.pi/2+np.pi/26.5,  #-np.pi/2-np.pi/3,
#                    'angle_JJ': 0,
#                    'length':10,
#                    'width':5,
#                   'ic1': 0.8 * 1 * jc,
#                   'ic2': 1.2 * 1.4 * jc,
#                   'lm': 4.2e-12
#                   }


jj_coaxmon_2JJ_mergemon = {'length_pad_up_to_squid':10,
                   'jj1_width':1,  #0.16,
                   'jj1_height':1,  #0.2,
                  'jj2_width': 1.4,  # 0.45,
                   'jj2_height':1.4,  #0.486,
                   'pad_up_length': 10,
                   'pad_up_width' : 5,
                   'width_pad_up_to_squid': 1,
                    'squid_shift_left': 2,
                    'squid_shift_right': 1.5,
                    'base_width': 0.8,
                    'jj_gap': 0.3,
                    'jj_vertical_length': 2,
                    'jj_horizontal_length': 2.5,
                    'side_pad_length': 5,
                    'side_pad_width': 5,
                     'base_shift':0.6,
                    'side_pads_distance': 18,

                   'angle_qubit':-np.pi/2+np.pi/26.5,  #-np.pi/2-np.pi/3,
                   'angle_JJ': 0,
                   'length':10,
                   'width':5,
                  'ic1': 0.8 * 1 * jc,
                  'ic2': 1.2 * 1.4 * jc,
                  'lm': 4.2e-12,

                    'separate_JJs':True,
                   'jj3_height': None,  # 0.2,
                   'jj3_width': None,  # 0.486,
                  }


jj_coaxmon_2JJ_mergemon2 = {'length_pad_up_to_squid':20,
                   'jj1_width':0.8,  #0.16,
                   'jj1_height':1,  #0.2,
                  'jj2_width': 1.2,  # 0.45,
                   'jj2_height':1.4,  #0.486,
                   'pad_up_length': 10,
                   'pad_up_width' : 5,
                   'width_pad_up_to_squid': 1,
                    'squid_shift_left': 2,
                    'squid_shift_right': 1.5,
                    'base_width': 0.8,
                    'jj_gap': 0.3,
                    'jj_vertical_length': 2,
                    'jj_horizontal_length': 2.5,
                    'side_pad_length': 5,
                    'side_pad_width': 5,
                     'base_shift':0.6,
                    'side_pads_distance': 18,

                   'angle_qubit':-np.pi/2+np.pi/26.5,  #-np.pi/2-np.pi/3,
                   'angle_JJ': 0,
                   'length':10,
                   'width':5,
                  'ic1': 0.8 * 1 * jc,
                  'ic2': 1.2 * 1.4 * jc,
                  'lm': 4.2e-12,

                    'separate_JJs':True,
                   'jj3_height': None,  # 0.2,
                   'jj3_width': None,  # 0.486,
                  }

bridge = elements.airbridge.AirBridgeGeometry(pad_width = 36,pad_length = 22,pad_distance = 62,
                                      narrow_width = 20, narrow_length = 46, sm_pad_length = 10,
                                      sm_pad_distance = 70, layer_configuration=sample.layer_configuration)



##################### Coaxmon 1 #####################
offset = 165
coaxmon1 = elements.coaxmon.Coaxmon(name='Coaxmon1', center=(coupler_start + offset, central_line_y - 1000),
                                    center_radius=20,
                                    inner_couplers_radius=50,
                                    outer_couplers_radius=80,
                                    inner_ground_radius=80,
                                    outer_ground_radius=95,
                                    layer_configuration=sample.layer_configuration,
                                    Couplers=Couplers_qubit_mergemon, jj_params=jj_coaxmon_2JJ_mergemon,
                                    transformations={},
                                    calculate_capacitance=True, third_JJ=False,
                                    hole_in_squid_pad=False, jj_cap=True,
                                    JJ_pad_connection_shift=True, draw_bandages=True, inner_couplers_radius2=25,
                                    outer_couplers_radius2=40
                                    )

##################### Coaxmon 2 #####################
transformations = {'mirror': [(coupler_start + offset, central_line_y), (coupler_start + offset + 10, central_line_y)]}

coaxmon2 = elements.coaxmon.Coaxmon(name='Coaxmon2', center=(coupler_start + offset, central_line_y - 1000),
                                    center_radius=20,
                                    inner_couplers_radius=50,
                                    outer_couplers_radius=80,
                                    inner_ground_radius=80,
                                    outer_ground_radius=95,
                                    layer_configuration=sample.layer_configuration,
                                    Couplers=Couplers_qubit_mergemon, jj_params=jj_coaxmon_2JJ_mergemon,
                                    transformations=transformations,
                                    calculate_capacitance=True, third_JJ=False, hole_in_squid_pad=False, jj_cap=True,
                                    JJ_pad_connection_shift=True, draw_bandages=True, inner_couplers_radius2=25,
                                    outer_couplers_radius2=40)




######  3 and 4 actually (was 2 and 3)

offset = 1085
offset1 = 1080
transformations={'mirror':[(coupler_start+offset,central_line_y),(coupler_start+offset+10,central_line_y)]}
coaxmon3= elements.coaxmon.Coaxmon(name='Coaxmon3',center=(coupler_start+offset1,central_line_y-1000),
                          center_radius = 20,
                          inner_couplers_radius = 55,
                          outer_couplers_radius = 80,
                          inner_ground_radius = 50,
                          outer_ground_radius = 80,
                          layer_configuration = sample.layer_configuration,
                          Couplers=Couplers_qubit_mergemon,jj_params= jj_coaxmon_2JJ_mergemon,transformations={},
                          calculate_capacitance = True, third_JJ=False,
                                   hole_in_squid_pad=False, jj_cap= True,
                                   JJ_pad_connection_shift=True, draw_bandages=True, inner_couplers_radius2=25,
                          outer_couplers_radius2=40)


coaxmon4= elements.coaxmon.Coaxmon(name='Coaxmon4',center=(coupler_start+offset,central_line_y-1000),
                          center_radius = 20,
                          inner_couplers_radius = 55,
                          outer_couplers_radius = 80,
                          inner_ground_radius = 50,
                          outer_ground_radius = 80,
                          layer_configuration = sample.layer_configuration,
                          Couplers=Couplers_qubit_mergemon,jj_params= jj_coaxmon_2JJ_mergemon,transformations=transformations,
                          calculate_capacitance = True, third_JJ=False, hole_in_squid_pad=False, jj_cap = True,
                                   JJ_pad_connection_shift=True, draw_bandages = True, inner_couplers_radius2=25,
                          outer_couplers_radius2=40)

#coaxmon4= elements.coaxmon.Coaxmon(name='Coaxmon4',center=(coupler_start+offset,central_line_y-1000),
#                           center_radius = 20,
#                           inner_couplers_radius = 60,
#                           outer_couplers_radius = 80,
#                           inner_ground_radius = 80,
#                           outer_ground_radius = 95,
#                           layer_configuration = sample.layer_configuration,
#                           Couplers=Couplers_qubit_mergemon,jj_params= jj_coaxmon_2JJ_mergemon,transformations=transformations,
#                           calculate_capacitance = True, third_JJ=False, hole_in_squid_pad=False, jj_cap = True,
#                                    JJ_pad_connection_shift=True, draw_bandages = True, inner_couplers_radius2=25,
#                           outer_couplers_radius2=40)


offset = 1950
transformations={'mirror':[(coupler_start+offset,central_line_y),(coupler_start+offset+10,central_line_y)]}
coaxmon5= elements.coaxmon.Coaxmon(name='Coaxmon5',center=(coupler_start+offset,central_line_y-1000),
                          center_radius = 20,
                          inner_couplers_radius = 50,
                          outer_couplers_radius = 80,
                          inner_ground_radius = 80,
                          outer_ground_radius = 95,
                          layer_configuration = sample.layer_configuration,
                          Couplers=Couplers_qubit_mergemon,jj_params= jj_coaxmon_2JJ_mergemon,transformations={},
                          calculate_capacitance = True, third_JJ=False,
                                   hole_in_squid_pad=False, jj_cap= True,
                                   JJ_pad_connection_shift=True, draw_bandages=True, inner_couplers_radius2=25,
                          outer_couplers_radius2=40)


transformations = {'mirror': [(coupler_start + offset, central_line_y), (coupler_start + offset + 10, central_line_y)]}
coaxmon6 = elements.coaxmon.Coaxmon(name='Coaxmon6', center=(coupler_start + offset, central_line_y - 1000),
                                    center_radius=20,
                                    inner_couplers_radius=50,
                                    outer_couplers_radius=80,
                                    inner_ground_radius=80,
                                    outer_ground_radius=95,
                                    layer_configuration=sample.layer_configuration,
                                    Couplers=Couplers_qubit_mergemon, jj_params=jj_coaxmon_2JJ_mergemon,
                                    transformations=transformations,
                                    calculate_capacitance=True, third_JJ=False,
                                    hole_in_squid_pad=False, jj_cap=True,
                                    JJ_pad_connection_shift=True, draw_bandages=True, inner_couplers_radius2=25,
                                    outer_couplers_radius2=40)


