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

resonator_core = 8
resonator_gap = 7
resonator_ground = 20

pad_offset = 800
pad_element_offset = 400
qubit_position_offset = 800

coupler_start = pad_offset + pad_element_offset
coupler_delta = 500
coupler_length = 320
num_couplers = 1

jc = 0.5e-6 # uA/um^2

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
Couplers_qubit_alone=[elements.coaxmon.CoaxmonCoupler(arc_start=-1/6-1/100+shift,arc_finish=-3/6+1/100+shift,phi=-1/2,
                                          coupler_type= 'grounded', w =7,s=4, g=20),
        elements.coaxmon.CoaxmonCoupler(arc_start=3/6+1/100+shift,arc_finish=5/6-1/100+shift,phi=1/2,
                                          coupler_type=None, w =8,g=20,s=7), #one upper
          elements.coaxmon.CoaxmonCoupler(arc_start=1/6+1/100+shift+1/20,arc_finish=3/6-1/100+shift-1/10,phi=phi1,
                                          coupler_type='coupler', w =8,g=20,s=7),# for resonator
          elements.coaxmon.CoaxmonCoupler(arc_start=-1/6+1/100+1+shift,arc_finish=1/6-1/100+1+shift,phi=1,
                                          coupler_type=None, w =10),
          elements.coaxmon.CoaxmonCoupler(arc_start=-1/6+1/100+shift,arc_finish=1/6-1/100+shift,phi=0,
                                          coupler_type=None, w =10,g=20),
          elements.coaxmon.CoaxmonCoupler(arc_start=-5/6+1/100+shift,arc_finish=-3/6-1/100+shift,phi=-3/4,#1,
                                          coupler_type=None,w=10,g=20)
]
Couplers_qubit_alone1=[elements.coaxmon.CoaxmonCoupler(arc_start=-1/6-3/100+shift,arc_finish=-3/6-1/100+shift,phi=-1/2,
                                          coupler_type= 'grounded', w =7,s=4, g=20),
        elements.coaxmon.CoaxmonCoupler(arc_start=3/6+1/100+shift,arc_finish=5/6-1/100+shift,phi=1/2,
                                          coupler_type=None, w =8,g=10,s=7), #one upper
          elements.coaxmon.CoaxmonCoupler(arc_start=1/6+1/100+shift+1/20,arc_finish=3/6-1/100+shift-1/10,phi=phi1,
                                          coupler_type='coupler', w =8,g=20,s=7),# for resonator
          elements.coaxmon.CoaxmonCoupler(arc_start=-1/6+1/100+1+shift,arc_finish=1/6-1/100+1+shift,phi=1,
                                          coupler_type=None, w =10),
          elements.coaxmon.CoaxmonCoupler(arc_start=-1/6+1/100+shift,arc_finish=1/6-1/100+shift,phi=0,
                                          coupler_type=None, w =10,g=40),
          elements.coaxmon.CoaxmonCoupler(arc_start=-5/6+1/100+shift,arc_finish=-3/6-3/100+shift,phi=-3/4,#1,
                                          coupler_type=None,w=10,g=40)
]


jj_coaxmon_sm_SQUID = {'a1':20,
               'b1':0.2,
               'a2':0.45,
               'b2':0.243,
               'c1':0.45,
               'c2':10,
               'angle_qubit':-np.pi/2+np.pi/60,#-np.pi/2-np.pi/3+np.pi/16,
               'angle_JJ': 0,
               'length':10,
               'width':5,
              'ic1': 0.45 * 0.45 * jc,
              'ic2': -0.2 * 0.243 * jc,
              'ic3': 0.2 * 0.243 * jc,
              'lm': 4.2e-12
              }
jj_coaxmon_big_rad = {'a1':20,#65,
               'b1':0.3,
               'a2':0.55,
               'b2':0.3,
               'c1':0.55,
               'c2':10,
               'angle_qubit':-np.pi/2+np.pi/15,#-np.pi+np.pi/30,
               'angle_JJ': 0,
               'length':10,
               'width':5,
              'ic1': 0.55 * 0.55 * jc,
              'ic2': -0.3 * 0.3 * jc,
              'ic3': 0.3 * 0.3 * jc,
              'lm': 4.2e-12
              }
jj_coaxmon_2JJ = {'a1':20,
               'b1':0.15,#0.16,
               'a2':0.2,#0.45,
               'b2':0.15,#0.2,
               'c1':0.3,#0.486,
               'c2':10,
               'angle_qubit':-np.pi/2+np.pi/26.5,#-np.pi/2-np.pi/3,
               'angle_JJ': 0,
               'length':10,
               'width':5,
              'ic1': 0.2 * 0.3 * jc,
              'ic2': 0.15 * 0.15 * jc,
              'lm': 4.2e-12
              }

bridge = elements.airbridge.AirBridgeGeometry(pad_width = 36,pad_length = 22,pad_distance = 62,
                                      narrow_width = 20, narrow_length = 46, sm_pad_length = 10,
                                      sm_pad_distance = 70, layer_configuration=sample.layer_configuration)
offset=165#200
transformations={'mirror':[(coupler_start+offset,central_line_y),(coupler_start+offset+10,central_line_y)]}
coaxmon1= elements.coaxmon.Coaxmon(name='Coaxmon1',center=(coupler_start+offset,central_line_y-1000),
                          center_radius = 100,
                          inner_couplers_radius = 140,
                          outer_couplers_radius = 200,
                          inner_ground_radius = 230,
                          outer_ground_radius = 250,
                          layer_configuration = sample.layer_configuration,
                          Couplers=Couplers_qubit_alone1,jj_params= jj_coaxmon_sm_SQUID,transformations=transformations,
                          calculate_capacitance = True, third_JJ=True, hole_in_squid_pad=False,
                                   JJ_pad_connection_shift=True, draw_bandages=True)

offset = 1085
offset1 = 1080
transformations={'mirror':[(coupler_start+offset,central_line_y),(coupler_start+offset+10,central_line_y)]}
coaxmon2= elements.coaxmon.Coaxmon(name='Coaxmon2',center=(coupler_start+offset1,central_line_y-1080),
                          center_radius = 170,
                          inner_couplers_radius = 210,
                          outer_couplers_radius = 270,
                          inner_ground_radius = 300,
                          outer_ground_radius = 320,
                          layer_configuration = sample.layer_configuration,
                          Couplers=Couplers_qubit_alone,jj_params= jj_coaxmon_big_rad,transformations={},
                          calculate_capacitance = True, third_JJ=True,
                                   hole_in_squid_pad=False,
                                   JJ_pad_connection_shift=True, draw_bandages=True)

coaxmon3= elements.coaxmon.Coaxmon(name='Coaxmon3',center=(coupler_start+offset,central_line_y-1000),
                          center_radius = 100,
                          inner_couplers_radius = 140,
                          outer_couplers_radius = 200,
                          inner_ground_radius = 230,
                          outer_ground_radius = 250,
                          layer_configuration = sample.layer_configuration,
                          Couplers=Couplers_qubit_alone1,jj_params= jj_coaxmon_sm_SQUID,transformations=transformations,
                          calculate_capacitance = True, third_JJ=True, hole_in_squid_pad=False,
                                   JJ_pad_connection_shift=True, draw_bandages = True)


offset = 1950
transformations={'mirror':[(coupler_start+offset,central_line_y),(coupler_start+offset+10,central_line_y)]}
coaxmon4= elements.coaxmon.Coaxmon(name='Coaxmon4',center=(coupler_start+offset,central_line_y-1000),
                          center_radius = 100,
                          inner_couplers_radius = 140,
                          outer_couplers_radius = 200,
                          inner_ground_radius = 230,
                          outer_ground_radius = 250,
                          layer_configuration = sample.layer_configuration,
                          Couplers=Couplers_qubit_alone,jj_params= jj_coaxmon_2JJ,transformations=transformations,
                          calculate_capacitance = True, third_JJ=False, hole_in_squid_pad=False,
                                   JJ_pad_connection_shift=True, draw_bandages = True)

#############################################add xmons
jj_geometry3 = {
    'gwidth': 40,#60,
    'gheight': 28,
    'iwidth': 32,#52,
    'iheight': 6.5,
    'ithick': 4,
    'iopen': 21,#20,
    'fheight1': 20,
    'fheight2': 40,
    'hdist': 4,
    'fshoulder': 15,
    'fcore': 4,
    'fgap': 4,
    'gter': 4,
    'lm': 4e-12
        }
jj_geometry = {
    'gwidth': 40,#60,
    'gheight': 24,
    'iwidth': 32,#52,
    'iheight': 10,
    'ithick': 4,
    'iopen': 22,#20,
    'fheight1': 20,
    'fheight2': 40,
    'hdist': 4,
    'fshoulder': 15,
    'fcore': 4,
    'fgap': 4,
    'gter': 4,
    'bandages': True,
    'lm': 4e-12
        }

bandage_geometry = {
    'bandage_side_height': 3,
    'bandage_side_width': 5,
    'bandage_up_height': 8,
    'bandage_up_width': 5,
    'up_shift': 2,
    'side_shift': 1
}


jj = {
    'type': 2,
    'up_rect_h': 8,
    'up_rect_w': 8,
    'side_rect_h': 5,
    'side_rect_w': 5,
    'side_l_thick': 0.15,#0.2,
    'side_r_thick': 0.2,#0.486,
    'up_l_thick': 0.15,#0.16,
    'up_r_thick': 0.3,#0.45,
    'side_l_length': 7,#9,
    'side_r_length': 7,#9,
    'up_l_length': 9 - 0.32,
    'up_r_length': 9 - 0.32,
    'up_rect_shift': 4,
    'side_rect_shift': 3,
    'ic_l': 0.15*0.15*jc,
    'ic_r': 0.2*0.3*jc
}

jj3 = {
    'type': 3,
    'up_rect_h': 12,
    'up_rect_w': 12,
    'side_rect_h': 6,
    'side_rect_w': 6,
    'side_l_thick': 0.2,#0.2,
    'side_r_thick': 0.45,#0.486,
    'up_l_thick': 0.243,#0.16,
    'up_r_thick': 0.45,#0.45,
    'side_l_length': 4,#9,
    'side_r_length': 4,#9,
    'up_l_length': 7 - 0.16,
    'up_r_length': 7 - 0.16,
    'ic_l': 0.243*0.2*jc,
    'ic_r': 0.45*0.45*jc,
    'ic3': 0.243 * 0.2 * jc
}

# this will be changed in the future
crab_terminals = {
    'up_w':8,
    'up_s':7,
    'up_g':20,
    'down_w':8,
    'down_s':7,
    'down_g':20,
    'left_w':10,
    'left_s':10,
    'left_g':20,
    'right_w':10,
    'right_s':10,
    'right_g':20
}



xmon1 = elements.xmon.Xmon(name = 'Xmon1',
                         center=(coupler_start+coupler_length+resonator_core/2+resonator_gap, central_line_y-1050),
                          length = 130,
                          width_gap = 15,
                          center_width = 15,
                          crab_position = ('up',),
                          crab_shoulder = 40,
                          crab_thickness = 40,
                          crab_terminals = crab_terminals,
                          ground_thickness = 20,
                          delete_ground = '',
                          jj_position = 'down',
                          jj_params1 = jj_geometry3,
                          jj_params2 = jj3,
                          aux_jj_params = {},
                          layer_configuration = sample.layer_configuration, hole_in_squid_pad=False)
xmon2 = elements.xmon.Xmon(name = 'Xmon2',
                           center=(coupler_start+1750+coupler_length+resonator_core/2+resonator_gap, central_line_y-1050),
                           length = 130,
                           width_gap = 15,
                           center_width = 15,
                           crab_position = ('up',),
                           crab_shoulder = 40,
                           crab_thickness = 40,
                           crab_terminals = crab_terminals,
                           ground_thickness = 20,
                           delete_ground = '',
                           jj_position = 'down',
                           jj_params1 = jj_geometry,
                           jj_params2 = jj,
                           aux_jj_params = bandage_geometry,
                           layer_configuration = sample.layer_configuration)



