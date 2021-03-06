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
### to have 50 Oms impedance with eps=11.45
tl_core = 22.
tl_gap = 12.
tl_ground = 6.#<-- changed from 10. to 5.

resonator_core = 15
resonator_gap = 10
resonator_ground = 15 #5
resonator_tl_ground=12

pad_offset = 800


jc = 1e-6 # uA/um^2

layers_configuration = {
    'total':0,
    'restricted area':10,
    'for removing':100,
    'JJs':1,
    'air bridges': 2,
    'air bridge pads': 3,
    'air bridge sm pads': 18,
    'vertical gridlines':15,
    'horizontal gridlines':16,
    'inverted':17,
    'bandages':19,
}

sample = creator.Sample('Single_qubits_PP_2',layers_configuration)

#specify sample vertical and horizontal lengths
sample.chip_geometry.sample_vertical_size=6e3
sample.chip_geometry.sample_horizontal_size=10e3

chip_edge_ground = elements.ChipEdgeGround(sample.chip_geometry, sample.layer_configuration, sample.pads)
sample.add(chip_edge_ground)

# 1. Create contact pads for 6*10 pcb WMI from Huns:
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
huns_offset=2470
pad_bottom_1 = elements.Pad('pad-bottom-' + str(1),
                   (sample.chip_geometry.sample_horizontal_size / 2+ huns_offset *(- 1), sample.chip_geometry.sample_vertical_size -pad_offset),
                   np.pi / 2, tl_core, tl_gap, tl_ground,
                   layer_configuration=sample.layer_configuration, chip_geometry=sample.chip_geometry,
                   **elements.default_pad_geometry())
pad_bottom_2 = elements.Pad('pad-bottom-' + str(2),
                   (sample.chip_geometry.sample_horizontal_size / 2+ huns_offset, sample.chip_geometry.sample_vertical_size -pad_offset),
                   np.pi / 2, tl_core, tl_gap, tl_ground,
                   layer_configuration=sample.layer_configuration, chip_geometry=sample.chip_geometry,
                   **elements.default_pad_geometry())
pads_top.append(pad_bottom_1)
pads_top.append(pad_bottom_2)
sample.add(pad_bottom_1)
sample.add(pad_bottom_2)
pad = elements.Pad('pad-top-' + str(pad_side_id),
                   (sample.chip_geometry.sample_horizontal_size / 2,
                     pad_offset),
                   -np.pi / 2, tl_core, tl_gap, tl_ground,
                   layer_configuration=sample.layer_configuration, chip_geometry=sample.chip_geometry,
                   **elements.default_pad_geometry())
pads_bottom.append(pad)
sample.add(pad)

p1 = pads_left[0]
p2 = pads_right[0]

################################

Couplers=[elements.pp_transmon.PP_Transmon_Coupler(0,0,50,'left',coupler_type = 'coupler',heightl = 0.6,
                                                   w=resonator_core,s=resonator_gap,g=resonator_ground)]

Couplers_flux=[elements.pp_transmon.PP_Transmon_Coupler(0,0,50,'right',coupler_type = 'coupler',heightr = 0.6,
                                                   w=resonator_core,s=resonator_gap,g=resonator_ground)]

CC1_mw = [elements.pp_transmon.PP_Transmon_Coupler(0,0,25,'right',coupler_type = 'coupler',heightr = 0.2,w=resonator_core,s=resonator_gap,g=resonator_ground,shift_to_qubit=175),
      elements.pp_transmon.PP_Transmon_Coupler(500,14,16,'top',coupler_type = 'coupler',w=resonator_core,s=resonator_gap,g=resonator_ground,shift_to_qubit=-25),
      ]

width = 250
height= 450
gap   = 50
ground_w = 680+40+30
ground_h   = 680+40+30
ground_t   = 50
# b_g   = 19 # from JJ Design for JJ4q
JJ_pad_offset_x = 16 # for JJ_manhatten #for the JJ connections pads between the PPs
JJ_pad_offset_y = 16 # JJ design

a1    = 0.15 #Junction height in um
a2    = 0.30 # Junction width in um


jj_pp = { 'a1':a1,"a2":a2,'angle_JJ':0,'manhatten':True,'h_w':5 ,'h_d':8,'squid':False }# hole sizes for the JJs




###### define parameters for the tunable qubits
jj_pp_flux = { 'a1':a1,"a2":a2,'angle_JJ':0,'manhatten':True,'h_w':5 ,'h_d':8,'squid':True }# hole sizes for the JJs
l, t_m, t_r, gp, l_arm, h_arm, s_gap = 100, resonator_core, 3, 5, 20, 50, resonator_gap
flux_distance = 20
flux = {'l':l,'t_m':t_m,'t_r':t_r,'flux_distance':flux_distance,'gap':gp,'l_arm':l_arm,'h_arm':h_arm,'s_gap':s_gap,'g':resonator_ground,'w':resonator_core,'s':resonator_gap}

# draw 2 tunable qubits
center=(3100,4500)
transmon1_left_flux = elements.pp_transmon.PP_Transmon(name='Q1_flux_left', center=center,
                                           width=width,
                                           height=height,
                                           bridge_gap=JJ_pad_offset_x,
                                           bridge_w=JJ_pad_offset_y,
                                           gap=gap,
                                           ground_w=ground_w,
                                           ground_h=ground_h,
                                           ground_t=ground_t,
                                           jj_params=jj_pp_flux,
                                           layer_configuration=sample.layer_configuration,
                                           Couplers=Couplers,
                                           calculate_capacitance=False,
                                           remove_ground={'left': 0, 'right': 0, 'top': 0, 'bottom': 0},
                                           shoes=[],
                                           transformations={'rotate':[np.pi/2,center]},
                                           fluxline_params=flux
                                           )
center=(6900,4500)
transmon2_right_flux = elements.pp_transmon.PP_Transmon(name='Q2_flux_right', center=center,
                                           width=width,
                                           height=height,
                                           bridge_gap=JJ_pad_offset_x,
                                           bridge_w=JJ_pad_offset_y,
                                           gap=gap,
                                           ground_w=ground_w,
                                           ground_h=ground_h,
                                           ground_t=ground_t,
                                           jj_params=jj_pp_flux,
                                           layer_configuration=sample.layer_configuration,
                                           Couplers=Couplers_flux,
                                           calculate_capacitance=False,
                                           remove_ground={'left': 0, 'right': 0, 'top': 0, 'bottom': 0},
                                           shoes=[],
                                           transformations={'rotate':[-np.pi/2,center]},
                                           fluxline_params=flux
                                           )

sample.add(transmon1_left_flux)
sample.add(transmon2_right_flux)

### add fixed frequency transmon qubit without microwave line
center = (2000,1500)

transmon1_left_fixed = elements.pp_transmon.PP_Transmon(name='Q1_fixed',center=center,
                          width = width,
                          height = height,
                          bridge_gap = JJ_pad_offset_x,
                          bridge_w   = JJ_pad_offset_y ,
                          gap = gap,
                          ground_w = ground_w,
                          ground_h = ground_h,
                          ground_t = ground_t,
                          jj_params= jj_pp,
                          layer_configuration = sample.layer_configuration,
                          Couplers = Couplers,
                          calculate_capacitance = False,
                          transformations = {'rotate':[-np.pi/2,center]}
                          )
sample.add(transmon1_left_fixed)

center = (6200,1500)

transmon1_right_fixed = elements.pp_transmon.PP_Transmon(name='Q2_fixed',center=center,
                          width = width,
                          height = height,
                          bridge_gap = JJ_pad_offset_x,
                          bridge_w   = JJ_pad_offset_y ,
                          gap = gap,
                          ground_w = ground_w,
                          ground_h = ground_h,
                          ground_t = ground_t,
                          jj_params= jj_pp_flux,
                          layer_configuration = sample.layer_configuration,
                          Couplers = Couplers_flux,
                          calculate_capacitance = False,
                          transformations = {'rotate':[np.pi/2,center]},
                          fluxline_params=flux
                          )
sample.add(transmon1_right_fixed)
################# add them for testing

center = (4000,1500)

transmon1_left_fixed = elements.pp_transmon.PP_Transmon(name='Q1_fixed',center=center,
                          width = width,
                          height = height,
                          bridge_gap = JJ_pad_offset_x,
                          bridge_w   = JJ_pad_offset_y ,
                          gap = gap,
                          ground_w = ground_w,
                          ground_h = ground_h,
                          ground_t = ground_t,
                          jj_params= jj_pp,
                          layer_configuration = sample.layer_configuration,
                          Couplers = Couplers,
                          calculate_capacitance = False,
                          transformations = {'rotate':[-np.pi/2,center]}
                          )
sample.add(transmon1_left_fixed)

center = (4900,4500)

transmon1_left_fixed = elements.pp_transmon.PP_Transmon(name='Q1_fixed',center=center,
                          width = width,
                          height = height,
                          bridge_gap = JJ_pad_offset_x,
                          bridge_w   = JJ_pad_offset_y ,
                          gap = gap,
                          ground_w = ground_w,
                          ground_h = ground_h,
                          ground_t = ground_t,
                          jj_params= jj_pp,
                          layer_configuration = sample.layer_configuration,
                          Couplers = Couplers,
                          calculate_capacitance = False,
                          transformations = {'rotate':[np.pi/2,center]}
                          )
sample.add(transmon1_left_fixed)




sample.draw_design()
#sample.watch()
