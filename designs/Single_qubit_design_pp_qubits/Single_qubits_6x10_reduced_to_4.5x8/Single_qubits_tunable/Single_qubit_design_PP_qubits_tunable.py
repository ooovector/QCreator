import sys
sys.path.append('..\\..\\..\\..')
import gdspy
import numpy as np
from importlib import reload
from copy import deepcopy
from QCreator import elements
from QCreator import general_sample_creator as creator
from QCreator import meshing
reload(gdspy)

### to have 50 Oms impedance with eps=11.45
#wet etching subtracts 0.5 um, so we add that to all structures where it matters, coplers,Junction region and Fluxline
d = 0.5
tl_core = 21.+2*d
tl_gap = 12.-2*d
tl_ground = 6.#<-- changed from 10. to 5.

resonator_core = 15+2*d
resonator_gap = 10-2*d
resonator_ground = 15 #5
resonator_tl_ground= 14 +2*d #changed to 14 from 13

pad_offset = 550


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

sample = creator.Sample('Single_qubits_PP_tunable_qubits',layers_configuration)

#specify sample vertical and horizontal lengths
sample.chip_geometry.sample_vertical_size=4.5e3
sample.chip_geometry.sample_horizontal_size=8e3

chip_edge_ground = elements.ChipEdgeGround(sample.chip_geometry, sample.layer_configuration, sample.pads,edge=10)
sample.add(chip_edge_ground)

# 1. Create contact pads for 4.5*8 pcb WMI from Hans:
pads_left = []
pads_right = []
for pad_side_id in range(1):
    pad = elements.Pad('pad-left-' + str(pad_side_id),
                       (pad_offset, sample.chip_geometry.sample_vertical_size / 2), np.pi, tl_core,
                       tl_gap, tl_ground,
                       layer_configuration=sample.layer_configuration, chip_geometry=sample.chip_geometry,
                       **elements.reduced_pad_geometry())
    pads_left.append(pad)
    sample.add(pad)
    pad = elements.Pad('pad-right-' + str(pad_side_id),
                       (sample.chip_geometry.sample_horizontal_size - pad_offset,
                        sample.chip_geometry.sample_vertical_size / 2), 0, tl_core,
                       tl_gap, tl_ground,
                       layer_configuration=sample.layer_configuration, chip_geometry=sample.chip_geometry,
                       **elements.reduced_pad_geometry())
    pads_right.append(pad)
    sample.add(pad)

pads_top = []
pads_bottom = []
huns_offset=2470
pad_bottom_1 = elements.Pad('pad-bottom-' + str(1),
                   (sample.chip_geometry.sample_horizontal_size / 2+ huns_offset *(- 1), sample.chip_geometry.sample_vertical_size -pad_offset),
                   np.pi / 2, tl_core, tl_gap, tl_ground,
                   layer_configuration=sample.layer_configuration, chip_geometry=sample.chip_geometry,
                   **elements.reduced_pad_geometry())
pad_bottom_2 = elements.Pad('pad-bottom-' + str(2),
                   (sample.chip_geometry.sample_horizontal_size / 2+ huns_offset, sample.chip_geometry.sample_vertical_size -pad_offset),
                   np.pi / 2, tl_core, tl_gap, tl_ground,
                   layer_configuration=sample.layer_configuration, chip_geometry=sample.chip_geometry,
                   **elements.reduced_pad_geometry())
pads_top.append(pad_bottom_1)
pads_top.append(pad_bottom_2)
sample.add(pad_bottom_1)
sample.add(pad_bottom_2)
pad = elements.Pad('pad-top-' + str(pad_side_id),
                   (sample.chip_geometry.sample_horizontal_size / 2,
                     pad_offset),
                   -np.pi / 2, tl_core, tl_gap, tl_ground,
                   layer_configuration=sample.layer_configuration, chip_geometry=sample.chip_geometry,
                   **elements.reduced_pad_geometry())
pads_bottom.append(pad)
sample.add(pad)

p1 = pads_left[0]
p2 = pads_right[0]



width = 250+2*d
height= 450+2*d
gap   = 50-2*d
ground_w = 780#-2*d
ground_h   = 780#-2*d
ground_t   = 50+d


tight = [True,6-2*d]
Couplers=[elements.pp_transmon.PP_Transmon_Coupler(0,0,25+2*d,'left',coupler_type = 'coupler',heightl = (331-(15+2*d))/(ground_h),
                                                   w=resonator_core,s=resonator_gap,g=resonator_ground,shift_to_qubit=46-d,tight =tight)]

Couplers_flux=[elements.pp_transmon.PP_Transmon_Coupler(0,0,25+2*d,'right',coupler_type = 'coupler',heightr = (331-(15+2*d))/(ground_h),
                                                   w=resonator_core,s=resonator_gap,g=resonator_ground,shift_to_qubit=46-d,tight = tight)]


# b_g   = 19 # from JJ Design for JJ4q
JJ_pad_offset_x = 10#-2*d # for JJ_manhatten #for the JJ connections pads between the PPs
JJ_pad_offset_y = 16+2*d # JJ design

a1    = np.sqrt(0.15*0.3)*1.03 #Junction height in um
a2    = a1 # Junction width in um


#jj_pp = { 'a1':a1,"a2":a2,'angle_JJ':0,'manhatten':True,'h_w':5 +2*d,'h_d':8+2*d,'squid':False,'bandages_extension':1.25,'connection_pad_width':0.6,'connection_pad_gap':0.5-d,'bandages_edge_shift':3.5,'bridge_translate':[0.5,0,-0.5,4*d],'translate':[4*d,0] }# hole sizes for the JJs

jj_pp = { 'a1':a1,"a2":a2,'angle_JJ':0,'manhatten':True,'h_w':5 +2*d,'h_d':8-d,'squid':False,'bandages_extension':1.25,'connection_pad_width':0.6,'connection_pad_gap':0.5-d,'bandages_edge_shift':3.5,'bridge_translate':[d,-d,-d,d],'translate':[0,-d,0,d] }# hole sizes for the JJs


a11 = 0.477297
a12 = 0.1002324
a2  = 0.1
###### define parameters for the tunable qubits
jj_pp_flux = { 'a12':a11,"a11":a12,"a2":a2,'angle_JJ':0,'manhatten':True,'h_w':3+2*d ,'h_d':8-d,'squid':True,'loop_h': 12,'bandages_extension':1.25,'connection_pad_width':0.6,'connection_pad_gap':0.5-d,'strip1_extension':15,'strip2_extension':8,'bandages_edge_shift':3.5,'bridge_translate':[d,0,-d,4*d],'translate':[0,0,0,4*d],'adjust_holes':True}# hole sizes for the JJs

flux_distance = 15-2*d
l, t_m, t_r, gp, l_arm, h_arm, s_gap,asymmetry = 160-flux_distance+0.5, 5+2*d, 3+2*d, 5, [40/3-2*d,58.85-2*d], 50, resonator_gap,15-4.7
flux = {'l':l,'t_m':t_m,'t_r':t_r,'flux_distance':flux_distance,'gap':gp,'l_arm':l_arm,'h_arm':h_arm,'s_gap':s_gap,'g':resonator_ground,'w':resonator_core,'s':resonator_gap,'asymmetry':asymmetry,'loop_h': 10 }

#define parameters for test SNAIl
a11 = 0.4
a2  = 0.4
a12 = a11*0.3
jj_pp_snail = { 'a11':a11,"a12":a12,"a2":a2,'angle_JJ':0,'manhatten':True,'h_w':3+2*d ,'h_d':8+2*d,'snail':True,'loop_h': 12,'bandages_extension':1.25,'connection_pad_width':0.6,'connection_pad_gap':0.5-d,'strip1_extension':15,'strip2_extension':8,'bandages_edge_shift':3.5,'snail_extension':2,'snail_reach':10}# hole sizes for the JJs


# draw 2 tunable qubits
offset_x=-1000
offset_y=-750
center=(3300,4500+offset_y)
transmon1_left_flux = elements.pp_transmon.PP_Transmon(name='Q1_flux_left', center=center,
                                           width=width,
                                           height=height,
                                           bridge_gap=JJ_pad_offset_x+13,
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
                                           fluxline_params=flux,
                                           )
center=(6700+offset_x-1000+200,4500+offset_y)
transmon2_right_flux = elements.pp_transmon.PP_Transmon(name='Q2_flux_right', center=center,
                                           width=width,
                                           height=height,
                                           bridge_gap=JJ_pad_offset_x+13,
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
                                           fluxline_params=flux,
                                           )
sample.add(transmon1_left_flux)
sample.add(transmon2_right_flux)

### add fixed frequency transmon qubit without microwave line
center = (2000,1700+offset_y)

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
                          transformations = {'rotate':[-np.pi/2,center]},
                          )
sample.add(transmon1_left_fixed)

center = (6200+offset_x,1700+offset_y)

transmon1_right_fixed = elements.pp_transmon.PP_Transmon(name='Q2_fixed',center=center,
                          width = width,
                          height = height,
                          bridge_gap = JJ_pad_offset_x+13,
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
                          fluxline_params=flux,
                          )
sample.add(transmon1_right_fixed)


#add test structures + a Test SNAIL
shiftx = 600
shifty = -530
center=(6400+shiftx,1200+shifty)
JJ_test_structure = elements.pp_transmon.PP_Transmon(name='JJ_test',center=center,
                          width = 300,
                          height = 300,
                          bridge_gap = JJ_pad_offset_x+13,
                          bridge_w   = JJ_pad_offset_y ,
                          gap = gap,
                          ground_w = 700,
                          ground_h = 500,
                          ground_t = 10,
                          jj_params= jj_pp_flux,
                          layer_configuration = sample.layer_configuration,
                          Couplers = [],
                          calculate_capacitance = False,
                          transformations = {'rotate':[np.pi/2,center]},
                          )
sample.add(JJ_test_structure)
center=(5900+shiftx,1200+shifty)
JJ_test_structure1 = elements.pp_transmon.PP_Transmon(name='JJ_test1',center=center,
                          width = 300,
                          height = 300,
                          bridge_gap = JJ_pad_offset_x,
                          bridge_w   = JJ_pad_offset_y ,
                          gap = gap,
                          ground_w = 700,
                          ground_h = 500,
                          ground_t = 10,
                          jj_params= jj_pp,
                          layer_configuration = sample.layer_configuration,
                          Couplers = [],
                          calculate_capacitance = False,
                          transformations = {'rotate':[np.pi/2,center]},
                          )
sample.add(JJ_test_structure1)

center=(6400+shiftx,1900+shifty)
JJ_test_structure2 = elements.pp_transmon.PP_Transmon(name='JJ_test1',center=center,
                          width = 300,
                          height = 300,
                          bridge_gap = JJ_pad_offset_x+13,
                          bridge_w   = JJ_pad_offset_y ,
                          gap = gap,
                          ground_w = 700,
                          ground_h = 500,
                          ground_t = 10,
                          jj_params= jj_pp_snail,
                          layer_configuration = sample.layer_configuration,
                          Couplers = [],
                          calculate_capacitance = False,
                          transformations = {'rotate':[np.pi/2,center]},
                          )
sample.add(JJ_test_structure2)



logos=elements.WMILogos((700,3500),(7300,3500),layers_configuration)
sample.add(logos)
#sample.draw_design()
markers = elements.AlignmentMarkers((470,470),(sample.chip_geometry.sample_horizontal_size,sample.chip_geometry.sample_vertical_size),10,sample.layer_configuration)
sample.add(markers)
markers2 = elements.AlignmentMarkers((485,485),(sample.chip_geometry.sample_horizontal_size,sample.chip_geometry.sample_vertical_size),4,sample.layer_configuration)
sample.add(markers2)
markers3 = elements.AlignmentMarkers((500,500),(sample.chip_geometry.sample_horizontal_size,sample.chip_geometry.sample_vertical_size),1,sample.layer_configuration)
sample.add(markers3)



#sample.draw_design()
#sample.watch()