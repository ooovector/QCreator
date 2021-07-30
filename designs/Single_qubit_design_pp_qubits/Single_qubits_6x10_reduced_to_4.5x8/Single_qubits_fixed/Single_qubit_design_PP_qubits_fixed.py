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

### to have 50 Oms impedance with eps=11.75
# tl_core = 20.
# tl_gap = 12.
### to have 50 Oms impedance with eps=11.45
#wet etching subtracts 0.5 um, so we add that to all structures where it matters, coplers,Junction region and Fluxline
d = 0.5
tl_core = 21+2*d
tl_gap = 12-2*d
tl_ground = 6.#<-- changed from 10. to 5.

resonator_core = 15+2*d
resonator_gap = 10-2*d
resonator_ground = 15#5
resonator_tl_ground=13+2*d

pad_offset = 550


jc = 0.5

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

sample = creator.Sample('Single_qubits_PP_fixed_freqs',layers_configuration)

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


p1 = pads_left[0]
p2 = pads_right[0]


################################
tight = [True,6-d]
Couplers=[elements.pp_transmon.PP_Transmon_Coupler(0,0,50+2*d,'left',coupler_type = 'coupler',heightl = 0.6,
                                                   w=resonator_core,s=resonator_gap,g=resonator_ground,shift_to_qubit=46-2*d,tight=tight)]

width = 250+2*d
height= 450+2*d
gap   = 50-2*d
ground_w = 680+40+30+30-2*d
ground_h   = 680+40+30+30-2*d
ground_t   = 50+2*d

JJ_pad_offset_x = 10 # for JJ_manhatten #for the JJ connections pads between the PPs
JJ_pad_offset_y = 16 # JJ design

a1    = 0.226 #Junction height in um
a2    = 0.226 # Junction width in um


jj_pp = { 'a1':a1,"a2":a2,'angle_JJ':0,'manhatten':True,'h_w':5 +2*d,'h_d':8+2*d,'squid':False,'bandages_extension':1.25,'connection_pad_width':0.6,'connection_pad_gap':0.5-d,'bandages_edge_shift':3.5, }# hole sizes for the JJs


# draw 4 fixed frequency qubits
offset_x=-1000
offset_y=-750
center=(3300,4500+offset_y)
transmon1_left_top = elements.pp_transmon.PP_Transmon(name='left_top', center=center,
                                           width=width,
                                           height=height,
                                           bridge_gap=JJ_pad_offset_x,
                                           bridge_w=JJ_pad_offset_y,
                                           gap=gap,
                                           ground_w=ground_w,
                                           ground_h=ground_h,
                                           ground_t=ground_t,
                                           jj_params=jj_pp,
                                           layer_configuration=sample.layer_configuration,
                                           Couplers=Couplers,
                                           calculate_capacitance=False,
                                           remove_ground={'left': 0, 'right': 0, 'top': 0, 'bottom': 0},
                                           shoes=[],
                                           transformations={'rotate':[np.pi/2,center]},
                                           )
center=(6700+offset_x-1000+200,4500+offset_y)
transmon2_right_top = elements.pp_transmon.PP_Transmon(name='right_top', center=center,
                                           width=width,
                                           height=height,
                                           bridge_gap=JJ_pad_offset_x,
                                           bridge_w=JJ_pad_offset_y,
                                           gap=gap,
                                           ground_w=ground_w,
                                           ground_h=ground_h,
                                           ground_t=ground_t,
                                           jj_params=jj_pp,
                                           layer_configuration=sample.layer_configuration,
                                           Couplers=Couplers,
                                           calculate_capacitance=False,
                                           remove_ground={'left': 0, 'right': 0, 'top': 0, 'bottom': 0},
                                           shoes=[],
                                           transformations={'rotate':[np.pi/2,center]},
                                           )

sample.add(transmon1_left_top)
sample.add(transmon2_right_top)

center = (2000,1500+offset_y)

transmon3_left_bottom = elements.pp_transmon.PP_Transmon(name='left_bottom',center=center,
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
sample.add(transmon3_left_bottom)

center = (6200+offset_x,1500+offset_y)

transmon4_right_bottom = elements.pp_transmon.PP_Transmon(name='right_bottom',center=center,
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
sample.add(transmon4_right_bottom)
################# add 2 test structures to measure the resistance of JJs
shiftx = 600
shifty = -530
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
                          jj_params= jj_pp,
                          layer_configuration = sample.layer_configuration,
                          Couplers = [],
                          calculate_capacitance = False,
                          transformations = {'rotate':[np.pi/2,center]},
                          )
sample.add(JJ_test_structure)



logos=elements.WMILogos((700,3500),(7300,3500),layers_configuration)
sample.add(logos)
sample.draw_design()
markers = elements.AlignmentMarkers((470,470),(sample.chip_geometry.sample_horizontal_size,sample.chip_geometry.sample_vertical_size),10,sample.layer_configuration)
sample.add(markers)
markers2 = elements.AlignmentMarkers((485,485),(sample.chip_geometry.sample_horizontal_size,sample.chip_geometry.sample_vertical_size),4,sample.layer_configuration)
sample.add(markers2)
markers3 = elements.AlignmentMarkers((500,500),(sample.chip_geometry.sample_horizontal_size,sample.chip_geometry.sample_vertical_size),1,sample.layer_configuration)
sample.add(markers3)
