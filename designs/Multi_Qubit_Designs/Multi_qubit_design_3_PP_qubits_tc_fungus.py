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
tl_ground = 5.#<-- changed from 10. to 5.

resonator_core = 8
resonator_gap = 7
resonator_ground = 5 #5

pad_offset = 800


jc = 1e-6 # uA/um^2

#Chip parameters

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

sample = creator.Sample('3Q_test',layers_configuration)

#specify sample vertical and horizontal lengths
sample.chip_geometry.sample_vertical_size=4.3e3
sample.chip_geometry.sample_horizontal_size=7e3
central_line_y = sample.chip_geometry.sample_vertical_size/2
central_line_x = 1000
resonator_coupler_length = 300


chip_edge_ground = elements.ChipEdgeGround(sample.chip_geometry, sample.layer_configuration, sample.pads)
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
eth_offset=2470
for pad_side_id in range(3):
    pad = elements.Pad('pad-bottom-' + str(pad_side_id),
                       (sample.chip_geometry.sample_horizontal_size / 2+ eth_offset * (pad_side_id - 1), pad_offset),
                       -np.pi / 2, tl_core, tl_gap, tl_ground,
                       layer_configuration=sample.layer_configuration, chip_geometry=sample.chip_geometry,
                       **elements.default_pad_geometry())
    pads_bottom.append(pad)
    sample.add(pad)
    pad = elements.Pad('pad-top-' + str(pad_side_id),
                       (sample.chip_geometry.sample_horizontal_size / 2 + eth_offset * (pad_side_id - 1),
                        sample.chip_geometry.sample_vertical_size - pad_offset),
                       np.pi / 2, tl_core, tl_gap, tl_ground,
                       layer_configuration=sample.layer_configuration, chip_geometry=sample.chip_geometry,
                       **elements.default_pad_geometry())
    pads_top.append(pad)
    sample.add(pad)

p1 = pads_left[0]
p2 = pads_right[0]

#Qubit parameters

Coupler1=[elements.pp_transmon.PP_Transmon_Coupler(0,0,16,'left',coupler_type = 'coupler',heightl = 0.3,
                                                   w=resonator_core,s=resonator_gap,g=resonator_ground)
         ]

Coupler2=[elements.pp_transmon.PP_Transmon_Coupler(0,0,16,'left',coupler_type = 'coupler',heightl = 0.3,
                                                   w=resonator_core,s=resonator_gap,g=resonator_ground)
         ]

Coupler3=[elements.pp_transmon.PP_Transmon_Coupler(0,0,16,'right',coupler_type = 'coupler',heightr = 0.3,
                                                   w=resonator_core,s=resonator_gap,g=resonator_ground)
         ]


Couplers_Squid1=[]

Couplers_Squid2=[]


width = 100
height= 250
gap   = 50
ground_w = 310
ground_h   = 300
ground_t   = 10

JJ_pad_offset_x = 10 # for JJ_manhatten
JJ_pad_offset_y = 16 # JJ design

#square junctions
a1    = np.sqrt(0.15*0.3) #Junction height in um
a2    = a1 # Junction width in um

#jj_pp = { 'a1':a1,"a2":a2,'angle_JJ':np.pi/2}
jj_pp = { 'a1':a1,"a2":a2,'angle_JJ':0,'manhatten':True,'h_w':5 ,'h_d':8 }# hole sizes for the JJs

#Coupler parameters
l, t_m, t_r, gp, l_arm, h_arm, s_gap = 500, resonator_core, 6, 5, 20, 50, resonator_gap
fluxline = {'l':l,'t_m':t_m,'t_r':t_r,'gap':gp,'l_arm':l_arm,'h_arm':h_arm,'s_gap':s_gap,'g':resonator_ground,'w':resonator_core,'s':resonator_gap}



lph,lg,lpw,lw = 85,0,55,0.5*200/7
rph,rg,rpw,rw = lph,lg,lpw,lw

arms = {}
width_tc = [140,60]
height_tc= [600,200]
gap_tc   = 1.9/7*200
ground_w_tc = 750
ground_h_tc   = 600
ground_t_tc   = 10





d = 30


transmon1 = elements.pp_transmon.PP_Transmon(name='PP_Transmon1',center=(2050-d,2000+ground_w_tc/4),
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
                          Couplers = Coupler1,
                          calculate_capacitance = False,
                          transformations = {},
                          #remove_ground = {'right':1,'top':1,'bottom':1},
                          )
transmon2 = elements.pp_transmon.PP_Transmon(name='PP_Transmon2',center=(2050+ground_w/2+ground_h_tc/2,2000+ground_w_tc/2),
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
                          Couplers = Coupler2,
                          calculate_capacitance = False,
                          transformations = {'rotate':[-np.pi/2,(2050+ground_w/2+ground_h_tc/2,2000+ground_w_tc/2)]},
                          #remove_ground = {'bottom':1,'right':1,'top':1},
                          )

transmon3 = elements.pp_transmon.PP_Transmon(name='PP_Transmon3',center=(2050+ground_w+ground_h_tc+d,2000+ground_w_tc/4),
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
                          Couplers = Coupler3,
                          calculate_capacitance = False,
                          transformations = {},
                          #remove_ground={'left': 1,'top':1,'bottom':1},
                          )

tc1 = elements.fungus_squid_coupler.Fungus_Squid_C(name='PP_Coupler1',center=(2050+ground_w/2+ground_h_tc/2,2000),
                          width = width_tc,
                          height = height_tc,
                          bridge_gap = JJ_pad_offset_x,
                          bridge_w   = JJ_pad_offset_y ,
                          gap = gap,
                          ground_w = ground_w_tc,
                          ground_h = ground_h_tc,
                          ground_t = ground_t_tc,
                          jj_params= jj_pp,
                          fluxline_params = fluxline,
                          arms = arms,
                          layer_configuration = sample.layer_configuration,
                          Couplers = Couplers_Squid1,
                          calculate_capacitance = False,
                          transformations = {'rotate':[-np.pi/2,(2050+ground_w/2+ground_h_tc/2,2000)]},
                          remove_ground = {'left':0.5,'top':1,'bottom':1,'right':1},
                          shoes = {1:(150,60),2:(150,60)},
                          asymmetry = 100
                          )



sample.add(transmon1)
sample.add(transmon2)
sample.add(transmon3)
sample.add(tc1)
sample.draw_design()
#sample.watch()