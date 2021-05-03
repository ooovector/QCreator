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

sample = creator.Sample('Two-qubits-PP',layers_configuration)

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
                   (sample.chip_geometry.sample_horizontal_size / 2+ huns_offset *(- 1), pad_offset),
                   -np.pi / 2, tl_core, tl_gap, tl_ground,
                   layer_configuration=sample.layer_configuration, chip_geometry=sample.chip_geometry,
                   **elements.default_pad_geometry())
pad_bottom_2 = elements.Pad('pad-bottom-' + str(2),
                   (sample.chip_geometry.sample_horizontal_size / 2+ huns_offset, pad_offset),
                   -np.pi / 2, tl_core, tl_gap, tl_ground,
                   layer_configuration=sample.layer_configuration, chip_geometry=sample.chip_geometry,
                   **elements.default_pad_geometry())
pads_bottom.append(pad_bottom_1)
pads_bottom.append(pad_bottom_2)
sample.add(pad_bottom_1)
sample.add(pad_bottom_2)
pad = elements.Pad('pad-top-' + str(pad_side_id),
                   (sample.chip_geometry.sample_horizontal_size / 2,
                    sample.chip_geometry.sample_vertical_size - pad_offset),
                   np.pi / 2, tl_core, tl_gap, tl_ground,
                   layer_configuration=sample.layer_configuration, chip_geometry=sample.chip_geometry,
                   **elements.default_pad_geometry())
pads_top.append(pad)
sample.add(pad)

p1 = pads_left[0]
p2 = pads_top[0]

################################
# resonator parameters:
resonator_core = 8
resonator_gap = 7
resonator_ground = 5
############### Draw single qubits
Couplers=[elements.pp_transmon.PP_Transmon_Coupler(0,0,16,'left',coupler_type = 'coupler',heightl = 0.3,heightr=1,
                                                   w=resonator_core,s=resonator_gap,g=resonator_ground)
         ]
width = 200
height= 200
gap   = 50
ground_w = 600
ground_h   = 400
ground_t   = 10
# b_g   = 19 # from JJ Design for JJ4q
JJ_pad_offset_x = 16 # for JJ_manhatten #for the JJ connections pads between the PPs
JJ_pad_offset_y = 16 # JJ design

a1    = 0.15 #Junction height in um
a2    = 0.30 # Junction width in um

#jj_pp = { 'a1':a1,"a2":a2,'angle_JJ':np.pi/2}
jj_pp = { 'a1':a1,"a2":a2,'angle_JJ':0,'manhatten':True,'h_w':5 ,'h_d':8 }# hole sizes for the JJs
empty_ground = 0.66
#Qubit 1
Qubit1 = elements.pp_transmon.PP_Transmon(name='PP_Transmon2',center=(4000,2500),
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
              transformations = {},
              remove_ground = {'right':empty_ground},
                                 calculate_capacitance=False
                                 )
#Qubit 2
Qubit2 = elements.pp_transmon.PP_Transmon(name='PP_Transmon2',center=(5000,2500),
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
              transformations = {},
              remove_ground = {'left':empty_ground},
              secret_shift = 3,
                             calculate_capacitance=False
                                 )


#
# ##################### add tunable couplers
# TunC = elements.pp_squid_coupler.PP_Squid_C(name=TC['name'],center=(self.center[0]+self.d1+Q1['g_w']/2+TC['g_w']/2,self.center[1]),
#                   width = TC['width'],
#                   height = TC['height'],
#                   bridge_gap = TC['b_g'],
#                   bridge_w   = TC['b_w'] ,
#                   gap = TC['gap'],
#                   g_w = TC['g_w'],
#                   g_h = TC['g_h'],
#                   g_t = TC['g_t'],
#                   jj_params= TC['jj_pp'],
#                   layer_configuration = self.layers_configuration,
#                   Couplers = TC['Couplers'],
#                   transformations = TC['transformations'],
#                   fluxline_params=TC['fluxline'],
#                   remove_ground = {'left':1,'right':1},
#                   secret_shift=3+3,
#                   calculate_capacitance=True,
#                   arms = TC['arms']
#                          )

