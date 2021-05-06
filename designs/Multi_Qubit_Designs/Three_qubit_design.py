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

sample = creator.Sample('Three-qubits-PP',layers_configuration)

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

############### Qubits and Coupler

#Qubit parameters
#origin of left bottom qubit
origin = [4000,2200]


width = 250
height= 400

gap   = 50

width  = 200
height = 2*width+gap


ground_t = 20
ground_w = 660+ground_t*2
ground_h = 660+ground_t*2

#square junctions
a1    = np.sqrt(0.15*0.3) #Junction height in um
a2    = a1 # Junction width in um

#jj_pp = { 'a1':a1,"a2":a2,'angle_JJ':np.pi/2}
jj_pp = { 'a1':a1,"a2":a2,'angle_JJ':0,'manhatten':True,'h_w':5 ,'h_d':8 }# hole sizes for the JJs

JJ_pad_offset_x = 10 # for JJ_manhatten
JJ_pad_offset_y = 16 # JJ design


sh = (70,20)

shoes1 = {1:sh,2:sh,3:sh,4:sh,'R1':np.pi/2,'R2':np.pi/4,'R3':np.pi/4,'R4':np.pi/4}

shoes2 = {1:sh,2:sh,3:sh,4:sh,'R1':np.pi/4,'R2':np.pi/2,'R3':np.pi/4,'R4':np.pi/4}

shoes3 = {1:sh,2:sh,3:sh,4:sh,'R':np.pi/4}

spacing = 740

center1 = (origin[0],origin[1])
center2 = (origin[0]+spacing+ground_w,origin[1])
center3 = (origin[0],origin[1]-spacing-ground_h)
center4 = (origin[0]+spacing+ground_w,origin[1]-spacing-ground_h)


#Coupler
arms = {}

width_tc    = [60,75]
height_tc   = [800,165]
height_tc2 = [860,165,1200]
gap_tc      = 70
ground_w_tc = 325
ground_h_tc = 950
ground_t_tc = 10

claw_tc = [10,50]

shift_y =gap_tc/2+width_tc[0]/2+claw_tc[0]

origin = [origin[0],origin[1]]
center_tc1 = (origin[0]+spacing/2+ground_w/2,origin[1]+shift_y)
center_tc2 = (origin[0]-shift_y,origin[1]-spacing/2-ground_w/2)
center_tc3 = (origin[0]+spacing+ground_w+shift_y,origin[1]-spacing/2-ground_w/2)
center_tc4 = (origin[0]+spacing/2+ground_w/2,origin[1]-shift_y-spacing-ground_w)

a = -250

air = [-20,40,100]

air2 = [200,40,100]


CC1 = [elements.pp_transmon.PP_Transmon_Coupler(0,0,16,'left',coupler_type = 'coupler',heightl = 0.2,w=resonator_core,s=resonator_gap,g=resonator_ground),
      elements.pp_transmon.PP_Transmon_Coupler(400,60,16,'bottom',coupler_type = 'coupler',w=resonator_core,s=resonator_gap,g=resonator_ground),
      ]

CC2 = [elements.pp_transmon.PP_Transmon_Coupler(0,0,16,'left',coupler_type = 'coupler',heightl = 0.2,w=resonator_core,s=resonator_gap,g=resonator_ground),
      elements.pp_transmon.PP_Transmon_Coupler(400,60,16,'top',coupler_type = 'coupler',w=resonator_core,s=resonator_gap,g=resonator_ground),
      ]

CC3 = [elements.pp_transmon.PP_Transmon_Coupler(0,0,16,'right',coupler_type = 'coupler',heightr = 0.2,w=resonator_core,s=resonator_gap,g=resonator_ground),
      elements.pp_transmon.PP_Transmon_Coupler(400,60,16,'bottom',coupler_type = 'coupler',w=resonator_core,s=resonator_gap,g=resonator_ground),
      ]



l, t_m, t_r, gp, l_arm, h_arm, s_gap = 100, resonator_core, 6, 5, 20, 50, resonator_gap
flux = {'l':l,'t_m':t_m,'t_r':t_r,'gap':gp,'l_arm':l_arm,'h_arm':h_arm,'s_gap':s_gap,'g':resonator_ground,'w':resonator_core,'s':resonator_gap}
#flux = {}


CC = [CC1,CC2,CC3]
X = 2
Y = 1
qubits   = []
couplers = []

center = (origin[0]+0*(spacing+ground_w),origin[1]+0*(spacing+ground_h))
Q1 = elements.pp_transmon.PP_Transmon(name='Q1', center=center,
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
                                           Couplers=CC[0],
                                           calculate_capacitance=False,
                                           remove_ground={'left': 1, 'right': 1, 'top': 1, 'bottom': 1},
                                           shoes=shoes1,
                                           transformations={}#transformations={'rotate': (np.pi / 4, center)},
                                           )

sample.add(Q1)
qubits.append(Q1)


center = (origin[0]+0*(spacing+ground_w),origin[1]+1*(spacing+ground_h))
Q2 = elements.pp_transmon.PP_Transmon(name='Q2', center=center,
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
                                           Couplers=CC[1],
                                           calculate_capacitance=False,
                                           remove_ground={'left': 1, 'right': 1, 'top': 1, 'bottom': 1},
                                           shoes=shoes2,
                                           transformations={}#transformations={'rotate': (np.pi / 4, center)},
                                           )

sample.add(Q2)
qubits.append(Q2)




center = (origin[0]+1*(spacing+ground_w)+150,origin[1]+1*(spacing+ground_h)/2)

Q3 = elements.pp_transmon.PP_Transmon(name='Q3', center=center,
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
                                           Couplers=CC[2],
                                           calculate_capacitance=False,
                                           remove_ground={'left': 1, 'right': 1, 'top': 1, 'bottom': 1},
                                           shoes=shoes3,
                                           transformations={'rotate': (np.pi / 4, center)},
                                           )

sample.add(Q3)
qubits.append(Q3)


center1 = (origin[0] - shift_y-gap/2-width, origin[1] +(spacing+ground_h)/2)

T1 = elements.fungus_squid_coupler.Fungus_Squid_C(name='PP_Coupler1', center=center1,
                                                  width=width_tc,
                                                  height=height_tc,
                                                  bridge_gap=JJ_pad_offset_x,
                                                  bridge_w=JJ_pad_offset_y,
                                                  gap=gap_tc,
                                                  ground_w=ground_w_tc,
                                                  ground_h=ground_h_tc,
                                                  ground_t=ground_t,
                                                  jj_params=jj_pp,
                                                  fluxline_params=flux,
                                                  layer_configuration=sample.layer_configuration,
                                                  Couplers=[],
                                                  calculate_capacitance=False,
                                                  transformations={'rotate': (-np.pi, center1)},
                                                  remove_ground={'left': 1, 'top': 1, 'bottom': 1, 'right': 1},
                                                  shoes={},
                                                  claw=claw_tc,
                                                  asymmetry=a,
                                                  air_bridge=air,
                                                  )

sample.add(T1)
couplers.append(T1)


center2 = (origin[0] +1*(spacing+ground_h)/2+width_tc[0]/2-3, origin[1] +(spacing+ground_h)/2+shift_y-claw_tc[0])


T2 = elements.y_squid_coupler.Y_Squid_C(name='Y_Coupler', center=center2,
                                                  width=width_tc,
                                                  height=height_tc2,
                                                  bridge_gap=JJ_pad_offset_x,
                                                  bridge_w=JJ_pad_offset_y,
                                                  gap=gap_tc,
                                                  ground_w=ground_w_tc,
                                                  ground_h=ground_h_tc,
                                                  ground_t=ground_t,
                                                  jj_params=jj_pp,
                                                  fluxline_params=flux,
                                                  layer_configuration=sample.layer_configuration,
                                                  Couplers=[],
                                                  calculate_capacitance=False,
                                                  transformations={'rotate': (np.pi/2, center2)},
                                                  remove_ground={'left': 1, 'top': 1, 'bottom': 1, 'right': 1},
                                                  shoes={},
                                                  claw=claw_tc,
                                                  asymmetry=a,
                                                  air_bridge=air2,
                                                  )

sample.add(T2)
couplers.append(T2)


#lets do it smartly
all_restricted = []
all_inverted = []
all_positive = []
for Q in qubits:
    Qu = Q.render()
    all_restricted.append(Qu['restrict'])
    all_inverted.append(Qu['pocket'])
    all_positive.append(Qu['positive'])
for Q in couplers:
    Qu = Q.render()
    all_restricted.append(Qu['restrict'])
    all_inverted.append(Qu['pocket'])
    all_positive.append(Qu['positive'])

one = gdspy.Rectangle((0,0),(0,0))
for i in all_restricted:
    one = gdspy.boolean(one,i,'or',layer=layers_configuration['vertical gridlines'])


for i in all_inverted:
    one = gdspy.boolean(one,i,'not',layer=layers_configuration['vertical gridlines'])

for i in all_positive:
    one = gdspy.boolean(one,i,'not',layer=layers_configuration['vertical gridlines'])


sample.total_cell.add(one)