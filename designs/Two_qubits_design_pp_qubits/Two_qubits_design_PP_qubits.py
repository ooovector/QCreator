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
    'air bridge sm pads': 18,
    'vertical gridlines':15,
    'horizontal gridlines':16,
    'inverted':17,
    'bandages':19,
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
p2 = pads_top[0]

################################
# resonator parameters:
resonator_core = 8
resonator_gap = 7
resonator_ground = 5

############### Qubits and Coupler

#Qubit parameters
#origin of left qubit
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
shoes1 = {1:sh,2:sh,3:sh,4:sh,'R':np.pi/4}
shoes2 = {}#{1:(70,50)}
spacing = 900
center1 = (origin[0],origin[1])
center2 = (origin[0]+spacing+ground_w,origin[1])
center3 = (origin[0],origin[1]-spacing-ground_h)
center4 = (origin[0]+spacing+ground_w,origin[1]-spacing-ground_h)


#Coupler
arms = {}

width_tc    = [60,75]
height_tc   = [800,165]
gap_tc      = 70
ground_w_tc = 325
ground_h_tc = 950
ground_t_tc = 10

claw_tc = [10,50]

shift_y =gap_tc/2+width_tc[0]/2

origin = [origin[0],origin[1]]
center_tc1 = (origin[0]+spacing/2+ground_w/2,origin[1]+shift_y)
center_tc2 = (origin[0]-shift_y,origin[1]-spacing/2-ground_w/2)
center_tc3 = (origin[0]+spacing+ground_w+shift_y,origin[1]-spacing/2-ground_w/2)
center_tc4 = (origin[0]+spacing/2+ground_w/2,origin[1]-shift_y-spacing-ground_w)

a = -250

air = [-20,40,100]


CC1 = [elements.pp_transmon.PP_Transmon_Coupler(0,0,25,'right',coupler_type = 'coupler',heightr = 0.2,w=resonator_core,s=resonator_gap,g=resonator_ground,shift_to_qubit=120),
       elements.pp_transmon.PP_Transmon_Coupler(0,0,16,'left',coupler_type = 'coupler',heightl = 0.02*0,w=resonator_core,s=resonator_gap,g=resonator_ground,shift_to_qubit=-25),
      ]

CC2 = [elements.pp_transmon.PP_Transmon_Coupler(0,0,16,'bottom',coupler_type = 'coupler',heightr = 0.06*0,w=resonator_core,s=resonator_gap,g=resonator_ground,shift_to_qubit=0),
      elements.pp_transmon.PP_Transmon_Coupler(450,160,25,'top',coupler_type = 'coupler',w=resonator_core,s=resonator_gap,g=resonator_ground,shift_to_qubit=0),
      ]

CCc = [elements.pp_transmon.PP_Transmon_Coupler(0,0,25,'top',coupler_type = 'coupler',heightr = 0.2,w=resonator_core,s=resonator_gap,g=resonator_ground,shift_to_qubit=0),
      ]

l, t_m, t_r, gp, l_arm, h_arm, s_gap = 100, resonator_core, 3, 5, 20, 50, resonator_gap
flux_distance = 20
flux = {'l':l,'t_m':t_m,'t_r':t_r,'flux_distance':flux_distance,'gap':gp,'l_arm':l_arm,'h_arm':h_arm,'s_gap':s_gap,'g':resonator_ground,'w':resonator_core,'s':resonator_gap}
#flux = {}


CC = [CC1,CC2]
X = 2
Y = 1
qubits   = []
couplers = []
for i in range(Y):
    for j in range(X):
        center = (origin[0]+j*(spacing+ground_w),origin[1]+i*(spacing+ground_h))
        element = elements.pp_transmon.PP_Transmon(name='Q'+str(j)+'_'+str(i), center=center,
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
                                           Couplers=CC[j],
                                           calculate_capacitance=False,
                                           remove_ground={'left': 1, 'right': 1, 'top': 1, 'bottom': 1},
                                           shoes=shoes1,
                                           transformations={'rotate': (np.pi / 4, center)},
                                           )
        sample.add(element)
        qubits.append(element)

for i in range(Y):
    for j in range(X):
        center1 = (origin[0] + (spacing / 2 + ground_w / 2)*(2*j+1), origin[1] - shift_y+(spacing+ground_w)*i)
        center2 = (origin[0] - shift_y+j*(spacing+ground_w), origin[1] + (spacing / 2 + ground_w / 2)*(2*i+1))
        T1 = elements.fungus_squid_coupler.Fungus_Squid_C(name='PP_Coupler1',center=center1,
                          width = width_tc,
                          height = height_tc,
                          bridge_gap = JJ_pad_offset_x,
                          bridge_w   = JJ_pad_offset_y ,
                          gap = gap_tc,
                          ground_w = ground_w_tc,
                          ground_h = ground_h_tc,
                          ground_t = ground_t,
                          jj_params= jj_pp,
                          fluxline_params =flux,
                          layer_configuration = sample.layer_configuration,
                          Couplers = CCc,
                          calculate_capacitance = False,
                          transformations = {'rotate':(-np.pi/2,center1)},
                          remove_ground = {'left':1,'top':1,'bottom':1,'right':1},
                          shoes ={},
                          claw  = claw_tc,
                          asymmetry = a,
                          air_bridge=air,
                          )

        T2 = elements.fungus_squid_coupler.Fungus_Squid_C(name='PP_Coupler2', center=center2,
                                                           width=width_tc,
                                                           height=height_tc,
                                                           bridge_gap=JJ_pad_offset_x,
                                                           bridge_w=JJ_pad_offset_y,
                                                           gap=gap_tc,
                                                           ground_w=ground_w_tc,
                                                           ground_h=ground_h_tc,
                                                           ground_t=ground_t,
                                                           jj_params=jj_pp,
                                                           fluxline_params={},
                                                           layer_configuration=sample.layer_configuration,
                                                           Couplers=[],
                                                           calculate_capacitance=False,
                                                           transformations={'rotate': (np.pi, center2)},
                                                           remove_ground={'left': 1, 'top': 1, 'bottom': 1, 'right': 1},
                                                           shoes={},
                                                           claw=claw_tc,
                                                           asymmetry=a,
                                                           air_bridge=air,
                                                           )
        if(j!=X-1):
            sample.add(T1)
            couplers.append(T1)
        if(i!=Y-1):
            sample.add(T2)
            couplers.append(T2)


#lets do it smartly
all_restricted = []
for Q in qubits:
    Qu = Q.render()
    all_restricted.append(Qu['restrict'])
for Q in couplers:
    Qu = Q.render()
    all_restricted.append(Qu['restrict'])

one = gdspy.Rectangle((0,0),(0,0))
for i in all_restricted:
    one = gdspy.boolean(one,i,'or',layer=layers_configuration['air bridges'])

all_inverted = []

for Q in qubits:
    Qu = Q.render()
    all_inverted.append(Qu['pocket'])

for Q in couplers:
    Qu = Q.render()
    all_inverted.append(Qu['pocket'])

for i in all_inverted:
    one = gdspy.boolean(one,i,'not',layer=layers_configuration['vertical gridlines'])

#Logos
sample.logo[0] = True

sample.logo[2] = (1550,4730)
sample.logo[1] = (8580,4730)

sample.total_cell.add(one)