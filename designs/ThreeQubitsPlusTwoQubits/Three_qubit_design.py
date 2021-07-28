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
tl_core = 21.
tl_gap = 12.
tl_ground = 6.#<-- changed from 10. to 5.

resonator_core = 15
resonator_gap = 10
resonator_ground = 15 #5
resonator_tl_ground=13
pad_offset = 550
fluxline_core, fluxline_gap, fluxline_ground=9,5,10

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

sample = creator.Sample('Three-qubits-PP',layers_configuration)

#specify sample vertical and horizontal lengths
sample.chip_geometry.sample_vertical_size=4.5e3
sample.chip_geometry.sample_horizontal_size=8e3

chip_edge_ground = elements.ChipEdgeGround(sample.chip_geometry, sample.layer_configuration, sample.pads,edge=10)
sample.add(chip_edge_ground)

# 1. Create contact pads for 6*10 pcb WMI from Hans:
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

p1 = pads_top[0]
p2 = pads_top[1]

################################


################################
# resonator parameters:
resonator_core = 8
resonator_gap = 7
resonator_ground = 10

############### Qubits and Coupler

#Qubit parameters
#origin of left bottom qubit
origin = [2200,1400]


width = 250
height= 400

gap   = 50

width  = 200
height = 2*width+gap


ground_t = 48
ground_w = 660+ground_t*2
ground_h = 660+ground_t*2

#square junctions
a1    = np.sqrt(0.15*0.3) #Junction height in um
a2    = a1 # Junction width in um

#jj_pp = { 'a1':a1,"a2":a2,'angle_JJ':np.pi/2}
jj_pp = { 'a1':a1,"a2":a2,'angle_JJ':0,'manhatten':True,'h_w':5 ,'h_d':8, 'squid':False,'bandages_extension':2.5,'connection_pad_width':0.9,'connection_pad_gap':0.5}# hole sizes for the JJs



jj_pp_rotated = { 'a1':a1,"a2":a2,'angle_JJ':-np.pi/4,'manhatten':True,'h_w':5 ,'h_d':8, 'squid':False,'bandages_extension':2.5,'connection_pad_width':0.9,'connection_pad_gap':0.5,'rotation':np.pi/4,'translate':(-7,0),'bridge_translate':(-5,-12,0,0),'paddingx':0,'paddingy':10}# hole sizes for the JJs

jj_pp_2 = { 'a1':a1,"a2":a2,'angle_JJ':0,'manhatten':True,'h_w':5 ,'h_d':8, 'squid':False,'inverted_extension':0,'strip1_extension':20,'strip2_extension':25,'loop_h':10,'bandages_extension':2.5,'connection_pad_width':0.9,'connection_pad_gap':0.5}
jj_pp_3 = { 'a1':a1,"a2":a2,'angle_JJ':0,'manhatten':True,'h_w':5 ,'h_d':8, 'squid':False,'bandages_extension':2.5,'connection_pad_width':0.9,'connection_pad_gap':0.5}



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
height_tc   = [910-20,500]

height_tc2 = [860+3*28,280,1200+3*28]

gap_tc      = 70
ground_w_tc = 325+2*ground_t
ground_h_tc = 950+2*ground_t


claw_tc = [10,50]

shift_y =gap_tc/2+width_tc[0]/2+claw_tc[0]

origin = [origin[0],origin[1]]
center_tc1 = (origin[0]+spacing/2+ground_w/2,origin[1]+shift_y)
center_tc2 = (origin[0]-shift_y,origin[1]-spacing/2-ground_w/2)
center_tc3 = (origin[0]+spacing+ground_w+shift_y,origin[1]-spacing/2-ground_w/2)
center_tc4 = (origin[0]+spacing/2+ground_w/2,origin[1]-shift_y-spacing-ground_w)

a = +30

air = [-20,40,100]

air2 = [[200,40,100],[400,40,100]]

tight = [True,6]

CC1 = [elements.pp_transmon.PP_Transmon_Coupler(450,160,25,'bottom',coupler_type = 'coupler',w=resonator_core,s=resonator_gap,g=resonator_ground,shift_to_qubit=120,tight = tight),
       #elements.pp_transmon.PP_Transmon_Coupler(0,0,16,'left',coupler_type = 'coupler',heightl = 0.02*0,w=resonator_core,s=resonator_gap,g=resonator_ground,shift_to_qubit=-25,tight = tight),
      ]

CC2 = [elements.pp_transmon.PP_Transmon_Coupler(0,0,25,'left',coupler_type = 'coupler',heightl = 0.2,w=resonator_core,s=resonator_gap,g=resonator_ground,shift_to_qubit=120,tight = tight),
      ]

CC3 = [#elements.pp_transmon.PP_Transmon_Coupler(0,0,16,'right',coupler_type = 'coupler',heightr = 0.06*0,w=resonator_core,s=resonator_gap,g=resonator_ground,shift_to_qubit=-25,tight = tight),
     elements.pp_transmon.PP_Transmon_Coupler(450,160,25,'bottom',coupler_type = 'coupler',w=resonator_core,s=resonator_gap,g=resonator_ground,shift_to_qubit=100,tight = tight),
      ]



l, t_m, t_r, gp, l_arm, h_arm, s_gap = 100, fluxline_core, 3, 5, 20, 50, fluxline_gap

fluxline_core, fluxline_gap, fluxline_ground=9,5,10
flux_distance = 7
flux = {'l':l,'t_m':t_m,'t_r':t_r,'flux_distance':flux_distance,'gap':gp,'l_arm':l_arm,'h_arm':h_arm,'s_gap':s_gap,'g':fluxline_ground,'w':fluxline_core,'s':fluxline_gap,'rotation':0,'inverted_extension':0}

CC = [CC1,CC2,CC3]

CC_tc1 = [elements.pp_transmon.PP_Transmon_Coupler(0,0,25,'right',coupler_type = 'coupler',heightr = 0.2,w=resonator_core,s=resonator_gap,g=resonator_ground,shift_to_qubit=60,tight = tight),]

CC_tc2 = [elements.pp_transmon.PP_Transmon_Coupler(0,0,25,'left',coupler_type = 'coupler',heightl = 0.13,w=resonator_core,s=resonator_gap,g=resonator_ground,shift_to_qubit=60,tight = tight),
      ]



CC_tc = [CC_tc1,CC_tc2]

X = 2
Y = 1
qubits   = []
couplers = []

x_offset=1500
y_offset=-800
center1 = (3860-120+x_offset,3000+20+y_offset)
Q1 = elements.pp_transmon.PP_Transmon(name='Q1', center=center1,
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
                                           transformations={'rotate': (np.pi, center1)},
                                           return_inverted=False,
                                           )

sample.add(Q1)
qubits.append(Q1)


center2 = (3860-120+x_offset,1600-70+y_offset)
Q2 = elements.pp_transmon.PP_Transmon(name='Q2', center=center2,
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
                                           transformations={'rotate': (np.pi, center2)},
                                            return_inverted=False,
                                           )

sample.add(Q2)
qubits.append(Q2)




center3 = (2258-170+x_offset,2298-20+y_offset)

Q3 = elements.pp_transmon.PP_Transmon(name='Q3', center=center3,
                                           width=width,
                                           height=height,
                                           bridge_gap=JJ_pad_offset_x,
                                           bridge_w=JJ_pad_offset_y,
                                           gap=gap,
                                           ground_w=ground_w,
                                           ground_h=ground_h,
                                           ground_t=ground_t,
                                           jj_params=jj_pp_rotated,
                                           layer_configuration=sample.layer_configuration,
                                           Couplers=CC[2],
                                           calculate_capacitance=False,
                                           remove_ground={'left': 1, 'right': 1, 'top': 1, 'bottom': 1},
                                           shoes=shoes3,
                                           transformations={'rotate': (-np.pi / 4, center3)},
                                            return_inverted=False,
                                           )

sample.add(Q3)
qubits.append(Q3)


center2tc = (origin[0] +1*(spacing+ground_h)/2+width_tc[0]/2-3+x_offset, origin[1] +(spacing+ground_h)/2+shift_y-claw_tc[0]+y_offset)


a2 = -300
a_coupl = -300
T2 = elements.y_squid_coupler.Y_Squid_C(name='Y_Coupler', center=center2tc,
                                                  width=[60,110],
                                                  height=height_tc2,
                                                  bridge_gap=JJ_pad_offset_x,
                                                  bridge_w=JJ_pad_offset_y,
                                                  gap=gap_tc,
                                                  ground_w=ground_w_tc,
                                                  ground_h=ground_h_tc,
                                                  ground_t=ground_t,
                                                  jj_params=jj_pp_3,
                                                  fluxline_params=flux,
                                                  layer_configuration=sample.layer_configuration,
                                                  Couplers=CC_tc[1],
                                                  calculate_capacitance=False,
                                                  transformations={'rotate': (-np.pi/2, center2tc),},
                                                  remove_ground={'left': 1, 'top': 1, 'bottom': 1, 'right': 1},
                                                  shoes={},
                                                  claw=claw_tc,
                                                  asymmetry=a2,
                                                  air_bridge=air2,
                                                  y_gap = 80,
                                                  asymmetry_coupler = a_coupl,
                                                return_inverted=False,
                                                  )

sample.add(T2)
couplers.append(T2)

"""
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

#removes the middle island

#island = gdspy.Polygon([(center1[0]-15,center1[1]),center1tc,(center2[0]-15,center2[1]),center2tc])
#one = gdspy.boolean(one, island, 'not', layer=layers_configuration['vertical gridlines'])
sample.total_cell.add(one)
"""

print('uploaded')


def create_restricted(check = False):
    if check:
        print('already run restricted once')
        return 0

    all_restricted = []
    for Q in qubits:
        Qu = Q.render()
        all_restricted.append(Qu['restrict'])
    for Q in couplers:
        Qu = Q.render()
        all_restricted.append(Qu['restrict'])

    restricted = gdspy.Rectangle((0,0),(0,0))
    for i in all_restricted:
        restricted = gdspy.boolean(restricted,i,'or',layer=layers_configuration['bandages'])

    one = gdspy.Rectangle((0, 0), (0, 0))
    for i in all_restricted:
        one = gdspy.boolean(one, i, 'or', layer=layers_configuration['air bridges'])
    #print(gdspy.Cell('Two-Qubits-PP').get_polygons())
    sample_all = [i for i in sample.objects]
    for object in sample_all:
        if not hasattr(object.render()['positive'],'layers'):
            continue
        restricted = gdspy.boolean(restricted,object.render()['positive'],'not',layer=layers_configuration['bandages'])
        one = gdspy.boolean(one,object.render()['positive'],'not',layer=layers_configuration['vertical gridlines'])

    all_inverted = []

    for Q in qubits:
        Qu = Q.render()
        all_inverted.append(Qu['pocket'])

    for Q in couplers:
        Qu = Q.render()
        all_inverted.append(Qu['pocket'])

    for i in all_inverted:
        one = gdspy.boolean(one, i, 'not', layer=layers_configuration['vertical gridlines'])

    restricted = gdspy.boolean(restricted,one,'not',layer=layers_configuration['inverted'])

    sample.total_cell.add(one)
    sample.total_cell.add(restricted)

    return 0