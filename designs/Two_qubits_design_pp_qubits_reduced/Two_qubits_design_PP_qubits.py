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

resonator_ground = 15


resonator_tl_ground=8
pad_offset = 550

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

############### Qubits and Coupler

#Qubit parameters
#origin of left qubit
origin = [2400-30,1600]

# parameters for Stefan's two-qubit setup
gap   = 70
width  = 150
height = 500


ground_t = 50
ground_w = 710+ground_t*2+100
ground_h = 750+ground_t*2+100

#square junctions
a1    = np.sqrt(0.17*0.3) #Junction height in um
a2    = a1 # Junction width in um

#jj_pp = { 'a1':a1,"a2":a2,'angle_JJ':np.pi/2}
jj_pp2 = { 'a1':a1,"a2":a2,'angle_JJ':0,'manhatten':True,'h_w':5 ,'h_d':8,'squid':True,'loop_h': 10,'bandages_extension':2.5,'connection_pad_width':0.9,'connection_pad_gap':0.5}# hole sizes for the JJs
jj_pp1 = { 'a1':a1,"a2":a2,'angle_JJ':0,'manhatten':True,'h_w':5 ,'h_d':8,'squid':False,'loop_h': 10 ,'bandages_extension':2.5,'connection_pad_width':0.9,'connection_pad_gap':0.5}# hole sizes for the JJs

jj_pp_c = { 'a1':a1,"a2":a2,'angle_JJ':0,'manhatten':True,'h_w':5 ,'h_d':8,'squid':False,'loop_h': 10,'bandages_extension':2.5,'connection_pad_width':0.9,'connection_pad_gap':0.5,'rotation':np.pi/4,'translate':(-6.5,6.5-3),'loop_w_shift':10,'strip1_extension':20,'strip2_extension':35,'padding':10}# hole sizes for the JJs


JJ_pad_offset_x = 10 # for JJ_manhatten
JJ_pad_offset_y = 16## JJ design
JJ_pad_offset_x_flux = 20 # for JJ_manhatten
JJ_pad_offset_y_flux = 10 # JJ design

sh = (130,30)
shoes1 = {1:sh,2:sh,3:sh,4:sh,'R':np.pi/4,'fake_claws': {'gap1':10,'gap2':10,'claw_t':25,'l1':140,'l2':55,'box_l':210,'box_h':340,'position':'1111'}}#(gap1,gap2,claw_t,l1,l2,box_l,box_h,binary number indicating where the fake claws are top left top right bottom left bottm right)

shoes2 = {1:sh,2:sh,3:sh,4:sh,'R':np.pi/4,'fake_claws': {'gap1':10,'gap2':10,'claw_t':25,'l1':140,'l2':55,'box_l':210,'box_h':340,'position':'1111'}}#{1:(70,50)}

shoes = [shoes1,shoes2]
# how to place qubits
spacing = 1200-30+60

center1 = (origin[0],origin[1])
center2 = (origin[0]+spacing+ground_w,origin[1])
center3 = (origin[0],origin[1]-spacing-ground_h)
center4 = (origin[0]+spacing+ground_w,origin[1]-spacing-ground_h)


#Coupler
arms = {}
width_tc    = [15,215]
height_tc   = [700,570]

height_tc2 = [860+3*28,280,1200+3*28]

gap_tc      = 160
ground_w_tc = 325+2*ground_t+90
ground_h_tc = 950+2*ground_t-300 #buffer for the claws is the +200

# width_tc    = [60,75]
# height_tc   = [800,165]
# gap_tc      = 70
# ground_w_tc = 325
# ground_h_tc = 950
# ground_t_tc = 10

claw_tc = [25,185+50]

shift_y =gap_tc/2+width_tc[0]/2

origin = [origin[0],origin[1]]
center_tc1 = (origin[0]+spacing/2+ground_w/2,origin[1]+shift_y)
center_tc2 = (origin[0]-shift_y,origin[1]-spacing/2-ground_w/2)
center_tc3 = (origin[0]+spacing+ground_w+shift_y,origin[1]-spacing/2-ground_w/2)
center_tc4 = (origin[0]+spacing/2+ground_w/2,origin[1]-shift_y-spacing-ground_w)

a = -250*0

air = [-20,40,100]



CC2 = [elements.pp_transmon.PP_Transmon_Coupler(550,5,100,'top',coupler_type = 'coupler',w=5,s=3,g=resonator_ground,shift_to_qubit=0,tight = [True,3],line_same_as_coupler= True),
      elements.pp_transmon.PP_Transmon_Coupler(10,10,25,'left',coupler_type = 'coupler',heightl = 0.2,w=resonator_core,s=resonator_gap,g=resonator_ground,shift_to_qubit=225,tight = [True,10]),
      ]

CC1_flux = [elements.pp_transmon.PP_Transmon_Coupler(0,0,25,'right',coupler_type = 'coupler',heightr = 0.2,w=resonator_core,s=resonator_gap,g=resonator_ground,shift_to_qubit=225,tight =[True,10]),
      # elements.pp_transmon.PP_Transmon_Coupler(500,14,16,'top',coupler_type = 'coupler',w=resonator_core,s=resonator_gap,g=resonator_ground,shift_to_qubit=-25),
      ]


#CC1_mw = [elements.pp_transmon.PP_Transmon_Coupler(0,0,25,'right',coupler_type = 'coupler',heightr = 0.2,w=resonator_core,s=resonator_gap,g=resonator_ground,shift_to_qubit=175),
#      elements.pp_transmon.PP_Transmon_Coupler(500,14,16,'top',coupler_type = 'coupler',w=resonator_core,s=resonator_gap,g=resonator_ground,shift_to_qubit=-25),
#      ]


CCc = [elements.pp_transmon.PP_Transmon_Coupler(0,0,25,'left',coupler_type = 'coupler',heightl = 0.35,w=resonator_core,s=resonator_gap,g=resonator_ground,shift_to_qubit=+110,tight = [True,10]),
      ]


flux_distance = 15
l, t_m, t_r, gp, l_arm, h_arm, s_gap,asymmetry = 160-flux_distance+0.5, 5, 3, 5, [40/3,58.85-0.2-0.156], 50, resonator_gap,15-4.7
flux1 = {'l':l,'t_m':t_m,'t_r':t_r,'flux_distance':flux_distance,'gap':gp,'l_arm':l_arm,'h_arm':h_arm,'s_gap':s_gap,'g':resonator_ground,'w':resonator_core,'s':resonator_gap,'asymmetry':asymmetry,'loop_h': 10 }
#for coupler
flux_distance = 0
l, t_m, t_r, gp, l_arm, h_arm, s_gap = 110-8-ground_t, 4, 3, 5, [58.85-0.2-0.156,40/3], 50, resonator_gap
flux2 = {'l':150,'t_m':5,'t_r':t_r,'flux_distance':flux_distance,'gap':gp,'l_arm':l_arm,'h_arm':h_arm,'s_gap':3,'g':resonator_ground,'w':resonator_core,'s':resonator_gap,'asymmetry':0,'rotation':np.pi/4,'translation':(-58-50/np.sqrt(2),-50/np.sqrt(2)),
         'bandages_extension':2.5,'connection_pad_width':0.9,'connection_pad_gap':0.5,'inverted_extension':0}



CC = [CC1_flux,CC2]
X = 2
Y = 1
qubits   = []
couplers = []
for i in range(Y):
    for j in range(X):
        center = (origin[0]+j*(spacing+ground_w),origin[1]+i*(spacing+ground_h)-100)
        if j%2==1:
            jj_pp=jj_pp1
            fluxq={}
            transformations={'rotate': (-np.pi / 4, center)}
            JJ_pad_offset_x_qubit=JJ_pad_offset_x
            JJ_pad_offset_y_qubit=JJ_pad_offset_y
        else:
            fluxq=flux1
            jj_pp = jj_pp2
            transformations={'rotate': (np.pi / 4, center)}
            JJ_pad_offset_x_qubit = JJ_pad_offset_x_flux
            JJ_pad_offset_y_qubit = JJ_pad_offset_y_flux
        element = elements.pp_transmon.PP_Transmon(name='Q'+str(j)+'_'+str(i), center=center,
                                           width=width,
                                           height=height,
                                           bridge_gap=JJ_pad_offset_x_qubit,
                                           bridge_w=JJ_pad_offset_y_qubit,
                                           gap=gap,
                                           ground_w=ground_w,
                                           ground_h=ground_h,
                                           ground_t=ground_t,
                                           jj_params=jj_pp,
                                           layer_configuration=sample.layer_configuration,
                                           Couplers=CC[j],
                                           calculate_capacitance=False,
                                           remove_ground={'left': 1, 'right': 1, 'top': 1, 'bottom': 1},
                                           shoes=shoes[j],
                                           transformations=transformations,
                                           fluxline_params = fluxq,
                                           return_inverted= False,
                                           )
        sample.add(element)
        qubits.append(element)

for i in range(Y):
    for j in range(X):
        additional_y_coupler_shift = -90+55-20+10-1-100
        shit_shift=0.124
        center1 = (origin[0] + (spacing / 2 + ground_w / 2)*(2*j+1), origin[1] - shift_y+(spacing+ground_w)*i+additional_y_coupler_shift)
        center2 = (origin[0] - shift_y+j*(spacing+ground_w), origin[1] + (spacing / 2 + ground_w / 2)*(2*i+1)+additional_y_coupler_shift)
        T1 = elements.fungus_squid_coupler.Fungus_Squid_C(name='PP_Coupler1',center=center1,
                          width = width_tc,
                          height = height_tc,
                          bridge_gap = JJ_pad_offset_x,
                          bridge_w   = JJ_pad_offset_y,
                          gap = gap_tc,
                          ground_w = ground_w_tc,
                          ground_h = ground_h_tc,
                          ground_t = ground_t,
                          jj_params= jj_pp_c,
                          fluxline_params =flux2,
                          layer_configuration = sample.layer_configuration,
                          Couplers = CCc,
                          calculate_capacitance = False,
                          transformations = {'rotate':(-np.pi/2,center1)},
                          #transformations={},
                          remove_ground = {'left':1,'top':1,'bottom':1,'right':1},
                          shoes ={},
                          claw  = claw_tc,
                          asymmetry = a,
                          air_bridge=air,
                          return_inverted=False,
                          )
        if(j!=X-1):
            sample.add(T1)
            couplers.append(T1)




center=(6400,1200)
JJ_test_structure = elements.pp_transmon.PP_Transmon(name='JJ_test',center=center,
                          width = 300,
                          height = 300,
                          bridge_gap = JJ_pad_offset_x,
                          bridge_w   = JJ_pad_offset_y ,
                          gap = gap,
                          ground_w = 700,
                          ground_h = 600,
                          ground_t = 10,
                          jj_params= jj_pp_c,
                          layer_configuration = sample.layer_configuration,
                          Couplers = [],
                          calculate_capacitance = False,
                          transformations = {'rotate':[np.pi/2,center]},
                          )
sample.add(JJ_test_structure)
center=(5900,1200)
JJ_test_structure1 = elements.pp_transmon.PP_Transmon(name='JJ_test1',center=center,
                          width = 300,
                          height = 300,
                          bridge_gap = JJ_pad_offset_x,
                          bridge_w   = JJ_pad_offset_y ,
                          gap = gap,
                          ground_w = 700,
                          ground_h = 600,
                          ground_t = 10,
                          jj_params= jj_pp_c,
                          layer_configuration = sample.layer_configuration,
                          Couplers = [],
                          calculate_capacitance = False,
                          transformations = {'rotate':[np.pi/2,center]},
                          )
sample.add(JJ_test_structure1)




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
    all_inverted = []
    restricted = gdspy.Rectangle((0,0),(0,0))

    rect1 = gdspy.Rectangle((3062-70, 1300), (3835, 1030))
    rect2 = gdspy.Rectangle((3900,1624), (2986, 1284))
    restricted = gdspy.boolean(restricted, [rect1,rect2], 'or', layer=layers_configuration['inverted'])
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


    for Q in qubits:
        Qu = Q.render()
        all_inverted.append(Qu['pocket'])

    for Q in couplers:
        Qu = Q.render()
        all_inverted.append(Qu['pocket'])

    for i in all_inverted:
        one = gdspy.boolean(one, i, 'not', layer=layers_configuration['vertical gridlines'])


    restricted = gdspy.boolean(restricted,one,'not',layer=layers_configuration['inverted'])
    rect_substr = gdspy.Rectangle((2992,1284),(3133,1030))
    restricted = gdspy.boolean(restricted,rect_substr,'not',layer=layers_configuration['inverted'])


    # for airbridges
    rect_substr = gdspy.Rectangle((3100, 1420), (3060, 1284))
    restricted = gdspy.boolean(restricted, rect_substr, 'not', layer=layers_configuration['inverted'])
    rect_substr = gdspy.Rectangle((3100, 1480), (3060, 1630))
    restricted = gdspy.boolean(restricted, rect_substr, 'not', layer=layers_configuration['inverted'])
    rect_substr = gdspy.Rectangle((3900, 1420), (3940, 1284))
    restricted = gdspy.boolean(restricted, rect_substr, 'not', layer=layers_configuration['inverted'])
    rect_substr = gdspy.Rectangle((3900, 1480), (3940, 1630))
    restricted = gdspy.boolean(restricted, rect_substr, 'not', layer=layers_configuration['inverted'])


    sample.total_cell.add(one)
    sample.total_cell.add(restricted)

    return 0

### add logos and markers
logos=elements.WMILogos((700,3500),(7300,3500),layers_configuration)
sample.add(logos)

###add 3 markers
markers = elements.AlignmentMarkers((470,470),(sample.chip_geometry.sample_horizontal_size,sample.chip_geometry.sample_vertical_size),10,sample.layer_configuration)
sample.add(markers)
markers2 = elements.AlignmentMarkers((485,485),(sample.chip_geometry.sample_horizontal_size,sample.chip_geometry.sample_vertical_size),4,sample.layer_configuration)
sample.add(markers2)
markers3 = elements.AlignmentMarkers((500,500),(sample.chip_geometry.sample_horizontal_size,sample.chip_geometry.sample_vertical_size),1,sample.layer_configuration)
sample.add(markers3)

sample.draw_design()
