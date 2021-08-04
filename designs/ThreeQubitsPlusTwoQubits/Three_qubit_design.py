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

p11 = pads_top[0]
p12 = pads_left[0]
p21 = pads_top[1]
p22 = pads_right[0]


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



gap   = 50

width  = 200
height = 2*width+gap


ground_t = 48
ground_w = 660+ground_t*2+100
ground_h = 660+ground_t*2+100

#square junctions
a1    = np.sqrt(0.15*0.3) #Junction height in um
a2    = a1 # Junction width in um


jj_pp = { 'a1':a1,"a2":a2,'angle_JJ':0,'manhatten':True,'h_w':5 ,'h_d':8, 'squid':False,'bandages_extension':1.25,'connection_pad_width':0.6,'connection_pad_gap':0.5}# hole sizes for the JJs



jj_pp_rotated = { 'a1':a1,"a2":a2,'angle_JJ':0,'manhatten':True,'h_w':5 ,'h_d':8, 'squid':False,'bandages_extension':1.25,'connection_pad_width':0.6,'connection_pad_gap':0.,'rotation':np.pi/4,'translate':(-5,-6),'bridge_translate':(-5,-16,0,0),'paddingx':0,'paddingy':10,'bandages_edge_shift':3.5}# hole sizes for the JJs

jj_pp_2 = { 'a1':a1,"a2":a2,'angle_JJ':0,'manhatten':True,'h_w':5 ,'h_d':8, 'squid':False,'inverted_extension':0,'strip1_extension':20,'strip2_extension':25,'loop_h':10,'bandages_extension':1.25,'connection_pad_width':0.6,'connection_pad_gap':0.,'bandages_edge_shift':3.5}

jj_pp_3 = { 'a11':a1,"a12":a2,"a2":a2,'angle_JJ':0,'manhatten':True,'h_w':3 ,'h_d':8,'squid':True,'loop_h': 13,'bandages_extension':1.25,'connection_pad_width':0.6,'connection_pad_gap':0,'strip1_extension':17,'strip2_extension':26,'bandages_edge_shift':3.5}# hole sizes for the JJs


JJ_pad_offset_x = 10 # for JJ_manhatten
JJ_pad_offset_y = 16 # JJ design


sh = (70*2.6+25,25)
sh_klein = (60,10)

shoes1 = {1:sh_klein,2:sh_klein,3:sh,4:sh_klein,'R1':np.pi/2,'R2':np.pi/4,'R3':np.pi/4,'R4':np.pi/4}

shoes2 = {1:sh_klein,2:sh_klein,3:sh_klein,4:sh,'R1':np.pi/4,'R2':np.pi/2,'R3':np.pi/4,'R4':np.pi/4}

shoes3 = {1:sh_klein,2:sh_klein,3:sh,4:sh_klein,'R1':np.pi/4,'R2':np.pi/4,'R3':np.pi/4,'R4':np.pi/4}

spacing = 740

center1 = (origin[0],origin[1])
center2 = (origin[0]+spacing+ground_w,origin[1])
center3 = (origin[0],origin[1]-spacing-ground_h)
center4 = (origin[0]+spacing+ground_w,origin[1]-spacing-ground_h)


#Coupler
arms = {}

width_tc    = [60,75]
height_tc   = [910-20,600]


reduced_length = 350
height_tc2 = [1600,500,1200+3*28-reduced_length]

gap_tc      = 70
ground_w_tc = 630+2*ground_t
ground_h_tc = 950+2*ground_t+600

c_t = 27
l1,l2,t = 400, 180,30
extension = {'left_arm':{'t1':c_t*np.sqrt(2),'l1':170,'t2':c_t*np.sqrt(2),'l2':370,},'right_arm_1':{'t2':c_t*np.sqrt(2),'l2':170,'t1':c_t*np.sqrt(2),'l1':370,},'right_arm_2':{'t2':c_t*np.sqrt(2),'l2':370,'t1':c_t*np.sqrt(2),'l1':170,}}

claw_tc = [c_t,[80*2.6,80*2.6+c_t,100],extension]

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

CC1 = [elements.pp_transmon.PP_Transmon_Coupler(450,160,25,'bottom',coupler_type = 'coupler',w=resonator_core,s=resonator_gap,g=resonator_ground,shift_to_qubit=40,tight = tight),
       #elements.pp_transmon.PP_Transmon_Coupler(0,0,16,'left',coupler_type = 'coupler',heightl = 0.02*0,w=resonator_core,s=resonator_gap,g=resonator_ground,shift_to_qubit=-25,tight = tight),
      ]

CC2 = [elements.pp_transmon.PP_Transmon_Coupler(0,0,25,'left',coupler_type = 'coupler',heightl = 0.4,w=resonator_core,s=resonator_gap,g=resonator_ground,shift_to_qubit=40,tight = tight),
      ]

CC3 = [#elements.pp_transmon.PP_Transmon_Coupler(0,0,16,'right',coupler_type = 'coupler',heightr = 0.06*0,w=resonator_core,s=resonator_gap,g=resonator_ground,shift_to_qubit=-25,tight = tight),
     elements.pp_transmon.PP_Transmon_Coupler(0,0,25,'left',coupler_type = 'coupler',heightl = 0.4,w=resonator_core,s=resonator_gap,g=resonator_ground,shift_to_qubit=40,tight = tight),
      ]


"""
l, t_m, t_r, gp, l_arm, h_arm, s_gap = 100, fluxline_core, 3, 5, 20, 50, fluxline_gap

fluxline_core, fluxline_gap, fluxline_ground=9,5,10
flux_distance = 7
flux = {'l':l,'t_m':t_m,'t_r':t_r,'flux_distance':flux_distance,'gap':gp,'l_arm':l_arm,'h_arm':h_arm,'s_gap':s_gap,'g':fluxline_ground,'w':fluxline_core,'s':fluxline_gap,'rotation':0,'inverted_extension':0}
"""
l, t_m, t_r, gp, l_arm, h_arm, s_gap = 110-8-ground_t, 5, 3, 5, 40, 50, 3
flux_distance = 20
#for coupler
flux2 = {'l':150,'t_m':t_m,'t_r':t_r,'flux_distance':flux_distance,'gap':gp,'l_arm':l_arm,'h_arm':h_arm,'s_gap':s_gap,'g':resonator_ground,'w':resonator_core,'s':resonator_gap,'asymmetry':0,'rotation':np.pi/4,
         'bandages_extension':2.5,'connection_pad_width':0.9,'connection_pad_gap':0.5,'inverted_extension':0}








CC = [CC1,CC2,CC3]

#CC_tc1 = [elements.pp_transmon.PP_Transmon_Coupler(0,0,25,'right',coupler_type = 'coupler',heightr = 0.2,w=resonator_core,s=resonator_gap,g=resonator_ground,shift_to_qubit=20,tight = tight),]

CC_tc2 = [elements.pp_transmon.PP_Transmon_Coupler(0,0,25,'left',coupler_type = 'coupler',heightl = 0.20,w=resonator_core,s=resonator_gap,g=resonator_ground,shift_to_qubit=43,tight = tight),]



CC_tc = [CC_tc2]

X = 2
Y = 1
qubits   = []
couplers = []

x_offset=1500
y_offset=-800

s1 = 100-reduced_length/np.sqrt(2)+150/np.sqrt(2)+11+5/np.sqrt(2)


s2 = 10/np.sqrt(2)
center1 = (3860-117+x_offset+s1+s2+56,3000+32+y_offset+s1-s2+66)

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

s1 = 78-reduced_length/np.sqrt(2)+198/np.sqrt(2)
s2= -2.5+18/np.sqrt(2)
center2 = (3860-120+x_offset+s1+s2+56,1600-70+y_offset-s1+s2+66)
Q2 = elements.pp_transmon.PP_Transmon(name='Q2', center=center2,
                                           width=width,
                                           height=height,
                                           bridge_gap=JJ_pad_offset_x,
                                           bridge_w=JJ_pad_offset_y,
                                           gap=gap,
                                           ground_w=ground_w,
                                           ground_h=ground_h,
                                           ground_t=ground_t,
                                           jj_params=jj_pp_2,
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
s1 = 0
center3 = (2258-110+x_offset-158+s1-300,2298-20+y_offset+81+s1)

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


center2tc = (origin[0] +1*(spacing+ground_h)/2+width_tc[0]/2-3+x_offset, origin[1] +(spacing+ground_h)/2+shift_y-claw_tc[0]+y_offset-10)


a2 = -395
a_coupl = -300
T2 = elements.y_squid_coupler.Y_Squid_C(name='Y_Coupler', center=center2tc,
                                                  width=[100,200],
                                                  height=height_tc2,
                                                  bridge_gap=JJ_pad_offset_x+5,
                                                  bridge_w=JJ_pad_offset_y,
                                                  gap=gap_tc+20,
                                                  ground_w=ground_w_tc,
                                                  ground_h=ground_h_tc+400,
                                                  ground_t=ground_t,
                                                  jj_params=jj_pp_3,
                                                  fluxline_params=flux2,
                                                  layer_configuration=sample.layer_configuration,
                                                  Couplers=CC_tc[0],
                                                  calculate_capacitance=False,
                                                  transformations={'rotate': (-np.pi/2, center2tc),},
                                                  #transformations = {},
                                                  remove_ground={'left': 1, 'top': 1, 'bottom': 1, 'right': 1},
                                                  shoes={},
                                                  claw=claw_tc,
                                                  asymmetry=a2,
                                                  air_bridge=air2,
                                                  y_gap = 170,
                                                  asymmetry_coupler = a_coupl,
                                                return_inverted=False,
                                                    thin_coupler = [True,32.5]
                                                  )

sample.add(T2)
couplers.append(T2)


#add markers,logos and test structures
#add test structures + a Test SNAIL





shiftx = 600-3100-2300
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
                          jj_params= jj_pp_3,
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

#define parameters for test SNAIl
a11 = 0.4
a2  = 0.4
a12 = a11*0.3
jj_pp_snail = { 'a11':a11,"a12":a12,"a2":a2,'angle_JJ':0,'manhatten':True,'h_w':3 ,'h_d':8,'snail':True,'loop_h': 12,'bandages_extension':1.25,'connection_pad_width':0.6,'connection_pad_gap':0,'strip1_extension':15,'strip2_extension':8,'bandages_edge_shift':3.5,'snail_extension':2,'snail_reach':10}# hole sizes for the JJs

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


logos=elements.WMILogos((700,3600),(700,3000),layers_configuration)
sample.add(logos)
#sample.draw_design()
markers = elements.AlignmentMarkers((470,470),(sample.chip_geometry.sample_horizontal_size,sample.chip_geometry.sample_vertical_size),10,sample.layer_configuration)
sample.add(markers)
markers2 = elements.AlignmentMarkers((485,485),(sample.chip_geometry.sample_horizontal_size,sample.chip_geometry.sample_vertical_size),4,sample.layer_configuration)
sample.add(markers2)
markers3 = elements.AlignmentMarkers((500,500),(sample.chip_geometry.sample_horizontal_size,sample.chip_geometry.sample_vertical_size),1,sample.layer_configuration)
sample.add(markers3)

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
    triangle = gdspy.Polygon([center1, center2, center2tc, center1])
    restricted = gdspy.boolean(restricted, triangle, 'or', layer=layers_configuration['inverted'])
    for i in all_restricted:
        restricted = gdspy.boolean(restricted,i,'or',layer=layers_configuration['bandages'])

    one = gdspy.Rectangle((0, 0), (0, 0))
    for i in all_restricted:
        one = gdspy.boolean(one, i, 'or', layer=layers_configuration['air bridges'])
    #print(gdspy.Cell('Two-Qubits-PP').get_polygons())

   ##################
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
    triangle = gdspy.Polygon([center1, center2, center2tc, center1])
    restricted = gdspy.boolean(restricted, triangle, 'or', layer=layers_configuration['inverted'])

    rect = gdspy.Rectangle((center2tc[0]-(ground_h_tc+400)/2,center2tc[1]-ground_w_tc/2),(center2tc[0]-(ground_h_tc+400)/2+290,center2tc[1]-ground_w_tc/2+250))
    restricted = gdspy.boolean(restricted, rect, 'not', layer=layers_configuration['inverted'])

    sample_all = [i for i in sample.objects]
    for object in sample_all:
        if not hasattr(object.render()['positive'],'layers'):
            continue
        restricted = gdspy.boolean(restricted,object.render()['positive'],'not',layer=layers_configuration['inverted'])
        one = gdspy.boolean(one,object.render()['positive'],'not',layer=layers_configuration['vertical gridlines'])

    sample.total_cell.add(one)
    sample.total_cell.add(restricted)
    return 0