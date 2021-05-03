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

sample = creator.Sample('QGrid_Garching',layers_configuration)


#Qubit parameters

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
center1 = (2050,2000)
center2 = (2050+spacing+ground_w,2000)
center3 = (2050,2000-spacing-ground_h)
center4 = (2050+spacing+ground_w,2000-spacing-ground_h)


#Coupler
arms = {}

width_tc    = [60,75]
height_tc   = [800,165]
gap_tc      = 70
ground_w_tc = 325
ground_h_tc = 850
ground_t_tc = 10

claw_tc = [10,50]

shift_y =gap_tc/2+width_tc[0]/2

center_tc1 = (2050+spacing/2+ground_w/2,2000+shift_y)
center_tc2 = (2050-shift_y,2000-spacing/2-ground_w/2)
center_tc3 = (2050+spacing+ground_w+shift_y,2000-spacing/2-ground_w/2)
center_tc4 = (2050+spacing/2+ground_w/2,2000-shift_y-spacing-ground_w)

a = -250

air = [-20,40,100]


CC = []#[elements.pp_transmon.PP_Transmon_Coupler(0,0,16,'left',coupler_type = 'coupler',heightl = 0.3,w=resonator_core,s=resonator_gap,g=resonator_ground)]

"""
transmon1 = elements.pp_transmon.PP_Transmon(name='Q1',center=center1,
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
                          Couplers = [],
                          calculate_capacitance = False,
                          remove_ground = {'left':1,'right':1,'top':1,'bottom':1},
                          shoes = shoes1,
                          transformations = {'rotate':(np.pi/4,center1)},
                          )


transmon2 = elements.pp_transmon.PP_Transmon(name='Q2',center=center2,
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
                          Couplers = [],
                          calculate_capacitance = False,
                          remove_ground = {'left':1,'right':1,'top':1,'bottom':1},
                          shoes = shoes1,
                          transformations = {'rotate':(np.pi/4,center2)},
                          )

transmon3 = elements.pp_transmon.PP_Transmon(name='Q3',center=center3,
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
                          Couplers = [],
                          calculate_capacitance = False,
                          remove_ground = {'left':1,'right':1,'top':1,'bottom':1},
                          shoes = shoes1,
                          transformations = {'rotate':(np.pi/4,center3)},
                          )

transmon4 = elements.pp_transmon.PP_Transmon(name='Q4',center=center4,
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
                          Couplers = [],
                          calculate_capacitance = False,
                          remove_ground = {'left':1,'right':1,'top':1,'bottom':1},
                          shoes = shoes1,
                          transformations = {'rotate':(np.pi/4,center4)},
                          )



tc1 = elements.fungus_squid_coupler.Fungus_Squid_C(name='PP_Coupler1',center=center_tc1,
                          width = width_tc,
                          height = height_tc,
                          bridge_gap = JJ_pad_offset_x,
                          bridge_w   = JJ_pad_offset_y ,
                          gap = gap_tc,
                          ground_w = ground_w_tc,
                          ground_h = ground_h_tc,
                          ground_t = ground_t_tc,
                          jj_params= jj_pp,
                          fluxline_params ={},
                          arms = arms,
                          layer_configuration = sample.layer_configuration,
                          Couplers = [],
                          calculate_capacitance = False,
                          transformations = {'rotate':(np.pi/2,center_tc1)},
                          remove_ground = {'left':1,'top':1,'bottom':1,'right':1},
                          shoes ={},
                          claw  = claw_tc,
                          asymmetry = a,
                          air_bridge=air,
                          )

tc2 = elements.fungus_squid_coupler.Fungus_Squid_C(name='PP_Coupler2',center=center_tc2,
                          width = width_tc,
                          height = height_tc,
                          bridge_gap = JJ_pad_offset_x,
                          bridge_w   = JJ_pad_offset_y ,
                          gap = gap_tc,
                          ground_w = ground_w_tc,
                          ground_h = ground_h_tc,
                          ground_t = ground_t_tc,
                          jj_params= jj_pp,
                          fluxline_params ={},
                          arms = arms,
                          layer_configuration = sample.layer_configuration,
                          Couplers = [],
                          calculate_capacitance = False,
                          transformations = {'rotate':(np.pi,center_tc2)},
                          remove_ground = {'left':1,'top':1,'bottom':1,'right':1},
                          shoes ={},
                          claw  = claw_tc,
                          asymmetry = a,
                          air_bridge=air,
                          )

tc3 = elements.fungus_squid_coupler.Fungus_Squid_C(name='PP_Coupler3',center=center_tc3,
                          width = width_tc,
                          height = height_tc,
                          bridge_gap = JJ_pad_offset_x,
                          bridge_w   = JJ_pad_offset_y ,
                          gap = gap_tc,
                          ground_w = ground_w_tc,
                          ground_h = ground_h_tc,
                          ground_t = ground_t_tc,
                          jj_params= jj_pp,
                          fluxline_params ={},
                          arms = arms,
                          layer_configuration = sample.layer_configuration,
                          Couplers = [],
                          calculate_capacitance = False,
                          transformations = {'rotate':(4*np.pi/2,center_tc3)},
                          remove_ground = {'left':1,'top':1,'bottom':1,'right':1},
                          shoes ={},
                          claw  = claw_tc,
                          asymmetry = a,
                          air_bridge=air,
                          )

tc4 = elements.fungus_squid_coupler.Fungus_Squid_C(name='PP_Coupler4',center=center_tc4,
                          width = width_tc,
                          height = height_tc,
                          bridge_gap = JJ_pad_offset_x,
                          bridge_w   = JJ_pad_offset_y ,
                          gap = gap_tc,
                          ground_w = ground_w_tc,
                          ground_h = ground_h_tc,
                          ground_t = ground_t_tc,
                          jj_params= jj_pp,
                          fluxline_params ={},
                          arms = arms,
                          layer_configuration = sample.layer_configuration,
                          Couplers = [],
                          calculate_capacitance = False,
                          transformations = {'rotate':(3*np.pi/2,center_tc4)},
                          remove_ground = {'left':1,'top':1,'bottom':1,'right':1},
                          shoes ={},
                          claw  = claw_tc,
                          asymmetry = a,
                          air_bridge=air,
                          )




sample.add(transmon1)
sample.add(transmon2)
sample.add(transmon3)
sample.add(transmon4)
sample.add(tc1)
sample.add(tc2)
sample.add(tc3)
sample.add(tc4)
"""


X = 3
Y = 3

for i in range(Y):
    for j in range(X):
        center = (2050+j*(spacing+ground_w),2000+i*(spacing+ground_h))
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
                                           Couplers=CC,
                                           calculate_capacitance=False,
                                           remove_ground={'left': 1, 'right': 1, 'top': 1, 'bottom': 1},
                                           shoes=shoes1,
                                           transformations={'rotate': (np.pi / 4, center)},
                                           )
        sample.add(element)

for i in range(Y):
    for j in range(X):
        center1 = (2050 + (spacing / 2 + ground_w / 2)*(2*j+1), 2000 + shift_y+(spacing+ground_w)*i)
        center2 = (2050 - shift_y+j*(spacing+ground_w), 2000 + (spacing / 2 + ground_w / 2)*(2*i+1))
        T1 = elements.fungus_squid_coupler.Fungus_Squid_C(name='PP_Coupler1',center=center1,
                          width = width_tc,
                          height = height_tc,
                          bridge_gap = JJ_pad_offset_x,
                          bridge_w   = JJ_pad_offset_y ,
                          gap = gap_tc,
                          ground_w = ground_w_tc,
                          ground_h = ground_h_tc,
                          ground_t = ground_t_tc,
                          jj_params= jj_pp,
                          fluxline_params ={},
                          arms = arms,
                          layer_configuration = sample.layer_configuration,
                          Couplers = [],
                          calculate_capacitance = False,
                          transformations = {'rotate':(np.pi/2,center1)},
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
                                                           ground_t=ground_t_tc,
                                                           jj_params=jj_pp,
                                                           fluxline_params={},
                                                           arms=arms,
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
        if(i!=Y-1):
            sample.add(T2)

sample.draw_design()
#sample.watch()