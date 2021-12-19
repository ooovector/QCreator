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

from importlib import reload
import numpy as np
from matplotlib import pyplot as plt
import QCreator.elements as elements
import QCreator.auxiliary_functions as auxfuncs


from QCreator.elements.gflux_squid import GFluxControllableSquid
from QCreator.elements.fluxonium import Fluxonium


### to have 50 Oms impedance with eps=11.45
#wet etching subtracts 0.5 um, so we add that to all structures where it matters, coplers,Junction region and Fluxline

# used in pads
tl_width = 4.
tl_gap = 3
tl_ground = 21.

# connect qubit ports and meanders with cpw:
tl_width_qubit_to_meander = 7
tl_gap_qubit_to_meander = 4
tl_ground_qubit_to_meander = tl_ground 


# should be the same as tl_width, tl_gap, tl_ground parameters 
# resonator_width = 10
# resonator_gap = 6
# tl_res_ground = 6

resonator_width = tl_width
resonator_gap = tl_gap
tl_res_ground = tl_ground


 #changed to 14 from 13




pad_offset = 550
pad_element_offset = 800


qubit_position_offset = 800

coupler_start = pad_offset + pad_element_offset
coupler_delta = 500
coupler_length = 500
num_couplers = 1

jc = 1e-6#1e-6 # uA/um^2

layers_configuration = {
    'total':0,
    'restricted area':10,
    'for removing':100,
    'JJs':1,
    'air bridges':2,
    'air bridge pads':3,
    'air bridge sm pads':4,
    'vertical gridlines':0,
    'horizontal gridlines':0,
    'inverted':101,
    'bandages':20
}

sample = creator.Sample('single qubits fluxonium',layers_configuration)
#specify sample vertical and horizontal lengths
sample.chip_geometry.sample_vertical_size=4.5e3
sample.chip_geometry.sample_horizontal_size=7.4e3

chip_edge_ground = elements.ChipEdgeGround(sample.chip_geometry, sample.layer_configuration, sample.pads,edge=10)
sample.add(chip_edge_ground)


# 1. Create contact pads for 4.5*8 pcb WMI from Hans:
pads_left = []
pads_right = []
for pad_side_id in range(1):
    pad = elements.Pad('pad-left-' + str(pad_side_id),
                       (pad_offset, sample.chip_geometry.sample_vertical_size / 2), np.pi, tl_width,
                       tl_gap, tl_ground,
                       layer_configuration=sample.layer_configuration, chip_geometry=sample.chip_geometry,
                       **elements.reduced_pad_geometry())
    pads_left.append(pad)
    sample.add(pad)

    pad = elements.Pad('pad-right-' + str(pad_side_id),
                       (sample.chip_geometry.sample_horizontal_size - pad_offset,
                        sample.chip_geometry.sample_vertical_size / 2), 0, tl_width,
                       tl_gap, tl_ground,
                       layer_configuration=sample.layer_configuration, chip_geometry=sample.chip_geometry,
                       **elements.reduced_pad_geometry())
    pads_right.append(pad)
    sample.add(pad)

pads_top = []
pads_bottom = []
huns_offset=1750

pad_bottom_1 = elements.Pad('pad-bottom-' + str(1),
                   (sample.chip_geometry.sample_horizontal_size / 2+ huns_offset *(- 1), sample.chip_geometry.sample_vertical_size -pad_offset),
                   np.pi / 2, tl_width, tl_gap, tl_ground,
                   layer_configuration=sample.layer_configuration, chip_geometry=sample.chip_geometry,
                   **elements.reduced_pad_geometry())

pad_bottom_2 = elements.Pad('pad-bottom-' + str(2),
                   (sample.chip_geometry.sample_horizontal_size / 2+ huns_offset, sample.chip_geometry.sample_vertical_size -pad_offset),
                   np.pi / 2, tl_width, tl_gap, tl_ground,
                   layer_configuration=sample.layer_configuration, chip_geometry=sample.chip_geometry,
                   **elements.reduced_pad_geometry())

pads_top.append(pad_bottom_1)
pads_top.append(pad_bottom_2)
sample.add(pad_bottom_1)
sample.add(pad_bottom_2)

pad = elements.Pad('pad-top-' + str(pad_side_id),
                   (sample.chip_geometry.sample_horizontal_size / 2,
                     pad_offset),
                   -np.pi / 2, tl_width, tl_gap, tl_ground,
                   layer_configuration=sample.layer_configuration, chip_geometry=sample.chip_geometry,
                   **elements.reduced_pad_geometry())
pads_bottom.append(pad)
sample.add(pad)

p1 = pads_left[0]
p2 = pads_right[0]
#################################################
center12 = (2200, 2300)

# jc = 2
# coupler_s = 3 
# squid_coupler_vertical = 2
# flux_w = 0 
# flux_s = 0 
# flux_g = 0  

# squid_coupler_vertical =  {'jjs_height': 0.3,
#                              'jjs_width': 0.3,
#                              'jj_lead_width':0.5,
#                              'chain_width':0.5,
#                              'ics': 0.3*0.3*jc,
#                              'icb': 0.5*0.5*jc,
#                              'chain_junctions': 11,
#                              'lm': 3.3e-12,
#                              'squid_orientation': 'horizontal'}

# squid_line12 = GFluxControllableSquid(name='squid_inline', 
#                                       position = center12, 
#                                       w=8, 
#                                       s=coupler_s, 
#                                       g=30, 
#                                       invert_x=True, 
#                                       invert_y=True,
#                                       layer_configuration = sample.layer_configuration, 
#                                       squid_params = squid_coupler_vertical,                       
#                                       flux_w = flux_w, 
#                                       flux_s = flux_s, 
#                                       flux_g=flux_g)
# sample.add(squid_line12)

qubits=[]
"""
        :param center: center of fluxonium like (x_coordinate, y_coordinate)
        :param distance: distance from center to the borders of inner rectangles
        :param rectang_params: parameters like (width_rectang,height_rectang) for big inner rectangles
        :param gap: distance between inner rectangles and ground
        :param ground_width: width of ground
        :param slit_width: width of small area at the top of fluxonium
        :param rect_in_slit_params: parameters like (width_rectang,height_rectang) for rectangle in slit
        :param ledge: the depth of penetration of the rectangle into the cavity
"""

########################### 1 
groove_params = {"width": 50,
                "height": 15,
                "distance_from_center": 60}

port_params = {"width": 20,
                "distance_from_center": 80}


couplers_params = {"height": 6,
                   "width": 30,
                   "distance_from_center": 70}
center_left_up= (sample.chip_geometry.sample_horizontal_size / 2 + huns_offset, 2900)

fluxonium_left_up = Fluxonium(name = "flux", 
                      layer_configuration = sample.layer_configuration, 
                      center = center_left_up, 
                      distance = 30, 
                      rectang_params = (200, 6), 
                      gap = 10,
                      ground_width = 30, 
                      slit_width = 50,
                      rect_in_slit_params = (6, 9), 
                      ledge = 2, 
                      groove_params = groove_params,
                      port_params = port_params, 
                      couplers = {"left": True,
                                  "right": True},
                      couplers_params = couplers_params,
                      calculate_capacitance = True,
                      left_cpw_params={'w':tl_width_qubit_to_meander,'s':tl_gap_qubit_to_meander,'g':tl_ground_qubit_to_meander},
                      right_cpw_params={'w':tl_width_qubit_to_meander,'s':tl_gap_qubit_to_meander,'g':tl_ground_qubit_to_meander},
                      qubit_cpw_params={'w':tl_width,'s':tl_gap,'g':tl_ground},
                      transformations = {'rotate':(0,center_left_up)}
                     )
sample.add(fluxonium_left_up)
qubits.append(fluxonium_left_up)

################################ 2
groove_params = {"width": 50,
                "height": 15,
                "distance_from_center": 60}

port_params = {"width": 20,
                "distance_from_center": 80}


couplers_params = {"height": 6,
                   "width": 30,
                   "distance_from_center": 70}

center_right_up= (sample.chip_geometry.sample_horizontal_size / 2 + huns_offset *(- 1), 2900)
fluxonium_right_up = Fluxonium(name = "flux", 
                      layer_configuration = sample.layer_configuration, 
                      center =center_right_up, 
                      distance = 30, 
                      rectang_params = (200, 6), 
                      gap = 10,
                      ground_width = 30, 
                      slit_width = 50,
                      rect_in_slit_params = (6, 9), 
                      ledge = 2, 
                      groove_params = groove_params,
                      port_params = port_params, 
                      couplers = {"left": True,
                                  "right": True},
                      couplers_params = couplers_params,
                      calculate_capacitance = True,
                      left_cpw_params={'w':tl_width_qubit_to_meander,'s':tl_gap_qubit_to_meander,'g':tl_ground_qubit_to_meander},
                      right_cpw_params={'w':tl_width_qubit_to_meander,'s':tl_gap_qubit_to_meander,'g':tl_ground_qubit_to_meander},
                      qubit_cpw_params={'w':tl_width,'s':tl_gap,'g':tl_ground},
                      transformations = {'rotate':(0,center_right_up)}
                     )
sample.add(fluxonium_right_up)
qubits.append(fluxonium_right_up)

####################################### 3
groove_params = {"width": 50,
                "height": 15,
                "distance_from_center": 60}

port_params = {"width": 20,
                "distance_from_center": 80}


couplers_params = {"height": 6,
                   "width": 30,
                   "distance_from_center": 70}

center_down= (sample.chip_geometry.sample_horizontal_size / 2, sample.chip_geometry.sample_vertical_size / 2 - ( center_left_up[1] - sample.chip_geometry.sample_vertical_size / 2))

fluxonium_down = Fluxonium(name = "flux", 
                      layer_configuration = sample.layer_configuration, 
                      center=center_down, 
                      distance = 30, 
                      rectang_params = (200, 6), 
                      gap = 10,
                      ground_width = 30, 
                      slit_width = 50,
                      rect_in_slit_params = (6, 9), 
                      ledge = 2, 
                      groove_params = groove_params,
                      port_params = port_params, 
                      couplers = {"left": True,
                                  "right": True},
                      couplers_params = couplers_params,
                      calculate_capacitance = True,
                      left_cpw_params={'w':tl_width_qubit_to_meander,'s':tl_gap_qubit_to_meander,'g':tl_ground_qubit_to_meander},
                      right_cpw_params={'w':tl_width_qubit_to_meander,'s':tl_gap_qubit_to_meander,'g':tl_ground_qubit_to_meander},
                      qubit_cpw_params={'w':tl_width,'s':tl_gap,'g':tl_ground},
                      transformations = {'rotate':(np.pi,center_down)}
                     )
sample.add(fluxonium_down)
qubits.append(fluxonium_down)


# logos=elements.WMILogos((700,3500),(7300,3500),layers_configuration)
# sample.add(logos)
# #sample.draw_design()
# markers = elements.AlignmentMarkers((470,470),(sample.chip_geometry.sample_horizontal_size,sample.chip_geometry.sample_vertical_size),10,sample.layer_configuration)
# sample.add(markers)
# markers2 = elements.AlignmentMarkers((485,485),(sample.chip_geometry.sample_horizontal_size,sample.chip_geometry.sample_vertical_size),4,sample.layer_configuration)
# sample.add(markers2)
# markers3 = elements.AlignmentMarkers((500,500),(sample.chip_geometry.sample_horizontal_size,sample.chip_geometry.sample_vertical_size),1,sample.layer_configuration)
# sample.add(markers3)