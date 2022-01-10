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

# 0. Define geometry
# define transmission line parameters
# to have 50 Oms impedance with eps=11.75
tl_core = 10
tl_gap = 6
tl_ground = 10

# define coplanar resonator parameters
resonator_core = 10
resonator_gap = 6
resonator_ground = 10

# define pads parameters
pad_offset = 550
pad_element_offset = 400
pad_geometry = {'pad_w': 250, 'pad_s': 146, 'pad_g': 8, 'pad_length': 250, 'narrowing_length': 100, 'stub_length': 100,
                'z0': 50}

qubit_position_offset = 900
tunable_coupler_length = 1600

coupler_start = pad_offset + pad_element_offset
coupler_delta = 500
coupler_length = 500
num_couplers = 1

jc = 1e-6  # 1e-6 # uA/um^2

layers_configuration = {
    'total': 0,
    'restricted area': 10,
    'for removing': 100,
    'JJs': 1,
    'air bridges': 2,
    'air bridge pads': 3,
    'air bridge sm pads': 4,
    'vertical gridlines': 0,
    'horizontal gridlines': 0,
    'inverted': 101,
    'bandages': 20
}

sample = creator.Sample('SQC8-script', layers_configuration)

# specify sample vertical and horizontal lengths
# substrate size is 5*10 mm
sample.chip_geometry.sample_vertical_size = 4.7e3
sample.chip_geometry.sample_horizontal_size = 4.7e3 * 2
central_line_y = sample.chip_geometry.sample_vertical_size / 2
chip_edge_ground = elements.ChipEdgeGround(sample.chip_geometry, sample.layer_configuration, sample.pads, 350)
sample.add(chip_edge_ground)

# 1. Create contact pads:
nu_pads_left = 3
nu_pads_right = 3
pads_left = []
pads_right = []

for pad_side_id in range(nu_pads_left):
    pad = elements.Pad('pad-left-' + str(pad_side_id),
                       (pad_offset, sample.chip_geometry.sample_vertical_size / 2 * (-0.5 * pad_side_id + 1 + 1 / 2)),
                       np.pi,
                       tl_core,
                       tl_gap, tl_ground,
                       layer_configuration=sample.layer_configuration, chip_geometry=sample.chip_geometry,
                       **pad_geometry)
    pads_left.append(pad)
    sample.add(pad)

for pad_side_id in range(nu_pads_right):
    pad = elements.Pad('pad-right-' + str(pad_side_id),
                       (sample.chip_geometry.sample_horizontal_size - pad_offset,
                        sample.chip_geometry.sample_vertical_size / 2 * (-0.5 * pad_side_id + 1 + 1 / 2)),
                       0,
                       tl_core,
                       tl_gap, tl_ground,
                       layer_configuration=sample.layer_configuration, chip_geometry=sample.chip_geometry,
                       **pad_geometry)
    pads_right.append(pad)
    sample.add(pad)

nu_pads_top = 7
nu_pads_bottom = 7
pads_top = []
pads_bottom = []

for pad_side_id in range(nu_pads_bottom):
    pad = elements.Pad('pad-bottom-' + str(pad_side_id),
                       (sample.chip_geometry.sample_horizontal_size / 8 * (pad_side_id + 1), pad_offset),
                       -np.pi / 2,
                       tl_core,
                       tl_gap, tl_ground,
                       layer_configuration=sample.layer_configuration, chip_geometry=sample.chip_geometry,
                       **pad_geometry)
    pads_bottom.append(pad)
    sample.add(pad)

for pad_side_id in range(nu_pads_top):
    pad = elements.Pad('pad-top-' + str(pad_side_id),
                       (sample.chip_geometry.sample_horizontal_size / 8 * (pad_side_id + 1),
                        sample.chip_geometry.sample_vertical_size - pad_offset),
                       np.pi / 2,
                       tl_core,
                       tl_gap, tl_ground,
                       layer_configuration=sample.layer_configuration, chip_geometry=sample.chip_geometry,
                       **pad_geometry)
    pads_top.append(pad)
    sample.add(pad)

# 3. Add qubits:
shift = -1 / 10
phi1 = 1 / 5
jj_coaxmon_sm_SQUID = {
    'a1': 30,
    'jj1_width': 0.5,
    'jj1_height': 0.5,
    'jj2_width': 0.15,
    'jj2_height': 0.5,
    'jj3_width': 0.15,
    'jj3_height': 0.15,
    'c2': 10,
    'angle_qubit': -np.pi / 2 - np.pi / 3,
    'angle_JJ': 0,
    'length': 10,
    'width': 5,
    'ic1': 0.5 * 0.5 * jc,
    'ic2': 0.15 * 0.15 * jc,
    'ic3': 0.15 * 0.15 * jc,
    'lm': 4.2e-12
}
jj_coaxmon_sm_SQUID1 = {
    'a1': 30,
    'jj1_width': 0.5,
    'jj1_height': 0.5,
    'jj2_width': 0.15,
    'jj2_height': 0.5,
    'jj3_width': 0.15,
    'jj3_height': 0.15,
    'c2': 10,
    'angle_qubit': -np.pi / 2 + np.pi / 3,
    'angle_JJ': 0,
    'length': 10,
    'width': 5,
    'ic1': 0.5 * 0.5 * jc,
    'ic2': 0.15 * 0.15 * jc,
    'ic3': 0.15 * 0.15 * jc,
    'lm': 4.2e-12
}
# couplers = claws: grounded coupler for the flux line
Couplers_qubit_left_down = [
    elements.coaxmon.CoaxmonCoupler(arc_start=-1 / 6 - 1 / 100 + shift, arc_finish=-3 / 6 + 1 / 100 + shift, phi=-1 / 2,
                                    coupler_type=None, w=10, g=10),
    elements.coaxmon.CoaxmonCoupler(arc_start=1 / 6 + 1 / 100 + shift, arc_finish=3 / 6 - 1 / 100 + shift,
                                    phi=phi1 + 1 / 12,
                                    coupler_type='coupler', w=10, g=10, s=6),  # for resonator
    elements.coaxmon.CoaxmonCoupler(arc_start=3 / 6 + 2 / 100 + shift, arc_finish=5 / 6 - 13 / 100 + shift, phi=1 / 2,
                                    coupler_type='coupler', w=8, g=30, s=26),  # one upper
    elements.coaxmon.CoaxmonCoupler(arc_start=-1 / 6 + 11 / 100 + shift, arc_finish=1 / 6 - 4 / 100 + shift, phi=0,
                                    coupler_type='coupler', w=8, g=30, s=26),
    elements.coaxmon.CoaxmonCoupler(arc_start=-1 / 6 + 1 / 100 + 1 + shift, arc_finish=1 / 6 - 1 / 100 + 1 + shift,
                                    phi=1,
                                    coupler_type=None, w=10, g=10),
    elements.coaxmon.CoaxmonCoupler(arc_start=-5 / 6 + 1 / 100 + shift, arc_finish=-3 / 6 - 1 / 100 + shift, phi=1,
                                    coupler_type='grounded', w=10, s=6, g=20)
]

Couplers_qubit_left_down_double = [elements.coaxmon.CoaxmonCoupler(arc_start=-1 / 6 - 1 / 100 + shift + 1 / 6,
                                                                   arc_finish=-3 / 6 + 1 / 100 + shift + 1 / 6,
                                                                   phi=-1 / 2, coupler_type=None, w=10, g=40
                                                                   ),
                                   elements.coaxmon.CoaxmonCoupler(arc_start=1 / 6 + 1 / 100 + shift,
                                                                   arc_finish=3 / 6 - 1 / 100 + shift, phi=phi1,
                                                                   coupler_type='coupler', w=10, g=10, s=6),
                                   # for resonator
                                   elements.coaxmon.CoaxmonCoupler(arc_start=3 / 6 + 2 / 100 + shift,
                                                                   arc_finish=5 / 6 - 13 / 100 + shift, phi=1 / 2,
                                                                   coupler_type='coupler', w=8, g=30, s=26),
                                   # one upper
                                   elements.coaxmon.CoaxmonCoupler(arc_start=-1 / 6 + 11 / 100 + 1 + shift,
                                                                   arc_finish=1 / 6 - 4 / 100 + 1 + shift, phi=1,
                                                                   coupler_type='coupler', w=8, g=30, s=26),
                                   elements.coaxmon.CoaxmonCoupler(arc_start=-1 / 6 + 1 / 100 + shift + 1 / 6,
                                                                   arc_finish=1 / 6 - 1 / 100 + shift, phi=0,
                                                                   coupler_type='coupler', w=8, g=30, s=26),
                                   elements.coaxmon.CoaxmonCoupler(arc_start=-5 / 6 + 1 / 100 + shift,
                                                                   arc_finish=-3 / 6 - 1 / 100 + shift + 1 / 6,
                                                                   phi=-1 / 2 - 1 / 3,
                                                                   coupler_type='grounded', w=10, s=6, g=20
                                                                   )
                                   ]

Couplers_qubit_left = [elements.coaxmon.CoaxmonCoupler(arc_start=-5 / 6 + 1 / 100 + shift,
                                                       arc_finish=-3 / 6 - 1 / 100 + shift + 1 / 6, phi=-1 / 2 - 1 / 3,
                                                       coupler_type=None, w=10, g=40),
                       elements.coaxmon.CoaxmonCoupler(arc_start=1 / 6 + 1 / 100 + shift,
                                                       arc_finish=3 / 6 - 1 / 100 + shift, phi=phi1,
                                                       coupler_type='coupler', w=10, g=10, s=6),  # for resonator
                       elements.coaxmon.CoaxmonCoupler(arc_start=3 / 6 + 2 / 100 + shift,
                                                       arc_finish=5 / 6 - 13 / 100 + shift, phi=1 / 2,
                                                       coupler_type='coupler', w=8, g=30, s=26),  # one upper
                       elements.coaxmon.CoaxmonCoupler(arc_start=-1 / 6 + 11 / 100 + 1 + shift,
                                                       arc_finish=1 / 6 - 4 / 100 + 1 + shift, phi=1,
                                                       coupler_type='coupler', w=8, g=30, s=26),
                       elements.coaxmon.CoaxmonCoupler(arc_start=-1 / 6 + 1 / 100 + shift + 1 / 6,
                                                       arc_finish=1 / 6 - 1 / 100 + shift, phi=0,
                                                       coupler_type=None, w=10, g=40),
                       elements.coaxmon.CoaxmonCoupler(arc_start=-1 / 6 - 1 / 100 + shift + 1 / 6,
                                                       arc_finish=-3 / 6 + 1 / 100 + shift + 1 / 6, phi=-1 / 2,
                                                       coupler_type='grounded', w=10, s=6, g=20)]

Couplers_qubit_left_double = [elements.coaxmon.CoaxmonCoupler(arc_start=-5 / 6 + 1 / 100 + shift,
                                                              arc_finish=-3 / 6 - 1 / 100 + shift + 1 / 6,
                                                              phi=-1 / 2 - 1 / 3,
                                                              coupler_type=None, w=10, g=40),
                              elements.coaxmon.CoaxmonCoupler(arc_start=1 / 6 + 1 / 100 + shift,
                                                              arc_finish=3 / 6 - 1 / 100 + shift, phi=phi1,
                                                              coupler_type='coupler', w=10, g=10, s=6),
                              # for resonator
                              elements.coaxmon.CoaxmonCoupler(arc_start=3 / 6 + 2 / 100 + shift,
                                                              arc_finish=5 / 6 - 13 / 100 + shift, phi=1 / 2,
                                                              coupler_type='coupler', w=8, g=30, s=26),  # one upper
                              elements.coaxmon.CoaxmonCoupler(arc_start=-1 / 6 + 11 / 100 + 1 + shift,
                                                              arc_finish=1 / 6 - 4 / 100 + 1 + shift, phi=1,
                                                              coupler_type='coupler', w=8, g=30, s=26),
                              elements.coaxmon.CoaxmonCoupler(arc_start=-1 / 6 + 1 / 100 + shift + 1 / 6,
                                                              arc_finish=1 / 6 - 1 / 100 + shift, phi=0,
                                                              coupler_type='coupler', w=8, g=30, s=26),
                              elements.coaxmon.CoaxmonCoupler(arc_start=-1 / 6 - 1 / 100 + shift + 1 / 6,
                                                              arc_finish=-3 / 6 + 1 / 100 + shift + 1 / 6, phi=-1 / 2,
                                                              coupler_type='grounded', w=10, s=6, g=20)

                              ]

Couplers_qubit_right_down_double = [elements.coaxmon.CoaxmonCoupler(arc_start=-1 / 6 - 1 / 100 + shift + 1 / 6,
                                                                    arc_finish=-3 / 6 + 1 / 100 + shift + 1 / 6,
                                                                    phi=-1 / 2,
                                                                    coupler_type='grounded', w=10, s=6, g=20),
                                    elements.coaxmon.CoaxmonCoupler(arc_start=1 / 6 + 1 / 100 + shift,
                                                                    arc_finish=3 / 6 - 1 / 100 + shift, phi=phi1,
                                                                    coupler_type='coupler', w=10, g=10, s=6),
                                    # for resonator
                                    elements.coaxmon.CoaxmonCoupler(arc_start=3 / 6 + 2 / 100 + shift,
                                                                    arc_finish=5 / 6 - 13 / 100 + shift, phi=1 / 2,
                                                                    coupler_type='coupler', w=8, g=30, s=26),
                                    # one upper
                                    elements.coaxmon.CoaxmonCoupler(arc_start=-1 / 6 + 11 / 100 + 1 + shift,
                                                                    arc_finish=1 / 6 - 4 / 100 + 1 + shift, phi=1,
                                                                    coupler_type='coupler', w=8, g=30, s=26),
                                    elements.coaxmon.CoaxmonCoupler(arc_start=-1 / 6 + 1 / 100 + shift + 1 / 6,
                                                                    arc_finish=1 / 6 - 1 / 100 + shift, phi=0,
                                                                    coupler_type='coupler', w=8, g=30, s=26),
                                    elements.coaxmon.CoaxmonCoupler(arc_start=-5 / 6 + 1 / 100 + shift,
                                                                    arc_finish=-3 / 6 - 1 / 100 + shift + 1 / 6,
                                                                    phi=-1 / 2 - 1 / 3,
                                                                    coupler_type=None, w=10, g=40)
                                    ]

Couplers_qubit_right_down = [elements.coaxmon.CoaxmonCoupler(arc_start=-1 / 6 - 1 / 100 + shift + 1 / 6,
                                                             arc_finish=-3 / 6 + 1 / 100 + shift + 1 / 6,
                                                             phi=-1 / 2,
                                                             coupler_type='grounded', w=10, s=6, g=20),
                             elements.coaxmon.CoaxmonCoupler(arc_start=1 / 6 + 1 / 100 + shift,
                                                             arc_finish=3 / 6 - 1 / 100 + shift, phi=phi1,
                                                             coupler_type='coupler', w=10, g=10, s=6),
                             # for resonator
                             elements.coaxmon.CoaxmonCoupler(arc_start=3 / 6 + 2 / 100 + shift,
                                                             arc_finish=5 / 6 - 13 / 100 + shift, phi=1 / 2,
                                                             coupler_type='coupler', w=8, g=30, s=26),
                             # one upper
                             elements.coaxmon.CoaxmonCoupler(arc_start=-1 / 6 + 11 / 100 + 1 + shift,
                                                             arc_finish=1 / 6 - 4 / 100 + 1 + shift, phi=1,
                                                             coupler_type='coupler', w=8, g=30, s=26),
                             elements.coaxmon.CoaxmonCoupler(arc_start=-1 / 6 + 1 / 100 + shift + 1 / 6,
                                                             arc_finish=1 / 6 - 1 / 100 + shift, phi=0,
                                                             coupler_type=None, w=10, g=40),
                             elements.coaxmon.CoaxmonCoupler(arc_start=-5 / 6 + 1 / 100 + shift,
                                                             arc_finish=-3 / 6 - 1 / 100 + shift + 1 / 6,
                                                             phi=-1 / 2 - 1 / 3,
                                                             coupler_type=None, w=10, g=40)
                             ]

Couplers_qubit_right = [
    elements.coaxmon.CoaxmonCoupler(arc_start=-5 / 6 + 1 / 100 + shift, arc_finish=-3 / 6 - 1 / 100 + shift, phi=1,
                                    coupler_type='grounded', w=10, s=6, g=20),
    elements.coaxmon.CoaxmonCoupler(arc_start=1 / 6 + 1 / 100 + shift, arc_finish=3 / 6 - 1 / 100 + shift, phi=phi1,
                                    coupler_type='coupler', w=10, g=10, s=6),  # for resonator
    elements.coaxmon.CoaxmonCoupler(arc_start=3 / 6 + 2 / 100 + shift, arc_finish=5 / 6 - 13 / 100 + shift, phi=1 / 2,
                                    coupler_type='coupler', w=8, g=30, s=26),  # one upper
    elements.coaxmon.CoaxmonCoupler(arc_start=-1 / 6 + 11 / 100 + shift, arc_finish=1 / 6 - 4 / 100 + shift, phi=0,
                                    coupler_type='coupler', w=8, g=30, s=26),
    elements.coaxmon.CoaxmonCoupler(arc_start=-1 / 6 + 1 / 100 + 1 + shift, arc_finish=1 / 6 - 1 / 100 + 1 + shift,
                                    phi=1,
                                    coupler_type=None, w=10, g=10),
    elements.coaxmon.CoaxmonCoupler(arc_start=-1 / 6 - 1 / 100 + shift, arc_finish=-3 / 6 + 1 / 100 + shift, phi=-1 / 2,
                                    coupler_type=None, w=10, g=10)]

Couplers_qubit_right_double = [elements.coaxmon.CoaxmonCoupler(arc_start=-5 / 6 + 1 / 100 + shift,
                                arc_finish=-3 / 6 - 1 / 100 + shift + 1 / 6,
                                phi=-1 / 2 - 1 / 3,
                                coupler_type='grounded', w=10, s=6, g=20),
                               elements.coaxmon.CoaxmonCoupler(arc_start=1 / 6 + 1 / 100 + shift,
                                                               arc_finish=3 / 6 - 1 / 100 + shift, phi=phi1,
                                                               coupler_type='coupler', w=10, g=10, s=6),
                               # for resonator
                               elements.coaxmon.CoaxmonCoupler(arc_start=3 / 6 + 2 / 100 + shift,
                                                               arc_finish=5 / 6 - 13 / 100 + shift, phi=1 / 2,
                                                               coupler_type='coupler', w=8, g=30, s=26),
                               # one upper
                               elements.coaxmon.CoaxmonCoupler(arc_start=-1 / 6 + 11 / 100 + 1 + shift,
                                                               arc_finish=1 / 6 - 4 / 100 + 1 + shift, phi=1,
                                                               coupler_type='coupler', w=8, g=30, s=26),
                               elements.coaxmon.CoaxmonCoupler(arc_start=-1 / 6 + 1 / 100 + shift + 1 / 6,
                                                               arc_finish=1 / 6 - 1 / 100 + shift, phi=0,
                                                               coupler_type='coupler', w=8, g=30, s=26),
elements.coaxmon.CoaxmonCoupler(arc_start=-1 / 6 - 1 / 100 + shift + 1 / 6,
                                                               arc_finish=-3 / 6 + 1 / 100 + shift + 1 / 6,
                                                               phi=-1 / 2, coupler_type=None, w=10, g=40
                                                               )

                               ]

coaxmon1 = elements.coaxmon.Coaxmon(name='Coaxmon1',
                                    center=(
                                        sample.chip_geometry.sample_horizontal_size / 2 - tunable_coupler_length * 3 / 2,
                                        sample.chip_geometry.sample_vertical_size / 2 - qubit_position_offset),
                                    center_radius=100,
                                    inner_couplers_radius=140,
                                    outer_couplers_radius=200,
                                    inner_ground_radius=230,
                                    outer_ground_radius=250,
                                    layer_configuration=sample.layer_configuration,
                                    Couplers=Couplers_qubit_left_down, jj_params=jj_coaxmon_sm_SQUID,
                                    transformations={},
                                    calculate_capacitance=True, third_JJ=True)

coaxmon2 = elements.coaxmon.Coaxmon(name='Coaxmon2', center=(
    sample.chip_geometry.sample_horizontal_size / 2 - tunable_coupler_length / 2,
    sample.chip_geometry.sample_vertical_size / 2 - qubit_position_offset),
                                    center_radius=100,
                                    inner_couplers_radius=140,
                                    outer_couplers_radius=200,
                                    inner_ground_radius=230,
                                    outer_ground_radius=250,
                                    layer_configuration=sample.layer_configuration,
                                    Couplers=Couplers_qubit_left_down_double, jj_params=jj_coaxmon_sm_SQUID,
                                    transformations={},
                                    calculate_capacitance=True, third_JJ=True)

transformations3 = {'rotate': [np.pi, (sample.chip_geometry.sample_horizontal_size / 2 - tunable_coupler_length / 2,
                                       sample.chip_geometry.sample_vertical_size / 2 + qubit_position_offset)]}

coaxmon3 = elements.coaxmon.Coaxmon(name='Coaxmon3', center=(
    sample.chip_geometry.sample_horizontal_size / 2 - tunable_coupler_length / 2,
    sample.chip_geometry.sample_vertical_size / 2 + qubit_position_offset),
                                    center_radius=100,
                                    inner_couplers_radius=140,
                                    outer_couplers_radius=200,
                                    inner_ground_radius=230,
                                    outer_ground_radius=250,
                                    layer_configuration=sample.layer_configuration,
                                    Couplers=Couplers_qubit_left_double, jj_params=jj_coaxmon_sm_SQUID1,
                                    transformations=transformations3,
                                    calculate_capacitance=True, third_JJ=True)

transformations4 = {'rotate': [np.pi, (sample.chip_geometry.sample_horizontal_size / 2 - tunable_coupler_length * 3 / 2,
                                       sample.chip_geometry.sample_vertical_size / 2 + qubit_position_offset)]}

coaxmon4 = elements.coaxmon.Coaxmon(name='Coaxmon4',
                                    center=(
                                        sample.chip_geometry.sample_horizontal_size / 2 - tunable_coupler_length * 3 / 2,
                                        sample.chip_geometry.sample_vertical_size / 2 + qubit_position_offset),
                                    center_radius=100,
                                    inner_couplers_radius=140,
                                    outer_couplers_radius=200,
                                    inner_ground_radius=230,
                                    outer_ground_radius=250,
                                    layer_configuration=sample.layer_configuration,
                                    Couplers=Couplers_qubit_left, jj_params=jj_coaxmon_sm_SQUID1,
                                    transformations=transformations4,
                                    calculate_capacitance=True, third_JJ=True)

coaxmon5 = elements.coaxmon.Coaxmon(name='Coaxmon5',
                                    center=(
                                        sample.chip_geometry.sample_horizontal_size / 2 + tunable_coupler_length / 2,
                                        sample.chip_geometry.sample_vertical_size / 2 - qubit_position_offset),
                                    center_radius=100,
                                    inner_couplers_radius=140,
                                    outer_couplers_radius=200,
                                    inner_ground_radius=230,
                                    outer_ground_radius=250,
                                    layer_configuration=sample.layer_configuration,
                                    Couplers=Couplers_qubit_right_down_double, jj_params=jj_coaxmon_sm_SQUID1,
                                    transformations={},
                                    calculate_capacitance=True, third_JJ=True)

coaxmon6 = elements.coaxmon.Coaxmon(name='Coaxmon6',
                                    center=(
                                        sample.chip_geometry.sample_horizontal_size / 2 + tunable_coupler_length * 3 / 2,
                                        sample.chip_geometry.sample_vertical_size / 2 - qubit_position_offset),
                                    center_radius=100,
                                    inner_couplers_radius=140,
                                    outer_couplers_radius=200,
                                    inner_ground_radius=230,
                                    outer_ground_radius=250,
                                    layer_configuration=sample.layer_configuration,
                                    Couplers=Couplers_qubit_right_down, jj_params=jj_coaxmon_sm_SQUID1,
                                    transformations={},
                                    calculate_capacitance=True, third_JJ=True)

transformations7 = {'rotate': [np.pi, (sample.chip_geometry.sample_horizontal_size / 2 + tunable_coupler_length * 3 / 2,
                                       sample.chip_geometry.sample_vertical_size / 2 + qubit_position_offset)]}

coaxmon7 = elements.coaxmon.Coaxmon(name='Coaxmon7',
                                    center=(
                                        sample.chip_geometry.sample_horizontal_size / 2 + tunable_coupler_length * 3 / 2,
                                        sample.chip_geometry.sample_vertical_size / 2 + qubit_position_offset),
                                    center_radius=100,
                                    inner_couplers_radius=140,
                                    outer_couplers_radius=200,
                                    inner_ground_radius=230,
                                    outer_ground_radius=250,
                                    layer_configuration=sample.layer_configuration,
                                    Couplers=Couplers_qubit_right, jj_params=jj_coaxmon_sm_SQUID,
                                    transformations=transformations7,
                                    calculate_capacitance=True, third_JJ=True)

transformations8 = {'rotate': [np.pi, (sample.chip_geometry.sample_horizontal_size / 2 + tunable_coupler_length / 2,
                                       sample.chip_geometry.sample_vertical_size / 2 + qubit_position_offset)]}

coaxmon8 = elements.coaxmon.Coaxmon(name='Coaxmon8',
                                    center=(
                                        sample.chip_geometry.sample_horizontal_size / 2 + tunable_coupler_length / 2,
                                        sample.chip_geometry.sample_vertical_size / 2 + qubit_position_offset),
                                    center_radius=100,
                                    inner_couplers_radius=140,
                                    outer_couplers_radius=200,
                                    inner_ground_radius=230,
                                    outer_ground_radius=250,
                                    layer_configuration=sample.layer_configuration,
                                    Couplers=Couplers_qubit_right_double, jj_params=jj_coaxmon_sm_SQUID,
                                    transformations=transformations8,
                                    calculate_capacitance=True, third_JJ=True)
