import numpy as np
from copy import deepcopy
from . import elements
import math

def draw_purcell(sample, coupler_start_x, coupler_start_y, coupler_length,
                          resonator_core,resonator_gap, resonator_ground,
                          tl_core, tl_gap, tl_ground, grounding_width,
                          closed_end_meander_length=None, length_left=None, length_right=None,
                        open_end_length_left=None, open_end_length_right=None,open_end_first_step_orientation = 'right',
                          min_bridge_spacing = None,
                          airbridge = None,
                          open_end_length = None,
                          port_orientation='left', direction_orientation='down', meander_r=None,
                 first_step_orientation='left', end_point_closed_end = None, end_orientation_closed_end=None,
                 coupler2_length=None, closed_end_res_meander_length=None, res_length_left=None,
                 res_length_right=None,object1=None, port=None, push_resonator=False,
                 narrow_length_left=None, narrow_length_right=None, bridge_part_decimation=1, open_end_angle = None,
                 pr_coupler_offset = 0):

    coupler_w = [resonator_core, tl_core]
    coupler_s = [resonator_gap, tl_gap, tl_gap]

    # 2. Create main coupler:
    angle = 0
    if open_end_angle is None:
        open_end_angle = np.pi/2
        if direction_orientation == 'up':
            open_end_angle = -np.pi / 2
    if direction_orientation == 'up':
        coupler_start_x, coupler_length = coupler_start_x + coupler_length, -coupler_length
        angle = np.pi



    main_coupler = elements.CPWCoupler('TL-purcell coupler',
                                       [(coupler_start_x, coupler_start_y),
                                        (coupler_start_x + coupler_length, coupler_start_y)],
                                       coupler_w, coupler_s, resonator_ground, sample.layer_configuration, r=meander_r)
    sample.add(main_coupler)
    total_length = [coupler_length]

    # 3. Create fanout to create closed and opened ends of resonator
    fanout1 = sample.fanout(o=main_coupler, port='port1', name='closed end purcell fanout', grouping=[1, 2])
    # g1 = sample.ground(o=fanout1, port='center', name='cl1', grounding_width=grounding_width, grounding_between=[(0, 0)])
    fanout2 = sample.fanout(o=main_coupler, port='port2', name='open end purcell fanout', grouping=[1, 2])
    # g2 = sample.ground(o=fanout2, port='center', name='cl2', grounding_width=grounding_width, grounding_between=[(0, 0)])
    fanout1_port = 'up'
    fanout2_port = 'down'

    if port_orientation == 'right':
        fanout1, fanout2 = fanout2, fanout1
        fanout1_port, fanout2_port = fanout2_port, fanout1_port
    if direction_orientation == 'up':
        fanout1, fanout2 = fanout2, fanout1
        fanout1_port, fanout2_port = fanout2_port, fanout1_port

    # 6. Create closed meander of resonator
    closed_end_meander = sample.connect_meander(name='closed end purcell', o1=fanout1, port1=fanout1_port,
                                                meander_length=closed_end_meander_length,
                                                length_left=length_left,
                                                length_right=length_right,
                                                first_step_orientation=first_step_orientation,
                                                meander_orientation=angle,
                                                meander_type='round',
                                                airbridge=airbridge,
                                                min_spacing=min_bridge_spacing, r=meander_r,
                                                end_point=end_point_closed_end, end_orientation=end_orientation_closed_end,
                                                bridge_part_decimation=bridge_part_decimation)

    total_length.append(sum([line.length for line in closed_end_meander]))

    coupler2_w = [resonator_core, 2*resonator_ground, resonator_core]
    coupler2_s = [resonator_gap, resonator_gap, resonator_gap, resonator_gap]

    coupler2_orientation = closed_end_meander[-1].get_terminals()['port2'].orientation

    coupler2_start_y = closed_end_meander[-1].get_terminals()['port2'].position[1] -\
                       np.cos(angle)*(2*meander_r+2*resonator_gap+resonator_core)
    coupler2_start_x = closed_end_meander[-1].get_terminals()['port2'].position[0] + \
                       np.cos(coupler2_orientation)*(resonator_core + 2*resonator_gap + grounding_width - pr_coupler_offset)

    main_coupler2 = elements.CPWCoupler('purcell-resonator coupler',
                                       [(coupler2_start_x, coupler2_start_y),
                                        (coupler2_start_x + np.cos(coupler2_orientation)*coupler2_length,
                                         coupler2_start_y)],
                                       coupler2_w, coupler2_s, resonator_ground, sample.layer_configuration, r=meander_r,
                                        )
    sample.add(main_coupler2)
    total_length.append(coupler2_length)

    if (direction_orientation=='down' and coupler2_orientation ==0) or \
            (direction_orientation=='up' and coupler2_orientation !=0):
        fanout3_grouping = [1, 3]
        fanout3_grouinding = [(2, 3)]
        fanout3_port = 'up'
    else:
        fanout3_grouping = [0,2]
        fanout3_grouinding = [(0, 1)]
        fanout3_port = 'down'

    fanout3 = sample.fanout(o=main_coupler2, port='port1', name='open end resonator fanout', grouping=fanout3_grouping)
    g3 = sample.ground(o=fanout3, port='center', name='cl2', grounding_width=grounding_width,
                       grounding_between=fanout3_grouinding)


    main_coupler2_ground = sample.ground(main_coupler2, 'port2', 'closed end purcell coupler', grounding_width,
                                            [(0, 4)])

    if pr_coupler_offset > 0:
        terminal = closed_end_meander[-1].terminals['port2']
        airbridge_position = np.asarray([g3.terminals['narrow'].position[0] - airbridge.pad_width / 2, terminal.position[1]])
        bridge = elements.airbridge.AirbridgeOverCPW(
            name='Purcell after-short-meander extension final airbridge',
            position=airbridge_position,
            orientation=terminal.orientation+np.pi, w=terminal.w,
            s=terminal.s, g=terminal.g,
            geometry=airbridge)
        sample.add(bridge)
        meander_extra = sample.connect_cpw(bridge, closed_end_meander[-1], 'port1', 'port2',
                                           name='Purcell after-short-meander extension', points= [],
                                           airbridge=airbridge, min_spacing=min_bridge_spacing)

        coupler_connection_object = bridge
    else:
        coupler_connection_object = closed_end_meander[-1]

    coupler_connection = sample.connect_cpw(coupler_connection_object, g3, 'port2', 'narrow',
                                             name='coupler-meander connection',
                                    points=[], r=meander_r)

    total_length.append(coupler_connection[0].length)

    if open_end_length !=0:
        open_end_meander = sample.connect_meander(name='opened end meander of purcell', o1=fanout2, port1=fanout2_port,
                                                    meander_length=open_end_length,
                                                    length_left=open_end_length_left,
                                                    length_right=open_end_length_right,
                                                    first_step_orientation=open_end_first_step_orientation,
                                                    meander_orientation=open_end_angle,
                                                    meander_type='round',
                                                    airbridge=airbridge,
                                                    min_spacing=min_bridge_spacing, r=meander_r,
                                                  bridge_part_decimation=bridge_part_decimation)

        total_length.append(sum([line.length for line in open_end_meander]))
        open_end = sample.open_end(open_end_meander[-1], 'port2', 'open end purcell')
    else:
        open_end = sample.open_end(fanout2, fanout2_port, 'open end purcell')
    if push_resonator==True:
        num_turns = coupler2_length / 2 / meander_r
        num_turns = math.ceil(num_turns)
        if num_turns % 2 == 0:
            num_turns = num_turns + 1
        narrow_length = np.pi * meander_r * (num_turns+0.5) + 2*(narrow_length_left+narrow_length_right)


        closed_end_res_meander_narrow = sample.connect_meander(name='I closed end resonator', o1=fanout3,
                                                               port1=fanout3_port,
                                                            meander_length=narrow_length,
                                                            length_left=narrow_length_left,
                                                            length_right=narrow_length_right,
                                                            first_step_orientation='left',
                                                            meander_orientation=-fanout3.get_terminals()[fanout3_port].orientation,
                                                            meander_type='round',
                                                            airbridge=airbridge,
                                                            min_spacing=min_bridge_spacing, r=meander_r,
                                                            end_orientation=angle,
                                                               bridge_part_decimation=bridge_part_decimation)
        closed_end_res_meander = sample.connect_meander(name='II closed end resonator', o1=closed_end_res_meander_narrow[-1],
                                                             port1='port2',
                                                             meander_length=closed_end_res_meander_length-narrow_length,
                                                             length_left=res_length_left,
                                                             length_right=res_length_right,
                                                             first_step_orientation='right',
                                                             meander_orientation= angle-np.pi/2,
                                                             meander_type='round',
                                                             airbridge=airbridge,
                                                             min_spacing=min_bridge_spacing, r=meander_r,
                                                        bridge_part_decimation=bridge_part_decimation)

        total_length.append(sum([line.length for line in closed_end_res_meander_narrow]))
        total_length.append(sum([line.length for line in closed_end_res_meander]))

    else:
        closed_end_res_meander = sample.connect_meander(name='closed end resonator', o1=fanout3, port1=fanout3_port,
                                                  meander_length=closed_end_res_meander_length,
                                                  length_left=res_length_left,
                                                  length_right=res_length_right,
                                                  first_step_orientation='left',
                                                  meander_orientation=-fanout3.get_terminals()[fanout3_port].orientation,
                                                  meander_type='round',
                                                  airbridge=airbridge,
                                                  min_spacing=min_bridge_spacing, r=meander_r,
                                                        bridge_part_decimation=bridge_part_decimation)

        total_length.append(sum([line.length for line in closed_end_res_meander]))

    if object1 is None:
        object1 = sample.open_end(closed_end_res_meander[-1], 'port2', 'open end')
    else:
        open_end_res = sample.connect_cpw(closed_end_res_meander[-1], object1, 'port2', port, name='open end connection',
                                      points=[], airbridge=airbridge, min_spacing=min_bridge_spacing, r=meander_r)
        total_length.append(open_end_res[0].length)

    return fanout1, fanout2, fanout3, main_coupler, main_coupler2

def draw_purcelled_single_resonator_plus_qubit(sample,
                          coupler_start_x, coupler_start_y, coupler_length,
                          resonator_core,resonator_gap, resonator_ground, grounding_width,
                          closed_end_meander_length, length_left, length_right,
                          min_bridge_spacing = None,
                          airbridge = None, object1=None, port=None,
                          port_orientation='left', direction_orientation='down', first_step_orientation='left', meander_r = 55):


    coupler_w = [resonator_core]
    coupler_s = [resonator_gap, resonator_gap]

    # 2. Create main coupler:
    angle = 0
    if direction_orientation == 'up':
        angle = np.pi


    main_coupler = elements.CPWCoupler('purcell-resonator coupler',
                                       [(coupler_start_x, coupler_start_y),
                                        (coupler_start_x + np.cos(angle)*coupler_length, coupler_start_y)],
                                       coupler_w, coupler_s, resonator_ground, sample.layer_configuration, r=meander_r)

    sample.add(main_coupler)
    total_length = [coupler_length]

    # 3. Create fanout to create closed and opened ends of resonator
    fanout2_grouping = [1, 1]
    fanout2_port = 'down'

    if (port_orientation == 'right' and direction_orientation== 'down') or\
            (port_orientation == 'left' and direction_orientation == 'up'):
        coupler_port_for_fanout1 = 'port1'
        coupler_port_for_fanout2 = 'port2'
    else:
        coupler_port_for_fanout1 = 'port2'
        coupler_port_for_fanout2 = 'port1'

    coupler_closed_end = sample.ground(main_coupler, coupler_port_for_fanout1, 'closed end coupler', grounding_width,
                                       [(0, 2)])

    fanout2 = sample.fanout(o=main_coupler, port=coupler_port_for_fanout2, name='open end resonator fanout', grouping=fanout2_grouping)
    g2 = sample.ground(o=fanout2, port='middle', name='cl2', grounding_width=grounding_width,
                       grounding_between=[(0, 0)])

    open_end_meander = sample.connect_meander(name='open end', o1=fanout2, port1=fanout2_port,
                                                meander_length=closed_end_meander_length,
                                                length_left=length_left,
                                                length_right=length_right,
                                                first_step_orientation=first_step_orientation,
                                                meander_orientation=-fanout2.get_terminals()[fanout2_port].orientation,
                                                meander_type='round',
                                                airbridge=airbridge,
                                                min_spacing=min_bridge_spacing, r=meander_r)

    total_length.append(sum([line.length for line in open_end_meander]))

    # 10. Connect open end with the coupler part of the qubit

    open_end = sample.connect_cpw(open_end_meander[-1], object1, 'port2', port, name='open end connection',
                                    points=[], airbridge=airbridge, min_spacing=min_bridge_spacing, r=meander_r)

    cl, ll = open_end[0].cm(sample.epsilon)
    total_length.append(sum([line.length for line in open_end]))
    z01 = np.sqrt(ll[0] / cl[0])[0]
    res_params = (sum(total_length), z01, cl[0, 0])

    return res_params

def draw_single_resonator_plus_qubit(sample,
                          coupler_start_x, coupler_start_y, coupler_length,
                          resonator_core,resonator_gap, resonator_ground,
                          tl_core, tl_gap, tl_ground, grounding_width,
                          closed_end_meander_length, length_left, length_right,
                          min_bridge_spacing = None,
                          airbridge = None, object1=None, port=None,
                          open_end_length = None,
                          port_orientation='left', direction_orientation='down', points_for_the_open_end=[]):

    #left-> open end will be done for the left port
    coupler_w = [resonator_core, resonator_ground, tl_core]
    coupler_s = [resonator_gap, resonator_gap, tl_gap, tl_gap]

    # 2. Create main coupler:
    angle = 0
    if direction_orientation == 'up':
        coupler_start_x, coupler_length = coupler_start_x + coupler_length, -coupler_length
        angle = np.pi
    coupler_start_y = coupler_start_y - np.cos(angle) * (tl_gap / 2 + tl_core / 2)
    main_coupler = elements.CPWCoupler('TL-resonator coupler',
                                       [(coupler_start_x, coupler_start_y),
                                        (coupler_start_x + coupler_length, coupler_start_y)],
                                       coupler_w, coupler_s, tl_ground, sample.layer_configuration, r=100)

    sample.add(main_coupler)
    total_length = [coupler_length]

    # 3. Create fanout to create closed and opened ends of resonator
    fanout1 = sample.fanout(o=main_coupler, port='port1', name='closed end resonator fanout', grouping=[1, 3])
    # g1 = sample.ground(o=fanout1, port='center', name='cl1', grounding_width=grounding_width, grounding_between=[(2, 3)])
    fanout2 = sample.fanout(o=main_coupler, port='port2', name='open end resonator fanout', grouping=[1, 3])
    # g2 = sample.ground(o=fanout2, port='center', name='cl2', grounding_width=grounding_width, grounding_between=[(0, 1)])
    fanout1_port = 'up'
    fanout2_port = 'down'

    if port_orientation == 'right':
        fanout1, fanout2 = fanout2, fanout1
        fanout1_port, fanout2_port = fanout2_port, fanout1_port
    if direction_orientation == 'up':
        fanout1, fanout2 = fanout2, fanout1
        fanout1_port, fanout2_port = fanout2_port, fanout1_port

    # 6. Create closed meander of resonator

    closed_end_meander = sample.connect_meander(name='closed end', o1=fanout1, port1=fanout1_port,
                                                meander_length=closed_end_meander_length,
                                                length_left=length_left,
                                                length_right=length_right,
                                                first_step_orientation='right',
                                                meander_orientation=angle,
                                                meander_type='round',
                                                airbridge=airbridge,
                                                min_spacing=min_bridge_spacing, r=80)

    total_length.append(sum([line.length for line in closed_end_meander]))
    # # 7. Create grounding of resonator
    resonator_ground_ = sample.ground(o=closed_end_meander[-1], port='port2', name='resonator ground', grounding_width=30,
                                      grounding_between=[(0, 2)])

    # 10. Connect open end with the coupler part of the qubit
    if object1 is None:
        object1 = elements.OpenEnd(name='open end',
                                     position=(fanout2.get_terminals()[fanout2_port].position[0],
                                               fanout2.get_terminals()[fanout2_port].position[1]
                                               -np.cos(angle)*open_end_length),
                                     w=[resonator_core],
                                     s=[resonator_gap, resonator_gap],
                                     g=tl_ground,
                                     orientation=fanout2.get_terminals()[fanout2_port].orientation+np.pi,
                                     layer_configuration=sample.layer_configuration,
                                     h1=20,
                                     h2=10,
                                     )
        port = 'wide'
        sample.add(object1)

    open_end = sample.connect_cpw(fanout2, object1, fanout2_port, port, name='right open end',
                                    points=points_for_the_open_end, airbridge=airbridge, min_spacing=min_bridge_spacing)

    cl, ll = open_end[0].cm(sample.epsilon)
    total_length.append(sum([line.length for line in open_end]))
    z01 = np.sqrt(ll[0] / cl[0])[0]
    res_params = (sum(total_length), z01, cl[0, 0])
    if direction_orientation == 'up':
        g1, g2 = g2, g1

    return g1, g2, res_params


def draw_single_resonator(sample,
                          coupler_start_x, coupler_start_y, coupler_length,
                          resonator_core,resonator_gap, resonator_ground,
                          tl_core, tl_gap, tl_ground, grounding_width,
                          closed_end_meander_length, length_left, length_right,
                          open_end_length, min_bridge_spacing=None, airbridge=None,
                          port_orientation='left', direction_orientation='down'):
    return draw_single_resonator_plus_qubit(sample,
                                            coupler_start_x, coupler_start_y, coupler_length,
                                            resonator_core,resonator_gap, resonator_ground,
                                            tl_core, tl_gap, tl_ground, grounding_width,
                                            closed_end_meander_length, length_left, length_right,
                                            min_bridge_spacing=min_bridge_spacing,
                                            airbridge=airbridge, object1=None, port=None,
                                            open_end_length = open_end_length,
                                            port_orientation=port_orientation,
                                            direction_orientation=direction_orientation)


######################################################################################################################
def draw_double_resonator(sample,
                        coupler_start_x, coupler_start_y, coupler_length,
                        resonator_core,resonator_gap, resonator_ground,
                        tl_core,  tl_gap, tl_ground,
                        grounding_width,
                        closed_end_meander_length1, length_left1, length_right1,
                        closed_end_meander_length2, length_left2, length_right2,
                        open_end_length1, open_end_length2,
                        min_bridge_spacing_closed_end=None, min_bridge_spacing_open_end=None,
                        airbridge = None, port_orientation='left', meander_r = None):

    return draw_double_resonator_plus_double_qubit(sample,
                        coupler_start_x, coupler_start_y, coupler_length,
                        resonator_core,resonator_gap, resonator_ground,
                        tl_core,  tl_gap, tl_ground,
                        grounding_width,
                        closed_end_meander_length1, length_left1, length_right1,
                        closed_end_meander_length2,length_left2, length_right2,
                        object1=None, port1=None, open_end_length1=open_end_length1,
                        object2=None, port2=None, open_end_length2=open_end_length2,
                        min_bridge_spacing_closed_end=min_bridge_spacing_closed_end,
                        min_bridge_spacing_open_end=min_bridge_spacing_open_end, airbridge=airbridge,
                        port_orientation=port_orientation, meander_r=meander_r)


def draw_double_resonator_plus_double_qubit(sample,
                        coupler_start_x, coupler_start_y, coupler_length,
                        resonator_core,resonator_gap, resonator_ground,
                        tl_core,  tl_gap, tl_ground,
                        grounding_width,
                        closed_end_meander_length1, length_left1, length_right1,
                        closed_end_meander_length2,length_left2, length_right2,
                        open_end_shift_length1=None, open_end_shift_length2=None,
                        object1=None, port1=None, open_end_length1=None,
                        object2=None, port2=None, open_end_length2=None,
                        min_bridge_spacing_closed_end=None, min_bridge_spacing_open_end=None,
                        airbridge=None,
                        port_orientation='left',
                        meander_first_intend_orientation='left', meander_r = None,
                        object1_airbridges=False, object2_airbridges=False):
    # 2. Create main copler:

    main_coupler = elements.CPWCoupler('TL-resonator coupler', [(coupler_start_x, coupler_start_y),
                                                                (coupler_start_x + coupler_length, coupler_start_y)],
                                       [resonator_core, resonator_ground, tl_core, resonator_ground, resonator_core],
                                       [resonator_gap, resonator_gap, tl_gap, tl_gap, resonator_gap, resonator_gap],
                                       tl_ground, sample.layer_configuration, r=100)
    sample.add(main_coupler)

    fanout1 = sample.fanout(o=main_coupler, port='port1', name='closed end resonator fanout', grouping=[1, 4])
    fanout2 = sample.fanout(o=main_coupler, port='port2', name='open end resonator fanout', grouping=[1, 4])
    fanout1_port = 'up'
    fanout2_port = 'down'

    g1 = sample.ground(o=fanout1, port='center', name='cl1', grounding_width=grounding_width,
                       grounding_between=[(0, 1), (3, 4)])
    g2 = sample.ground(o=fanout2, port='center', name='cl2', grounding_width=grounding_width,
                       grounding_between=[(0, 1), (3, 4)])

    if port_orientation == 'right':
        fanout1, fanout2 = fanout2, fanout1
        fanout1_port, fanout2_port = fanout2_port, fanout1_port
    angle1 = np.pi
    angle2 = 0

    meander_first_intend_orientation2 = 'right' if meander_first_intend_orientation == 'left' else 'left'

    # 6. Create closed meander of resonator
    closed_end_meander1 = sample.connect_meander(name='closed end 1', o1=fanout1,
                                                 port1=fanout1_port,
                                                 meander_length=closed_end_meander_length1,
                                                 length_left=length_left1,
                                                 length_right=length_right1,
                                                 first_step_orientation=meander_first_intend_orientation,
                                                 meander_orientation=angle2, meander_type='round',
                                                 min_spacing=min_bridge_spacing_closed_end,
                                                 airbridge=airbridge, r=meander_r)
    closed_end_meander2 = sample.connect_meander(name='closed end 2', o1=fanout1,
                                                 port1=fanout2_port,
                                                 meander_length=closed_end_meander_length2,
                                                 length_left=length_left2,
                                                 length_right=length_right2,
                                                 first_step_orientation=meander_first_intend_orientation2,
                                                 meander_orientation=angle1, meander_type='round',
                                                 min_spacing=min_bridge_spacing_closed_end,
                                                 airbridge=airbridge, r=meander_r)

    if object1 is None:
        object1 = elements.OpenEnd(name='open end 1',
                                     position=(fanout2.get_terminals()[fanout2_port].position[0],
                                               fanout2.get_terminals()[fanout2_port].position[1] - open_end_length1),
                                     w=[resonator_core],
                                     s=[resonator_gap, resonator_gap],
                                     g=tl_ground,
                                     orientation=fanout2.get_terminals()[fanout1_port].orientation,
                                     layer_configuration=sample.layer_configuration,
                                     h1=20,
                                     h2=10,
                                     )
        sample.add(object1)
        port1 = 'wide'
        open_end_shift1 = []
    elif airbridge is not None:  # add an airbridge over resonator port as close as possible
        # terminal = object1.get_terminals()[port1]
        # airbridge_position = sample.cpw_shift(object1, port1, airbridge.pad_width / 2)[0]
        # bridge_resonator = elements.airbridge.AirbridgeOverCPW(
        #     name='Airbridge over %s qubit flux coupler' % object1.name,
        #     position=airbridge_position,
        #     orientation=terminal.orientation, w=terminal.w,
        #     s=terminal.s, g=terminal.g,
        #     geometry=airbridge)
        # sample.add(bridge_resonator)
        # sample.connect(bridge_resonator, 'port2', object1, port1)
        # port1 = 'port1'
        # object1 = bridge_resonator
        object1, port1 = sample.airbridge(
            object1, port1, name='Airbridge over %s readout resonator' % object1.name, geometry=airbridge)
        open_end_shift1 = sample.cpw_shift(fanout2, fanout2_port, open_end_shift_length1)
    else:
        open_end_shift1 = sample.cpw_shift(fanout2, fanout2_port, open_end_shift_length1)

    if object2 is None:
        object2 = elements.OpenEnd(name='open end 2',
                                     position=(fanout2.get_terminals()[fanout1_port].position[0],
                                               fanout2.get_terminals()[fanout1_port].position[1] + open_end_length2),
                                     w=[resonator_core],
                                     s=[resonator_gap, resonator_gap],
                                     g=tl_ground,
                                     orientation=fanout2.get_terminals()[fanout2_port].orientation,
                                     layer_configuration=sample.layer_configuration,
                                     h1=20,
                                     h2=10,
                                     )
        sample.add(object2)
        port2 = 'wide'
        open_end_shift2 = []
    elif airbridge is not None:  # add an airbridge over resonator port as close as possible
        object2, port2 = sample.airbridge(
            object2, port2, name='Airbridge over %s readout resonator' % object2.name, geometry=airbridge)
        open_end_shift2 = sample.cpw_shift(fanout2, fanout1_port, open_end_shift_length2)
    else:
        open_end_shift2 = sample.cpw_shift(fanout2, fanout1_port, open_end_shift_length2)

    # 11. Connect open end with the coupler part of the resonator

    open_end_resonator1 = sample.connect_cpw(fanout2, object1, fanout2_port, port1, name='right open end',
                                             points=open_end_shift1, min_spacing=min_bridge_spacing_open_end,
                                             airbridge=airbridge, r=meander_r)


    open_end_resonator2 = sample.connect_cpw(fanout2, object2, fanout1_port, port2, name='right open end',
                                             points=open_end_shift2, min_spacing=min_bridge_spacing_open_end,
                                             airbridge=airbridge, r=meander_r)


    # 11. Create grounding of resonator
    resonator_ground_1 = sample.ground(o=closed_end_meander1[-1], port='port2', name='resonator ground 1',
                                       grounding_width=30, grounding_between=[(0, 2)])
    resonator_ground_2 = sample.ground(o=closed_end_meander2[-1], port='port2', name='resonator ground 2',
                                       grounding_width=30, grounding_between=[(0, 2)])

    cl, ll = open_end_resonator1[0].cm(sample.epsilon)
    z01 = np.sqrt(ll[0] / cl[0])[0]
    z02 = np.sqrt(ll[0] / cl[0])[0]
    # get some resonator parameters
    res_params1 = (closed_end_meander_length1+open_end_resonator1[0].length+coupler_length, z01, cl[0,0])
    res_params2 = (closed_end_meander_length2+open_end_resonator2[0].length+coupler_length, z02, cl[0,0])
    return g1, g2, res_params1, res_params2


def search_for_resonators_qubits(f,delta,min_freq,max_freq):
    res_modes = []
    qs=f/delta/2
    min_freq=min_freq*1e9*2*np.pi
    max_freq=max_freq*1e9*2*np.pi
    min_Q=0#1e3
    max_Q=1e12#1e9
    for mode_id in range(len(qs)):
        if min_Q<=qs[mode_id]<=max_Q and min_freq<=f[mode_id]/2/np.pi<=max_freq:
            res_modes.append(mode_id)
    print('Resonance frequencies are, GHz:',f[res_modes]/(2*np.pi)/1e9)
    print('Kappas are, us^-1:',2*delta[res_modes]/1e6)
    print('Quality factors are:',qs[res_modes])
    return (f[res_modes]/(2*np.pi)/1e9, delta[res_modes], qs[res_modes])

def get_grounded_qubit_resonator_coupling(resonator,qubit,coupler_name,res_fr,qubit_fr):
    cap_scal_factor=1e-15
    claw_cap_total = qubit.C[coupler_name][0]+qubit.C[coupler_name][1]
    #resonator's lenght should be in microns
    res_total_cap = resonator[0]*1/2*resonator[2]+claw_cap_total*cap_scal_factor # in case of lambda/4 resonators
    qubit_total_cap = (qubit.C[coupler_name][1] + qubit.C['qubit'])*cap_scal_factor
    coupling_cap= qubit.C[coupler_name][1]*cap_scal_factor
    coupling = coupling_cap/np.sqrt(qubit_total_cap*res_total_cap)*np.sqrt(res_fr*qubit_fr)/2
    return {'g':(coupling*1e3,'MHz/2pi')}

# TODO: make it better?
def get_grounded_qubit_resonator_parameters(resonator,qubit,coupler_name,res_fr,qubit_fr,kappa):
    from scipy.constants import hbar,h,e,c
    cap_scal_factor=1e-15
    coupling = get_grounded_qubit_resonator_coupling(resonator,qubit,coupler_name,res_fr,qubit_fr)['g'][0]
    qubit_total_cap = (qubit.C[coupler_name][1] + qubit.C['qubit'])*cap_scal_factor
    alpha = -e**2/(2*qubit_total_cap)/h/1e9
    disp_shift=(coupling/1e3)**2/((qubit_fr-res_fr))/(1+(qubit_fr-res_fr)/(alpha))
    T=(1/kappa)*((qubit_fr-res_fr)/(coupling/1e3))**2
    print("Qubit- resonator detuning:",qubit_fr-res_fr)
    return {'g':(coupling,'MHz/2pi'),
            'alpha':(alpha*1e3,'MHz/2pi'),
            'chi':(disp_shift*1e3,'MHz/2pi'),
            'T':(T*1e6,'us'),
            'protection ratio':((qubit_fr-res_fr)/(coupling/1e3))**2}


def draw_rounded_single_resonator_plus_qubit_orientation(sample,
                          coupler_start_x, coupler_start_y, coupler_length,
                          resonator_core,resonator_gap, resonator_ground,
                          tl_core, tl_gap, tl_ground, resonator_tl_ground,
                          closed_end_meander_length, length_left, length_right,
                          min_bridge_spacing = None,
                          airbridge = None, object1=None, port=None,
                          open_end_length = None,
                          port_orientation='left',meander_orientation='right', direction_orientation='down', points_for_the_open_end=[],orient_angle=np.pi/4):

    #left-> open end will be done for the left port
    coupler_w = [resonator_core, resonator_tl_ground, tl_core]
    coupler_s = [resonator_gap, resonator_gap, tl_gap, tl_gap]

    # 2. Create main coupler:
    angle = 0
    if direction_orientation == 'up':
        coupler_start_x, coupler_length = coupler_start_x + coupler_length, -coupler_length
        angle = np.pi
    coupler_start_y_initial = deepcopy(coupler_start_y)
    coupler_start_x_initial = deepcopy(coupler_start_x)
    coupler_start_y = coupler_start_y + np.sin(orient_angle-np.pi/2)*((resonator_gap*2+tl_gap*2+resonator_tl_ground+tl_core+resonator_core+tl_ground*2)/2-tl_ground-tl_gap-tl_core/2)
    coupler_start_x = coupler_start_x + np.cos(orient_angle-np.pi/2)*((resonator_gap*2+tl_gap*2+resonator_tl_ground+tl_core+resonator_core+tl_ground*2)/2-tl_ground-tl_gap-tl_core/2)

    main_coupler = elements.CPWCoupler('TL-resonator coupler',
                                       [(coupler_start_x, coupler_start_y),
                                        (coupler_start_x + np.cos(orient_angle)*coupler_length,
                                         coupler_start_y + np.sin(orient_angle)*coupler_length)],
                                       coupler_w, coupler_s, tl_ground, sample.layer_configuration, r=100)
    temp_value = deepcopy(main_coupler.terminals['port1'])
    main_coupler.terminals['port1'].position = [coupler_start_x_initial, coupler_start_y_initial]
    main_coupler.terminals['port1'].type='cpw'
    main_coupler.terminals['port1'].w = tl_core
    main_coupler.terminals['port1'].s = tl_gap
    main_coupler.terminals['port1'].g = tl_ground

    main_coupler.terminals['port2'].position = [coupler_start_x_initial+ np.cos(orient_angle)*coupler_length,
                                                coupler_start_y_initial+ np.sin(orient_angle)*coupler_length]
    main_coupler.terminals['port2'].type = 'cpw'
    main_coupler.terminals['port2'].w = tl_core
    main_coupler.terminals['port2'].s = tl_gap
    main_coupler.terminals['port2'].g = tl_ground
    print(main_coupler.terminals['port1'].position,main_coupler.terminals['port2'].position)
    main_coupler.terminals.update({'res1':deepcopy(temp_value),'res2':deepcopy(temp_value)})
    print(coupler_start_x,coupler_start_y)
    main_coupler.terminals['res1'].position = [coupler_start_x_initial+np.cos(orient_angle-np.pi/2)*(tl_core/2+resonator_core/2+resonator_gap+resonator_tl_ground+tl_gap),
                                                coupler_start_y_initial +np.sin(orient_angle-np.pi/2)*(tl_core/2+resonator_core/2+resonator_gap+resonator_tl_ground+tl_gap)]
    main_coupler.terminals['res1'].type = 'cpw'
    main_coupler.terminals['res1'].orientation=main_coupler.terminals['res1'].orientation+np.pi*0
    main_coupler.terminals['res1'].w = resonator_core
    main_coupler.terminals['res1'].s = resonator_gap
    main_coupler.terminals['res1'].g = resonator_ground

    main_coupler.terminals['res2'].position = [coupler_start_x_initial + np.cos(orient_angle)*coupler_length+np.cos(orient_angle-np.pi/2)*(tl_core/2+resonator_core/2+resonator_gap+resonator_tl_ground+tl_gap),
                                                coupler_start_y_initial + np.sin(orient_angle)*coupler_length +np.sin(orient_angle-np.pi/2)*(tl_core/2+resonator_core/2+resonator_gap+resonator_tl_ground+tl_gap)]
    main_coupler.terminals['res2'].orientation=main_coupler.terminals['res2'].orientation+np.pi
    main_coupler.terminals['res2'].type = 'cpw'
    main_coupler.terminals['res2'].w = resonator_core
    main_coupler.terminals['res2'].s = resonator_gap
    main_coupler.terminals['res2'].g = resonator_ground
    print(main_coupler.terminals['res1'].position,main_coupler.terminals['res2'].position)

    sample.add(main_coupler)
    total_length = [abs(coupler_length)]
    ################# first fanout
    fanout_offset=180
    left_rounded_fanout = elements.CPWCoupler('TL-resonator coupler',
                                       [
                                       (coupler_start_x+ np.cos(orient_angle+np.pi) * fanout_offset*np.cos(angle)+ np.cos(orient_angle+np.pi+np.pi/2) * fanout_offset*np.cos(angle),
                                        coupler_start_y+ np.sin(orient_angle+np.pi) * fanout_offset*np.cos(angle)+ np.sin(orient_angle+np.pi+np.pi/2) * fanout_offset*np.cos(angle)),
                                       (coupler_start_x+ np.cos(orient_angle+np.pi) * fanout_offset*np.cos(angle)+ np.cos(orient_angle+np.pi+np.pi/2) * (fanout_offset+50)*np.cos(angle),
                                        coupler_start_y+ np.sin(orient_angle+np.pi) * fanout_offset*np.cos(angle)+ np.sin(orient_angle+np.pi+np.pi/2) * (fanout_offset+50)*np.cos(angle))
                                       ],
                                       [resonator_core], [resonator_gap, resonator_gap], resonator_ground, sample.layer_configuration, r=100)
    left_rounded_fanout.terminals['port1'].type = 'cpw'
    left_rounded_fanout.terminals['port1'].w = resonator_core
    left_rounded_fanout.terminals['port1'].s = resonator_gap

    left_rounded_fanout.terminals['port2'].type = 'cpw'
    left_rounded_fanout.terminals['port2'].w = resonator_core
    left_rounded_fanout.terminals['port2'].s = resonator_gap
    sample.add(left_rounded_fanout)

    left_fanout_length = sample.connect_cpw(o1=main_coupler, o2=left_rounded_fanout, port1='res1', port2='port1', name='feedline',
                           points=[])
    total_length.append(sum([line.length for line in left_fanout_length]))
    # 7. Create grounding of resonator

    resonator_ground_ = sample.ground(o=main_coupler, port='res2', name='resonator ground',
                                      grounding_width=30,
                                      grounding_between=[(0, 2)])
    # return 0

    total_length.append(50)

    # 6. Create closed meander of resonator
    closed_end_meander = sample.connect_meander(name='closed end', o1=left_rounded_fanout, port1='port2',
                                                meander_length=closed_end_meander_length,
                                                length_left=length_left,
                                                length_right=length_right,
                                                first_step_orientation=meander_orientation,
                                                meander_orientation=orient_angle,
                                                meander_type='round',
                                                airbridge=airbridge,
                                                min_spacing=min_bridge_spacing, r=80)

    total_length.append(sum([line.length for line in closed_end_meander]))

    # resonator_ground_ = sample.ground(o=closed_end_meander[-1], port='port2', name='resonator ground', grounding_width=30,
    #                                   grounding_between=[(0, 2)])


    # 10. Connect open end with the coupler part of the qubit
    if object1 is None:
        object1 = elements.OpenEnd(name='open end',
                                     position=(closed_end_meander[-1].get_terminals()['port2'].position[0],
                                               closed_end_meander[-1].get_terminals()['port2'].position[1]
                                               -np.cos(angle)*open_end_length*0),
                                     w=[resonator_core],
                                     s=[resonator_gap, resonator_gap],
                                     g=resonator_ground,
                                     orientation=closed_end_meander[-1].get_terminals()['port2'].orientation+np.pi,
                                     layer_configuration=sample.layer_configuration,
                                     h1=20,
                                     h2=10,
                                     )
        port = 'wide'
        sample.add(object1)
    else:
        open_end = sample.connect_cpw(closed_end_meander[-1], object1, 'port2', port, name='right open end',
                                  points=points_for_the_open_end, airbridge=airbridge, min_spacing=min_bridge_spacing)
        total_length.append(sum([line.length for line in open_end]))
    # print(total_length)
    cl, ll = left_fanout_length[0].cm(sample.epsilon)
    z01 = np.sqrt(ll[0] / cl[0])[0]
    res_params = (sum(total_length), z01, cl[0, 0])
    return main_coupler,res_params

def draw_rounded_single_resonator_orientaion(sample,
                          coupler_start_x, coupler_start_y, coupler_length,
                          resonator_core,resonator_gap, resonator_ground,
                          tl_core, tl_gap, tl_ground, grounding_width,
                          closed_end_meander_length, length_left, length_right,
                          open_end_length, min_bridge_spacing=None, airbridge=None,
                          port_orientation='left',meander_orientation='right', direction_orientation='down',transmission=True):
    return draw_rounded_single_resonator_plus_qubit_orientation(sample,
                                            coupler_start_x, coupler_start_y, coupler_length,
                                            resonator_core,resonator_gap, resonator_ground,
                                            tl_core, tl_gap, tl_ground, grounding_width,
                                            closed_end_meander_length, length_left, length_right,
                                            min_bridge_spacing=min_bridge_spacing,
                                            airbridge=airbridge, object1=None, port=None,
                                            open_end_length = 0,
                                            port_orientation=port_orientation,
                                            meander_orientation = meander_orientation,
                                            direction_orientation=direction_orientation,
                                            points_for_the_open_end=[],
                                            transmission=transmission)


def draw_rounded_single_resonator_plus_qubit(sample,
                          coupler_start_x, coupler_start_y, coupler_length,
                          resonator_core,resonator_gap, resonator_ground,
                          tl_core, tl_gap, tl_ground, resonator_tl_ground,
                          closed_end_meander_length, length_left, length_right,
                          min_bridge_spacing = None,
                          airbridge = None, object1=None, port=None,
                          open_end_length = None,
                          port_orientation='left',meander_orientation='right', direction_orientation='down', points_for_the_open_end=[],transmission=False):

    #left-> open end will be done for the left port
    coupler_w = [resonator_core, resonator_tl_ground, tl_core]
    coupler_s = [resonator_gap, resonator_gap, tl_gap, tl_gap]

    # 2. Create main coupler:
    angle = 0
    if direction_orientation == 'up':
        coupler_start_x, coupler_length = coupler_start_x + coupler_length, -coupler_length
        angle = np.pi
    coupler_start_y_initial=deepcopy(coupler_start_y)
    coupler_start_y = coupler_start_y - np.cos(angle)*((resonator_gap*2+tl_gap*2+resonator_tl_ground+tl_core+resonator_core+tl_ground*2)/2-tl_ground-tl_gap-tl_core/2)

    main_coupler = elements.CPWCoupler('TL-resonator coupler',
                                       [(coupler_start_x, coupler_start_y),
                                        (coupler_start_x + coupler_length, coupler_start_y)],
                                       coupler_w, coupler_s, tl_ground, sample.layer_configuration, r=100)
    temp_value = deepcopy(main_coupler.terminals['port1'])
    main_coupler.terminals['port1'].position = [coupler_start_x, coupler_start_y_initial]
    main_coupler.terminals['port1'].type='cpw'
    main_coupler.terminals['port1'].w = tl_core
    main_coupler.terminals['port1'].s = tl_gap
    main_coupler.terminals['port1'].g = tl_ground

    main_coupler.terminals['port2'].position = [coupler_start_x+coupler_length,
                                                coupler_start_y_initial]
    main_coupler.terminals['port2'].type = 'cpw'
    main_coupler.terminals['port2'].w = tl_core
    main_coupler.terminals['port2'].s = tl_gap
    main_coupler.terminals['port2'].g = tl_ground

    main_coupler.terminals.update({'res1':deepcopy(temp_value),'res2':deepcopy(temp_value)})
    main_coupler.terminals['res1'].position = [coupler_start_x,
                                                coupler_start_y_initial - np.cos(angle)*(tl_core/2+resonator_core/2+resonator_gap+resonator_tl_ground+tl_gap)]
    main_coupler.terminals['res1'].type = 'cpw'
    main_coupler.terminals['res1'].orientation=main_coupler.terminals['res1'].orientation+np.pi*0
    main_coupler.terminals['res1'].w = resonator_core
    main_coupler.terminals['res1'].s = resonator_gap
    main_coupler.terminals['res1'].g = resonator_ground

    main_coupler.terminals['res2'].position = [coupler_start_x + coupler_length,
                                                coupler_start_y_initial - np.cos(angle)*(tl_core/2+resonator_core/2+resonator_gap+resonator_tl_ground+tl_gap)]
    main_coupler.terminals['res2'].orientation=main_coupler.terminals['res2'].orientation+np.pi
    main_coupler.terminals['res2'].type = 'cpw'
    main_coupler.terminals['res2'].w = resonator_core
    main_coupler.terminals['res2'].s = resonator_gap
    main_coupler.terminals['res2'].g = resonator_ground
    sample.add(main_coupler)
    total_length = [abs(coupler_length)]
    ################# first fanout
    fanout_offset=150
    left_rounded_fanout = elements.CPWCoupler('TL-resonator coupler',
                                       [(coupler_start_x-np.cos(angle)*fanout_offset, coupler_start_y-np.cos(angle)*200),
                                        (coupler_start_x -np.cos(angle)*fanout_offset, coupler_start_y-np.cos(angle)*250)],
                                       [resonator_core], [resonator_gap, resonator_gap], resonator_ground, sample.layer_configuration, r=100)
    left_rounded_fanout.terminals['port1'].type = 'cpw'
    left_rounded_fanout.terminals['port1'].w = resonator_core
    left_rounded_fanout.terminals['port1'].s = resonator_gap

    left_rounded_fanout.terminals['port2'].type = 'cpw'
    left_rounded_fanout.terminals['port2'].w = resonator_core
    left_rounded_fanout.terminals['port2'].s = resonator_gap

    sample.add(left_rounded_fanout)
    left_fanout_length = sample.connect_cpw(o1=main_coupler, o2=left_rounded_fanout, port1='res1', port2='port1', name='feedline',
                           points=[])
    total_length.append(sum([line.length for line in left_fanout_length]))
    total_length.append(50)

    # 6. Create closed meander of resonator
    closed_end_meander = sample.connect_meander(name='closed end', o1=left_rounded_fanout, port1='port2',
                                                meander_length=closed_end_meander_length,
                                                length_left=length_left,
                                                length_right=length_right,
                                                first_step_orientation=meander_orientation,
                                                meander_orientation=angle,
                                                meander_type='round',
                                                airbridge=airbridge,
                                                min_spacing=min_bridge_spacing, r=80)

    total_length.append(sum([line.length for line in closed_end_meander]))
    # # 7. Create grounding of resonator
    resonator_ground_ = sample.ground(o=main_coupler, port='res2', name='resonator ground',
                                      grounding_width=30,
                                      grounding_between=[(0, 2)])
    # resonator_ground_ = sample.ground(o=closed_end_meander[-1], port='port2', name='resonator ground', grounding_width=30,
    #                                   grounding_between=[(0, 2)])


    # 10. Connect open end with the coupler part of the qubit
    if object1 is None:
        object1 = elements.OpenEnd(name='open end',
                                     position=(closed_end_meander[-1].get_terminals()['port2'].position[0],
                                               closed_end_meander[-1].get_terminals()['port2'].position[1]
                                               -np.cos(angle)*open_end_length*0),
                                     w=[resonator_core],
                                     s=[resonator_gap, resonator_gap],
                                     g=resonator_ground,
                                     orientation=closed_end_meander[-1].get_terminals()['port2'].orientation+np.pi,
                                     layer_configuration=sample.layer_configuration,
                                     h1=20,
                                     h2=10,
                                     )
        port = 'wide'
        sample.add(object1)
    else:
        open_end = sample.connect_cpw(closed_end_meander[-1], object1, 'port2', port, name='right open end',
                                  points=points_for_the_open_end, airbridge=airbridge, min_spacing=min_bridge_spacing)
        total_length.append(sum([line.length for line in open_end]))
    # print(total_length)
    cl, ll = left_fanout_length[0].cm(sample.epsilon)
    z01 = np.sqrt(ll[0] / cl[0])[0]
    res_params = (sum(total_length), z01, cl[0, 0])
    return main_coupler,res_params

def draw_rounded_single_resonator(sample,
                          coupler_start_x, coupler_start_y, coupler_length,
                          resonator_core,resonator_gap, resonator_ground,
                          tl_core, tl_gap, tl_ground, grounding_width,
                          closed_end_meander_length, length_left, length_right,
                          open_end_length, min_bridge_spacing=None, airbridge=None,
                          port_orientation='left',meander_orientation='right', direction_orientation='down',transmission=True):
    return draw_rounded_single_resonator_plus_qubit(sample,
                                            coupler_start_x, coupler_start_y, coupler_length,
                                            resonator_core,resonator_gap, resonator_ground,
                                            tl_core, tl_gap, tl_ground, grounding_width,
                                            closed_end_meander_length, length_left, length_right,
                                            min_bridge_spacing=min_bridge_spacing,
                                            airbridge=airbridge, object1=None, port=None,
                                            open_end_length = 0,
                                            port_orientation=port_orientation,
                                            meander_orientation = meander_orientation,
                                            direction_orientation=direction_orientation,
                                            points_for_the_open_end=[],
                                            transmission=transmission)