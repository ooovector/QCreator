import numpy as np

def draw_single_resonator_plus_qubit(sample, elements, coupler_start_x, coupler_start_y, coupler_length,
                                     resonator_core, resonator_ground, tl_core, resonator_gap, tl_gap, grounding_width,
                                     closed_end_meander_length, qubit, coupler_name):
    # 2. Create main copler:
    total_length = 0
    main_coupler = elements.CPWCoupler('TL-resonator coupler', [(coupler_start_x, central_line_y),
                                                                (coupler_start_x + coupler_length, central_line_y)],
                                       [resonator_core, resonator_ground, tl_core],
                                       [resonator_gap, resonator_gap, tl_gap, tl_gap],
                                       tl_ground, sample.layer_configuration, r=100)
    sample.add(main_coupler)
    total_length = total_length + coupler_length

    # 3. Create fanout to create closed end of resonator
    fanout_for_closed_end = sample.fanout(o=main_coupler, port='port1', name='closed end resonator fanout',
                                          grouping=[1, 3])

    # 4.
    g1 = sample.ground(o=fanout_for_closed_end, port='center', name='cl1', grounding_width=grounding_width,
                       grounding_between=[(2, 3)])

    # 6. Create closed meander of resonator
    closed_end_meander = sample.connect_meander(name='closed end', o1=fanout_for_closed_end, port1='up',
                                                meander_length=closed_end_meander_length,
                                                length_left=150,
                                                length_right=300,
                                                first_step_orientation='left',
                                                meander_orientation=0, meander_type='round')
    total_length = total_length + closed_end_meander.length
    # 7. Create grounding of resonator
    resonator_ground_ = sample.ground(o=closed_end_meander, port='port2', name='resonator ground', grounding_width=30,
                                      grounding_between=[(0, 2)])

    # 8. Create fanout to create open end of resonator
    fanout_for_open_end = sample.fanout(o=main_coupler, port='port2', name='open end resonator fanout', grouping=[1, 3])

    # 9.
    g2 = sample.ground(o=fanout_for_open_end, port='center', name='cl2', grounding_width=10, grounding_between=[(0, 1)])

    # 10. Connect open end with the coupler part of the qubit
    open_end = sample.connect_cpw(fanout_for_open_end, qubit, 'down', coupler_name, name='right open end', points=[])
    total_length = total_length + open_end.length
    res_params = total_length

    return g1, g2, res_params


######################################################################################################################
def draw_double_resonator(sample, elements,
                        coupler_start_x, coupler_start_y, coupler_length,
                        resonator_core,resonator_gap, resonator_ground,
                        tl_core,  tl_gap, tl_ground,
                        grounding_width,
                        closed_end_meander_length1, length_left1, length_right1,
                        closed_end_meander_length2,length_left2, length_right2,
                        open_length1, open_length2, orientation='left'):
    # 2. Create main copler:

    main_coupler = elements.CPWCoupler('TL-resonator coupler', [(coupler_start_x, coupler_start_y),
                                                                (coupler_start_x + coupler_length, coupler_start_y)],
                                       [resonator_core, resonator_ground, tl_core, resonator_ground, resonator_core],
                                       [resonator_gap, resonator_gap, tl_gap, tl_gap, resonator_gap, resonator_gap],
                                       tl_ground, sample.layer_configuration, r=100)
    sample.add(main_coupler)

    # 3. Create fanout to create closed end of resonator
    if orientation == 'left':
        port1 = 'port1'
        port2 = 'port2'
        direction2 = 'up'
        direction1 = 'down'
        closed_end_direction2 = 'up'
        closed_end_direction1 = 'down'
        angle2 = np.pi
        angle1 = 0
    else:
        port1 = 'port2'
        port2 = 'port1'
        closed_end_direction2 = 'up'
        closed_end_direction1 = 'down'
        direction2 = 'down'
        direction1 = 'up'
        angle1 = np.pi
        angle2 = 0

    fanout_for_closed_end = sample.fanout(o=main_coupler, port=port1, name='closed end resonator fanout',
                                          grouping=[1, 4])

    # 4.
    g1 = sample.ground(o=fanout_for_closed_end, port='center', name='cl1', grounding_width=grounding_width,
                       grounding_between=[(0, 1), (3, 4)])

    # 6. Create closed meander of resonator
    closed_end_meander1 = sample.connect_meander(name='closed end 1', o1=fanout_for_closed_end,
                                                 port1=closed_end_direction1,
                                                 meander_length=closed_end_meander_length1,
                                                 length_left=length_left1,
                                                 length_right=length_right1,
                                                 first_step_orientation='left',
                                                 meander_orientation=angle2, meander_type='round')
    closed_end_meander2 = sample.connect_meander(name='closed end 2', o1=fanout_for_closed_end,
                                                 port1=closed_end_direction2,
                                                 meander_length=closed_end_meander_length2,
                                                 length_left=length_left2,
                                                 length_right=length_right2,
                                                 first_step_orientation='left',
                                                 meander_orientation=angle1, meander_type='round')

    # 7. Create fanout to create closed enfd of resonator
    fanout_for_open_end = sample.fanout(o=main_coupler, port=port2, name='open end resonator fanout', grouping=[1, 4])

    # 8.
    g2 = sample.ground(o=fanout_for_open_end, port='center', name='cl2', grounding_width=grounding_width,
                       grounding_between=[(0, 1), (3, 4)])

    # 10. Create open end of resonators
    open_end1 = elements.OpenEnd(name='open end 1',
                                 position=(fanout_for_open_end.get_terminals()[direction1].position[0],
                                           fanout_for_open_end.get_terminals()[direction1].position[1] - open_length1),
                                 w=[resonator_core],
                                 s=[resonator_gap, resonator_gap],
                                 g=tl_ground,
                                 orientation=fanout_for_open_end.get_terminals()[direction2].orientation,
                                 layer_configuration=sample.layer_configuration,
                                 h1=20,
                                 h2=10,
                                 )
    sample.add(open_end1)
    open_end2 = elements.OpenEnd(name='open end 2',
                                 position=(fanout_for_open_end.get_terminals()[direction2].position[0],
                                           fanout_for_open_end.get_terminals()[direction2].position[1] + open_length2),
                                 w=[resonator_core],
                                 s=[resonator_gap, resonator_gap],
                                 g=tl_ground,
                                 orientation=fanout_for_open_end.get_terminals()[direction1].orientation,
                                 layer_configuration=sample.layer_configuration,
                                 h1=20,
                                 h2=10,
                                 )
    sample.add(open_end2)

    # 11. Connect open end with the coupler part of the resonator
    open_end_resonator1 = sample.connect_cpw(fanout_for_open_end, open_end1, direction1, 'wide', name='right open end',
                                             points=[])

    open_end_resonator2 = sample.connect_cpw(fanout_for_open_end, open_end2, direction2, 'wide', name='right open end',
                                             points=[])

    # 11. Create grounding of resonator
    resonator_ground_1 = sample.ground(o=closed_end_meander1, port='port2', name='resonator ground 1',
                                       grounding_width=30, grounding_between=[(0, 2)])
    resonator_ground_2 = sample.ground(o=closed_end_meander2, port='port2', name='resonator ground 2',
                                       grounding_width=30, grounding_between=[(0, 2)])

    return g1, g2


#######################################################################################################################
def draw_double_resonator_plus_double_qubit(sample, elements,
                                            coupler_start_x, coupler_start_y, coupler_length,
                                            resonator_core,resonator_gap, resonator_ground,
                                            tl_core,  tl_gap, tl_ground,
                                            grounding_width,
                                            closed_end_meander_length1, length_left1, length_right1,
                                            closed_end_meander_length2,length_left2, length_right2,
                                            open_end_shift_length1, open_end_shift_length2,
                                            qubit1, coupler_name1,
                                            qubit2, coupler_name2,
                                            orientation='left'):
    # 2. Create main copler:

    main_coupler = elements.CPWCoupler('TL-resonator coupler', [(coupler_start_x, coupler_start_y),
                                                                (coupler_start_x + coupler_length, coupler_start_y)],
                                       [resonator_core, resonator_ground, tl_core, resonator_ground, resonator_core],
                                       [resonator_gap, resonator_gap, tl_gap, tl_gap, resonator_gap, resonator_gap],
                                       tl_ground, sample.layer_configuration, r=100)
    sample.add(main_coupler)

    # 3. Create fanout to create closed end of resonator
    if orientation == 'left':
        port1 = 'port1'
        port2 = 'port2'
        direction2 = 'up'
        direction1 = 'down'
        closed_end_direction2 = 'up'
        closed_end_direction1 = 'down'
        angle2 = np.pi
        angle1 = 0
    else:
        port1 = 'port2'
        port2 = 'port1'
        closed_end_direction2 = 'up'
        closed_end_direction1 = 'down'
        direction2 = 'down'
        direction1 = 'up'
        angle1 = np.pi
        angle2 = 0

    fanout_for_closed_end = sample.fanout(o=main_coupler, port=port1, name='closed end resonator fanout',
                                          grouping=[1, 4])

    # 4.
    g1 = sample.ground(o=fanout_for_closed_end, port='center', name='cl1', grounding_width=grounding_width,
                       grounding_between=[(0, 1), (3, 4)])

    # 6. Create closed meander of resonator
    closed_end_meander1 = sample.connect_meander(name='closed end 1', o1=fanout_for_closed_end,
                                                 port1=closed_end_direction1,
                                                 meander_length=closed_end_meander_length1,
                                                 length_left=length_left1,
                                                 length_right=length_right1,
                                                 first_step_orientation='left',
                                                 meander_orientation=angle2, meander_type='round')
    closed_end_meander2 = sample.connect_meander(name='closed end 2', o1=fanout_for_closed_end,
                                                 port1=closed_end_direction2,
                                                 meander_length=closed_end_meander_length2,
                                                 length_left=length_left2,
                                                 length_right=length_right2,
                                                 first_step_orientation='left',
                                                 meander_orientation=angle1, meander_type='round')

    # 7. Create fanout to create closed enfd of resonator
    fanout_for_open_end = sample.fanout(o=main_coupler, port=port2, name='open end resonator fanout', grouping=[1, 4])

    # 8.
    g2 = sample.ground(o=fanout_for_open_end, port='center', name='cl2', grounding_width=grounding_width,
                       grounding_between=[(0, 1), (3, 4)])

    # 10. Create closed meander of resonator
    open_end_shift1 = sample.cpw_shift(fanout_for_open_end, direction1, open_end_shift_length1)
    open_end_shift2 = sample.cpw_shift(fanout_for_open_end, direction2, open_end_shift_length2)

    open_end = sample.connect_cpw(fanout_for_open_end, qubit1, direction1, coupler_name1, name='right open end 1',
                                  points=open_end_shift1)
    open_end = sample.connect_cpw(fanout_for_open_end, qubit2, direction2, coupler_name2, name='right open end 2',
                                  points=open_end_shift2)

    # 11. Create grounding of resonator
    resonator_ground_1 = sample.ground(o=closed_end_meander1, port='port2', name='resonator ground 1',
                                       grounding_width=30, grounding_between=[(0, 2)])
    resonator_ground_2 = sample.ground(o=closed_end_meander2, port='port2', name='resonator ground 2',
                                       grounding_width=30, grounding_between=[(0, 2)])

    return g1, g2