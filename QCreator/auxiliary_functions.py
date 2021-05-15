import numpy as np
from . import elements


def draw_single_resonator_plus_qubit(sample,
                          coupler_start_x, coupler_start_y, coupler_length,
                          resonator_core,resonator_gap, resonator_ground,
                          tl_core, tl_gap, tl_ground, grounding_width,
                          closed_end_meander_length, length_left, length_right,
                          min_bridge_spacing = None,
                          airbridge = None, object1=None, port=None,
                          open_end_length = None,
                          port_orientation='left', direction_orientation='down'):

    #left-> open end will be done for the left port
    coupler_w = [resonator_core, resonator_ground, tl_core]
    coupler_s = [resonator_gap, resonator_gap, tl_gap, tl_gap]

    # 2. Create main coupler:
    angle = 0
    if direction_orientation == 'up':
        coupler_start_x, coupler_length = coupler_start_x + coupler_length, -coupler_length
        angle = np.pi

    main_coupler = elements.CPWCoupler('TL-resonator coupler',
                                       [(coupler_start_x, coupler_start_y),
                                        (coupler_start_x + coupler_length, coupler_start_y)],
                                       coupler_w, coupler_s, tl_ground, sample.layer_configuration, r=100)

    sample.add(main_coupler)
    total_length = [coupler_length]

    # 3. Create fanout to create closed and opened ends of resonator
    fanout1 = sample.fanout(o=main_coupler, port='port1', name='closed end resonator fanout', grouping=[1, 3])
    g1 = sample.ground(o=fanout1, port='center', name='cl1', grounding_width=grounding_width, grounding_between=[(2, 3)])
    fanout2 = sample.fanout(o=main_coupler, port='port2', name='open end resonator fanout', grouping=[1, 3])
    g2 = sample.ground(o=fanout2, port='center', name='cl2', grounding_width=grounding_width, grounding_between=[(0, 1)])
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
                                    points=[], airbridge=airbridge, min_spacing=min_bridge_spacing)

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
    min_freq=min_freq*1e9
    max_freq=max_freq*1e9
    min_Q=1e3
    max_Q=1e9
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
