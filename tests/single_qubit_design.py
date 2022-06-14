from designs.single_qubit_design import single_qubit_design as sqd
import numpy as np

import QCreator.auxiliary_functions as auxfuncs
import QCreator.elements as elements

import unittest


class SingleQubitDesign(unittest.TestCase):
    def test_sqd(self):
        # draw qubits
        sqd.sample.add(sqd.xmon1)
        sqd.sample.add(sqd.coaxmon1)
        sqd.sample.add(sqd.coaxmon2)
        sqd.sample.add(sqd.coaxmon3)
        sqd.sample.add(sqd.xmon2)
        sqd.sample.add(sqd.coaxmon4)

        sqd.sample.draw_design()

        airbridge = elements.AirBridgeGeometry(pad_width=36, pad_length=22, pad_distance=62,
                                               narrow_width=20, narrow_length=46, sm_pad_length=10,
                                               sm_pad_distance=70, layer_configuration=sqd.sample.layer_configuration)

        # draw resonators
        g1, g2, par1, par2, res12 = auxfuncs.draw_double_resonator_plus_double_qubit(
            sqd.sample, sqd.coupler_start, sqd.central_line_y, sqd.coupler_length, sqd.resonator_core,
            sqd.resonator_gap, sqd.resonator_ground, sqd.tl_core, sqd.tl_gap, sqd.tl_ground, grounding_width=10,
            closed_end_meander_length1=3400, length_left1=250, length_right1=360, closed_end_meander_length2=2850,
            length_left2=360, length_right2=260, open_end_shift_length1=600, open_end_shift_length2=400,
            object1=sqd.xmon1, port1='crab_up', object2=sqd.coaxmon1, port2='coupler2', airbridge=airbridge,
            min_bridge_spacing_closed_end=150, min_bridge_spacing_open_end=150, port_orientation='left',
            meander_first_intend_orientation='right', object2_airbridges=True, meander_r=55)
        res1 = res12['resonator1']
        res2 = res12['resonator2']

        g3, g4, par3, par4, res34 = auxfuncs.draw_double_resonator_plus_double_qubit(
            sqd.sample, sqd.coupler_start + 900, sqd.central_line_y, sqd.coupler_length, sqd.resonator_core,
            sqd.resonator_gap, sqd.resonator_ground, sqd.tl_core, sqd.tl_gap, sqd.tl_ground, grounding_width=10,
            closed_end_meander_length1=2950, length_left1=260, length_right1=360,
            closed_end_meander_length2=2700, length_left2=360, length_right2=240,
            open_end_shift_length1=400, open_end_shift_length2=400,
            object1=sqd.coaxmon2, port1='coupler2',
            object2=sqd.coaxmon3, port2='coupler2',
            airbridge=airbridge,
            min_bridge_spacing_closed_end=150, min_bridge_spacing_open_end=150,
            port_orientation='left', object1_airbridges=True, object2_airbridges=True, meander_r=55)
        res3 = res34['resonator1']
        res4 = res34['resonator2']

        g5,g6, par5, par6, res56 = auxfuncs.draw_double_resonator_plus_double_qubit(
            sqd.sample, sqd.coupler_start + 1750, sqd.central_line_y, sqd.coupler_length,
            sqd.resonator_core, sqd.resonator_gap, sqd.resonator_ground,
            sqd.tl_core, sqd.tl_gap, sqd.tl_ground, grounding_width=10,
            closed_end_meander_length1=2400, length_left1=250, length_right1=360,
            closed_end_meander_length2=1900, length_left2=360, length_right2=250,
            open_end_shift_length1=400, open_end_shift_length2=400,
            object1=sqd.xmon2, port1='crab_up',
            object2=sqd.coaxmon4, port2='coupler2',
            airbridge=airbridge,
            min_bridge_spacing_closed_end=150, min_bridge_spacing_open_end=150,
            port_orientation='left', object2_airbridges=True, meander_r=55)
        res5 = res56['resonator1']
        res6 = res56['resonator2']
        g7, g8, _1, _2, res78 = auxfuncs.draw_double_resonator(
            sqd.sample, sqd.coupler_start + 2450, sqd.central_line_y, 180,
            sqd.resonator_core, sqd.resonator_gap, 15,
            sqd.tl_core, sqd.tl_gap, sqd.tl_ground, grounding_width=20,
            closed_end_meander_length1=2200, length_left1=130, length_right1=140,
            closed_end_meander_length2=2300, length_left2=140, length_right2=130,
            open_end_length1=1000, open_end_length2=1000,
            airbridge=airbridge,
            port_orientation='left',
            min_bridge_spacing_closed_end=100, min_bridge_spacing_open_end=150, meander_r=55)
        res7 = res78['resonator1']
        res8 = res78['resonator2']

        # Connect resonators into feedline
        sqd.sample.connect_cpw(o1=sqd.p1, o2=g1, port1='port', port2='narrow', name='right TL', points=[],
                               airbridge=airbridge, min_spacing=150)
        sqd.sample.connect_cpw(o1=g2, o2=g3, port1='narrow', port2='narrow', name='right TL', points=[],
                               airbridge=airbridge, min_spacing=150)
        sqd.sample.connect_cpw(o1=g4, o2=g5, port1='narrow', port2='narrow', name='right TL', points=[],
                               airbridge=airbridge, min_spacing=150)
        sqd.sample.connect_cpw(o1=g6, o2=g7, port1='narrow', port2='narrow', name='right TL', points=[],
                               airbridge=airbridge, min_spacing=150)
        sqd.sample.connect_cpw(o1=g8, o2=sqd.p2, port1='narrow', port2='port', name='right TL', points=[],
                               airbridge=airbridge, min_spacing=150)

        # Calculate capacitances by FastCap
        sqd.sample.draw_cap()
        for i, qubit in enumerate(sqd.sample.qubits):
            caps = sqd.sample.calculate_qubit_capacitance(cell=sqd.sample.qubit_cap_cells[i],
                                                          qubit=sqd.sample.qubits[i],
                                                          mesh_volume=10)

        # Connect contact pads
        flux_pads_qubits = [
            (sqd.sample.qubits[1], sqd.pads_top[0], []),
            (sqd.sample.qubits[2], sqd.pads_bottom[1], []),
            (sqd.sample.qubits[3], sqd.pads_top[1], []),
            (sqd.sample.qubits[5], sqd.pads_top[2], []),
            (sqd.sample.qubits[0], sqd.pads_bottom[0], [(1531, 929)]),
            (sqd.sample.qubits[4], sqd.pads_bottom[2], [])]

        for qubit, pad, points in flux_pads_qubits:
            bridge, port = sqd.sample.airbridge(
                qubit, 'flux', name='Airbridge over %s qubit flux coupler' % qubit.name, geometry=airbridge)

            narrowing_length = 25
            flux_line_narrowing_position = sqd.sample.cpw_shift(bridge, port, narrowing_length / 2)[0]

            flux_line_narrowing = sqd.elements.Narrowing(name='flux_line_narrowing',
                                                         position=flux_line_narrowing_position,
                                                         orientation=bridge.terminals[port].orientation + np.pi,
                                                         w1=bridge.terminals[port].w,
                                                         s1=bridge.terminals[port].s, g1=bridge.terminals[port].g,
                                                         w2=sqd.tl_core, s2=sqd.tl_gap, g2=sqd.tl_ground,
                                                         layer_configuration=sqd.sample.layer_configuration,
                                                         length=narrowing_length)
            sqd.sample.add(flux_line_narrowing)
            sqd.sample.connect(flux_line_narrowing, 'port1', bridge, port)
            sqd.sample.connect_cpw(pad, flux_line_narrowing, 'port', 'port2', 'flux_control_line_qubit_' + qubit.name,
                                   points=points,
                                   airbridge=airbridge, min_spacing=100, r=80)

        #Add ground grid
        grid_ground = elements.GridGround(sqd.sample.chip_geometry, sqd.sample.layer_configuration, sqd.sample.objects,
                                          width=4, period=25)
        sqd.sample.add(grid_ground)

        #Render design
        sqd.sample.draw_design()

        #Render negative layer
        sqd.sample.render_negative([0], sqd.sample.layer_configuration.inverted)

        # Render layer expansion
        sqd.sample.layer_expansion(0.8, sqd.sample.layer_configuration.inverted, 101, 8)

        #Write GDS file
        # sqd.sample.write_to_gds()

        # Create transmission-line-simulator and calculate eigenfrequencies
        sys, connections, elements_ = sqd.sample.get_tls(cutoff=1e11)
        # omegas, delta, modes = sys.get_modes()

if __name__ == '__main__':
    unittest.main()
