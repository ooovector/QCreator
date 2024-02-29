import numpy as np
from QCreator.elements.core import DesignElement
from QCreator.elements.squid_in_line import SquidInLine
from QCreator.elements.JJ_in_line import JJInLine
from QCreator.elements.airbridge import AirbridgeOverCPW
from QCreator.elements.cpw import Narrowing


class TunableCouplerThreeMode:
    """
    Tunable coupler object consists of:
    :param qubit1: the first qubit is a DesignElement object in the tunable coupling cell
    :param qubit2: the second qubit is a DesignElement object
    :param port1: a name of port for a qubit1
    :param port2: a name of port for a qubit2
    :param sample: a sample object
    """
    def __init__(self, qubit1: DesignElement, qubit2: DesignElement, port1: str, port2: str, sample,
                 coupler_params, jj_inline_params, squid_inline_params):
        self.qubit1 = qubit1
        self.qubit2 = qubit2
        self.port1 = port1
        self.port2 = port2
        self.sample = sample
        self.name = 'tunable coupler between {} and {}'.format(self.qubit1.name, self.qubit2.name)

        self.airbridge_geometry = coupler_params['airbridge_geometry']
        self.coupler_w = coupler_params['coupler_w']
        self.coupler_s = coupler_params['coupler_s']
        self.coupler_g = coupler_params['coupler_g']

        self.jj_inline_params = jj_inline_params
        self.squid_inline_params = squid_inline_params

        self.connection1 = qubit1.get_terminals()[port1].position
        self.connection2 = qubit2.get_terminals()[port2].position
        self.coupler_orientation = np.arctan2(self.connection1[0]-self.connection2[0],
                                              self.connection1[1]-self.connection2[1])

    def generate(self):
        """
        Render tunable coupler system
        """
        # Add JJ inlines
        jj_inline1 = JJInLine('JJ inline 1 for ' + self.name,
                              self.qubit1.get_terminals()[self.port1],
                              self.coupler_w, self.coupler_g, self.coupler_s, self.jj_inline_params['length'],
                              self.jj_inline_params['jj_params'],
                              self.sample.layer_configuration,
                              import_jj=True, file_jj=self.jj_inline_params['file_name_jj_inline'],
                              cell_jj=self.jj_inline_params['cell_name_jj_inline'], bandages=True)

        jj_inline2 = JJInLine('JJ inline 2 for ' + self.name,
                              self.qubit2.get_terminals()[self.port2],
                              self.coupler_w, self.coupler_g, self.coupler_s, self.jj_inline_params['length'],
                              self.jj_inline_params['jj_params'],
                              self.sample.layer_configuration,
                              import_jj=True, file_jj=self.jj_inline_params['file_name_jj_inline'],
                              cell_jj=self.jj_inline_params['cell_name_jj_inline'], bandages=True)

        self.sample.add(jj_inline1)
        self.sample.add(jj_inline2)

        # Add SQUID inline
        qubit1_position = self.qubit1.get_center()
        qubit2_position = self.qubit2.get_center()
        squid_inline_position = ((qubit1_position[0] + qubit2_position[0]) / 2,
                                 (qubit1_position[1] + qubit2_position[1]) / 2)

        squid_inline = SquidInLine(name='SQUID inline for ' + self.name, center=squid_inline_position,
                                   core=self.coupler_w, gap=self.coupler_s, ground=self.coupler_g,
                                   layer_configuration=self.sample.layer_configuration,
                                   squid_params=self.squid_inline_params['squid_params'],
                                   fluxline=self.squid_inline_params['fluxline_params'], import_jj=True,
                                   file_jj=self.squid_inline_params['file_name_squid_inline'], bandages=True,
                                   cell_jj=self.squid_inline_params['cell_name_squid_inline'])
        self.sample.add(squid_inline)
        self.sample.draw_design()

        ########## Add airbridges ##########
        lamda = 1 / 2
        bridge1_position = ((qubit1_position[0] + lamda * qubit2_position[0]) / (lamda + 1),
                            (qubit1_position[1] + lamda * qubit2_position[1]) / (lamda + 1))
        bridge1 = AirbridgeOverCPW(name='bridge1', position=bridge1_position,
                                   orientation=self.coupler_orientation + np.pi / 2,
                                   w=7, s=20, g=self.coupler_g,
                                   geometry=self.airbridge_geometry,
                                   with_ground=True)
        self.sample.add(bridge1)

        lamda = 2
        bridge2_position = ((qubit1_position[0] + lamda * qubit2_position[0]) / (lamda + 1),
                            (qubit1_position[1] + lamda * qubit2_position[1]) / (lamda + 1))
        bridge2 = AirbridgeOverCPW(name='bridge1', position=bridge2_position,
                                   orientation=self.coupler_orientation + np.pi / 2,
                                   w=7, s=20, g=self.coupler_g,
                                   geometry=self.airbridge_geometry,
                                   with_ground=True)

        self.sample.add(bridge2)

        ########## Add narrowings ##########
        narrowing_length = 40
        narrowings = []
        for bridge_id, bridge in enumerate([bridge1, bridge2]):
            for port in ['port1', 'port2']:
                narrowing_bridge_position = self.sample.cpw_shift(bridge, port, narrowing_length/2)[0]
                narrowing = Narrowing(name='narroing', position=narrowing_bridge_position,
                                                     orientation=bridge.terminals[port].orientation+np.pi,
                                                     w1=bridge.terminals[port].w,
                                                     s1=bridge.terminals[port].s, g1=bridge1.terminals[port].g,
                                                     w2=self.coupler_w, s2=self.coupler_s, g2=self.coupler_g,
                                                     layer_configuration=self.sample.layer_configuration, length=narrowing_length)
                self.sample.add(narrowing)
                narrowings.append(narrowing)

        ########## Connect all elements ##########
        self.sample.connect_cpw(jj_inline1, narrowings[0], 'port2', 'port2', 'jj-squid cpw', [], airbridge=None)
        self.sample.connect_cpw(narrowings[1], squid_inline, 'port2', 'port1', 'jj-squid cpw', [], airbridge=None)
        self.sample.connect_cpw(squid_inline, narrowings[2], 'port2', 'port2', 'jj-squid cpw', [], airbridge=None)
        self.sample.connect_cpw(narrowings[3], jj_inline2, 'port2', 'port2', 'jj-squid cpw', [], airbridge=None)
        self.sample.draw_design()




class TunableCouplerThreeMode2:
    """
    Tunable coupler object consists of:
    :param qubit1: the first qubit is a DesignElement object in the tunable coupling cell
    :param qubit2: the second qubit is a DesignElement object
    :param port1: a name of port for a qubit1
    :param port2: a name of port for a qubit2
    :param sample: a sample object
    """
    def __init__(self, qubit1: DesignElement, qubit2: DesignElement, port1: str, port2: str, sample,
                 coupler_params, jj_inline_params, squid_inline_params):
        self.qubit1 = qubit1
        self.qubit2 = qubit2
        self.port1 = port1
        self.port2 = port2
        self.sample = sample
        self.name = 'tunable coupler between {} and {}'.format(self.qubit1.name, self.qubit2.name)

        self.airbridge_geometry = coupler_params['airbridge_geometry']
        self.coupler_w = coupler_params['coupler_w']
        self.coupler_s = coupler_params['coupler_s']
        self.coupler_g = coupler_params['coupler_g']

        self.jj_inline_params = jj_inline_params
        self.squid_inline_params = squid_inline_params

        self.connection1 = qubit1.get_terminals()[port1].position
        self.connection2 = qubit2.get_terminals()[port2].position
        self.coupler_orientation = np.arctan2(self.connection1[0]-self.connection2[0],
                                              self.connection1[1]-self.connection2[1])

    def generate(self):
        """
        Render tunable coupler system
        """
        # Add JJ inlines
        jj_inline1 = JJInLine('JJ inline 1 for ' + self.name,
                              self.qubit1.get_terminals()[self.port1],
                              self.coupler_w, self.coupler_g, self.coupler_s, self.jj_inline_params['length'],
                              self.jj_inline_params['jj_params'],
                              self.sample.layer_configuration,
                              import_jj=True, file_jj=self.jj_inline_params['file_name_jj_inline'],
                              cell_jj=self.jj_inline_params['cell_name_jj_inline'], bandages=True)

        jj_inline2 = JJInLine('JJ inline 2 for ' + self.name,
                              self.qubit2.get_terminals()[self.port2],
                              self.coupler_w, self.coupler_g, self.coupler_s, self.jj_inline_params['length'],
                              self.jj_inline_params['jj_params'],
                              self.sample.layer_configuration,
                              import_jj=True, file_jj=self.jj_inline_params['file_name_jj_inline'],
                              cell_jj=self.jj_inline_params['cell_name_jj_inline'], bandages=True)

        self.sample.add(jj_inline1)
        self.sample.add(jj_inline2)

        # Add SQUID inline
        qubit1_position = self.qubit1.get_center()
        qubit2_position = self.qubit2.get_center()
        squid_inline_position = ((qubit1_position[0] + qubit2_position[0]) / 2,
                                 (qubit1_position[1] + qubit2_position[1]) / 2)

        squid_inline = SquidInLine(name='SQUID inline for ' + self.name, center=squid_inline_position,
                                   core=self.coupler_w, gap=self.coupler_s, ground=self.coupler_g,
                                   layer_configuration=self.sample.layer_configuration,
                                   squid_params=self.squid_inline_params['squid_params'],
                                   fluxline=self.squid_inline_params['fluxline_params'], import_jj=True,
                                   file_jj=self.squid_inline_params['file_name_squid_inline'], bandages=True,
                                   cell_jj=self.squid_inline_params['cell_name_squid_inline'])
        self.sample.add(squid_inline)
        self.sample.draw_design()

        # Connect all elements
        self.sample.connect_cpw(jj_inline1, squid_inline, 'port2', 'port1', 'jj squid cpw for ' + self.name, [], airbridge=None)
        self.sample.connect_cpw(squid_inline, jj_inline2, 'port2', 'port2', 'jj squid cpw for ' + self.name, [], airbridge=None)

        # Add airbridges
        lamda = 1 / 2
        bridge1_position = ((qubit1_position[0] + lamda * qubit2_position[0]) / (lamda + 1),
                            (qubit1_position[1] + lamda * qubit2_position[1]) / (lamda + 1))
        lamda = 2
        bridge2_position = ((qubit1_position[0] + lamda * qubit2_position[0]) / (lamda + 1),
                            (qubit1_position[1] + lamda * qubit2_position[1]) / (lamda + 1))
        bridge1 = AirbridgeOverCPW(name='airbridge1 for ' + self.name, position=bridge1_position,
                                   orientation=self.coupler_orientation + np.pi / 2,
                                   w=self.coupler_w, s=self.coupler_s, g=self.coupler_g,
                                   geometry=self.airbridge_geometry,
                                   with_ground=True)

        bridge2 = AirbridgeOverCPW(name='airbridge2 for ' + self.name, position=bridge2_position,
                                   orientation=self.coupler_orientation + np.pi / 2,
                                   w=self.coupler_w, s=self.coupler_s, g=self.coupler_g,
                                   geometry=self.airbridge_geometry,
                                   with_ground=True)

        self.sample.add(bridge1)
        self.sample.add(bridge2)

        self.sample.draw_design()





