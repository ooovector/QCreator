from .core import DesignElement, LayerConfiguration, DesignTerminal
from .. import conformal_mapping as cm
from .. import transmission_line_simulator as tlsim
import gdspy
import numpy as np
from typing import List, Tuple, Mapping, Dict
from . import squid3JJ #TODO make this qubit class suitable for any squid types
from . import JJ4q
from copy import deepcopy
from . import pp_squid
from . import pp_squid_coupler
from . import pp_transmon





class TWOQTC(DesignElement):
    """
    two Qubits coupled to a common tunable coupler
    """
    def __init__(self, name: str, center: Tuple[float, float],layers_configuration:dict,transformations:dict,Q1:dict,Q2:dict,TC:dict,d1:float,d2:float):
        super().__init__(type='qubit', name=name)
        #qubit parameters
        self.Q1 = Q1
        self.Q2 = Q2
        self.TC = TC
        self.d1,self.d2 = d1,d2
        self.transformations = transformations# to mirror the structure
        self.center = center
        #layers
        self.layers_configuration = layers_configuration

        self.couplers = {}
        self.tls_cache = []
        self.L = 15e-9  # 20nHr


        # Filling the dictionary with dummy values for the calculation of the Capacity matrix
        self.C = {'qubit': None}
        for i in range(0,25):
            self.C['coupler'+str(i)] = None


    def render(self):
        #Qubit 1
        Q1 = self.Q1
        Qubit1 = pp_transmon.PP_Transmon(name=Q1['name'],center=self.center,
                          width = Q1['width'],
                          height = Q1['height'],
                          bridge_gap = Q1['b_g'],
                          bridge_w   = Q1['b_w'] ,
                          gap = Q1['gap'],
                          ground_w = Q1['g_w'],
                          ground_h = Q1['g_h'],
                          ground_t = Q1['g_t'],
                          jj_params= Q1['jj_pp'],
                          layer_configuration = self.layers_configuration,
                          Couplers = Q1['Couplers'],
                          transformations = Q1['transformations'],
                          remove_ground = {'right':Q1['empty_ground']},
                                         calculate_capacitance=True
                                         )
        #Qubit 2
        Q2 = self.Q2
        Qubit2 = pp_transmon.PP_Transmon(name=Q2['name'],center=(self.center[0]+self.d1+self.d2+Q1['g_w']/2+Q2['g_w']/2+self.TC['g_w'],self.center[1]),
                          width = Q2['width'],
                          height = Q2['height'],
                          bridge_gap = Q2['b_g'],
                          bridge_w   = Q2['b_w'] ,
                          gap = Q2['gap'],
                          ground_w = Q2['g_w'],
                          ground_h = Q2['g_h'],
                          ground_t = Q2['g_t'],
                          jj_params= Q2['jj_pp'],
                          layer_configuration = self.layers_configuration,
                          Couplers = Q2['Couplers'],
                          transformations = Q2['transformations'],
                          remove_ground = {'left':Q2['empty_ground']},
                          secret_shift = 3,
                                         calculate_capacitance=True
                                         )



        #TC
        TC = self.TC
        TunC = pp_squid_coupler.PP_Squid_C(name=TC['name'],center=(self.center[0]+self.d1+Q1['g_w']/2+TC['g_w']/2,self.center[1]),
                          width = TC['width'],
                          height = TC['height'],
                          bridge_gap = TC['b_g'],
                          bridge_w   = TC['b_w'] ,
                          gap = TC['gap'],
                          g_w = TC['g_w'],
                          g_h = TC['g_h'],
                          g_t = TC['g_t'],
                          jj_params= TC['jj_pp'],
                          layer_configuration = self.layers_configuration,
                          Couplers = TC['Couplers'],
                          transformations = TC['transformations'],
                          fluxline_params=TC['fluxline'],
                          remove_ground = {'left':1,'right':1},
                          secret_shift=3+3,
                          calculate_capacitance=True,
                          arms = TC['arms']
                                 )






        Q1Render = Qubit1.render()
        Q2Render = Qubit2.render()
        TCRender = TunC.render()

        RESULT   = Q1Render#{}

        for key in Q1Render:
            if key == 'qubit_cap':
                for i in range(len(Q2Render[key])):
                    RESULT[key].append(Q2Render[key][i])

                for i in range(len(TCRender[key])):
                    RESULT[key].append(TCRender[key][i])

                continue

            for i in range(len(Q2Render[key].polygons)):
                RESULT[key].polygons.append(Q2Render[key].polygons[i])
                RESULT[key].layers.append(Q2Render[key].layers[i])
                RESULT[key].datatypes.append(Q2Render[key].datatypes[i])

            for i in range(len(TCRender[key].polygons)):
                RESULT[key].polygons.append(TCRender[key].polygons[i])
                RESULT[key].layers.append(TCRender[key].layers[i])
                RESULT[key].datatypes.append(TCRender[key].datatypes[i])
            #RESULT[key] = gdspy.boolean(Q1Render[key],Q2Render[key],'or')

        return RESULT




    def add_to_tls(self, tls_instance: tlsim.TLSystem, terminal_mapping: dict,
                   track_changes: bool = True, cutoff: float = np.inf) -> list:
        #scaling factor for C
        scal_C = 1e-15
        JJ = tlsim.Inductor(self.L)
        C = tlsim.Capacitor(c=self.C['qubit']*scal_C, name=self.name+' qubit-ground')
        tls_instance.add_element(JJ, [0, terminal_mapping['qubit']])
        tls_instance.add_element(C, [0, terminal_mapping['qubit']])
        mut_cap = []
        cap_g = []
        for id, coupler in enumerate(self.couplers):
            if coupler.coupler_type == 'coupler':
                c0 = tlsim.Capacitor(c=self.C['coupler'+str(id)][1]*scal_C, name=self.name+' qubit-coupler'+str(id))
                c0g = tlsim.Capacitor(c=self.C['coupler'+str(id)][0]*scal_C, name=self.name+' coupler'+str(id)+'-ground')
                tls_instance.add_element(c0, [terminal_mapping['qubit'], terminal_mapping['coupler'+str(id)]])
                tls_instance.add_element(c0g, [terminal_mapping['coupler'+str(id)], 0])
                mut_cap.append(c0)
                cap_g.append(c0g)
            # elif coupler.coupler_type =='grounded':
            #     tls_instance.add_element(tlsim.Short(), [terminal_mapping['flux line'], 0])

        if track_changes:
            self.tls_cache.append([JJ, C]+mut_cap+cap_g)
        return [JJ, C]+mut_cap+cap_g

