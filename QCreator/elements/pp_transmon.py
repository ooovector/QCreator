from .core import DesignElement, LayerConfiguration, DesignTerminal
from .. import conformal_mapping as cm
from .. import transmission_line_simulator as tlsim
import gdspy
import numpy as np
from typing import List, Tuple, Mapping, Dict
from . import squid3JJ #TODO make this qubit class suitable for any squid types
from . import JJ4q
from copy import deepcopy
import sys

class PP_Transmon(DesignElement):
    """
    PP-Transmon consists of several parts:
    1) Central part - central circuit
    params: center = center of the qubit, w,h,gap = width,height,gap of the Parallel Plate Transmon in the ground cavity
    2) Couplers - claw like couplers for the left and right, rectangular pad couplers on top and bottom
    3) Ground = Ground rectangle around qubit, ground_w,ground_h,ground_t = width,height and thickness of ground frame
    4)layer_configuration
    5)Couplers - coupler classes
    6) jj_params - parameters of the SQUID which here is 3JJ SQUID.#TODO add more information
    """
    def __init__(self, name: str, center: Tuple[float, float],
                 width: float, height: float,gap: float,
                 bridge_gap:float,bridge_w:float,
                 ground_w: float, ground_h: float,ground_t: float,
                 layer_configuration: LayerConfiguration,
                 jj_params: Dict,Couplers,
                 transformations: Dict,
                 calculate_capacitance: False,
                 remove_ground = {},
                 shoes = {},
                 holes = False,
                 fluxline_params = [],
                 secret_shift = 0):
        super().__init__(type='qubit', name=name)
        #qubit parameters
        self.transformations = transformations# to mirror the structure
        self.center = center
        self.w = width # defines the width of the qubit's plate
        self.h = height # defines the height of the qubit's plate
        self.gap = gap # distance between qubit's plates
        #the ground rectangular with the hole inside and with the thickness==ground_t
        self.g_w = ground_w
        self.g_h = ground_h
        self.g_t = ground_t

        # the position of the JJ for manhatan style
        self.b_g = bridge_gap
        self.b_w = bridge_w
        #layers
        self.layer_configuration = layer_configuration

        #couplers
        self.couplers = Couplers

        #shoes
        self.shoes = shoes

        # JJs and fluxline
        self.JJ_params = jj_params
        self.JJ = None
        self.layers = []
        self.fluxline_params = fluxline_params

        #terminals
        self.terminals = {  # 'coupler0': None,
            # 'coupler1': None,
            # 'coupler2': None,
            # 'coupler3': None,
            # 'coupler4': None,
            # 'flux line': None,
            'qubit': None}

        self.calculate_capacitance = calculate_capacitance
        self.tls_cache = []
        self.L = 15e-9  # 20nHr
        self.C = {   'coupler0': None,
             'coupler1': None,
             'coupler2': None,
             'coupler3': None,
             'coupler4': None,
             'qubit': None}

        #remove ground on these sites
        self.remove_ground = remove_ground
        #To introduce rectangular holes in the ground around the qubit
        self.holes = holes

        #for calculating cacities
        self.secret_shift = secret_shift

    def render(self):
        """
        This function draws everything: qubit,ground,couplers
        """
        qubit_cap_parts=[]
        ground = self.generate_ground()
        # restricted area for a future grid lines
        result_restricted = gdspy.Rectangle((self.center[0]-self.g_w/2,self.center[1]-self.g_h/2),(self.center[0]+self.g_w/2,self.center[1]+self.g_h/2),layer=self.layer_configuration.restricted_area_layer)

        P1 = gdspy.Rectangle((self.center[0]-self.gap/2-self.w,self.center[1]-self.h/2),(self.center[0]-self.gap/2,self.center[1]+self.h/2))
        P2 = gdspy.copy(P1,self.w+self.gap,0)

        # adding the shoe caps here
        if self.shoes != {}:
            Rot = 0
            if 'R' in self.shoes:
                Rot = self.shoes['R']
            for key in self.shoes:
                if 'R' not in self.shoes:
                    Rot = 0
                if key == 1:
                    Shoe = gdspy.Rectangle(
                        (self.center[0] - self.gap / 2 - self.w+30, self.center[1] + self.h / 2), (
                        self.center[0] - self.gap / 2 - self.w - self.shoes[key][0],
                        self.center[1] + self.h / 2 - self.shoes[key][1]))
                    if 'R1' in self.shoes:
                        Rot = self.shoes['R1']
                    if 'R' in self.shoes:
                        Shoe.translate(0,self.shoes[key][1]/2)

                    Shoe.rotate(-Rot,(self.center[0] - self.gap / 2 - self.w, self.center[1] + self.h / 2))

                    P1 = gdspy.boolean(P1, Shoe, 'or')
                if key == 2:
                    Shoe = gdspy.Rectangle(
                        (self.center[0] - self.gap / 2 - self.w+30, self.center[1] - self.h / 2), (
                        self.center[0] - self.gap / 2 - self.w - self.shoes[key][0],
                        self.center[1] - self.h / 2 + self.shoes[key][1]))
                    if 'R2' in self.shoes:
                        Rot = self.shoes['R2']
                    if 'R' in self.shoes:
                        Shoe.translate(0,-self.shoes[key][1]/2)

                    Shoe.rotate(Rot, (self.center[0] - self.gap / 2 - self.w, self.center[1] - self.h / 2))
                    P1 = gdspy.boolean(P1, Shoe, 'or')
                if key == 3:
                    Shoe = gdspy.Rectangle(
                        (self.center[0] + self.gap / 2 + self.w-30, self.center[1] + self.h / 2), (
                        self.center[0] + self.gap / 2 + self.w + self.shoes[key][0],
                        self.center[1] + self.h / 2 - self.shoes[key][1]))
                    if 'R3' in self.shoes :
                        Rot = self.shoes['R3']
                    if 'R' in self.shoes:
                        Shoe.translate(0,self.shoes[key][1]/2)

                    Shoe.rotate(+Rot, (self.center[0] + self.gap / 2 + self.w, self.center[1] + self.h / 2))
                    P2 = gdspy.boolean(P2, Shoe, 'or')
                if key == 4:
                    Shoe =  gdspy.Rectangle(
                        (self.center[0] + self.gap / 2 + self.w-30, self.center[1] - self.h / 2), (
                        self.center[0] + self.gap / 2 + self.w + self.shoes[key][0],
                        self.center[1] - self.h / 2 + self.shoes[key][1]))
                    if 'R4' in self.shoes :
                        Rot = self.shoes['R4']
                    if 'R' in self.shoes:
                        Shoe.translate(0,-self.shoes[key][1]/2)

                    Shoe.rotate(-Rot, (self.center[0] + self.gap / 2 + self.w, self.center[1] - self.h / 2))
                    P2 = gdspy.boolean(P2, Shoe, 'or')

        self.layers.append(9)
        result = gdspy.boolean(ground, P1, 'or', layer=self.layer_configuration.total_layer)
        result = gdspy.boolean(result, P2, 'or', layer=self.layer_configuration.total_layer)

        #change here if case to allow Manhatten-style junctions
        if self.JJ_params['manhatten'] and self.JJ_params['squid'] == False:
            P1_bridge = gdspy.Rectangle((self.center[0]-self.gap/2,self.center[1]),(self.center[0]-self.b_g/2,self.center[1]+self.b_w))
            P2_bridge = gdspy.Rectangle((self.center[0] + self.gap / 2, self.center[1] ),(self.center[0] + self.b_g / 2, self.center[1] - self.b_w))
            hole1     = gdspy.Rectangle((self.center[0]-self.b_g/2-self.JJ_params['h_w']-self.b_w/2,self.center[1]),(self.center[0]-self.b_g/2-self.b_w/2,self.center[1]+self.JJ_params['h_d']))
            hole2     = gdspy.Rectangle((self.center[0] + self.b_g / 2, self.center[1]-self.JJ_params['h_w']-self.b_w/2),(self.center[0] + self.b_g / 2 + self.JJ_params['h_d'], self.center[1] - self.b_w/2))
            P1_bridge = gdspy.boolean(P1_bridge, hole1, 'not', layer=8)
            P2_bridge = gdspy.boolean(P2_bridge, hole2, 'not', layer=8)

        if self.JJ_params['manhatten'] and self.JJ_params['squid'] == True:
            P1_bridge = gdspy.Rectangle((self.center[0] - self.gap / 2, self.center[1] + self.h / 2),
                                        (self.center[0] - self.b_g / 2, self.center[1] + self.h / 2 - self.b_w))
            P2_bridge = gdspy.Rectangle((self.center[0] + self.gap / 2, self.center[1] + self.h / 2 - 2 * self.b_w),
                                        (self.center[0] + self.b_g / 2, self.center[1] + self.h / 2 - 3 * self.b_w))



            f = self.fluxline_params
            l, t_m, t_r, gap, l_arm, h_arm, s_gap = f['l'],f['t_m'],f['t_r'],f['gap'],f['l_arm'],f['h_arm'],f['s_gap']
            flux_distance = f['flux_distance']
            #result_restricted, to ct off hanging parts from the fluxline, None for no cutoff
            flux = PP_Squid_Fluxline(l, t_m, t_r, gap, l_arm, h_arm, s_gap,flux_distance,self.w,self.h,self.gap,self.b_w,self.b_g,ground = None,asymmetry = 0,g = f.get('g'),w = f.get('w'),s = f.get('s'),extend = f.get('extend_to_ground'))

            fluxline = flux.render(self.center, self.w, self.h,self.g_h,self.g_t)['positive']

            r_flux = flux.render(self.center, self.w, self.h,self.g_h,self.g_t)['restricted']
            #adding fluxline restricted area
            result_restricted = gdspy.boolean(result_restricted,r_flux,'or', layer=self.layer_configuration.restricted_area_layer)

            self.couplers.append(flux)
            result = gdspy.boolean(result, fluxline, 'or', layer=self.layer_configuration.total_layer)

            # removing ground where the fluxline is
            ground_fluxline = True
            if ground_fluxline == False:
                result = gdspy.boolean(result, gdspy.Rectangle(
                    (self.center[0] - l_arm / 2 - t_r - self.g_t, self.center[1] + self.h / 2 + 0.01),
                    (self.center[0] + 3 * l_arm / 2 + t_r + t_m + self.g_t, self.center[1] + self.h / 2 + 250)), 'not',
                                       layer=self.layer_configuration.total_layer)
            else:

                result = gdspy.boolean(result, gdspy.Rectangle(
                    (self.center[0]+l_arm/2-s_gap, self.center[1] + self.h / 2 + 0.01),
                    (self.center[0]+l_arm/2+t_m+s_gap , self.center[1] + self.h / 2 + 250)), 'not',
                                       layer=self.layer_configuration.total_layer)


            result = gdspy.boolean(result, fluxline, 'or', layer=self.layer_configuration.total_layer)

        qubit_cap_parts.append(gdspy.boolean(P1, P1_bridge, 'or', layer=8 + self.secret_shift))
        qubit_cap_parts.append(gdspy.boolean(P2, P2_bridge, 'or', layer=9 + self.secret_shift))

        result = gdspy.boolean(result, [P1_bridge,P2_bridge], 'or', layer=self.layer_configuration.total_layer)
        self.layers.append(self.layer_configuration.total_layer)

        # add couplers
        last_step_cap = [gdspy.boolean(gdspy.boolean(P2, P2_bridge, 'or'),gdspy.boolean(P1, P1_bridge, 'or'),'or')]
        self.layers.append(self.layer_configuration.total_layer)

        # Box for inverted Polygons
        box = gdspy.Rectangle((self.center[0] - self.g_w / 2, self.center[1] - self.g_h / 2),(self.center[0] + self.g_w / 2, self.center[1] + self.g_h / 2))
        pocket = gdspy.Rectangle((self.center[0] - self.g_w / 2 + self.g_t, self.center[1] - self.g_h / 2 + self.g_t),
                                 (self.center[0] + self.g_w / 2 - self.g_t, self.center[1] + self.g_h / 2 - self.g_t))

        if len(self.couplers) != 0:
            pockets = []
            for id, coupler in enumerate(self.couplers):
                if coupler.side == 'fluxline':
                    continue
                coupler_parts = coupler.render(self.center, self.g_w,self.g_h,self.g_t)

                #result = gdspy.boolean(coupler_parts['positive'], result, 'or',layer=self.layer_configuration.total_layer)

                #Extend ground around coupler
                l1   = coupler.l1
                l2   = coupler.l2
                t    = coupler.t
                gap  = coupler.gaps[1]
                gap2 = coupler.gaps[0]

                core = coupler.w
                side = coupler.side
                height_left = coupler.height_left
                height_right = coupler.height_right

                l1_check = True
                l2_check = True
                #to make sure ground is placed correctly, l1_check an l2_check is needed if one wants to remove the ground between coupler and qubit  
                if l1 < t:
                    l1_check = False
                    l1 = t
                if l2 < t:
                    l2_check = False
                    l2 = t

                if side =='right':
                    #upper
                    extended = gdspy.Rectangle((0, 0), (0, 0))  # empty object
                    if l1_check:
                        extended = gdspy.boolean(extended,gdspy.Rectangle((self.center[0]+self.g_w/2-l1-self.g_t+t,self.center[1]+height_right*self.g_h/2),(self.center[0]+self.g_w/2-l1+t,self.center[1]+gap+height_right*self.g_h/2+t+self.g_t+gap)),'or')

                    if coupler.tight == False:
                        extended = gdspy.boolean(extended, gdspy.Rectangle((self.center[0] + t + self.g_w / 2 - l1,
                                                                        self.center[
                                                                            1] + gap + height_right * self.g_h / 2  + gap),
                                                                       (self.center[
                                                                            0] + self.g_w / 2 + 2 * gap + t + self.g_t,
                                                                        self.center[
                                                                            1] + gap + height_right * self.g_h / 2  + self.g_t + gap)),
                                             'or')
                        extended = gdspy.boolean(extended, gdspy.Rectangle((self.center[0] + self.g_w / 2 + 2 * gap + t,
                                                                        self.center[
                                                                            1] + gap + height_right * self.g_h / 2  + gap),
                                                                       (self.center[
                                                                            0] + self.g_w / 2 + 2 * gap + t + self.g_t,
                                                                        self.center[1] + core/2 + gap2)), 'or')

                        extended = gdspy.boolean(extended, gdspy.Rectangle((self.center[0] + self.g_w / 2 ,
                                                                        self.center[1] + gap + height_right * self.g_h / 2  + gap),
                                                                       (self.center[0] + self.g_w / 2+self.g_t,
                                                                        self.center[1] + self.g_h/2)), 'or')
                    else:
                        extended = gdspy.boolean(extended,gdspy.Rectangle((self.center[0]+t+self.g_w/2-l1,self.center[1]+gap+height_right*self.g_h/2+t+gap),(self.center[0]+self.g_w/2+2*gap+t+self.g_t,self.center[1]+height_right*self.g_h/2+t+self.g_t+ core/2 + gap2)),'or')
                        extended = gdspy.boolean(extended,gdspy.Rectangle((self.center[0]+self.g_w/2+2*gap+t,self.center[1]+gap+height_right*self.g_h/2+t+gap),(self.center[0]+self.g_w/2+2*gap+t+self.g_t,self.center[1]+ core/2 + gap2)), 'or')

                    #lower
                    if l2_check:
                        #extended = gdspy.boolean(extended,gdspy.Rectangle((self.center[0]+t+self.g_w/2-l2-self.g_t,self.center[1]-height_right*self.g_h/2),(self.center[0]+t+self.g_w/2-l2,self.center[1]-gap-height_right*self.g_h/2-t-self.g_t-gap)), 'or')
                        extended = gdspy.boolean(extended, gdspy.Rectangle((self.center[0] + t + self.g_w / 2 - l2 ,self.center[1] - height_right * self.g_h / 2), (self.center[0] + t + self.g_w / 2 - l2, self.center[1] - gap - height_right * self.g_h / 2 - t - gap)),'or')

                    if coupler.tight == False:
                        extended = gdspy.boolean(extended, gdspy.Rectangle((self.center[0] + t + self.g_w / 2 - l2,
                                                                        self.center[
                                                                            1] - gap - height_right * self.g_h / 2  - gap),
                                                                       (self.center[
                                                                            0] + self.g_w / 2 + 2 * gap + t + self.g_t,
                                                                        self.center[
                                                                            1] - gap - height_right * self.g_h / 2  - self.g_t - gap)),
                                             'or')
                        extended = gdspy.boolean(extended, gdspy.Rectangle((self.center[0] + self.g_w / 2 + 2 * gap + t,
                                                                        self.center[
                                                                            1] - gap - height_right * self.g_h / 2  - gap),
                                                                       (self.center[
                                                                            0] + self.g_w / 2 + 2 * gap + t + self.g_t,
                                                                        self.center[1] - gap2 - core/2 )), 'or')

                        extended = gdspy.boolean(extended, gdspy.Rectangle((self.center[0] + self.g_w / 2 ,
                                                                        self.center[1] - gap - height_right * self.g_h / 2  - gap),
                                                                       (self.center[0] + self.g_w / 2+self.g_t,
                                                                        self.center[1] - self.g_h/2)), 'or')
                    else:
                        extended = gdspy.boolean(extended,gdspy.Rectangle((self.center[0]+t+self.g_w/2-l2,self.center[1]-gap-height_right*self.g_h/2-t-gap),(self.center[0]+self.g_w/2+2*gap+t+self.g_t,self.center[1]-height_right*self.g_h/2-t-self.g_t- core/2 - gap2)),'or')
                        extended = gdspy.boolean(extended,gdspy.Rectangle((self.center[0]+self.g_w/2+2*gap+t,self.center[1]-gap-height_right*self.g_h/2-t-gap),(self.center[0]+self.g_w/2+2*gap+t+self.g_t,self.center[1]- core/2 - gap2)), 'or')


                    # shift coupler to qubit and remove ground where necessary
                    remove = gdspy.Rectangle((self.center[0] + self.g_w / 2,
                                              self.center[1] - gap - height_right * self.g_h / 2 - t - self.g_t - gap),
                                             (self.center[0] + self.g_w / 2 + t + 2 * gap,
                                              self.center[1] + gap + height_right * self.g_h / 2 + t + self.g_t + gap))

                    result = gdspy.boolean(result, remove.translate(-coupler.sctq, 0), 'not')

                    extended.translate(-coupler.sctq,0)

                    #remove/add missing overhang:
                    #add missing piece
                    if coupler.sctq<0:
                        extended = gdspy.boolean(extended,gdspy.Rectangle((self.center[0]+self.g_w/2,self.center[1]+gap+height_right*self.g_h/2+t+self.g_t+gap),(self.center[0]+self.g_w/2-coupler.sctq,self.center[1]-gap-height_right*self.g_h/2-t-self.g_t-gap)),'or')

                    # remove additional pieces
                    if coupler.sctq > self.g_t:
                        extended = gdspy.boolean(extended, gdspy.Rectangle(
                            (self.center[0] + self.g_t - self.g_w / 2, self.center[1] - self.g_t + self.g_h / 2),
                            (self.center[0] - self.g_t + self.g_w / 2, self.center[1] + self.g_t - self.g_h / 2)),
                                                 'not')

                    result = gdspy.boolean(result,extended,'or')
                    # box for inverted polygon

                    if coupler.tight == True:
                        box = gdspy.boolean(box, gdspy.Rectangle((self.center[0] + self.g_w / 2 + self.g_t + 2 * gap + t,
                                                              self.center[
                                                                  1] - height_right * self.g_h / 2 - self.g_t - 2 * gap - t),
                                                             (self.center[0] + self.g_w / 2 - l1 + t, self.center[
                                                                 1] + height_right * self.g_h / 2 + self.g_t + 2 * gap + t)).translate(-coupler.sctq,0),
                                        'or', layer=self.layer_configuration.inverted)
                    else:
                        box = gdspy.boolean(box, gdspy.Rectangle((self.center[0] + self.g_w / 2 + self.g_t + 2 * gap + t,self.center[1] - height_right * self.g_h / 2 - self.g_t - 2 * gap ),(self.center[0] + self.g_w / 2 - l1 + t, self.center[1] + height_right * self.g_h / 2 + self.g_t + 2 * gap)).translate(-coupler.sctq, 0),'or', layer=self.layer_configuration.inverted)



                    pocket = gdspy.boolean(pocket, gdspy.Rectangle((self.center[0] + self.g_w / 2 + self.g_t + 2 * gap + t,self.center[1] - height_right * self.g_h / 2 - self.g_t - 2 * gap - t),(self.center[0] + self.g_w / 2 - l1 + t, self.center[1] + height_right * self.g_h / 2 + self.g_t + 2 * gap + t)).translate(-coupler.sctq, 0),'or')
                    if coupler.sctq > self.g_t:
                        pocket = gdspy.boolean(pocket,gdspy.Rectangle((self.center[0],self.center[1]+coupler.gap+coupler.w/2),(self.center[0]+self.g_w/2+1,self.center[1]-coupler.w/2-coupler.gap)),'or')

                if side =='left':
                    #upper
                    extended = gdspy.Rectangle((0,0),(0,0)) # empty object
                    if l1_check:
                        extended = gdspy.boolean(extended,gdspy.Rectangle((self.center[0]-self.g_w/2+l1+self.g_t-t,self.center[1]+height_left*self.g_h/2),(self.center[0]-t-self.g_w/2+l1,self.center[1]+gap+height_left*self.g_h/2+t+self.g_t+gap)),'or')

                    if coupler.tight == False:
                        extended = gdspy.boolean(extended, gdspy.Rectangle((self.center[0] - t - self.g_w / 2 + l1,
                                                                        self.center[
                                                                            1] + gap + height_left * self.g_h / 2 + gap),
                                                                       (self.center[
                                                                            0] - self.g_w / 2 - 2 * gap - t - self.g_t,
                                                                        self.center[
                                                                            1] + gap + height_left * self.g_h / 2 + self.g_t + gap)),
                                             'or')
                        extended = gdspy.boolean(extended, gdspy.Rectangle((self.center[0] - self.g_w / 2 - 2 * gap - t,
                                                                        self.center[
                                                                            1] + gap + height_left * self.g_h / 2 + gap),
                                                                       (self.center[
                                                                            0] - self.g_w / 2 - 2 * gap - t - self.g_t,
                                                                        self.center[1] + core/2 + gap2)), 'or')

                        extended = gdspy.boolean(extended, gdspy.Rectangle((self.center[0] - self.g_w / 2 ,
                                                                        self.center[1] + gap + height_left * self.g_h / 2  + gap),
                                                                       (self.center[0] - self.g_w / 2-self.g_t,
                                                                        self.center[1] + self.g_h/2)), 'or')

                    else:
                        extended = gdspy.boolean(extended,gdspy.Rectangle((self.center[0]-t-self.g_w/2+l1,self.center[1]+gap+height_left*self.g_h/2+t+gap),(self.center[0]-self.g_w/2-2*gap-t-self.g_t,self.center[1]+height_left*self.g_h/2+t+self.g_t+ core/2 + gap2)),'or')
                        extended = gdspy.boolean(extended,gdspy.Rectangle((self.center[0]-self.g_w/2-2*gap-t,self.center[1]+gap+height_left*self.g_h/2+t+gap),(self.center[0]-self.g_w/2-2*gap-t-self.g_t,self.center[1]+ core/2 + gap2)), 'or')

                    #lower
                    if l2_check:
                        extended = gdspy.boolean(extended,gdspy.Rectangle((self.center[0]-t-self.g_w/2+l2+self.g_t,self.center[1]-height_left*self.g_h/2),(self.center[0]-t-self.g_w/2+l2,self.center[1]-gap-height_left*self.g_h/2-t-self.g_t-gap)), 'or')

                    if coupler.tight == False:
                        extended = gdspy.boolean(extended, gdspy.Rectangle((self.center[0] - t - self.g_w / 2 + l2,
                                                                        self.center[
                                                                            1] - gap - height_left * self.g_h / 2- gap),
                                                                       (self.center[
                                                                            0] - self.g_w / 2 - 2 * gap - t - self.g_t,
                                                                        self.center[
                                                                            1] - gap - height_left * self.g_h / 2- self.g_t - gap)),
                                             'or')
                        extended = gdspy.boolean(extended, gdspy.Rectangle((self.center[0] - self.g_w / 2 - 2 * gap - t,
                                                                        self.center[
                                                                            1] - gap - height_left * self.g_h / 2- gap),
                                                                       (self.center[
                                                                            0] - self.g_w / 2 - 2 * gap - t - self.g_t,
                                                                        self.center[1] - core/2 - gap2)), 'or')
                        extended = gdspy.boolean(extended, gdspy.Rectangle((self.center[0] - self.g_w / 2,
                                                                            self.center[
                                                                                1] - gap - height_left * self.g_h / 2 - gap),
                                                                           (self.center[0] - self.g_w / 2 - self.g_t,
                                                                            self.center[1] - self.g_h / 2)), 'or')
                    else:
                        extended = gdspy.boolean(extended,gdspy.Rectangle((self.center[0]-t-self.g_w/2+l2,self.center[1]-gap-height_left*self.g_h/2-t-gap),(self.center[0]-self.g_w/2-2*gap-t-self.g_t,self.center[1]-height_left*self.g_h/2-t-self.g_t- core/2 - gap2)),'or')
                        extended = gdspy.boolean(extended,gdspy.Rectangle((self.center[0]-self.g_w/2-2*gap-t,self.center[1]-gap-height_left*self.g_h/2-t-gap),(self.center[0]-self.g_w/2-2*gap-t-self.g_t,self.center[1]- core/2 - gap2)), 'or')


                    # shift coupler to qubit and remove ground where necessary
                    remove = gdspy.Rectangle((self.center[0] - self.g_w / 2,
                                              self.center[1] + gap + height_left * self.g_h / 2 + t + self.g_t + gap),
                                             (self.center[0] - self.g_w / 2-t-2*gap,
                                              self.center[1] - gap - height_left * self.g_h / 2 - t - self.g_t - gap))
                    result = gdspy.boolean(result, remove.translate(+coupler.sctq, 0), 'not')

                    extended.translate(+coupler.sctq, 0)
                    # remove/add missing overhang:
                    # add missing piece
                    if coupler.sctq < 0:
                        extended = gdspy.boolean(extended, gdspy.Rectangle((self.center[0] - self.g_w / 2, self.center[1] + gap + height_left * self.g_h / 2 + t + self.g_t + gap), (self.center[0] - self.g_w / 2 + coupler.sctq,self.center[1] - gap - height_left * self.g_h / 2 - t - self.g_t - gap)),'or')

                    # remove additional pieces
                    if coupler.sctq > self.g_t:
                        extended = gdspy.boolean(extended, gdspy.Rectangle(
                            (self.center[0] - self.g_t + self.g_w / 2, self.center[1] + self.g_t - self.g_h / 2),(self.center[0] + self.g_t - self.g_w / 2, self.center[1] - self.g_t + self.g_h / 2)),'not')


                    result = gdspy.boolean(result,extended,'or')

                    #box for inverted polygon

                    if coupler.tight == True:
                        box = gdspy.boolean(box,
                                            gdspy.Rectangle((self.center[0] - self.g_w / 2 - self.g_t - 2 * gap - t,
                                                             self.center[
                                                                 1] - height_left * self.g_h / 2 - self.g_t - 2 * gap - t),
                                                            (self.center[0] - self.g_w / 2 + l1 - t, self.center[
                                                                1] + height_left * self.g_h / 2 + self.g_t + 2 * gap + t)).translate(
                                                +coupler.sctq, 0),
                                            'or', layer=self.layer_configuration.inverted)
                    else:
                        box = gdspy.boolean(box, gdspy.Rectangle((self.center[0] - self.g_w / 2 - self.g_t - 2 * gap - t,self.center[1] - height_left * self.g_h / 2 - self.g_t - 2 * gap ),(self.center[0] - self.g_w / 2 + l1 - t, self.center[1] + height_left * self.g_h / 2 + self.g_t + 2 * gap)).translate(+coupler.sctq,0),'or', layer=self.layer_configuration.inverted)

                    pocket = gdspy.boolean(pocket, gdspy.Rectangle((self.center[0] - self.g_w / 2 - self.g_t - 2 * gap - t,
                                                              self.center[
                                                                  1] - height_left * self.g_h / 2 - self.g_t - 2 * gap - t),
                                                             (self.center[0] - self.g_w / 2 + l1 - t, self.center[
                                                                 1] + height_left * self.g_h / 2 + self.g_t + 2 * gap + t)).translate(coupler.sctq,0),'or')


                    if coupler.sctq > self.g_t:
                        pocket = gdspy.boolean(pocket,gdspy.Rectangle((self.center[0],self.center[1]+coupler.gap/2+coupler.w),(self.center[0]-self.g_w/2-1,self.center[1]-coupler.w-coupler.gap/2)),'or')



                if side == 'top':
                    extended = gdspy.Rectangle((self.center[0]-self.g_w/2+l1-gap-self.g_t,self.center[1]+self.g_h/2),(self.center[0]-self.g_w/2+l1-gap,self.center[1]+self.g_h/2+t+gap+gap))
                    extended = gdspy.boolean(extended,gdspy.Rectangle((self.center[0]-self.g_w/2+l1-gap-self.g_t,self.center[1]+self.g_h/2+t+gap+gap),(self.center[0]-self.g_w/2+l1-gap+l2/2-gap,self.center[1]+self.g_h/2+t+gap+gap+self.g_t)),'or')
                    extended = gdspy.boolean(extended, gdspy.Rectangle((self.center[0]-self.g_w/2+l1+l2+gap,self.center[1]+self.g_h/2),(self.center[0]-self.g_w/2+l1+l2+self.g_t+gap,self.center[1]+self.g_h/2+t+gap+gap)), 'or')
                    extended = gdspy.boolean(extended, gdspy.Rectangle((self.center[0]-self.g_w/2+l1+l2+self.g_t+gap,self.center[1] + self.g_h / 2 + t + gap + gap),(self.center[0]-self.g_w/2+l1+l2/2+gap+gap,self.center[1] + self.g_h / 2 + t + gap + gap + self.g_t)),'or')
                    extended.translate(0,-coupler.sctq)
                    # remove/add missing overhang:
                    # add missing piece
                    if coupler.sctq < 0:
                        extended = gdspy.boolean(extended, gdspy.Rectangle((self.center[0]-self.g_w/2+l1-gap-self.g_t,self.center[1]+ self.g_h / 2),(self.center[0]-self.g_w/2+l1-gap+l2/2,self.center[1]+self.g_h/2+coupler.sctq)),'or')

                    # remove additional pieces
                    if coupler.sctq > self.g_t:
                        extended = gdspy.boolean(extended, gdspy.Rectangle((self.center[0] - self.g_t + self.g_w / 2, self.center[1] + self.g_t - self.g_h / 2),(self.center[0] + self.g_t - self.g_w / 2, self.center[1] - self.g_t + self.g_h / 2)),'not')

                    result = gdspy.boolean(result,extended,'or')

                    # box for inverted polygon
                    box = gdspy.boolean(box,gdspy.Rectangle((self.center[0]-self.g_w/2+l1-gap-self.g_t,self.center[1]+self.g_h/2),(self.center[0]-self.g_w/2+l1+l2+self.g_t+gap,self.center[1] + self.g_h / 2 + t + gap + gap + self.g_t)).translate(0,-coupler.sctq),'or', layer=self.layer_configuration.inverted)
                    pocket = gdspy.boolean(pocket, gdspy.Rectangle(
                        (self.center[0] - self.g_w / 2 + l1 - gap - self.g_t, self.center[1] + self.g_h / 2), (
                        self.center[0] - self.g_w / 2 + l1 + l2 + self.g_t + gap,
                        self.center[1] + self.g_h / 2 + t + gap + gap + self.g_t)).translate(0,-coupler.sctq), 'or',
                                        layer=self.layer_configuration.inverted)

                if side == 'bottom':
                    extended = gdspy.Rectangle((self.center[0]-self.g_w/2+l1-gap-self.g_t,self.center[1]-self.g_h/2),(self.center[0]-self.g_w/2+l1-gap,self.center[1]-self.g_h/2-t-gap-gap))
                    extended = gdspy.boolean(extended,gdspy.Rectangle((self.center[0]-self.g_w/2+l1-gap-self.g_t,self.center[1]-self.g_h/2-t-gap-gap),(self.center[0]-self.g_w/2+l1-gap+l2/2-5,self.center[1]-self.g_h/2-t-gap-gap-self.g_t)),'or')
                    extended = gdspy.boolean(extended, gdspy.Rectangle((self.center[0]-self.g_w/2+l1+l2+gap,self.center[1]-self.g_h/2),(self.center[0]-self.g_w/2+l1+l2+self.g_t+gap,self.center[1]-self.g_h/2-t-gap-gap)), 'or')
                    extended = gdspy.boolean(extended, gdspy.Rectangle((self.center[0]-self.g_w/2+l1+l2+self.g_t+gap,self.center[1] - self.g_h / 2 - t - gap-gap),(self.center[0]-self.g_w/2+l1+l2/2+gap+5,self.center[1] - self.g_h / 2 - t - gap-gap- self.g_t)),'or')
                    extended.translate(0, +coupler.sctq)
                    # remove/add missing overhang:
                    # add missing piece
                    if coupler.sctq < 0:
                        extended = gdspy.boolean(extended, gdspy.Rectangle(
                            (self.center[0] - self.g_w / 2 + l1 - gap - self.g_t, self.center[1]- self.g_h / 2), (
                            self.center[0] - self.g_w / 2 + l1 - gap + l2 / 2,
                            self.center[1] - self.g_h / 2 - coupler.sctq)), 'or')

                    # remove additional pieces
                    if coupler.sctq > self.g_t:
                        extended = gdspy.boolean(extended, gdspy.Rectangle(
                            (self.center[0] - self.g_t + self.g_w / 2, self.center[1] + self.g_t - self.g_h / 2),
                            (self.center[0] + self.g_t - self.g_w / 2, self.center[1] - self.g_t + self.g_h / 2)),
                                                 'not')

                    result = gdspy.boolean(result,extended,'or')

                    # box for inverted polygon
                    box = gdspy.boolean(box, gdspy.Rectangle(
                        (self.center[0] - self.g_w / 2 + l1 - gap - self.g_t, self.center[1] - self.g_h / 2), (self.center[0] - self.g_w / 2 + l1 + l2 + self.g_t + gap,self.center[1] - self.g_h / 2 - t - gap - gap - self.g_t)).translate(0,+coupler.sctq), 'or',layer=self.layer_configuration.inverted)
                    pocket = gdspy.boolean(pocket,gdspy.Rectangle((self.center[0] - self.g_w / 2 + l1 - gap - self.g_t, self.center[1] - self.g_h / 2), (self.center[0] - self.g_w / 2 + l1 + l2 + self.g_t + gap,self.center[1] - self.g_h / 2 - t - gap - gap - self.g_t)).translate(0,coupler.sctq),'or')

                result = gdspy.boolean(coupler_parts['positive'], result, 'or',layer=self.layer_configuration.total_layer)
                pockets.append(pocket)
                result_restricted = gdspy.boolean(pocket, result_restricted, 'or',layer=self.layer_configuration.total_layer)
                if coupler.coupler_type == 'coupler':
                        qubit_cap_parts.append(gdspy.boolean(coupler.result_coupler, coupler.result_coupler, 'or',layer=10+id+self.secret_shift))
                        self.layers.append(10+id+self.secret_shift)
                        last_step_cap.append(coupler.result_coupler)
        qubit_cap_parts.append(gdspy.boolean(result,last_step_cap,'not'))

        inverted = gdspy.boolean(box, result, 'not',layer=self.layer_configuration.inverted)

        #for p in pockets:
            #result_restricted = gdspy.boolean(p, result_restricted, 'or', layer=self.layer_configuration.total_layer)


        # add JJs
        if self.JJ_params is not None:
            self.JJ_coordinates = (self.center[0],self.center[1])
            JJ = self.generate_JJ()
        '''
        return {'positive': result,
                    'restricted': result_restricted,
                    'qubit': result,
                    'qubit_cap': qubit_cap_parts,
                    'JJ': JJ,
                    'inverted': inverted
                    }
        '''
        qubit=deepcopy(result)

        # set terminals for couplers
        self.set_terminals()
        if self.calculate_capacitance is False:
            qubit_cap_parts = None
            qubit = None

        if 'mirror' in self.transformations:
            return {'positive': result.mirror(self.transformations['mirror'][0], self.transformations['mirror'][1]),
                    'restrict': result_restricted,
                    'qubit': qubit.mirror(self.transformations['mirror'][0],
                                          self.transformations['mirror'][1]) if qubit is not None else None,
                    'qubit_cap': qubit_cap_parts,
                    'JJ': JJ.mirror(self.transformations['mirror'][0], self.transformations['mirror'][1]),
                    'inverted': inverted.mirror(self.transformations['mirror'][0], self.transformations['mirror'][1]),
                    'pocket': pocket.mirror(self.transformations['mirror'][0], self.transformations['mirror'][1]),
                    }

        if 'rotate' in self.transformations:
            return {'positive': result.rotate(self.transformations['rotate'][0], self.transformations['rotate'][1]),
                    'restrict': result_restricted.rotate(self.transformations['rotate'][0], self.transformations['rotate'][1]),
                    'qubit': qubit.rotate(self.transformations['rotate'][0],
                                          self.transformations['rotate'][1]) if qubit is not None else None,
                    'qubit_cap': qubit_cap_parts,
                    'JJ': JJ.rotate(self.transformations['rotate'][0], self.transformations['rotate'][1]),
                    'inverted': inverted.rotate(self.transformations['rotate'][0], self.transformations['rotate'][1]),
                    'pocket': pocket.rotate(self.transformations['rotate'][0], self.transformations['rotate'][1]),
                    }
        elif self.transformations == {}:
            return {'positive': result,
                    'restrict': result_restricted,
                    'qubit': qubit,
                    'qubit_cap': qubit_cap_parts,
                    'JJ': JJ,
                    'inverted': inverted,
                    'pocket':pocket,
                    }

    def generate_ground(self):
        x = self.g_w
        y = self.g_h
        z = self.center
        t = self.g_t
        ground1 = gdspy.Rectangle((z[0] - x / 2, z[1] - y / 2), (z[0] + x / 2, z[1] + y / 2))
        ground2 = gdspy.Rectangle((z[0] - x / 2 + t, z[1] - y / 2 + t), (z[0] + x / 2 - t, z[1] + y / 2 - t))
        ground = gdspy.fast_boolean(ground1, ground2, 'not')
        for key in self.remove_ground:
            factor = 1
            if key == 'left':
                if self.remove_ground[key] != None:
                    factor = self.remove_ground[key]
                ground = gdspy.fast_boolean(ground,gdspy.Rectangle((z[0] - x / 2,z[1] - factor*y / 2), (z[0] - x / 2 +t, z[1] + factor*y / 2)) , 'not')
            if key == 'right':
                if self.remove_ground[key] != None:
                    factor = self.remove_ground[key]
                ground = gdspy.fast_boolean(ground,gdspy.Rectangle((z[0] + x / 2,z[1] - factor*y / 2), (z[0] + x / 2 -t, z[1] + factor*y / 2)) , 'not')
            if key == 'top':
                if self.remove_ground[key] != None:
                    factor = self.remove_ground[key]
                ground = gdspy.fast_boolean(ground,gdspy.Rectangle((z[0] - factor*x / 2,z[1] + y / 2), (z[0] + factor*x / 2, z[1] + y / 2-t)) , 'not')
            if key == 'bottom':
                if self.remove_ground[key] != None:
                    factor = self.remove_ground[key]
                ground = gdspy.fast_boolean(ground,gdspy.Rectangle((z[0] - factor*x / 2,z[1] - y / 2), (z[0] + factor*x / 2, z[1] - y / 2+t)) , 'not')

        if self.holes :
            print('importing file')
            hole_mask = gdspy.GdsLibrary(infile=".\\QCreator\\QCreator\\elements\\masks\\Holes.gds")
            print('imported holes')
            for holes in hole_mask:
                ground = gdspy.fast_boolean(ground,holes,'not')
                print('done subtracting holes')
                print(ground)
            ground = gdspy.fast_boolean(None, ground, 'or')
            print(ground)
        return ground


    def set_terminals(self):
        for id, coupler in enumerate(self.couplers):
            if 'mirror' in self.transformations:
                if coupler.connection is not None:
                    coupler_connection = mirror_point(coupler.connection, self.transformations['mirror'][0], self.transformations['mirror'][1])
                    qubit_center = mirror_point(deepcopy(self.center), self.transformations['mirror'][0], self.transformations['mirror'][1])
                    if coupler.side == "left":
                        coupler_phi = np.pi
                    if coupler.side == "right":
                        coupler_phi = 0
                    if coupler.side == "top":
                        coupler_phi = -np.pi / 2
                    if coupler.side == "bottom":
                        coupler_phi = np.pi / 2
                    if coupler.side == 'fluxline':
                        coupler_phi = -np.pi / 2

            if 'rotate' in self.transformations:
                if coupler.connection is not None:
                    coupler_connection = rotate_point(coupler.connection, self.transformations['rotate'][0], self.transformations['rotate'][1])
                    qubit_center = rotate_point(deepcopy(self.center), self.transformations['rotate'][0], self.transformations['rotate'][1])
                    if coupler.side == "left":
                        coupler_phi = 0+self.transformations['rotate'][0]
                    if coupler.side == "right":
                        coupler_phi = np.pi+self.transformations['rotate'][0]
                    if coupler.side == "top":
                        coupler_phi = -np.pi / 2+self.transformations['rotate'][0]
                    if coupler.side == "bottom":
                        coupler_phi = np.pi / 2+self.transformations['rotate'][0]
                    if coupler.side == "fluxline":
                        coupler_phi =  -np.pi / 2+self.transformations['rotate'][0]

            if self.transformations == {}:
                coupler_connection = coupler.connection
                if coupler.side == "left":
                    coupler_phi = 0
                if coupler.side == "right":
                    coupler_phi = np.pi
                if coupler.side == "top":
                    coupler_phi = -np.pi/2
                if coupler.side == "bottom":
                    coupler_phi = np.pi/2
                if coupler.side == "fluxline":
                    coupler_phi = -np.pi/2
            if coupler.connection is not None:
                self.terminals['coupler'+str(id)] = DesignTerminal(tuple(coupler_connection),
                                                                   coupler_phi, g=coupler.g, s=coupler.s,
                                                                w=coupler.w, type='cpw')
        return True


    def get_terminals(self):
        return self.terminals



    def generate_JJ(self):
        #change here to allow Manhatten style junctions
        if self.JJ_params['manhatten']:
            if self.JJ_params['squid']:
                reachy = 5
                reachx = 15
                loop_h = self.JJ_params['loop_h']
                result1 = gdspy.Rectangle((self.center[0] - self.b_g / 2,
                                          self.center[1] + self.h / 2 - loop_h/2-self.b_w/2 + self.JJ_params['a1'] / 2), (
                                         self.center[0] +reachx,
                                         self.center[1] + self.h / 2 - loop_h/2 -self.b_w/2 - self.JJ_params['a1'] / 2))
                result2 = gdspy.copy(result1,0,loop_h)
                result = gdspy.boolean(result1, result2, 'or')

                result = gdspy.boolean(result, gdspy.Rectangle(
                    (self.center[0] + self.b_g / 2, self.center[1] + self.h / 2 - 2 * self.b_w), (
                    self.center[0] + self.b_g / 2 + self.JJ_params['a2'],self.center[1] + self.h / 2 - self.b_w / 3 + self.JJ_params['a1'] / 2+reachy
                    )), 'or')
            else:
                #JJ_2 is the manhatten junction style
                ##############################################################################
                self.JJ = JJ4q.JJ_2(self.JJ_coordinates[0], self.JJ_coordinates[1],
                                    self.JJ_params['a1'], self.JJ_params['a2'],self.JJ_params['h_w'],self.JJ_params['h_d'])
                result = self.JJ.generate_jj()
                #############################################################################
        else:
            self.JJ = JJ4q.JJ_1(self.JJ_coordinates[0], self.JJ_coordinates[1],
                                    self.JJ_params['a1'], self.JJ_params['a2'])
            result = self.JJ.generate_jj()

        result = gdspy.boolean(result, result, 'or', layer=self.layer_configuration.jj_layer)

        angle = self.JJ_params['angle_JJ']
        result.rotate(angle, (self.JJ_coordinates[0], self.JJ_coordinates[1]))

        return result


    #for the capacity
    def add_to_tls(self, tls_instance: tlsim.TLSystem, terminal_mapping: dict, track_changes: bool = True,
                   cutoff: float = np.inf, epsilon: float = 11.45) -> list:
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
                c0 = tlsim.Capacitor(c=self.C['coupler'+str(id)][1]*scal_C, name=self.name+' qubit-coupler'+str(id)+self.secret_shift)
                c0g = tlsim.Capacitor(c=self.C['coupler'+str(id)][0]*scal_C, name=self.name+' coupler'+str(id)+'-ground'+self.secret_shift)
                tls_instance.add_element(c0, [terminal_mapping['qubit'], terminal_mapping['coupler'+str(id)+self.secret_shift]])
                tls_instance.add_element(c0g, [terminal_mapping['coupler'+str(id)+self.secret_shift], 0])
                mut_cap.append(c0)
                cap_g.append(c0g)

        if track_changes:
            self.tls_cache.append([JJ, C]+mut_cap+cap_g)
        return [JJ, C]+mut_cap+cap_g


class PP_Transmon_Coupler:
    """
    This class represents a coupler for a PP_transmon qubit.
    There are several parameters:
    1) l1 - length of the upper claw finger
    2) l2 - length of the lower claw finger
    3) t  - thickness of the coupler
    4) gap - gap between ground and coupler
    5) side - which side the coupler is on
    6) heightl / heightr - height as a fraction of total length
    """
    def __init__(self, l1,l2,t,side = 'left',coupler_type = 'none',heightl = 1,heightr=1,w= None, g=None, s=None,shift_to_qubit=0,tight = [False]):
        self.l1 = l1
        self.l2 = l2
        self.t = t

        self.gap = s
        if tight[0]:
            self.gaps = [s,tight[1]]
            self.gap = tight[1]
        else:
            self.gaps = [s,s]
        self.tight = tight[0]
        self.side = side
        self.coupler_type = coupler_type
        self.ground_t = 0 # the lenght of the coupler connection part to the resonator
        self.height_left = heightl
        self.height_right = heightr
        self.connection = None
        #for defining the terminals
        self.w = w
        self.g = g
        self.s = s
        #put coupler closer to qubit
        self.sctq = shift_to_qubit

    def render(self, center, g_w,g_h,g_t = 0):
        result = 0
        if self.side == "left":
            result = gdspy.Rectangle((center[0]-g_w/2-self.t-self.gap,center[1]-self.height_left*g_h/2-self.gap),(center[0]-g_w/2-self.gap,center[1]+self.height_left*g_h/2+self.gap))
            if self.height_left == 1:
                upper  = gdspy.Rectangle((center[0]-g_w/2-self.t-self.gap,center[1]+g_h/2+self.gap),(center[0]-g_w/2+self.l1-self.gap-self.t,center[1]+g_h/2+self.t+self.gap))
                lower  = gdspy.Rectangle((center[0]-g_w/2-self.t-self.gap,center[1]-g_h/2-self.gap-self.t),(center[0]-g_w/2+self.l2-self.gap-self.t,center[1]-g_h/2-self.gap))
            if self.sctq > g_t:
                line = gdspy.Rectangle(
                    (center[0] - g_w / 2 + g_t -self.sctq, center[1] - self.w / 2), (
                    center[0] - g_w / 2 - self.t - self.gap,
                    center[1] + self.w / 2))  # modified here ;), remove ground_t

            else:
                line   = gdspy.Rectangle((center[0]-g_w/2-self.t-self.gap-self.gap-self.ground_t-g_t,center[1]-self.w/2),(center[0]-g_w/2-self.t-self.gap,center[1]+self.w/2))#modified here ;), remove ground_t


            if self.height_left ==1:
                result = gdspy.boolean(result, upper, 'or')
                result = gdspy.boolean(result, lower, 'or')
            result = gdspy.boolean(result, line, 'or')
            result.translate(self.sctq,0)

            if self.sctq > g_t:
                self.connection = (center[0] - g_w / 2 + g_t, center[1])
            else:
                self.connection = (center[0]-g_w/2-self.t-self.gap-self.gap+self.sctq-g_t,center[1])




        if self.side == "right":
            result = gdspy.Rectangle((center[0]+g_w/2+self.t+self.gap,center[1]-self.height_right*g_h/2-self.gap),(center[0]+g_w/2+self.gap,center[1]+self.height_right*g_h/2+self.gap))
            if self.height_right == 1:
                upper  = gdspy.Rectangle((center[0]+g_w/2+self.t+self.gap,center[1]+g_h/2+self.gap),(center[0]+g_w/2-self.l1+self.gap+self.t,center[1]+g_h/2+self.t+self.gap))
                lower  = gdspy.Rectangle((center[0]+g_w/2+self.t+self.gap,center[1]-g_h/2-self.gap-self.t),(center[0]+g_w/2-self.l2+self.gap+self.t,center[1]-g_h/2-self.gap))

            if self.sctq > g_t:
                line = gdspy.Rectangle((center[0] + g_w / 2 - g_t +self.sctq, center[1] - self.w / 2),
                                       (center[0] + g_w / 2 + self.t + self.gap, center[1] + self.w / 2))
            else:
                line = gdspy.Rectangle((center[0]+g_w/2+self.t+self.gap+self.gap+g_t,center[1]-self.w/2),(center[0]+g_w/2+self.t+self.gap,center[1]+self.w/2))


            if self.height_right == 1:
                result = gdspy.boolean(result, upper, 'or')
                result = gdspy.boolean(result, lower, 'or')
            result = gdspy.boolean(result, line, 'or')
            result.translate(-self.sctq, 0)
            if self.sctq > g_t:
                self.connection = (center[0] + g_w / 2 - g_t, center[1])
            else:
                self.connection = (center[0] + g_w / 2 + self.t + self.gap + self.gap-self.sctq+g_t, center[1] )

        if self.side == "top":
            result = gdspy.Rectangle((center[0]-g_w/2+self.l1,center[1]+g_h/2+self.gap),(center[0]-g_w/2+self.l1+self.l2,center[1]+g_h/2+self.gap+self.t))
            line = gdspy.Rectangle(
                (center[0] - g_w / 2 + self.l1 + self.l2 / 2 - self.w / 2, center[1] + g_h / 2 + self.gap + self.t), (
                center[0] - g_w / 2 + self.l1 + self.l2 / 2 + self.w / 2,
                center[1] + g_h / 2 + self.gap + self.t + self.gap+g_t))

            result = gdspy.boolean(result, line, 'or')
            result.translate(0,-self.sctq)
            self.connection = (center[0]-g_w/2+self.l1+self.l2/2, center[1]+g_h/2+self.gap+self.t+self.gap-self.sctq+g_t)




        if self.side == "bottom":
            result = gdspy.Rectangle((center[0]-g_w/2+self.l1,center[1]-g_h/2-self.gap),(center[0]-g_w/2+self.l1+self.l2,center[1]-g_h/2-self.gap-self.t))
            line   = gdspy.Rectangle((center[0]-g_w/2+self.l1+self.l2/2-self.w/2,center[1]-g_h/2-self.gap-self.t),(center[0]-g_w/2+self.l1+self.l2/2+self.w/2,center[1]-g_h/2-self.gap-self.t-self.gap-g_t))
            result = gdspy.boolean(result, line, 'or')
            result.translate(0,+self.sctq)
            self.connection = (center[0]-g_w/2+self.l1+self.l2/2, center[1]-g_h/2-self.gap-self.t-self.gap+self.sctq-g_t)

        self.result_coupler = result

        return {
            'positive': result
                        }


def rotate_point(point, angle, origin):
    """
    Rotate a point counterclockwise by a given angle around a given origin.
    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point
    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return qx, qy




def mirror_point(point,ref1,ref2):
    """
       Mirror a point by a given line specified by 2 points ref1 and ref2.
    """
    [x1, y1] =ref1
    [x2, y2] =ref2

    dx = x2-x1
    dy = y2-y1
    a = (dx * dx - dy * dy) / (dx * dx + dy * dy)
    b = 2 * dx * dy / (dx * dx + dy * dy)
    x2 = round(a * (point[0] - x1) + b * (point[1] - y1) + x1)
    y2 = round(b * (point[0] - x1) - a * (point[1] - y1) + y1)
    return x2, y2


class PP_Squid_Fluxline:
    """
    This class represents a Flux_line for a PP_Squid. Design inspired from  Vivien Schmitt. Design, fabrication and test of a four superconducting quantum-bit processor. Physics[physics]
    There are several parameters:
    1) l     - total length of the flux line to the Squid
    2) t_m   - main line thickness, standard is 2*t_r
    3) t_r   - return line thickness
    4) gap   - gap between Squid and line
    5) l_arm - length of one sidearm
    6) h_arm - height of the return arm
    7) s_gap - gap between main and return fluxline
    """
    def __init__(self, l,t_m,t_r,gap,l_arm,h_arm,s_gap,flux_distance,pad_w,pad_h,pad_g,b_w,b_g,ground,asymmetry = 0,w= None, g=None, s=None,extend = None):
        self.l      = l
        self.t_m    = t_m
        self.t_r    = t_r
        self.gap    = gap
        self.l_arm  = l_arm
        self.h_arm  = h_arm
        self.s_gap  = s_gap
        self.asymmetry = asymmetry
        self.flux_distance = flux_distance
        self.side   = 'fluxline'
        self.connection = None
        #for the terminals:
        self.g = g
        self.w = w
        self.s = s
        self.extend = extend
        #pad parameters
        self.pad_w = pad_w
        self.pad_h = pad_h
        self.b_w = b_w
        self.b_g = b_g
        self.pad_g = pad_g
        self.ground = ground
    def render(self, center, width,height,ground_height,ground_t):
        if not self.extend:
            ground_t = ground_height/2
        factor = 2.18
        #the shift is necessary for the diagonal parts to have the same thickness as the arms
        x1 = (self.l_arm-self.s_gap)/self.h_arm
        x2 =  (factor*self.l_arm-self.s_gap)/self.h_arm
        shift_right = self.t_r*np.tan(0.5*np.arctan(x1))
        shift_left  = self.t_r*np.tan(0.5*np.arctan(x2))

        start  = [0,0]
        points = [start+[0,0],start+[self.t_m,0],start+[self.t_m,-self.l],start+[self.t_m+self.l_arm-shift_right*(self.l_arm-self.s_gap)/self.h_arm,-self.l]]
        points.append(start+[self.t_m+self.s_gap,-self.l+self.h_arm-shift_right])
        points.append(start+[self.t_m+self.s_gap,0])
        points.append(start + [self.t_m + self.s_gap+self.t_r, 0])
        points.append(start + [self.t_m + self.s_gap + self.t_r, -self.l+self.h_arm])
        points.append(start + [self.t_m + self.l_arm+ self.t_r, -self.l])
        points.append(start + [self.t_m + self.l_arm+ self.t_r, -self.l-self.t_r])
        points.append(start + [- factor*self.l_arm- self.t_r, -self.l-self.t_r])
        points.append(start + [- factor*self.l_arm - self.t_r, -self.l])
        points.append(start + [-self.t_r-self.s_gap, -self.l+self.h_arm])
        points.append(start + [-self.t_r - self.s_gap, 0])
        points.append(start + [- self.s_gap, 0])
        points.append(start + [- self.s_gap, -self.l+self.h_arm-shift_left])
        points.append(start + [- factor*self.l_arm+shift_left*(factor*self.l_arm-self.s_gap)/self.h_arm, -self.l])
        points.append(start + [0, -self.l])
        points = [(i[0]+i[2],i[1]+i[3]) for i in points]
        result = gdspy.Polygon(points)

        #restricted area:
        points2 = [points[13],points[6],points[7],points[8],points[9],points[10],points[11],points[12],points[13]]

        restrict = gdspy.Polygon(points2)



        #cutouts of the ground
        cutout1 = gdspy.Rectangle(
                (0,ground_height / 2 - ground_t),
                (self.l_arm + self.t_m,ground_height / 2 + 250))
        cutout2 = gdspy.Rectangle(
                (-self.s_gap-self.t_m/2, center[1] + ground_height / 2 ),
                (+self.s_gap+self.t_m/2, center[1] + ground_height / 2 +100))

        result = gdspy.boolean(result,cutout1,'not')

        result = gdspy.boolean(result,cutout2,'not')

        result.translate(-self.t_m/2,+self.l+self.t_r)
        #result.rotate(-np.pi/2)

        restrict.translate(-self.t_m/2,+self.l+self.t_r)
        #restrict.rotate(-np.pi / 2)

        # move fluxline to correct position
        result.translate(center[0], center[1])
        result.translate(0+(self.l_arm + self.t_m)/2,self.pad_h/2+self.flux_distance)#self.pad_g/2+self.pad_w-self.b_w/2+self.flux_distance,self.asymmetry+self.pad_h/2+3.5*self.b_w)

        restrict.translate(center[0], center[1])

        #restrict.translate(self.pad_g/2+self.pad_w-self.b_w/2+self.flux_distance,self.asymmetry+self.pad_h/2+3.5*self.b_w)
        restrict.translate(0+(self.l_arm + self.t_m)/2, self.pad_h / 2 + self.flux_distance)
        #cuttng off the hanging rest:
        if self.ground != None:
            result = gdspy.boolean(result,self.ground,'and')

        self.result_coupler = result

        #point = (center[0]+self.pad_g/2+self.pad_w-self.b_w/2+self.flux_distance+self.t_r+self.l,center[1]+self.asymmetry+self.pad_h/2+3.5*self.b_w)
        point = (center[0]+(self.l_arm + self.t_m)/2,center[1]+self.pad_h / 2+self.flux_distance+self.t_r+self.l)
        self.connection = point

        return {
            'positive': result,
            'restricted':restrict,
                        }