from .core import DesignElement, LayerConfiguration, DesignTerminal
from .. import conformal_mapping as cm
from .. import transmission_line_simulator as tlsim
import gdspy
import numpy as np
from typing import List, Tuple, Mapping, Dict
from . import squid3JJ #TODO make this qubit class suitable for any squid types
from . import JJ4q
from copy import deepcopy


class Fungus_Squid_C(DesignElement):
    """
    Fungus-Transmon (asymmetric pad size) consists of several parts:
    1) Central part - central circuit
    params: center = center of the qubit, w,h,gap = width,height,gap of the Parallel Plate Transmon in the ground cavity
    2) Couplers - claw like couplers for the left and right, rectangular pad couplers on top and bottom
    3) Ground = Ground rectangle around qubit, g_w,g_h,g_t = width,height and thickness of ground frame
    4)layer_configuration
    5)Couplers - coupler classes
    6) jj_params - parameters of the SQUID which here is 3JJ SQUID.#TODO add more information
    7) remove_ground - removes the ground on the specified site (left,right, top, bottom)
    8) arms - the parameters for the squid arms for coupling left and right qubits
    9) shoes - caps for the two pads
    """
    def __init__(self, name: str,
                 center: Tuple[float, float],
                 width: Tuple[float,float],
                 height: Tuple[float,float],
                 gap: float,
                 bridge_gap:float,
                 bridge_w:float,
                 ground_w: float,
                 ground_h: float,
                 ground_t: float,
                 layer_configuration: LayerConfiguration,
                 jj_params: Dict,
                 Couplers,
                 transformations:Dict,
                 fluxline_params= {},
                 remove_ground = {},
                 secret_shift = 0,
                 calculate_capacitance = False,
                 shoes = {},
                 claw = [],
                 asymmetry = 0,
                 air_bridge=[],
                 use_bandages = True,
                 return_inverted = True):
        super().__init__(type='qubit', name=name)
        #qubit parameters
        self.transformations = transformations# to mirror the structure
        self.center = center
        self.w_pads = width
        self.h_pads = height
        self.w = width[1]
        self.h = height[1]
        self.shoes = shoes
        self.claw = claw
        self.asymmetry = asymmetry # to shift the right smaller pad
        self.gap = gap
        self.g_w = ground_w
        self.g_h = ground_h
        self.g_t = ground_t
        self.b_g = bridge_gap
        self.b_w = bridge_w
        self.air = air_bridge
        #layers
        self.layer_configuration = layer_configuration

        #couplers
        self.couplers = Couplers

        # JJs and fluxline
        self.JJ_params = jj_params
        self.JJ = None
        self.layers = []
        self.fluxline_params = fluxline_params

        self.tls_cache = []
        self.L = 15e-9  # 20nHr
        self.C = {   'coupler0': None,
             'coupler1': None,
             'coupler2': None,
             'coupler3': None,
             'coupler4': None,
             'qubit': None}

        #terminals
        self.terminals = {  # 'coupler0': None,
            # 'coupler1': None,
            # 'coupler2': None,
            # 'coupler3': None,
            # 'coupler4': None,
            # 'flux line': None,
            'qubit': None}


        # use bandages?
        self.use_bandages = use_bandages
        #check if inverted region should be returned
        self.return_inverted = return_inverted

        # remove ground on these sites
        self.remove_ground = remove_ground

        self.secret_shift = secret_shift

        self.calculate_capacitance = calculate_capacitance

    def render(self):
        """
        This function draws everything: qubit,ground,couplers
        """
        qubit_cap_parts=[]
        ground = self.generate_ground()
        # restricted area for a future grid lines
        result_restricted = gdspy.Rectangle((self.center[0]-self.g_w/2,self.center[1]-self.g_h/2),(self.center[0]+self.g_w/2,self.center[1]+self.g_h/2),layer=self.layer_configuration.restricted_area_layer)

        P1 = gdspy.Rectangle((self.center[0]-self.gap/2-self.w_pads[0],self.center[1]-self.h_pads[0]/2),(self.center[0]-self.gap/2,self.center[1]+self.h_pads[0]/2))
        P2 = gdspy.Rectangle((self.center[0] + self.gap / 2 + self.w_pads[1], self.center[1] - self.h_pads[1] / 2+self.asymmetry),(self.center[0] + self.gap / 2, self.center[1] + self.h_pads[1] / 2+self.asymmetry))
        # adding the shoe caps here
        for key in self.shoes:
            factor = 0
            if 'T' in self.shoes:
                if 'T':
                    factor = 1
            Rot = 0
            if 'R' in self.shoes:
                Rot = self.shoes['R']

            if key == 1:
                Shoe = gdspy.Rectangle(
                    (self.center[0] - self.gap / 2 - self.w_pads[0]+factor*(self.w_pads[0]+self.shoes[key][0]), self.center[1] + self.h_pads[0] / 2), (
                    self.center[0] - self.gap / 2 - self.w_pads[0] - self.shoes[key][0],
                    self.center[1] + self.h_pads[0] / 2 - self.shoes[key][1]))

                if 'R' in self.shoes:
                    Shoe.translate(0, self.shoes[key][1] / 2)
                Shoe.rotate(-Rot, (self.center[0] - self.gap / 2 - self.w, self.center[1] + self.h / 2))

                P1 = gdspy.boolean(P1, Shoe, 'or')

            if key == 2:
                Shoe = gdspy.Rectangle(
                    (self.center[0] - self.gap / 2 - self.w_pads[0]+factor*(self.w_pads[0]+self.shoes[key][0]), self.center[1] - self.h_pads[0] / 2), (
                    self.center[0] - self.gap / 2 - self.w_pads[0] - self.shoes[key][0],
                    self.center[1] - self.h_pads[0] / 2 + self.shoes[key][1]))

                if 'R' in self.shoes:
                    Shoe.translate(0, -self.shoes[key][1] / 2)
                Shoe.rotate(Rot, (self.center[0] - self.gap / 2 - self.w, self.center[1] - self.h / 2))

                P1 = gdspy.boolean(P1, Shoe, 'or')

            if key == 3:
                Shoe = gdspy.Rectangle(
                    (self.center[0] + self.gap / 2 + self.w-factor*(self.w+self.shoes[key][0]), self.center[1] + self.h / 2), (
                    self.center[0] + self.gap / 2 + self.w + self.shoes[key][0],
                    self.center[1] + self.h / 2 - self.shoes[key][1]))

                if 'R' in self.shoes:
                    Shoe.translate(0, self.shoes[key][1] / 2)
                Shoe.rotate(+Rot, (self.center[0] + self.gap / 2 + self.w, self.center[1] + self.h / 2))

                P2 = gdspy.boolean(P2, Shoe, 'or')

            if key == 4:
                Shoe = gdspy.Rectangle(
                    (self.center[0] + self.gap / 2 + self.w-factor*(self.w+self.shoes[key][0]), self.center[1] - self.h / 2), (
                    self.center[0] + self.gap / 2 + self.w + self.shoes[key][0],
                    self.center[1] - self.h / 2 + self.shoes[key][1]))
                if 'R' in self.shoes:
                    Shoe.translate(0, -self.shoes[key][1] / 2)

                Shoe.rotate(-Rot, (self.center[0] + self.gap / 2 + self.w, self.center[1] - self.h / 2))
                P2 = gdspy.boolean(P2, Shoe, 'or')

        #adding the claws on P1:
        if self.claw != []:
            i = self.claw
            #upper
            P1 = gdspy.boolean(P1,gdspy.Rectangle((self.center[0]-self.gap/2-i[0],self.center[1]+self.h_pads[0]/2+i[1]),(self.center[0]-self.gap/2,self.center[1]-self.h_pads[0]/2-i[1])),'or')
            #lower
            P1 = gdspy.boolean(P1, gdspy.Rectangle((self.center[0]-self.gap/2+i[0]-self.w_pads[0],self.center[1]+self.h_pads[0]/2+i[1]),(self.center[0]-self.gap/2-self.w_pads[0],self.center[1]-self.h_pads[0]/2-i[1])),'or')



        self.layers.append(9)
        result = gdspy.boolean(ground, P1, 'or', layer=self.layer_configuration.total_layer)
        result = gdspy.boolean(result, P2, 'or', layer=self.layer_configuration.total_layer)



        P1_bridge = gdspy.Rectangle((self.center[0]-self.gap/2,self.center[1]+self.h/2+self.asymmetry+3*self.b_w),(self.center[0]+self.gap/2+self.w-self.b_w,self.center[1]+self.h/2+self.asymmetry+2*self.b_w))

        P2_bridge = gdspy.Rectangle((self.center[0] + self.gap / 2+self.w, self.center[1]+self.h/2+self.asymmetry),(self.center[0] + self.gap / 2+self.w-self.b_w, self.center[1]+self.h/2+self.asymmetry+2*self.b_w-self.b_g))
        if 'rotation' in self.JJ_params:
            P1_bridge = gdspy.Rectangle(
            (self.center[0] - self.gap / 2, self.center[1] + self.h / 2 + self.asymmetry + 3 * self.b_w), (
            self.center[0] + self.gap / 2 + self.w - np.sqrt(2)*self.b_w,
            self.center[1] + self.h / 2 + self.asymmetry + 2 * self.b_w))

            P2_bridge = gdspy.Rectangle(
            (self.center[0] + self.gap / 2 + self.w, self.center[1] + self.h / 2 + self.asymmetry), (
            self.center[0] + self.gap / 2 + self.w - self.b_w,
            self.center[1] + self.h / 2 + self.asymmetry + np.sqrt(2)*2 * self.b_w - self.b_g))

        qubit_cap_parts.append(gdspy.boolean(P1, P1_bridge, 'or', layer=8+self.secret_shift))
        qubit_cap_parts.append(gdspy.boolean(P2, P2_bridge, 'or', layer=9+self.secret_shift))

        result = gdspy.boolean(result, P1_bridge, 'or', layer=self.layer_configuration.total_layer)
        result = gdspy.boolean(result, P2_bridge, 'or', layer=self.layer_configuration.total_layer)
        self.layers.append(self.layer_configuration.total_layer)

        #change orientation of fluxline
        if self.fluxline_params != {}:
            f = self.fluxline_params
            print(f)
            l, t_m, t_r, gap, l_arm, h_arm, s_gap = f['l'],f['t_m'],f['t_r'],f['gap'],f['l_arm'],f['h_arm'],f['s_gap']
            flux_distance = f['flux_distance']
            #result_restricted, to ct off hanging parts from the fluxline, None for no cutoff
            flux = PP_Squid_Fluxline(l, t_m, t_r, gap, l_arm, h_arm, s_gap,flux_distance,self.w,self.h,self.gap,self.b_w,self.b_g,ground = None,asymmetry = self.asymmetry,g = f.get('g'),w = f.get('w'),s = f.get('s'),extend = f.get('extend_to_ground'),rotation = f.get('rotation'))

            fluxline = flux.render(self.center, self.w, self.h,self.g_h,self.g_t)['positive']

            r_flux = flux.render(self.center, self.w, self.h,self.g_h,self.g_t)['restricted']

            remove_ground_flux = flux.render(self.center, self.w, self.h,self.g_h,self.g_t)['remove_ground']
            if f['inverted_extension'] is not None:
                inverted_flux = flux.render(self.center, self.w, self.h, self.g_h, self.g_t,f['inverted_extension'])['inverted']
                r_flux = flux.render(self.center, self.w, self.h, self.g_h, self.g_t,f['inverted_extension'])['restricted']

            #adding fluxline restricted area
            result_restricted = gdspy.boolean(result_restricted,r_flux,'or', layer=self.layer_configuration.restricted_area_layer)

            self.couplers.append(flux)
            result = gdspy.boolean(result, fluxline, 'or', layer=self.layer_configuration.total_layer)
            #removing ground where the fluxline is
            ground_fluxline =True
            if ground_fluxline == False:
                result = gdspy.boolean(result, gdspy.Rectangle((self.center[0]-l_arm/2-t_r-self.g_t,self.center[1]+self.h/2+0.01),(self.center[0]+3*l_arm/2+t_r+t_m+self.g_t,self.center[1]+self.h/2)), 'not', layer=self.layer_configuration.total_layer)
            else:
                result = gdspy.boolean(result,remove_ground_flux, 'not',layer=self.layer_configuration.total_layer)



        #adding air bridges
        Air = None
        if self.air != []:
            Air = gdspy.Rectangle((self.center[0]-self.gap/2-self.w_pads[0]/2-self.air[2]/2,self.center[1]+self.air[0]),(self.center[0]-self.gap/2-self.w_pads[0]/2+self.air[2]/2,self.center[1]+self.air[0]+self.air[1]),self.layer_configuration.airbridges_layer)

        # add couplers
        last_step_cap = [gdspy.boolean(gdspy.boolean(P2, P2_bridge, 'or'),gdspy.boolean(P1, P1_bridge, 'or'),'or')]
        self.layers.append(self.layer_configuration.total_layer)

        # add junctions, bandages and holes
        if self.JJ_params is not None:
            self.JJ_coordinates = (self.center[0], self.center[1])
            JJ, bandages, holes = self.generate_JJ()
            result = gdspy.boolean(result, holes, 'not')


        # Box for inverted Polygons
        box = gdspy.Rectangle((self.center[0] - self.g_w / 2, self.center[1] - self.g_h / 2),(self.center[0] + self.g_w / 2, self.center[1] + self.g_h / 2))
        pocket = gdspy.Rectangle((self.center[0] - self.g_w / 2+self.g_t, self.center[1] - self.g_h / 2+self.g_t),(self.center[0] + self.g_w / 2-self.g_t, self.center[1] + self.g_h / 2-self.g_t))
        if self.fluxline_params != {}:
            pocket = gdspy.boolean(pocket,r_flux,'or', layer=self.layer_configuration.restricted_area_layer)

        if len(self.couplers) != 0:
            for id, coupler in enumerate(self.couplers):
                if coupler.side == 'fluxline':
                    continue
                coupler_parts = coupler.render(self.center, self.g_w,self.g_h)

                result = gdspy.boolean(coupler_parts['positive'], result, 'or',
                                       layer=self.layer_configuration.total_layer)

                #Extend ground around coupler
                l1   = coupler.l1
                l2   = coupler.l2
                t    = coupler.t
                gap  = coupler.gap
                side = coupler.side
                height_left = coupler.height_left
                height_right = coupler.height_right
                #to make sure ground is placed correctly
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
                    extended = gdspy.boolean(extended,gdspy.Rectangle((self.center[0]+t+self.g_w/2-l1,self.center[1]+gap+height_right*self.g_h/2+t+gap),(self.center[0]+self.g_w/2+2*gap+t+self.g_t,self.center[1]+gap+height_right*self.g_h/2+t+self.g_t+gap)),'or')
                    extended = gdspy.boolean(extended,gdspy.Rectangle((self.center[0]+self.g_w/2+2*gap+t,self.center[1]+gap+height_right*self.g_h/2+t+gap),(self.center[0]+self.g_w/2+2*gap+t+self.g_t,self.center[1]+5+gap)), 'or')
                    #lower
                    if l2_check:
                        extended = gdspy.boolean(extended,gdspy.Rectangle((self.center[0]+t+self.g_w/2-l2-self.g_t,self.center[1]-height_right*self.g_h/2),(self.center[0]+t+self.g_w/2-l2,self.center[1]-gap-height_right*self.g_h/2-t-self.g_t-gap)), 'or')
                    extended = gdspy.boolean(extended,gdspy.Rectangle((self.center[0]+t+self.g_w/2-l2,self.center[1]-gap-height_right*self.g_h/2-t-gap),(self.center[0]+self.g_w/2+2*gap+t+self.g_t,self.center[1]-gap-height_right*self.g_h/2-t-self.g_t-gap)),'or')
                    extended = gdspy.boolean(extended,gdspy.Rectangle((self.center[0]+self.g_w/2+2*gap+t,self.center[1]-gap-height_right*self.g_h/2-t-gap),(self.center[0]+self.g_w/2+2*gap+t+self.g_t,self.center[1]-5-gap)), 'or')
                    extended.translate(-coupler.sctq, 0)

                    # remove/add missing overhang:
                    # add missing piece
                    if coupler.sctq < 0:
                        extended = gdspy.boolean(extended, gdspy.Rectangle((self.center[0] + self.g_w / 2, self.center[
                            1] + gap + height_right * self.g_h / 2 + t + self.g_t + gap), (
                                                                           self.center[0] + self.g_w / 2 - coupler.sctq,
                                                                           self.center[
                                                                               1] - gap - height_right * self.g_h / 2 - t - self.g_t - gap)),
                                                 'or')

                    # remove additional pieces
                    if coupler.sctq > self.g_t:
                        extended = gdspy.boolean(extended, gdspy.Rectangle(
                            (self.center[0] + self.g_t - self.g_w / 2, self.center[1] + self.g_t - self.g_h / 2),
                            (self.center[0] - self.g_t + self.g_w / 2, self.center[1] - self.g_t + self.g_h / 2)),
                                                 'not')

                    result = gdspy.boolean(result,extended,'or')
                    # box for inverted polygon
                    box = gdspy.boolean(box, gdspy.Rectangle((self.center[0] + self.g_w / 2 + self.g_t + 2 * gap + t,
                                                              self.center[
                                                                  1] - height_right * self.g_h / 2 - self.g_t - 2 * gap - t),
                                                             (self.center[0] + self.g_w / 2 - l1 + t, self.center[
                                                                 1] + height_right * self.g_h / 2 + self.g_t + 2 * gap + t)).translate(-coupler.sctq,0),
                                        'or', layer=self.layer_configuration.inverted)

                    pocket = gdspy.boolean(pocket, gdspy.Rectangle((self.center[
                                                                        0] + self.g_w / 2 + self.g_t + 2 * gap + t,self.center[1] - height_right * self.g_h / 2 - self.g_t - 2 * gap - t),(self.center[0] + self.g_w / 2 - l1 + t, self.center[1] + height_right * self.g_h / 2 + self.g_t + 2 * gap + t)).translate(-coupler.sctq, 0), 'or')

                if side =='left':
                    #upper
                    extended = gdspy.Rectangle((0,0),(0,0)) # empty object
                    if l1_check:
                        extended = gdspy.boolean(extended,gdspy.Rectangle((self.center[0]-self.g_w/2+l1+self.g_t-t,self.center[1]+height_left*self.g_h/2),(self.center[0]-t-self.g_w/2+l1,self.center[1]+gap+height_left*self.g_h/2+t+self.g_t+gap)),'or')

                    extended = gdspy.boolean(extended,gdspy.Rectangle((self.center[0]-t-self.g_w/2+l1,self.center[1]+gap+height_left*self.g_h/2+t+gap),(self.center[0]-self.g_w/2-2*gap-t-self.g_t,self.center[1]+gap+height_left*self.g_h/2+t+self.g_t+gap)),'or')
                    extended = gdspy.boolean(extended,gdspy.Rectangle((self.center[0]-self.g_w/2-2*gap-t,self.center[1]+gap+height_left*self.g_h/2+t+gap),(self.center[0]-self.g_w/2-2*gap-t-self.g_t,self.center[1]+5+gap)), 'or')
                    #lower
                    if l2_check:
                        extended = gdspy.boolean(extended,gdspy.Rectangle((self.center[0]-t-self.g_w/2+l2+self.g_t,self.center[1]-height_left*self.g_h/2),(self.center[0]-t-self.g_w/2+l2,self.center[1]-gap-height_left*self.g_h/2-t-self.g_t-gap)), 'or')
                    extended = gdspy.boolean(extended,gdspy.Rectangle((self.center[0]-t-self.g_w/2+l2,self.center[1]-gap-height_left*self.g_h/2-t-gap),(self.center[0]-self.g_w/2-2*gap-t-self.g_t,self.center[1]-gap-height_left*self.g_h/2-t-self.g_t-gap)),'or')
                    extended = gdspy.boolean(extended,gdspy.Rectangle((self.center[0]-self.g_w/2-2*gap-t,self.center[1]-gap-height_left*self.g_h/2-t-gap),(self.center[0]-self.g_w/2-2*gap-t-self.g_t,self.center[1]-5-gap)), 'or')
                    extended.translate(+coupler.sctq, 0)
                    # remove/add missing overhang:
                    # add missing piece
                    if coupler.sctq < 0:
                        extended = gdspy.boolean(extended, gdspy.Rectangle((self.center[0] - self.g_w / 2, self.center[
                            1] + gap + height_left * self.g_h / 2 + t + self.g_t + gap), (
                                                                           self.center[0] - self.g_w / 2 + coupler.sctq,
                                                                           self.center[
                                                                               1] - gap - height_left * self.g_h / 2 - t - self.g_t - gap)),
                                                 'or')

                    # remove additional pieces
                    if coupler.sctq > self.g_t:
                        extended = gdspy.boolean(extended, gdspy.Rectangle(
                            (self.center[0] - self.g_t + self.g_w / 2, self.center[1] + self.g_t - self.g_h / 2),
                            (self.center[0] + self.g_t - self.g_w / 2, self.center[1] - self.g_t + self.g_h / 2)),
                                                 'not')

                    result = gdspy.boolean(result,extended,'or')

                    #box for inverted polygon
                    box = gdspy.boolean(box, gdspy.Rectangle((self.center[0] - self.g_w / 2 - self.g_t - 2 * gap - t,self.center[1] - height_left * self.g_h / 2 - self.g_t - 2 * gap - t),(self.center[0] - self.g_w / 2 + l1 - t, self.center[1] + height_left * self.g_h / 2 + self.g_t + 2 * gap + t)).translate(+coupler.sctq,0),'or', layer=self.layer_configuration.inverted)
                    pocket = gdspy.boolean(pocket,box,'or')

                if side == 'top':
                    extended = gdspy.Rectangle((self.center[0]-self.g_w/2+l1-gap-self.g_t,self.center[1]+self.g_h/2),(self.center[0]-self.g_w/2+l1-gap,self.center[1]+self.g_h/2+t+gap+gap))
                    extended = gdspy.boolean(extended,gdspy.Rectangle((self.center[0]-self.g_w/2+l1-gap-self.g_t,self.center[1]+self.g_h/2+t+gap+gap),(self.center[0]-self.g_w/2+l1-gap+l2/2-5,self.center[1]+self.g_h/2+t+gap+gap+self.g_t)),'or')
                    extended = gdspy.boolean(extended, gdspy.Rectangle((self.center[0]-self.g_w/2+l1+l2+gap,self.center[1]+self.g_h/2),(self.center[0]-self.g_w/2+l1+l2+self.g_t+gap,self.center[1]+self.g_h/2+t+gap+gap)), 'or')
                    extended = gdspy.boolean(extended, gdspy.Rectangle((self.center[0]-self.g_w/2+l1+l2+self.g_t+gap,self.center[1] + self.g_h / 2 + t + gap + gap),(self.center[0]-self.g_w/2+l1+l2/2+gap+5,self.center[1] + self.g_h / 2 + t + gap + gap + self.g_t)),'or')
                    extended.translate(0, -coupler.sctq)
                    # remove/add missing overhang:
                    # add missing piece
                    if coupler.sctq < 0:
                        extended = gdspy.boolean(extended, gdspy.Rectangle(
                            (self.center[0] - self.g_w / 2 + l1 - gap - self.g_t, self.center[1] + self.g_h / 2), (
                            self.center[0] - self.g_w / 2 + l1 - gap + l2 / 2,
                            self.center[1] + self.g_h / 2 + coupler.sctq)), 'or')

                    # remove additional pieces
                    if coupler.sctq > self.g_t:
                        extended = gdspy.boolean(extended, gdspy.Rectangle(
                            (self.center[0] - self.g_t + self.g_w / 2, self.center[1] + self.g_t - self.g_h / 2),
                            (self.center[0] + self.g_t - self.g_w / 2, self.center[1] - self.g_t + self.g_h / 2)),
                                                 'not')

                    result = gdspy.boolean(result,extended,'or')

                    # box for inverted polygon
                    box = gdspy.boolean(box,gdspy.Rectangle((self.center[0]-self.g_w/2+l1-gap-self.g_t,self.center[1]+self.g_h/2),(self.center[0]-self.g_w/2+l1+l2+self.g_t+gap,self.center[1] + self.g_h / 2 + t + gap + gap + self.g_t)).translate(0,-coupler.sctq),'or', layer=self.layer_configuration.inverted)
                    pocket = gdspy.boolean(pocket, gdspy.Rectangle(
                        (self.center[0] - self.g_w / 2 + l1 - gap - self.g_t, self.center[1] + self.g_h / 2), (
                            self.center[0] - self.g_w / 2 + l1 + l2 + self.g_t + gap,
                            self.center[1] + self.g_h / 2 + t + gap + gap + self.g_t)).translate(0, -coupler.sctq),
                                           'or',
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
                            (self.center[0] - self.g_w / 2 + l1 - gap - self.g_t, self.center[1] - self.g_h / 2), (
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
                    pocket = gdspy.boolean(pocket, gdspy.Rectangle(
                        (self.center[0] - self.g_w / 2 + l1 - gap - self.g_t, self.center[1] - self.g_h / 2), (
                        self.center[0] - self.g_w / 2 + l1 + l2 + self.g_t + gap,
                        self.center[1] - self.g_h / 2 - t - gap - gap - self.g_t)).translate(0, coupler.sctq), 'or')

                result_restricted = gdspy.boolean(pocket, result_restricted, 'or',
                                                  layer=self.layer_configuration.restricted_area_layer)
                if coupler.coupler_type == 'coupler':
                        qubit_cap_parts.append(gdspy.boolean(coupler.result_coupler, coupler.result_coupler, 'or',layer=10+id+self.secret_shift))
                        self.layers.append(10+id+self.secret_shift)
                        last_step_cap.append(coupler.result_coupler)
        qubit_cap_parts.append(gdspy.boolean(result,last_step_cap,'not'))

        inverted = gdspy.boolean(box, result, 'not',layer=self.layer_configuration.inverted)
        if self.fluxline_params != {} and f['inverted_extension'] is not None:
            inverted = gdspy.boolean(inverted, inverted_flux, 'not',layer=self.layer_configuration.inverted)

        if not self.return_inverted:
            inverted = gdspy.Rectangle((0,0),(0,0))


        qubit = deepcopy(result)

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
                    'airbridges': Air.mirror(self.transformations['mirror'][0], self.transformations['mirror'][1])if Air is not None else None,
                    'pocket': pocket.mirror(self.transformations['mirror'][0], self.transformations['mirror'][1]),
                    'bandages': bandages.mirror(self.transformations['mirror'][0], self.transformations['mirror'][1]),
                    }

        if 'rotate' in self.transformations:
            return {'positive': result.rotate(self.transformations['rotate'][0], self.transformations['rotate'][1]),
                    'restrict': result_restricted.rotate(self.transformations['rotate'][0], self.transformations['rotate'][1]),
                    'qubit': qubit.rotate(self.transformations['rotate'][0],
                                          self.transformations['rotate'][1]) if qubit is not None else None,
                    'qubit_cap': qubit_cap_parts,
                    'JJ': JJ.rotate(self.transformations['rotate'][0], self.transformations['rotate'][1]),
                    'inverted': inverted.rotate(self.transformations['rotate'][0], self.transformations['rotate'][1]),
                    'airbridges':Air.rotate(self.transformations['rotate'][0], self.transformations['rotate'][1])if Air is not None else None,
                    'pocket': pocket.rotate(self.transformations['rotate'][0], self.transformations['rotate'][1]),
                    'bandages': bandages.rotate(self.transformations['rotate'][0], self.transformations['rotate'][1]),
                    }
        elif self.transformations == {}:
            return {'positive': result,
                    'restrict': result_restricted,
                    'qubit': qubit,
                    'qubit_cap': qubit_cap_parts,
                    'JJ': JJ,
                    'inverted': inverted,
                    'airbridges': Air if Air is not None else None,
                    'pocket':pocket,
                    'bandages': bandages,
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
                ground = gdspy.fast_boolean(ground, gdspy.Rectangle((z[0] - x / 2, z[1] - factor * y / 2),
                                                                    (z[0] - x / 2 + t, z[1] + factor * y / 2)), 'not')
            if key == 'right':
                if self.remove_ground[key] != None:
                    factor = self.remove_ground[key]
                ground = gdspy.fast_boolean(ground, gdspy.Rectangle((z[0] + x / 2, z[1] - factor * y / 2),
                                                                    (z[0] + x / 2 - t, z[1] + factor * y / 2)), 'not')
            if key == 'top':
                if self.remove_ground[key] != None:
                    factor = self.remove_ground[key]
                ground = gdspy.fast_boolean(ground, gdspy.Rectangle((z[0] - factor * x / 2, z[1] + y / 2),
                                                                    (z[0] + factor * x / 2, z[1] + y / 2 - t)), 'not')
            if key == 'bottom':
                if self.remove_ground[key] != None:
                    factor = self.remove_ground[key]
                ground = gdspy.fast_boolean(ground, gdspy.Rectangle((z[0] - factor * x / 2, z[1] - y / 2),
                                                                    (z[0] + factor * x / 2, z[1] - y / 2 + t)), 'not')

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
                        coupler_phi = 0+coupler.rotation
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
                        coupler_phi =  np.pi+self.transformations['rotate'][0]+coupler.rotation

            if self.transformations == {}:
                coupler_connection = coupler.connection
                if coupler.side == "left":
                    coupler_phi = 0
                if coupler.side == "right":
                    coupler_phi = np.pi
                if coupler.side == "top":
                    coupler_phi = -np.pi/2
                if coupler.side == "fluxline":
                    coupler_phi = np.pi+coupler.rotation
                if coupler.side == "bottom":
                    coupler_phi = np.pi/2


            if coupler.connection is not None:
                self.terminals['coupler'+str(id)] = DesignTerminal(tuple(coupler_connection),
                                                                   coupler_phi, g=coupler.g, s=coupler.s,
                                                                w=coupler.w, type='cpw')
        return True

    def get_terminals(self):
        return self.terminals

    #change for new design
    def generate_JJ(self):
        #parameter to remove overhanging bridge parts
        padding = self.JJ_params['padding']  if ('padding' in self.JJ_params) else 20

        #cheap Manhatten style
        reach1 = self.JJ_params['strip1_extension']#20
        reach2 = self.JJ_params['strip2_extension']#25
        loop_h = self.JJ_params['loop_h']
        if 'loop_w_shift' in self.JJ_params:
            loop_extension = self.JJ_params['loop_w_shift']
        else:
            loop_extension = 0
        #single strip
        strip1 = gdspy.Rectangle((self.center[0]+self.gap/2+self.w-self.b_w,self.center[1]+self.h/2+self.JJ_params['a1']/2+self.asymmetry+3*self.b_w-self.b_w/2),(self.center[0]+self.gap/2+self.w-self.b_w+reach1,self.center[1]+self.h/2-self.b_w/2-self.JJ_params['a1']/2+self.asymmetry+3*self.b_w))
        #pad
        pad1 = gdspy.Rectangle((self.center[0]+self.gap/2+self.w-self.b_w,self.center[1]+self.h/2+0.45+self.asymmetry+3*self.b_w-self.b_w/2),(self.center[0]+self.gap/2+self.w-self.b_w-self.JJ_params['h_d']+0.5,self.center[1]+self.h/2-self.b_w/2-0.45+self.asymmetry+3*self.b_w))
        strip1.translate(0,loop_extension)
        pad1.translate(0,loop_extension)

        #double strip
        strip2 = gdspy.Rectangle((self.center[0]+self.gap/2+self.w-self.b_w/2-loop_h/2-self.JJ_params['a2']/2,self.center[1]+self.h/2+self.asymmetry+2*self.b_w-self.b_g),(self.center[0]+self.gap/2+self.w-self.b_w/2+self.JJ_params['a2']/2-loop_h/2,self.center[1]+self.h/2+self.asymmetry+2*self.b_w-self.b_g+reach2))
        strip3 = gdspy.copy(strip2, +loop_h,0)

        #pad2
        pad2 = gdspy.Rectangle((self.center[0] + self.gap / 2 + self.w - self.b_w / 2 - loop_h / 2-0.45,self.center[1] + self.h / 2 + self.asymmetry + 2 * self.b_w - self.b_g-self.JJ_params['h_d']+0.5),
                                                       (self.center[0] + self.gap / 2 + self.w - self.b_w / 2 + 0.45- loop_h / 2, self.center[1] + self.h / 2 + self.asymmetry + 2 * self.b_w - self.b_g))
        #pad3
        pad3   = gdspy.copy(pad2,+loop_h,0)

        result = gdspy.boolean(strip1,[strip2,strip3,pad1,pad2,pad3],'or', layer=self.layer_configuration.jj_layer)
        #placing the junctions in the correct layer
        result = gdspy.boolean(result, result, 'or', layer=self.layer_configuration.jj_layer)

        #add bandages
        if self.use_bandages:
            b_ex = self.JJ_params['bandages_extension']
            c_p_w = self.JJ_params['connection_pad_width']
            c_p_g = self.JJ_params['connection_pad_gap']

            bandage1 = gdspy.Rectangle((self.center[0] + self.gap / 2 + self.w - self.b_w,
                             self.center[1] + self.h / 2 - c_p_w/2 + self.asymmetry + 3 * self.b_w - self.b_w / 2), (
                            self.center[0] + self.gap / 2 + self.w - self.b_w - self.JJ_params['h_d'] + 0.5,
                            self.center[1] + self.h / 2 - self.b_w / 2 + c_p_w/2 + self.asymmetry + 3 * self.b_w+b_ex))
            bandage1.translate(0,loop_extension)
            bandage2 = gdspy.Rectangle((self.center[0] + self.gap / 2 + self.w - self.b_w / 2 - loop_h / 2 - c_p_w/2,
                             self.center[1] + self.h / 2 + self.asymmetry + 2 * self.b_w - self.b_g - self.JJ_params[
                                 'h_d'] + 0.5),
                            (self.center[0] + self.gap / 2 + self.w - self.b_w / 2 +c_p_w/2 - loop_h / 2+b_ex,
                             self.center[1] + self.h / 2 + self.asymmetry + 2 * self.b_w - self.b_g))

            bandage3 = gdspy.copy(bandage2, +self.JJ_params['loop_h'] - b_ex, 0)
            bandages = gdspy.boolean(bandage1, (bandage2, bandage3), 'or',
                                     layer=self.layer_configuration.bandages_layer)
        #create rectangles for holes
        hole1 = gdspy.Rectangle((self.center[0] + self.gap / 2 + self.w - self.b_w,
                             self.center[1] + self.h / 2 - c_p_w/2 + self.asymmetry + 2 * self.b_w), (
                            self.center[0] + self.gap / 2 + self.w - self.b_w - self.JJ_params['h_d'] + 0.5-c_p_g,
                            self.center[1] + self.h / 2 + self.b_w / 2 + c_p_w/2 + self.asymmetry + 2 * self.b_w+c_p_g))
        if 'rotation' in self.JJ_params:
            hole1 = gdspy.Rectangle((self.center[0] + self.gap / 2 + self.w - self.b_w,
                                     self.center[1] + self.h / 2 - c_p_w / 2 + self.asymmetry + 2 * self.b_w-padding), (
                                        self.center[0] + self.gap / 2 + self.w - self.b_w - self.JJ_params['h_d'] + 0.5 - c_p_g,
                                        self.center[1] + self.h / 2 + self.b_w / 2 + c_p_w / 2 + self.asymmetry + 2 * self.b_w + c_p_g))

        hole1.translate(0,loop_extension)
        hole2 = gdspy.Rectangle((self.center[0] + self.gap / 2 + self.w - self.b_w-padding,
                             self.center[1] + self.h / 2 + self.asymmetry + 2 * self.b_w - self.b_g - self.JJ_params[
                                 'h_d'] + 0.5-c_p_g),
                            (self.center[0] + self.gap / 2 + self.w - self.b_w / 2 + c_p_w/2 - loop_h / 2+c_p_g,
                             self.center[1] + self.h / 2 + self.asymmetry + 2 * self.b_w - self.b_g))
        hole3 = gdspy.Rectangle((self.center[0] + self.gap / 2 + self.w+padding,
                             self.center[1] + self.h / 2 + self.asymmetry + 2 * self.b_w - self.b_g - self.JJ_params[
                                 'h_d'] + 0.5-c_p_g),
                            (self.center[0] + self.gap / 2 + self.w - self.b_w / 2 - c_p_w/2 + loop_h / 2-c_p_g,
                             self.center[1] + self.h / 2 + self.asymmetry + 2 * self.b_w - self.b_g))
        hole4 = gdspy.Rectangle((self.center[0]+self.gap/2+self.w-self.b_w,self.center[1] + self.h / 2 + self.asymmetry + 2 * self.b_w - self.b_g),(self.center[0]+self.gap/2+self.w-self.b_w+reach1,self.center[1] + self.h / 2 + self.asymmetry + 2 * self.b_w - self.b_g + reach2+loop_extension))


        holes =  gdspy.boolean(hole1, (hole2, hole3,hole4), 'or')

        #rotate all
        if 'rotation' in self.JJ_params:
            point =  (self.center[0] + self.gap / 2 + self.w- self.b_w/2,self.center[1] + self.h / 2 + self.asymmetry + 2 * self.b_w - self.b_g +self.JJ_params['loop_h']/2)
            result   = result.rotate(self.JJ_params['rotation'],point)
            bandages = bandages.rotate(self.JJ_params['rotation'], point)
            holes    = holes.rotate(self.JJ_params['rotation'], point)

        if 'translate' in self.JJ_params:
            dx = self.JJ_params['translate'][0]
            dy = self.JJ_params['translate'][1]
            result = result.translate(dx,dy)
            bandages = bandages.translate(dx,dy)
            holes = holes.translate(dx,dy)

        return result,bandages,holes



    def add_to_tls(self, tls_instance: tlsim.TLSystem, terminal_mapping: dict,
                   track_changes: bool = True, cutoff: float = np.inf, epsilon: float = 11.45) -> list:
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
            # elif coupler.coupler_type =='grounded':
            #     tls_instance.add_element(tlsim.Short(), [terminal_mapping['flux line'], 0])

        if track_changes:
            self.tls_cache.append([JJ, C]+mut_cap+cap_g)
        return [JJ, C]+mut_cap+cap_g


class PP_Squid_Coupler:
    """
    This class represents a coupler for a PP_Squid, note that the top position is reserved for the fluxline.
    There are several parameters:
    1) l1 - length of the upper claw finger
    2) l2 - length of the lower claw finger
    3) t  - thickness of the coupler
    4) gap - gap between ground and coupler
    5) side - which side the coupler is on
    6) heightl / heightr - height as a fraction of total length
    """
    def __init__(self, l1,l2,t,side = 'left',coupler_type = 'none',heightl = 1,heightr=1,w= None, g=None, s=None):
        self.l1 = l1
        self.l2 = l2
        self.t = t
        self.gap = s
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

    def render(self, center, g_w,g_h):
        result = 0
        if self.side == "left":
            result = gdspy.Rectangle((center[0]-g_w/2-self.t-self.gap,center[1]-self.height_left*g_h/2-self.gap),(center[0]-g_w/2-self.gap,center[1]+self.height_left*g_h/2+self.gap))
            if self.height_left == 1:
                upper  = gdspy.Rectangle((center[0]-g_w/2-self.t-self.gap,center[1]+g_h/2+self.gap),(center[0]-g_w/2+self.l1-self.gap-self.t,center[1]+g_h/2+self.t+self.gap))
                lower  = gdspy.Rectangle((center[0]-g_w/2-self.t-self.gap,center[1]-g_h/2-self.gap-self.t),(center[0]-g_w/2+self.l2-self.gap-self.t,center[1]-g_h/2-self.gap))
            line   = gdspy.Rectangle((center[0]-g_w/2-self.t-self.gap-self.gap-self.ground_t,center[1]-self.w/2),(center[0]-g_w/2-self.t-self.gap,center[1]+self.w/2))#modified here ;), remove ground_t
            if self.height_left ==1:
                result = gdspy.boolean(result, upper, 'or')
                result = gdspy.boolean(result, lower, 'or')
            result = gdspy.boolean(result, line, 'or')

            self.connection = (center[0]-g_w/2-self.t-self.gap-self.gap,center[1])


        if self.side == "right":
            result = gdspy.Rectangle((center[0]+g_w/2+self.t+self.gap,center[1]-self.height_right*g_h/2-self.gap),(center[0]+g_w/2+self.gap,center[1]+self.height_right*g_h/2+self.gap))
            if self.height_right == 1:
                upper  = gdspy.Rectangle((center[0]+g_w/2+self.t+self.gap,center[1]+g_h/2+self.gap),(center[0]+g_w/2-self.l1+self.gap+self.t,center[1]+g_h/2+self.t+self.gap))
                lower  = gdspy.Rectangle((center[0]+g_w/2+self.t+self.gap,center[1]-g_h/2-self.gap-self.t),(center[0]+g_w/2-self.l2+self.gap+self.t,center[1]-g_h/2-self.gap))

            line = gdspy.Rectangle((center[0]+g_w/2+self.t+self.gap+self.gap,center[1]-self.w/2),(center[0]+g_w/2+self.t+self.gap,center[1]+self.w/2))
            if self.height_right == 1:
                result = gdspy.boolean(result, upper, 'or')
                result = gdspy.boolean(result, lower, 'or')
            result = gdspy.boolean(result, line, 'or')

            self.connection = (center[0] + g_w / 2 + self.t + self.gap + self.gap, center[1] )

        if self.side == "top":
            result = gdspy.Rectangle((center[0]-g_w/2+self.l1,center[1]+g_h/2+self.gap),(center[0]-g_w/2+self.l1+self.l2,center[1]+g_h/2+self.gap+self.t))
            line   = gdspy.Rectangle((center[0]-g_w/2+self.l1+self.l2/2-self.w/2,center[1]+g_h/2+self.gap+self.t),(center[0]-g_w/2+self.l1+self.l2/2+self.w/2,center[1]+g_h/2+self.gap+self.t+self.gap))
            result = gdspy.boolean(result, line, 'or')
            self.connection = (center[0]-g_w/2+self.l1+self.l2/2, center[1]+g_h/2+self.gap+self.t+self.gap)

        if self.side == "bottom":
            result = gdspy.Rectangle((center[0]-g_w/2+self.l1,center[1]-g_h/2-self.gap),(center[0]-g_w/2+self.l1+self.l2,center[1]-g_h/2-self.gap-self.t))
            line   = gdspy.Rectangle((center[0]-g_w/2+self.l1+self.l2/2-self.w/2,center[1]-g_h/2-self.gap-self.t),(center[0]-g_w/2+self.l1+self.l2/2+self.w/2,center[1]-g_h/2-self.gap-self.t-self.gap))
            result = gdspy.boolean(result, line, 'or')
            self.connection = (center[0]-g_w/2+self.l1+self.l2/2, center[1]-g_h/2-self.gap-self.t-self.gap)

        self.result_coupler = result

        return {
            'positive': result
                        }


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
    def __init__(self, l,t_m,t_r,gap,l_arm,h_arm,s_gap,flux_distance,pad_w,pad_h,pad_g,b_w,b_g,ground,asymmetry = 0,w= None, g=None, s=None,extend = None,rotation = 0):
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
        self.w = t_m
        self.s = s
        self.extend = extend
        #pad parameters
        self.pad_w = pad_w
        self.pad_h = pad_h
        self.b_w = b_w
        self.b_g = b_g
        self.pad_g = pad_g
        self.ground = ground
        self.rotation = rotation
        print(self.rotation)
    def render(self, center, width,height,ground_height,ground_t,inverted_extension = -1):
        g_t = ground_t
        g_h = ground_height
        if not self.extend:
            ground_t = ground_height/2

        start  = [0,0]#[self.l_arm/2,self.t_r+self.l+height/2+self.gap+self.asymmetry]
        points = [start+[0,0],start+[self.t_m,0],start+[self.t_m,-self.l],start+[self.t_m+self.l_arm,-self.l]]
        points.append(start+[self.t_m+self.s_gap,-self.l+self.h_arm])
        points.append(start+[self.t_m+self.s_gap,0])
        points.append(start + [self.t_m + self.s_gap+self.t_r, 0])
        points.append(start + [self.t_m + self.s_gap + self.t_r, -self.l+self.h_arm])
        points.append(start + [self.t_m + self.l_arm+ self.t_r, -self.l])
        points.append(start + [self.t_m + self.l_arm+ self.t_r, -self.l-self.t_r])
        points.append(start + [- self.l_arm- self.t_r, -self.l-self.t_r])
        points.append(start + [- self.l_arm - self.t_r, -self.l])
        points.append(start + [-self.t_r-self.s_gap, -self.l+self.h_arm])
        points.append(start + [-self.t_r - self.s_gap, 0])
        points.append(start + [- self.s_gap, 0])
        points.append(start + [- self.s_gap, -self.l+self.h_arm])
        points.append(start + [- self.l_arm, -self.l])
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
                (-2*self.l_arm, center[1] + ground_height / 2 ),
                (+2*self.l_arm, center[1] + ground_height / 2 +100))

        result = gdspy.boolean(result,cutout1,'not')

        result = gdspy.boolean(result,cutout2,'not')

        result.translate(-self.t_m/2,+self.l+self.t_r)
        result.rotate(-np.pi/2)

        restrict.translate(-self.t_m/2,+self.l+self.t_r)
        restrict.rotate(-np.pi / 2)

        # move fluxline to correct position
        result.translate(center[0], center[1])
        result.translate(self.pad_g/2+self.pad_w-self.b_w/2+self.flux_distance,self.asymmetry+self.pad_h/2+3.5*self.b_w)

        restrict.translate(center[0], center[1])
        restrict.translate(self.pad_g/2+self.pad_w-self.b_w/2+self.flux_distance,self.asymmetry+self.pad_h/2+3.5*self.b_w)

        #adding inverse to inverted layer
        inverted = gdspy.Rectangle((points[0][0],points[0][1]),(points[1][0],points[1][1]+inverted_extension))
        inverted.translate(-self.t_m/2,+self.l+self.t_r)
        inverted.rotate(-np.pi/2)
        inverted.translate(center[0], center[1])
        inverted.translate(self.pad_g/2+self.pad_w-self.b_w/2+self.flux_distance,self.asymmetry+self.pad_h/2+3.5*self.b_w)

        #adding restricted
        restricted  = gdspy.Rectangle((points[0][0]-self.s_gap,points[0][1]),(points[1][0]+self.s_gap,points[1][1]+inverted_extension))
        restricted .translate(-self.t_m/2,+self.l+self.t_r)
        restricted .rotate(-np.pi/2)
        restricted .translate(center[0], center[1])
        restricted .translate(self.pad_g/2+self.pad_w-self.b_w/2+self.flux_distance,self.asymmetry+self.pad_h/2+3.5*self.b_w)
        restrict = gdspy.boolean(restrict,restricted,'or')

        #cuttng off the hanging rest:
        if self.ground != None:
            result = gdspy.boolean(result,self.ground,'and')

        self.result_coupler = result

        point = (center[0]+self.pad_g/2+self.pad_w-self.b_w/2+self.flux_distance+self.t_r+self.l,center[1]+self.asymmetry+self.pad_h/2+3.5*self.b_w)

        self.connection =point

        remove1 = gdspy.Rectangle(
            (center[0] + self.t_r / 2, center[1] + g_h / 2 - g_t),
            (center[0] + self.l_arm + self.t_m - self.t_r / 2, center[1] + g_h / 2))
        remove2 = gdspy.Rectangle(
            (center[0] - 100, center[1] + g_h / 2),
            (center[0] + 100 + self.l_arm + self.t_m, center[1] + g_h / 2 + 2000))

        if self.rotation != 0 and self.rotation != None :
            self.connection = rotate_point(point, self.rotation, (point[0] - self.l - self.t_r, point[1]))
            return {
                'positive': result.rotate(self.rotation,(point[0]-self.l-self.t_r,point[1])),
                'restricted': restrict.rotate(self.rotation,(point[0]-self.l-self.t_r,point[1])),
                'remove_ground': [remove1.rotate(self.rotation,(point[0]-self.l-self.t_r,point[1])), remove2.rotate(self.rotation,(point[0]-self.l-self.t_r,point[1]))],
                'inverted': inverted.rotate(self.rotation, (point[0] - self.l - self.t_r, point[1])),
            }

        return {
            'positive': result,
            'restricted':restrict,
            'remove_ground':[remove1,remove2],
            'inverted':inverted,
                        }



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
