from .core import DesignElement, LayerConfiguration, DesignTerminal
from .. import conformal_mapping as cm
from .. import transmission_line_simulator as tlsim
import gdspy
import numpy as np
from typing import List, Tuple, Mapping, Dict
from . import squid3JJ #TODO make this qubit class suitable for any squid types
from . import JJ4q
from copy import deepcopy


class Y_Squid_C(DesignElement):
    """
    Y-Transmon (asymmetric pad size) consists of several parts:
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
    def __init__(self, name: str, center: Tuple[float, float],width: Tuple[float,float], height: Tuple[float,float,float],gap: float,bridge_gap:float,bridge_w:float, ground_w: float, ground_h: float,ground_t: float,layer_configuration: LayerConfiguration,
                 jj_params: Dict,Couplers,transformations:Dict,fluxline_params= {},remove_ground = {},secret_shift = 0,calculate_capacitance = False,shoes = {},claw = [],asymmetry = 0,asymmetry_coupler=0,air_bridge=[],y_gap=15,return_inverted = True):
        super().__init__(type='qubit', name=name)
        #qubit parameters
        self.transformations = transformations# to mirror the structure
        self.center = center
        self.w_pads = width
        self.h_pads = height
        if len(self.h_pads) !=3:
            self.h_pads = [self.h_pads[0],self.h_pads[1],self.h_pads[0]]

        self.w = width[1]
        self.h = height[1]
        self.shoes = shoes
        self.claw = claw
        self.asymmetry = asymmetry # to shift the right smaller pad
        self.asymmetry_coupler = asymmetry_coupler
        self.gap = gap
        self.g_w = ground_w
        self.g_h = ground_h
        self.g_t = ground_t
        self.b_g = bridge_gap
        self.b_w = bridge_w
        self.air = air_bridge #first parameter is the shift,secnd parameter is thickness, third parameter is the span length of the airbridge
        #layers
        self.y_gap = y_gap
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


        # remove ground on these sites
        self.remove_ground = remove_ground
        self.return_inverted = return_inverted

        self.secret_shift = secret_shift

        self.calculate_capacitance = calculate_capacitance

    def render(self):
        """
        This function draws everything: qubit,ground,couplers
        """
        qubit_cap_parts=[]
        ground = self.generate_ground()
        # restricted area for a future grid lines
        result_restricted = gdspy.Rectangle((self.center[0]-self.g_w/2,self.center[1]-self.g_h/2),(self.center[0]+self.g_w/2,self.center[1]),layer=self.layer_configuration.restricted_area_layer)

        left_restricted   = gdspy.Rectangle((self.center[0]+self.y_gap-self.gap/2,self.center[1]-self.w_pads[0]),(self.center[0]-self.gap/2-self.w_pads[0]-self.y_gap,self.center[1]+self.h_pads[2]/2+100)).rotate(+np.pi / 4,(self.center[0]-self.gap/2-self.w_pads[0],self.center[1]))
        right_restricted  = gdspy.Rectangle((self.center[0]+self.y_gap-self.gap/2,self.center[1]-self.w_pads[0]),(self.center[0]-self.gap/2-self.w_pads[0]-self.y_gap,self.center[1]+self.h_pads[2]/2+100)).rotate(-np.pi / 4,(self.center[0] - self.gap / 2 ,self.center[1]))

        result_restricted = gdspy.boolean(result_restricted,left_restricted,'or',layer=self.layer_configuration.restricted_area_layer)
        result_restricted = gdspy.boolean(result_restricted, right_restricted,'or',layer=self.layer_configuration.restricted_area_layer)



        P1 = gdspy.Rectangle((self.center[0]-self.gap/2-self.w_pads[0],self.center[1]-self.h_pads[0]/2),(self.center[0]-self.gap/2,self.center[1]+self.w_pads[0]/2))

        #adding the lower claws on P1:
        if self.claw != []:
            i = self.claw
            #lower
            P1 = gdspy.boolean(P1, gdspy.Rectangle((self.center[0]-self.gap/2+i[0]-self.w_pads[0],self.center[1]-self.h_pads[0]/2+i[1]),(self.center[0]-self.gap/2-self.w_pads[0],self.center[1]-self.h_pads[0]/2-i[1])),'or')
            P1 = gdspy.boolean(P1, gdspy.Rectangle((self.center[0]-self.gap/2-i[0],self.center[1]-self.h_pads[0]/2+i[1]),(self.center[0]-self.gap/2,self.center[1]-self.h_pads[0]/2-i[1])),'or')

        #adding upper half
        P11 = gdspy.Rectangle((self.center[0] - self.gap / 2 - self.w_pads[0], self.center[1] + self.h_pads[2] / 2),
                             (self.center[0] - self.gap / 2, self.center[1]))
        if self.claw != []:
            i = self.claw
            # upper
            P11 = gdspy.boolean(P11, gdspy.Rectangle(
                (self.center[0] - self.gap / 2 + i[0] - self.w_pads[0], self.center[1] + self.h_pads[2] / 2 - i[1]),
                (self.center[0] - self.gap / 2 - self.w_pads[0], self.center[1] + self.h_pads[2] / 2 + i[1])), 'or')

            P11 = gdspy.boolean(P11, gdspy.Rectangle(
                (self.center[0] - self.gap / 2 - i[0], self.center[1] + self.h_pads[2] / 2 + i[1]),
                (self.center[0] - self.gap / 2, self.center[1] + self.h_pads[2] / 2 - i[1])), 'or')

        left_y = gdspy.copy(P11)
        left_y.rotate(+np.pi / 4,(self.center[0]-self.gap/2-self.w_pads[0],self.center[1]))
        right_y = gdspy.copy(P11)
        right_y.rotate(-np.pi / 4,(self.center[0] - self.gap / 2 ,self.center[1]))

        P1 = gdspy.boolean(P1,left_y,'or')
        P1 = gdspy.boolean(P1, right_y, 'or')


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

        self.layers.append(9)
        result = gdspy.boolean(ground, P1, 'or', layer=self.layer_configuration.total_layer)
        result = gdspy.boolean(result, P2, 'or', layer=self.layer_configuration.total_layer)


        P1_bridge = gdspy.Rectangle((self.center[0]-self.gap/2,self.center[1]+self.h/2+self.asymmetry+3*self.b_w),(self.center[0]+self.gap/2+self.w-self.b_w,self.center[1]+self.h/2+self.asymmetry+2*self.b_w))

        P2_bridge = gdspy.Rectangle((self.center[0] + self.gap / 2+self.w, self.center[1]+self.h/2+self.asymmetry),(self.center[0] + self.gap / 2+self.w-self.b_w, self.center[1]+self.h/2+self.asymmetry+2*self.b_w-self.b_g))



        qubit_cap_parts.append(gdspy.boolean(P1, P1_bridge, 'or', layer=8+self.secret_shift))
        qubit_cap_parts.append(gdspy.boolean(P2, P2_bridge, 'or', layer=9+self.secret_shift))

        result = gdspy.boolean(result, P1_bridge, 'or', layer=self.layer_configuration.total_layer)
        result = gdspy.boolean(result, P2_bridge, 'or', layer=self.layer_configuration.total_layer)
        self.layers.append(self.layer_configuration.total_layer)

        #change orientation of fluxline
        if self.fluxline_params != {}:
            f = self.fluxline_params
            l, t_m, t_r, gap, l_arm, h_arm, s_gap = f['l'],f['t_m'],f['t_r'],f['gap'],f['l_arm'],f['h_arm'],f['s_gap']
            flux_distance = f['flux_distance']
            #result_restricted, to ct off hanging parts from the fluxline, None for no cutoff
            flux = PP_Squid_Fluxline(l, t_m, t_r, gap, l_arm, h_arm, s_gap,flux_distance,self.w,self.h,self.gap,self.b_w,self.b_g,ground = None,asymmetry = self.asymmetry,g = f.get('g'),w = f.get('w'),s = f.get('s'),extend = f.get('extend_to_ground'))

            fluxline = flux.render(self.center, self.w, self.h,self.g_h,self.g_t)['positive']

            r_flux = flux.render(self.center, self.w, self.h,self.g_h,self.g_t)['restricted']
            #adding fluxline restricted area
            result_restricted = gdspy.boolean(result_restricted,r_flux,'or', layer=self.layer_configuration.restricted_area_layer)

            self.couplers.append(flux)
            result = gdspy.boolean(result, fluxline, 'or', layer=self.layer_configuration.total_layer)
            #removing ground where the fluxline is
            ground_fluxline =True
            if ground_fluxline == False:
                result = gdspy.boolean(result, gdspy.Rectangle((self.center[0]-l_arm/2-t_r-self.g_t,self.center[1]+self.h/2+0.01),(self.center[0]+3*l_arm/2+t_r+t_m+self.g_t,self.center[1]+self.h/2)), 'not', layer=self.layer_configuration.total_layer)
            else:
                result = gdspy.boolean(result, gdspy.Rectangle(
                    (self.center[0]+t_r/2 , self.center[1] + self.g_h/2-self.g_t),
                    (self.center[0] +  l_arm + t_m-t_r/2, self.center[1] + self.g_h/2 )), 'not',
                                   layer=self.layer_configuration.total_layer)
                result = gdspy.boolean(result, gdspy.Rectangle(
                    (self.center[0]-100 , self.center[1] + self.g_h / 2 ),
                    (self.center[0]+100 +  l_arm + t_m, self.center[1] + self.g_h / 2 +2000)), 'not',
                                   layer=self.layer_configuration.total_layer)


        #adding air bridges
        Air = None

        if self.air != []:
            Air = None
            for A in self.air:
                left_air  = gdspy.Rectangle((self.center[0]-self.gap/2-self.w_pads[0]/2-A[2]/2,self.center[1]+A[0]),(self.center[0]-self.gap/2-self.w_pads[0]/2+A[2]/2,self.center[1]+A[0]+A[1]),self.layer_configuration.airbridges_layer).rotate(+np.pi / 4, (self.center[0] - self.gap / 2 - self.w_pads[0], self.center[1]))
                right_air = gdspy.Rectangle((self.center[0]-self.gap/2-self.w_pads[0]/2-A[2]/2,self.center[1]+A[0]),(self.center[0]-self.gap/2-self.w_pads[0]/2+A[2]/2,self.center[1]+A[0]+A[1]),self.layer_configuration.airbridges_layer).rotate(-np.pi / 4, (self.center[0] - self.gap / 2, self.center[1]))
                Air = gdspy.boolean(Air, left_air, 'or',layer = self.layer_configuration.airbridges_layer)
                Air = gdspy.boolean(Air, right_air, 'or', layer=self.layer_configuration.airbridges_layer)

        # add couplers
        last_step_cap = [gdspy.boolean(gdspy.boolean(P2, P2_bridge, 'or'),gdspy.boolean(P1, P1_bridge, 'or'),'or')]
        self.layers.append(self.layer_configuration.total_layer)

        # Box for inverted Polygons
        box = gdspy.Rectangle((self.center[0] - self.g_w / 2, self.center[1] - self.g_h / 2),(self.center[0] + self.g_w / 2, self.center[1] ))

        box = gdspy.boolean(box,left_restricted,'or')
        box = gdspy.boolean(box,right_restricted,'or')

        #internal pocket for calculating the ground around the structure
        pocket = gdspy.Rectangle((self.center[0] - self.g_w / 2+self.g_t, self.center[1] - self.g_h / 2+self.g_t),(self.center[0] + self.g_w / 2-self.g_t, self.center[1]-self.g_t))

        left_pocket = gdspy.Rectangle((self.center[0]-self.g_t+self.y_gap-self.gap/2, self.center[1]-50), (
        self.center[0] - self.gap/2 - self.w_pads[0]+self.g_t-self.y_gap, self.center[1] + self.h_pads[2] / 2 + 100)).rotate(+np.pi / 4, (
        self.center[0] - self.gap / 2 - self.w_pads[0], self.center[1]))


        right_pocket = gdspy.Rectangle((self.center[0]-self.g_t+self.y_gap-self.gap/2, self.center[1]-50), (
        self.center[0] - self.gap/2 - self.w_pads[0]+self.g_t-self.y_gap, self.center[1] + self.h_pads[2] / 2 + 100)).rotate(-np.pi / 4, (
        self.center[0] - self.gap / 2, self.center[1]))

        pocket = gdspy.boolean(pocket, left_pocket, 'or')
        pocket = gdspy.boolean(pocket, right_pocket, 'or')
        pocket = gdspy.boolean(pocket,gdspy.Rectangle((self.center[0]-self.gap/2+15, self.center[1]-50), (
        self.center[0] - self.gap/2 - self.w_pads[0]-15, self.center[1] + 50)) , 'or')
        if self.fluxline_params != {}:
            pocket = gdspy.boolean(pocket,r_flux,'or', layer=self.layer_configuration.restricted_area_layer)




        if len(self.couplers) != 0:
            center_save = self.center
            self.center = [self.center[0],self.center[1]+self.asymmetry_coupler]
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
                            1] + gap + height_right * self.g_h / 2 + t + self.g_t + gap), (self.center[0] + self.g_w / 2 - coupler.sctq,self.center[1] - gap - height_right * self.g_h / 2 - t - self.g_t - gap)),'or')

                    # remove additional pieces
                    if coupler.sctq > self.g_t:
                        extended = gdspy.boolean(extended, gdspy.Rectangle(
                            (self.center[0] + self.g_t - self.g_w / 2, self.center[1] + self.g_t - self.g_h / 2),
                            (self.center[0] - self.g_t + self.g_w / 2, self.center[1] - self.g_t + self.g_h / 2)),
                                                 'not')

                    result = gdspy.boolean(result, extended, 'or')
                    # box for inverted polygon
                    box = gdspy.boolean(box, gdspy.Rectangle((self.center[0] + self.g_w / 2 + self.g_t + 2 * gap + t,
                                                              self.center[1] - height_right * self.g_h / 2 - self.g_t - 2 * gap - t),(self.center[0] + self.g_w / 2 - l1 + t, self.center[1] + height_right * self.g_h / 2 + self.g_t + 2 * gap + t)).translate(-coupler.sctq, 0),
                                        'or', layer=self.layer_configuration.inverted)
                    pocket = gdspy.boolean(pocket, gdspy.Rectangle((self.center[0] + self.g_w / 2 + self.g_t + 2 * gap + t,self.center[1] - height_right * self.g_h / 2 - self.g_t - 2 * gap - t),(self.center[0] + self.g_w / 2 - l1 + t, self.center[1] + height_right * self.g_h / 2 + self.g_t + 2 * gap + t)).translate(-coupler.sctq, 0), 'or')

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
                                                                               self.center[
                                                                                   0] - self.g_w / 2 + coupler.sctq,
                                                                               self.center[
                                                                                   1] - gap - height_left * self.g_h / 2 - t - self.g_t - gap)),
                                                 'or')

                    # remove additional pieces
                    if coupler.sctq > self.g_t:
                        extended = gdspy.boolean(extended, gdspy.Rectangle(
                            (self.center[0] - self.g_t + self.g_w / 2, self.center[1] + self.g_t - self.g_h / 2),
                            (self.center[0] + self.g_t - self.g_w / 2, self.center[1] - self.g_t + self.g_h / 2)),
                                                 'not')

                    result = gdspy.boolean(result, extended, 'or')

                    # box for inverted polygon
                    box = gdspy.boolean(box, gdspy.Rectangle((self.center[0] - self.g_w / 2 - self.g_t - 2 * gap - t,self.center[1] - height_left * self.g_h / 2 - self.g_t - 2 * gap - t),(self.center[0] - self.g_w / 2 + l1 - t, self.center[1] + height_left * self.g_h / 2 + self.g_t + 2 * gap + t)).translate(+coupler.sctq, 0), 'or', layer=self.layer_configuration.inverted)
                    pocket = gdspy.boolean(pocket,
                                           gdspy.Rectangle((self.center[0] - self.g_w / 2 - self.g_t - 2 * gap - t,self.center[1] - height_left * self.g_h / 2 - self.g_t - 2 * gap - t),(self.center[0] - self.g_w / 2 + l1 - t, self.center[1] + height_left * self.g_h / 2 + self.g_t + 2 * gap + t)).translate(coupler.sctq, 0), 'or')

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
                    pocket = gdspy.boolean(pocket, gdspy.Rectangle((self.center[0] - self.g_w / 2 + l1 - gap - self.g_t, self.center[1] + self.g_h / 2), (self.center[0] - self.g_w / 2 + l1 + l2 + self.g_t + gap,self.center[1] + self.g_h / 2 + t + gap + gap + self.g_t)).translate(0, -coupler.sctq),'or',layer=self.layer_configuration.inverted)

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

                    result = gdspy.boolean(result, extended, 'or')

                    # box for inverted polygon
                    box = gdspy.boolean(box, gdspy.Rectangle(
                        (self.center[0] - self.g_w / 2 + l1 - gap - self.g_t, self.center[1] - self.g_h / 2), (
                        self.center[0] - self.g_w / 2 + l1 + l2 + self.g_t + gap,
                        self.center[1] - self.g_h / 2 - t - gap - gap - self.g_t)).translate(0, +coupler.sctq), 'or',
                                        layer=self.layer_configuration.inverted)
                    pocket = gdspy.boolean(pocket, gdspy.Rectangle((self.center[0] - self.g_w / 2 + l1 - gap - self.g_t, self.center[1] - self.g_h / 2), (self.center[0] - self.g_w / 2 + l1 + l2 + self.g_t + gap,self.center[1] - self.g_h / 2 - t - gap - gap - self.g_t)).translate(0, coupler.sctq), 'or')

                if coupler.coupler_type == 'coupler':
                    qubit_cap_parts.append(gdspy.boolean(coupler.result_coupler, coupler.result_coupler, 'or',layer=10+id+self.secret_shift))
                    self.layers.append(10+id+self.secret_shift)
                    last_step_cap.append(coupler.result_coupler)
                self.center = center_save
        qubit_cap_parts.append(gdspy.boolean(result,last_step_cap,'not'))

        inverted = gdspy.boolean(box, result, 'not',layer=self.layer_configuration.inverted)

        # add JJs
        if self.JJ_params is not None:
            self.JJ_coordinates = (self.center[0],self.center[1])
            JJ = self.generate_JJ()

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
                        coupler_phi = 0
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
                        coupler_phi =  np.pi+self.transformations['rotate'][0]

            if self.transformations == {}:
                coupler_connection = coupler.connection
                if coupler.side == "left":
                    coupler_phi = 0
                if coupler.side == "right":
                    coupler_phi = np.pi
                if coupler.side == "top":
                    coupler_phi = -np.pi/2
                if coupler.side == "fluxline":
                    coupler_phi = np.pi
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
        # cheap Manhatten style
        reach1 = 11
        reach2 = 25

        # double strip 1
        result = gdspy.Rectangle((self.center[0] + self.gap / 2 + self.w - self.b_w,
                                  self.center[1] + self.h / 2 - self.b_w / 3 + self.JJ_params[
                                      'a1'] / 2 + self.asymmetry + 3 * self.b_w), (
                                 self.center[0] + self.gap / 2 + self.w - self.b_w + reach1,
                                 self.center[1] + self.h / 2 - self.b_w / 3 - self.JJ_params[
                                     'a1'] / 2 + self.asymmetry + 3 * self.b_w))
        # double strip 2
        result = gdspy.boolean(result, gdspy.Rectangle((self.center[0] + self.gap / 2 + self.w - self.b_w,
                                                        self.center[1] + self.h / 2 - 2 * self.b_w / 3 +
                                                        self.JJ_params['a1'] / 2 + self.asymmetry + 3 * self.b_w), (
                                                       self.center[0] + self.gap / 2 + self.w - self.b_w + reach1,
                                                       self.center[1] + self.h / 2 - 2 * self.b_w / 3 -
                                                       self.JJ_params['a1'] / 2 + self.asymmetry + 3 * self.b_w))
                               , 'or')
        # single strip
        result = gdspy.boolean(result, gdspy.Rectangle((self.center[0] + self.gap / 2 + self.w - self.b_w / 2,
                                                        self.center[
                                                            1] + self.h / 2 + self.asymmetry + 2 * self.b_w - self.b_g),
                                                       (self.center[0] + self.gap / 2 + self.w - self.b_w / 2 +
                                                        self.JJ_params['a2'], self.center[
                                                            1] + self.h / 2 + self.asymmetry + 2 * self.b_w - self.b_g + reach2)),
                               'or')
        # placing the junctions in the correct layer
        result = gdspy.boolean(result, result, 'or', layer=self.layer_configuration.jj_layer)

        angle = self.JJ_params['angle_JJ']
        result.rotate(angle, (self.JJ_coordinates[0], self.JJ_coordinates[1]))
        return result

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

        #cuttng off the hanging rest:
        if self.ground != None:
            result = gdspy.boolean(result,self.ground,'and')

        self.result_coupler = result

        point = (center[0] + self.pad_g / 2 + self.pad_w - self.b_w / 2 + self.flux_distance + self.t_r + self.l,
                 center[1] + self.asymmetry + self.pad_h / 2 + 3.5 * self.b_w)

        self.connection = point

        return {
            'positive': result,
            'restricted':restrict,
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
