from .core import DesignElement, LayerConfiguration, DesignTerminal
from .. import conformal_mapping as cm
from .. import transmission_line_simulator as tlsim
import gdspy
import numpy as np
from typing import List, Tuple, Mapping, Dict
from . import squid3JJ #TODO make this qubit class suitable for any squid types
from . import JJ4q
from copy import deepcopy


class PP_Transmon(DesignElement):
    """
    PP-Transmon consists of several parts:
    1) Central part - central circuit
    params: center = center of the qubit, w,h,gap = width,height,gap of the Parallel Plate Transmon in the ground cavity
    2) Couplers - claw like couplers for the left and right, rectangular pad couplers on top and bottom
    3) Ground = Ground rectangle around qubit, g_w,g_h,g_t = width,height and thickness of ground frame
    4)layer_configuration
    5)Couplers - coupler classes
    6) jj_params - parameters of the SQUID which here is 3JJ SQUID.#TODO add more information
    """
    def __init__(self, name: str, center: Tuple[float, float],width: float, height: float,gap: float,bridge_gap:float,bridge_w:float, g_w: float, g_h: float,g_t: float,layer_configuration: LayerConfiguration,
                 jj_params: Dict,Couplers):
        super().__init__(type='qubit', name=name)
        #qubit parameters
        #self.transformations = transformations# to mirror the structure
        self.center = center
        self.w = width
        self.h = height
        self.gap = gap
        self.g_w = g_w
        self.g_h = g_h
        self.g_t = g_t
        self.b_g = bridge_gap
        self.b_w = bridge_w
        #layers
        self.layer_configuration = layer_configuration

        #couplers
        self.couplers = Couplers

        # JJs and fluxline
        self.JJ_params = jj_params
        self.JJ = None
        self.layers = []



    def render(self):
        """
        This function draws everything: qubit,ground,couplers
        """
        qubit_cap_parts=[]
        ground = self.generate_ground()
        # restricted area for a future grid lines
        result_restricted = gdspy.Rectangle((self.center[0]-self.g_w/2,self.center[1]-self.g_h/2),(self.center[0]+self.g_w/2,self.center[1]+self.g_h/2),layer=self.layer_configuration.restricted_area_layer)

        P1 = gdspy.Rectangle((self.center[0]-self.gap/2-self.w,self.center[1]-self.h/2),(self.center[0]-self.gap/2,self.center[1]+self.h/2))
        P2 = gdspy.Rectangle((self.center[0] + self.gap / 2 + self.w, self.center[1] - self.h / 2),(self.center[0] + self.gap / 2, self.center[1] + self.h / 2))


        self.layers.append(9)
        result = gdspy.boolean(ground, P1, 'or', layer=self.layer_configuration.total_layer)
        result = gdspy.boolean(result, P2, 'or', layer=self.layer_configuration.total_layer)

        #change here if case to allow Manhatten-style junctions
        if self.JJ_params['manhatten']:
            P1_bridge = gdspy.Rectangle((self.center[0]-self.gap/2,self.center[1]),(self.center[0]-self.b_g/2,self.center[1]+self.b_w))
            P2_bridge = gdspy.Rectangle((self.center[0] + self.gap / 2, self.center[1] ),(self.center[0] + self.b_g / 2, self.center[1] - self.b_w))
            hole1     = gdspy.Rectangle((self.center[0]-self.b_g/2-self.JJ_params['h_w']-self.b_w/2,self.center[1]),(self.center[0]-self.b_g/2-self.b_w/2,self.center[1]+self.JJ_params['h_d']))
            hole2     = gdspy.Rectangle((self.center[0] + self.b_g / 2, self.center[1]-self.JJ_params['h_w']-self.b_w/2),(self.center[0] + self.b_g / 2 + self.JJ_params['h_d'], self.center[1] - self.b_w/2))
            P1_bridge = gdspy.boolean(P1_bridge, hole1, 'not', layer=8)
            P2_bridge = gdspy.boolean(P2_bridge, hole2, 'not', layer=8)
        else:
            P1_bridge = gdspy.Rectangle((self.center[0]-self.gap/2,self.center[1]-self.b_w/2),(self.center[0]-self.b_g/2,self.center[1]+self.b_w/2))
            P2_bridge = gdspy.Rectangle((self.center[0]+self.gap/2,self.center[1]-self.b_w/2),(self.center[0]+self.b_g/2,self.center[1]+self.b_w/2))


        qubit_cap_parts.append(gdspy.boolean(P1, P1_bridge, 'or', layer=8))
        qubit_cap_parts.append(gdspy.boolean(P2, P2_bridge, 'or', layer=9))

        result = gdspy.boolean(result, P1_bridge, 'or', layer=self.layer_configuration.total_layer)
        result = gdspy.boolean(result, P2_bridge, 'or', layer=self.layer_configuration.total_layer)
        self.layers.append(self.layer_configuration.total_layer)

        # add couplers
        last_step_cap = [gdspy.boolean(gdspy.boolean(P2, P2_bridge, 'or'),gdspy.boolean(P1, P1_bridge, 'or'),'or')]
        self.layers.append(self.layer_configuration.total_layer)

        # Box for inverted Polygons
        box = gdspy.Rectangle((self.center[0] - self.g_w / 2, self.center[1] - self.g_h / 2),(self.center[0] + self.g_w / 2, self.center[1] + self.g_h / 2))

        if len(self.couplers) != 0:
            for id, coupler in enumerate(self.couplers):
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
                if l1 < t:
                    l1 = t
                if l2 < t:
                    l2 = t

                if side =='right':
                    #upper
                    extended = gdspy.Rectangle((self.center[0]+self.g_w/2-l1-self.g_t+t,self.center[1]+height_right*self.g_h/2),(self.center[0]+self.g_w/2-l1+t,self.center[1]+gap+height_right*self.g_h/2+t+self.g_t+gap))
                    extended = gdspy.boolean(extended,gdspy.Rectangle((self.center[0]+t+self.g_w/2-l1,self.center[1]+gap+height_right*self.g_h/2+t+gap),(self.center[0]+self.g_w/2+2*gap+t+self.g_t,self.center[1]+gap+height_right*self.g_h/2+t+self.g_t+gap)),'or')
                    extended = gdspy.boolean(extended,gdspy.Rectangle((self.center[0]+self.g_w/2+2*gap+t,self.center[1]+gap+height_right*self.g_h/2+t+gap),(self.center[0]+self.g_w/2+2*gap+t+self.g_t,self.center[1]+5+gap)), 'or')
                    #lower
                    extended = gdspy.boolean(extended,gdspy.Rectangle((self.center[0]+t+self.g_w/2-l2-self.g_t,self.center[1]-height_right*self.g_h/2),(self.center[0]+t+self.g_w/2-l2,self.center[1]-gap-height_right*self.g_h/2-t-self.g_t-gap)), 'or')
                    extended = gdspy.boolean(extended,gdspy.Rectangle((self.center[0]+t+self.g_w/2-l2,self.center[1]-gap-height_right*self.g_h/2-t-gap),(self.center[0]+self.g_w/2+2*gap+t+self.g_t,self.center[1]-gap-height_right*self.g_h/2-t-self.g_t-gap)),'or')
                    extended = gdspy.boolean(extended,gdspy.Rectangle((self.center[0]+self.g_w/2+2*gap+t,self.center[1]-gap-height_right*self.g_h/2-t-gap),(self.center[0]+self.g_w/2+2*gap+t+self.g_t,self.center[1]-5-gap)), 'or')
                    result = gdspy.boolean(result,extended,'or')
                    # box for inverted polygon
                    box = gdspy.boolean(box, gdspy.Rectangle((self.center[0] + self.g_w / 2 + self.g_t + 2 * gap + t,
                                                              self.center[
                                                                  1] - height_right * self.g_h / 2 - self.g_t - 2 * gap - t),
                                                             (self.center[0] + self.g_w / 2 - l1 + t, self.center[
                                                                 1] + height_right * self.g_h / 2 + self.g_t + 2 * gap + t)),
                                        'or', layer=self.layer_configuration.inverted)

                if side =='left':
                    #upper
                    extended = gdspy.Rectangle((self.center[0]-self.g_w/2+l1+self.g_t-t,self.center[1]+height_left*self.g_h/2),(self.center[0]-t-self.g_w/2+l1,self.center[1]+gap+height_left*self.g_h/2+t+self.g_t+gap))
                    extended = gdspy.boolean(extended,gdspy.Rectangle((self.center[0]-t-self.g_w/2+l1,self.center[1]+gap+height_left*self.g_h/2+t+gap),(self.center[0]-self.g_w/2-2*gap-t-self.g_t,self.center[1]+gap+height_left*self.g_h/2+t+self.g_t+gap)),'or')
                    extended = gdspy.boolean(extended,gdspy.Rectangle((self.center[0]-self.g_w/2-2*gap-t,self.center[1]+gap+height_left*self.g_h/2+t+gap),(self.center[0]-self.g_w/2-2*gap-t-self.g_t,self.center[1]+5+gap)), 'or')
                    #lower
                    extended = gdspy.boolean(extended,gdspy.Rectangle((self.center[0]-t-self.g_w/2+l2+self.g_t,self.center[1]-height_left*self.g_h/2),(self.center[0]-t-self.g_w/2+l2,self.center[1]-gap-height_left*self.g_h/2-t-self.g_t-gap)), 'or')
                    extended = gdspy.boolean(extended,gdspy.Rectangle((self.center[0]-t-self.g_w/2+l2,self.center[1]-gap-height_left*self.g_h/2-t-gap),(self.center[0]-self.g_w/2-2*gap-t-self.g_t,self.center[1]-gap-height_left*self.g_h/2-t-self.g_t-gap)),'or')
                    extended = gdspy.boolean(extended,gdspy.Rectangle((self.center[0]-self.g_w/2-2*gap-t,self.center[1]-gap-height_left*self.g_h/2-t-gap),(self.center[0]-self.g_w/2-2*gap-t-self.g_t,self.center[1]-5-gap)), 'or')
                    result = gdspy.boolean(result,extended,'or')

                    #box for inverted polygon
                    box = gdspy.boolean(box, gdspy.Rectangle((self.center[0] - self.g_w / 2 - self.g_t - 2 * gap - t,self.center[1] - height_left * self.g_h / 2 - self.g_t - 2 * gap - t),(self.center[0] - self.g_w / 2 + l1 - t, self.center[1] + height_left * self.g_h / 2 + self.g_t + 2 * gap + t)),'or', layer=self.layer_configuration.inverted)

                if side == 'top':
                    extended = gdspy.Rectangle((self.center[0]-self.g_w/2+l1-gap-self.g_t,self.center[1]+self.g_h/2),(self.center[0]-self.g_w/2+l1-gap,self.center[1]+self.g_h/2+t+gap+gap))
                    extended = gdspy.boolean(extended,gdspy.Rectangle((self.center[0]-self.g_w/2+l1-gap-self.g_t,self.center[1]+self.g_h/2+t+gap+gap),(self.center[0]-self.g_w/2+l1-gap+l2/2-5,self.center[1]+self.g_h/2+t+gap+gap+self.g_t)),'or')
                    extended = gdspy.boolean(extended, gdspy.Rectangle((self.center[0]-self.g_w/2+l1+l2+gap,self.center[1]+self.g_h/2),(self.center[0]-self.g_w/2+l1+l2+self.g_t+gap,self.center[1]+self.g_h/2+t+gap+gap)), 'or')
                    extended = gdspy.boolean(extended, gdspy.Rectangle((self.center[0]-self.g_w/2+l1+l2+self.g_t+gap,self.center[1] + self.g_h / 2 + t + gap + gap),(self.center[0]-self.g_w/2+l1+l2/2+gap+5,self.center[1] + self.g_h / 2 + t + gap + gap + self.g_t)),'or')
                    result = gdspy.boolean(result,extended,'or')

                    # box for inverted polygon
                    box = gdspy.boolean(box,gdspy.Rectangle((self.center[0]-self.g_w/2+l1-gap-self.g_t,self.center[1]+self.g_h/2),(self.center[0]-self.g_w/2+l1+l2+self.g_t+gap,self.center[1] + self.g_h / 2 + t + gap + gap + self.g_t)),'or', layer=self.layer_configuration.inverted)


                if side == 'bottom':
                    extended = gdspy.Rectangle((self.center[0]-self.g_w/2+l1-gap-self.g_t,self.center[1]-self.g_h/2),(self.center[0]-self.g_w/2+l1-gap,self.center[1]-self.g_h/2-t-gap-gap))
                    extended = gdspy.boolean(extended,gdspy.Rectangle((self.center[0]-self.g_w/2+l1-gap-self.g_t,self.center[1]-self.g_h/2-t-gap-gap),(self.center[0]-self.g_w/2+l1-gap+l2/2-5,self.center[1]-self.g_h/2-t-gap-gap-self.g_t)),'or')
                    extended = gdspy.boolean(extended, gdspy.Rectangle((self.center[0]-self.g_w/2+l1+l2+gap,self.center[1]-self.g_h/2),(self.center[0]-self.g_w/2+l1+l2+self.g_t+gap,self.center[1]-self.g_h/2-t-gap-gap)), 'or')
                    extended = gdspy.boolean(extended, gdspy.Rectangle((self.center[0]-self.g_w/2+l1+l2+self.g_t+gap,self.center[1] - self.g_h / 2 - t - gap-gap),(self.center[0]-self.g_w/2+l1+l2/2+gap+5,self.center[1] - self.g_h / 2 - t - gap-gap- self.g_t)),'or')
                    result = gdspy.boolean(result,extended,'or')

                    # box for inverted polygon
                    box = gdspy.boolean(box, gdspy.Rectangle(
                        (self.center[0] - self.g_w / 2 + l1 - gap - self.g_t, self.center[1] - self.g_h / 2), (self.center[0] - self.g_w / 2 + l1 + l2 + self.g_t + gap,self.center[1] - self.g_h / 2 - t - gap - gap - self.g_t)), 'or',layer=self.layer_configuration.inverted)

                if coupler.coupler_type == 'coupler':
                        qubit_cap_parts.append(gdspy.boolean(coupler.result_coupler, coupler.result_coupler, 'or',layer=10+id))
                        self.layers.append(10+id)
                        last_step_cap.append(coupler.result_coupler)
        qubit_cap_parts.append(gdspy.boolean(result,last_step_cap,'not'))
        print(qubit_cap_parts)

        inverted = gdspy.boolean(box, result, 'not',layer=self.layer_configuration.inverted)

        # add JJs
        if self.JJ_params is not None:
            self.JJ_coordinates = (self.center[0],self.center[1])
            JJ = self.generate_JJ()

        return {'positive': result,
                    'restricted': result_restricted,
                    'qubit': result,
                    'qubit_cap': qubit_cap_parts,
                    'JJ': JJ,
                    'inverted': inverted
                    }

    def generate_ground(self):
        x = self.g_w
        y = self.g_h
        z = self.center
        t = self.g_t
        ground1 = gdspy.Rectangle((z[0] - x / 2, z[1] - y / 2), (z[0] + x / 2, z[1] + y / 2))
        ground2 = gdspy.Rectangle((z[0] - x / 2 + t, z[1] - y / 2 + t), (z[0] + x / 2 - t, z[1] + y / 2 - t))
        ground = gdspy.fast_boolean(ground1, ground2, 'not')
        return ground

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

    def generate_JJ(self):
        #change here to allow Manhatten style junctions
        if self.JJ_params['manhatten']:
            #JJ_2 is the manhatten junction style
            ##############################################################################
            self.JJ = JJ4q.JJ_2(self.JJ_coordinates[0], self.JJ_coordinates[1],
                                self.JJ_params['a1'], self.JJ_params['a2'],self.JJ_params['h_w'],self.JJ_params['h_d'])
            #############################################################################
        else:
            self.JJ = JJ4q.JJ_1(self.JJ_coordinates[0], self.JJ_coordinates[1],
                                    self.JJ_params['a1'], self.JJ_params['a2'])

        result = self.JJ.generate_jj()

        result = gdspy.boolean(result, result, 'or', layer=self.layer_configuration.jj_layer)

        angle = self.JJ_params['angle_JJ']
        result.rotate(angle, (self.JJ_coordinates[0], self.JJ_coordinates[1]))

        return result


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
    def __init__(self, l1,l2,t,gap,side = 'left',coupler_type = 'none',heightl = 1,heightr=1):
        self.l1 = l1
        self.l2 = l2
        self.t = t
        self.gap = gap
        self.side = side
        self.coupler_type = coupler_type
        self.ground_t = 10 #<-- temporary fix for the gds creation
        self.height_left = heightl
        self.height_right = heightr
    def render(self, center, g_w,g_h):
        result = 0
        if self.side == "left":
            result = gdspy.Rectangle((center[0]-g_w/2-self.t-self.gap,center[1]-self.height_left*g_h/2-self.gap),(center[0]-g_w/2-self.gap,center[1]+self.height_left*g_h/2+self.gap))
            if self.height_left == 1:
                upper  = gdspy.Rectangle((center[0]-g_w/2-self.t-self.gap,center[1]+g_h/2+self.gap),(center[0]-g_w/2+self.l1-self.gap-self.t,center[1]+g_h/2+self.t+self.gap))
                lower  = gdspy.Rectangle((center[0]-g_w/2-self.t-self.gap,center[1]-g_h/2-self.gap-self.t),(center[0]-g_w/2+self.l2-self.gap-self.t,center[1]-g_h/2-self.gap))
            line   = gdspy.Rectangle((center[0]-g_w/2-self.t-self.gap-self.gap-self.ground_t,center[1]-5),(center[0]-g_w/2-self.t-self.gap,center[1]+5))#modified here ;), remove ground_t
            if self.height_left ==1:
                result = gdspy.boolean(result, upper, 'or')
                result = gdspy.boolean(result, lower, 'or')
            result = gdspy.boolean(result, line, 'or')


        if self.side == "right":
            result = gdspy.Rectangle((center[0]+g_w/2+self.t+self.gap,center[1]-self.height_right*g_h/2-self.gap),(center[0]+g_w/2+self.gap,center[1]+self.height_right*g_h/2+self.gap))
            if self.height_right == 1:
                upper  = gdspy.Rectangle((center[0]+g_w/2+self.t+self.gap,center[1]+g_h/2+self.gap),(center[0]+g_w/2-self.l1+self.gap+self.t,center[1]+g_h/2+self.t+self.gap))
                lower  = gdspy.Rectangle((center[0]+g_w/2+self.t+self.gap,center[1]-g_h/2-self.gap-self.t),(center[0]+g_w/2-self.l2+self.gap+self.t,center[1]-g_h/2-self.gap))

            line = gdspy.Rectangle((center[0]+g_w/2+self.t+self.gap+self.gap,center[1]-5),(center[0]+g_w/2+self.t+self.gap,center[1]+5))
            if self.height_right == 1:
                result = gdspy.boolean(result, upper, 'or')
                result = gdspy.boolean(result, lower, 'or')
            result = gdspy.boolean(result, line, 'or')

        if self.side == "top":
            result = gdspy.Rectangle((center[0]-g_w/2+self.l1,center[1]+g_h/2+self.gap),(center[0]-g_w/2+self.l1+self.l2,center[1]+g_h/2+self.gap+self.t))
            line   = gdspy.Rectangle((center[0]-g_w/2+self.l1+self.l2/2-5,center[1]+g_h/2+self.gap+self.t),(center[0]-g_w/2+self.l1+self.l2/2+5,center[1]+g_h/2+self.gap+self.t+self.gap))
            result = gdspy.boolean(result, line, 'or')

        if self.side == "bottom":
            result = gdspy.Rectangle((center[0]-g_w/2+self.l1,center[1]-g_h/2-self.gap),(center[0]-g_w/2+self.l1+self.l2,center[1]-g_h/2-self.gap-self.t))
            line   = gdspy.Rectangle((center[0]-g_w/2+self.l1+self.l2/2-5,center[1]-g_h/2-self.gap-self.t),(center[0]-g_w/2+self.l1+self.l2/2+5,center[1]-g_h/2-self.gap-self.t-self.gap))
            result = gdspy.boolean(result, line, 'or')

        self.result_coupler = result

        return {
            'positive': result
                        }