from .core import DesignElement, DesignTerminal, LayerConfiguration
# from .. import transmission_line_simulator as tlsim
import numpy as np
import gdspy
from typing import Tuple  # , Mapping, Any
from .airbridge import AirBridgeGeometry
from .airbridge import AirbridgeOverCPW
from .cpw import Narrowing


class CrossLinesViaAirbridges(DesignElement):
    def __init__(self, name: str, position: Tuple[float, float], orientation: float,
                 top_w: float, top_s: float, top_g: float, bot_w: float, bot_s: float, bot_g: float, narrowing_length: float, geometry: AirBridgeGeometry):
        """
        Cross two CPW lines via three Airbridges.
        :param name: element identifier
        :param position: position of the center point
        :param orientation: angle counted from default position
        :param top_w: top line core width
        :param top_s: top line gap width
        :param top_g: top line ground width
        :param bot_w: bottom line core width
        :param bot_s: bottom line gap width
        :param bot_g: bottom line ground width
        :param narrowing_length: length of the narrowing elements
        :param geometry: configuration and type of bridges used to cross lines
        """
        super().__init__('line crossing via airbridges', name)
        self.position = np.asarray(position)
        self.orientation = orientation
        self.top_w = top_w
        self.top_s = top_s
        self.top_g = top_g
        self.bot_w = bot_w
        self.bot_s = bot_s
        self.bot_g = bot_g
        self.nar_len = narrowing_length

        self.geometry = geometry

        self.terminals = {'bottom_1': None,
                          'bottom_2': None,
                          'top_1': None,
                          'top_2': None}

    def render(self):
        """
            IDEA: draw everything in horizontal orientation and THEN rotate the whole thing for one's liking.
            This way there's no sin-cos stuff in rendering. This is why there are fundamental constants in "orientation"
            like pi or 0.
        """
        b_len = self.geometry.bridge_length
        p_wid = self.geometry.pad_width

        # 1. Draw two Narrowings for airbridge pads, draft future connection points:
        nar_pos_1 = (self.position[0] - b_len/2 - self.nar_len/2, self.position[1])
        nar_pos_2 = (self.position[0] + b_len/2 + self.nar_len/2, self.position[1])
        top_1_connection = (nar_pos_1[0] - self.nar_len/2, self.position[1])
        top_2_connection = (nar_pos_2[0] + self.nar_len/2, self.position[1])
        bot_1_connection = (self.position[0], self.position[1] + p_wid/2 + p_wid + self.top_s)
        bot_2_connection = (self.position[0], self.position[1] - p_wid/2 - p_wid - self.top_s)

        narrowing_1 = Narrowing('narrowing_1', nar_pos_1,
                                orientation=0,
                                w1=self.top_w,
                                s1=self.top_s,
                                g1=self.top_g,
                                w2=p_wid,
                                s2=self.top_s,
                                g2=p_wid,
                                layer_configuration=self.geometry.layer_configuration,
                                length=self.nar_len)

        positive = narrowing_1.render()['positive']
        restrict = narrowing_1.render()['restrict']

        narrowing_2 = Narrowing('narrowing_2', nar_pos_2,
                                orientation=np.pi,
                                w1=self.top_w,
                                s1=self.top_s,
                                g1=self.top_g,
                                w2=p_wid,
                                s2=self.top_s,
                                g2=p_wid,
                                layer_configuration=self.geometry.layer_configuration,
                                length=self.nar_len)

        positive = gdspy.boolean(positive, narrowing_2.render()['positive'], 'or', layer=self.geometry.layer_configuration.total_layer)
        restrict = gdspy.boolean(restrict, narrowing_2.render()['restrict'], 'or', layer=self.geometry.layer_configuration.restricted_area_layer)

        # Draw bridges using existing Airbridge element:
        br_pos_center = (self.position[0], self.position[1])
        bridge_center = AirbridgeOverCPW('bridge_center', br_pos_center,
                                         orientation=np.pi/2,
                                         w=self.bot_w,
                                         s=self.bot_s,
                                         g=self.bot_g,
                                         geometry=self.geometry)

        positive = gdspy.boolean(positive, bridge_center.render()['positive'], 'or', layer=self.geometry.layer_configuration.total_layer)
        restrict = gdspy.boolean(restrict, bridge_center.render()['restrict'], 'or', layer=self.geometry.layer_configuration.restricted_area_layer)
        contacts = bridge_center.render()['airbridges_pads']
        contacts_sm = bridge_center.render()['airbridges_sm_pads']
        bridges = bridge_center.render()['airbridges']

        br_pos_up = (self.position[0], self.position[1] + p_wid + self.top_s)
        bridge_up = AirbridgeOverCPW('bridge_up', br_pos_up,
                                     orientation=np.pi/2,
                                     w=self.bot_w,
                                     s=self.bot_s,
                                     g=self.bot_g,
                                     geometry=self.geometry)

        positive = gdspy.boolean(positive, bridge_up.render()['positive'], 'or', layer=self.geometry.layer_configuration.total_layer)
        restrict = gdspy.boolean(restrict, bridge_up.render()['restrict'], 'or', layer=self.geometry.layer_configuration.restricted_area_layer)
        contacts = gdspy.boolean(contacts, bridge_up.render()['airbridges_pads'], 'or', layer=self.geometry.layer_configuration.airbridges_pad_layer)
        contacts_sm = gdspy.boolean(contacts_sm, bridge_up.render()['airbridges_sm_pads'], 'or', layer=self.geometry.layer_configuration.airbridges_sm_pad_layer)
        bridges = gdspy.boolean(bridges, bridge_up.render()['airbridges'], 'or', layer=self.geometry.layer_configuration.airbridges_layer)

        br_pos_down = (self.position[0], self.position[1] - p_wid - self.top_s)
        bridge_down = AirbridgeOverCPW('bridge_down', br_pos_down,
                                       orientation=np.pi/2,
                                       w=self.bot_w,
                                       s=self.bot_s,
                                       g=self.bot_g,
                                       geometry=self.geometry)

        positive = gdspy.boolean(positive, bridge_down.render()['positive'], 'or', layer=self.geometry.layer_configuration.total_layer)
        restrict = gdspy.boolean(restrict, bridge_down.render()['restrict'], 'or', layer=self.geometry.layer_configuration.restricted_area_layer)
        contacts = gdspy.boolean(contacts, bridge_down.render()['airbridges_pads'], 'or', layer=self.geometry.layer_configuration.airbridges_pad_layer)
        contacts_sm = gdspy.boolean(contacts_sm, bridge_down.render()['airbridges_sm_pads'], 'or', layer=self.geometry.layer_configuration.airbridges_sm_pad_layer)
        bridges = gdspy.boolean(bridges, bridge_down.render()['airbridges'], 'or', layer=self.geometry.layer_configuration.airbridges_layer)

        # There are blank spaces left between bridges, these aus rectangles fill them:
        aux_g_1_up = gdspy.Rectangle((self.position[0] - self.bot_w/2 - self.bot_s - self.bot_g, self.position[1] + p_wid/2),
                                     (self.position[0] - self.bot_w/2 - self.bot_s, self.position[1] + p_wid/2 + self.top_s))
        aux_g_2_up = gdspy.Rectangle((self.position[0] + self.bot_w/2 + self.bot_s + self.bot_g, self.position[1] + p_wid/2),
                                     (self.position[0] + self.bot_w/2 + self.bot_s, self.position[1] + p_wid/2 + self.top_s))
        aux_g_up = gdspy.boolean(aux_g_1_up, aux_g_2_up, 'or')

        aux_g_1_down = gdspy.Rectangle((self.position[0] - self.bot_w/2 - self.bot_s - self.bot_g, self.position[1] - p_wid/2),
                                       (self.position[0] - self.bot_w/2 - self.bot_s, self.position[1] - p_wid/2 - self.top_s))
        aux_g_2_down = gdspy.Rectangle((self.position[0] + self.bot_w/2 + self.bot_s + self.bot_g, self.position[1] - p_wid/2),
                                       (self.position[0] + self.bot_w/2 + self.bot_s, self.position[1] - p_wid/2 - self.top_s))
        aux_g_down = gdspy.boolean(aux_g_1_down, aux_g_2_down, 'or')

        aux_g = gdspy.boolean(aux_g_down, aux_g_up, 'or')
        positive = gdspy.boolean(positive, aux_g, 'or', layer=self.geometry.layer_configuration.total_layer)

        aux_w_up = gdspy.Rectangle((self.position[0] - self.bot_w/2, self.position[1] + p_wid/2),
                                   (self.position[0] + self.bot_w/2, self.position[1] + p_wid/2 + self.top_s))
        aux_w_down = gdspy.Rectangle((self.position[0] - self.bot_w/2, self.position[1] - p_wid/2),
                                     (self.position[0] + self.bot_w/2, self.position[1] - p_wid/2 - self.top_s))
        aux_w = gdspy.boolean(aux_w_down, aux_w_up, 'or')
        positive = gdspy.boolean(positive, aux_w, 'or', layer=self.geometry.layer_configuration.total_layer)

        # Rotate everything if needed:
        if self.orientation is not 0:
            positive.rotate(self.orientation, self.position)
            restrict.rotate(self.orientation, self.position)
            contacts.rotate(self.orientation, self.position)
            contacts_sm.rotate(self.orientation, self.position)
            bridges.rotate(self.orientation, self.position)
            top_1_connection = rotate_point(top_1_connection, self.orientation, self.position)
            top_2_connection = rotate_point(top_2_connection, self.orientation, self.position)
            bot_1_connection = rotate_point(bot_1_connection, self.orientation, self.position)
            bot_2_connection = rotate_point(bot_2_connection, self.orientation, self.position)

        # Set up terminals:
        self.terminals['top_1'] = DesignTerminal(top_1_connection, self.orientation,
                                                 g=self.top_g,
                                                 s=self.top_s,
                                                 w=self.top_w,
                                                 type='cpw')
        self.terminals['top_2'] = DesignTerminal(top_2_connection, self.orientation + np.pi,
                                                 g=self.top_g,
                                                 s=self.top_s,
                                                 w=self.top_w,
                                                 type='cpw')

        self.terminals['bottom_1'] = DesignTerminal(bot_1_connection, self.orientation + 3*np.pi/2,
                                                    g=self.bot_g,
                                                    s=self.bot_s,
                                                    w=self.bot_w,
                                                    type='cpw')

        self.terminals['bottom_2'] = DesignTerminal(bot_2_connection, self.orientation + np.pi/2,
                                                    g=self.bot_g,
                                                    s=self.bot_s,
                                                    w=self.bot_w,
                                                    type='cpw')

        return {'positive': positive, 'airbridges_pads': contacts, 'airbridges_sm_pads': contacts_sm,
                'airbridges': bridges, 'restrict': restrict}

    def get_terminals(self) -> dict:
        return self.terminals


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
