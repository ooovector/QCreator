import gdspy
from .core import DesignElement, LayerConfiguration, DesignTerminal
import numpy as np
from scipy.constants import epsilon_0



class micron_test_JJ(DesignElement):
    def __init__(self, name, x_upleft, y_upleft, jj_width, jj_height, pad_length,
                 pad_width, pad_distance, jj_shift_left, jj_shift_right, jj_gap,
                 layers_configuration, label_size=10, ic=None,    h=2e-9, empty_pads = False):
        super().__init__(type='micron test JJ', name=name)

        self.name = name
        self.x_upleft = x_upleft
        self.y_upleft = y_upleft
        self.jj_width = jj_width
        self.jj_height = jj_height
        self.pad_length = pad_length
        self.pad_width = pad_width
        self.pad_distance = pad_distance
        self.jj_shift_left = jj_shift_left
        self.jj_shift_right = jj_shift_right
        self.jj_gap = jj_gap
        self.layers_configuration = layers_configuration
        self.ic = ic
        self.label_size = label_size
        self.h = h
        self.empty_pads = empty_pads
        # self.jj_horizontal_turn = jj_horizontal_turn



    def render(self):

        pad_up = gdspy.Rectangle((self.x_upleft,  self.y_upleft),
                                 (self.x_upleft + self.pad_length,  self.y_upleft - self.pad_width),
                                 layer=self.layers_configuration['total'])
        pad_down = gdspy.Rectangle((self.x_upleft,  self.y_upleft - self.pad_width - self.pad_distance),
                                   (self.x_upleft + self.pad_length,
                                    self.y_upleft - self.pad_width - self.pad_distance - self.pad_width),
                                   layer=self.layers_configuration['total'])

        result = gdspy.boolean(pad_up, pad_down, 'or', layer=self.layers_configuration['total'])

        if not self.empty_pads:

            center_point_x = self.x_upleft + self.pad_length/2
            center_point_y = self.y_upleft - self.pad_width - self.pad_distance/2

            vertical_jj_part = gdspy.Rectangle((center_point_x - self.jj_width/2,
                                            self.y_upleft - self.pad_width),
                                           (center_point_x + self.jj_width / 2,
                                            center_point_y + self.jj_gap/2), layer=self.layers_configuration['total'])

            horizontal_jj_part = gdspy.Polygon([ (center_point_x + self.jj_width/2 +self.jj_shift_right,
                                                 center_point_y - self.jj_gap/2),
                                                (center_point_x + self.jj_width/2 +self.jj_shift_right,
                                                 center_point_y - self.jj_gap/2 - self.jj_height),
                                                (center_point_x - self.jj_width/2 - self.jj_shift_left + self.jj_height,
                                                 center_point_y - self.jj_gap / 2 - self.jj_height),
                                                (center_point_x - self.jj_width/2 - self.jj_shift_left + self.jj_height,
                                                 self.y_upleft - self.pad_width - self.pad_distance),
                                                (center_point_x - self.jj_width/2 - self.jj_shift_left,
                                                 self.y_upleft - self.pad_width - self.pad_distance),
                                                (center_point_x - self.jj_width / 2 - self.jj_shift_left,
                                                 center_point_y - self.jj_gap / 2) ],
                                               layer=self.layers_configuration['total'])


            result = gdspy.boolean(result, vertical_jj_part , 'or', layer=self.layers_configuration['total'])
            result = gdspy.boolean(result, horizontal_jj_part, 'or', layer=self.layers_configuration['total'])

        label = gdspy.Text(self.name, self.label_size, (self.x_upleft - self.pad_length, self.y_upleft + self.pad_width),
                           layer=self.layers_configuration['total'])
        result = gdspy.boolean(result, label, 'or', layer=self.layers_configuration['total'])

        render_result = {'positive': result}

        return render_result



    def find_jj_params(self):
        from scipy.constants import hbar

        s = 1e-12 * self.jj_width * self.jj_height
        epsilon = 10
        self.jj_Ic = self.ic * s
        self.jj_capacitance = epsilon_0 * epsilon * s / self.h


        return self.jj_Ic, self.jj_capacitance
