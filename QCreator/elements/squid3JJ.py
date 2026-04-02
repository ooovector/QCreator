
import warnings
import gdspy


class JJ_2_separate:
    def __init__(self, x0, y0, length_pad_up_to_squid, jj1_width, jj1_height, jj2_width, jj2_height, pad_up_length,
                 pad_up_width, width_pad_up_to_squid, squid_shift_left, squid_shift_right,
                 base_width, jj_gap, jj_vertical_length, jj_horizontal_length, side_pad_length, side_pad_width,
                 base_shift, side_pads_distance):
        self._x0 = x0
        self._y0 = y0

        self.length_pad_up_to_squid = length_pad_up_to_squid
        self.jj1_width = jj1_width
        self.jj1_height = jj1_height
        self.jj2_width = jj2_width
        self.jj2_height = jj2_height
        self.pad_up_length = pad_up_length
        self.pad_up_width = pad_up_width
        self.width_pad_up_to_squid = width_pad_up_to_squid
        self.squid_shift_right = squid_shift_right
        self.squid_shift_left = squid_shift_left
        self.base_width = base_width
        self.jj_gap = jj_gap
        self.jj_vertical_length = jj_vertical_length
        self.jj_horizontal_length = jj_horizontal_length
        self.side_pad_length = side_pad_length
        self.side_pad_width = side_pad_width
        self.base_shift = base_shift
        self.side_pads_distance = side_pads_distance

    def generate_jj(self):


        '''Add upper contact pad'''
        points0 = [(self._x0 - self.pad_up_length / 2, self._y0 - self.pad_up_width),
                   (self._x0 - self.pad_up_length / 2, self._y0),
                   (self._x0 + self.pad_up_length / 2, self._y0),
                   (self._x0 + self.pad_up_length / 2, self._y0 - self.pad_up_width)]

        x1 = self._x0
        y1 = self._y0 - self.pad_up_width


        # H_a = self.width_pad_up_to_squid
        # H_b = self.length_pad_up_to_squid

        '''Add connection between upper pad and SQUID'''
        points1 = [(x1 - self.width_pad_up_to_squid / 2, y1),
                   (x1 + self.width_pad_up_to_squid / 2, y1),
                   (x1 + self.width_pad_up_to_squid / 2, y1 - self.length_pad_up_to_squid),
                   (x1 - self.width_pad_up_to_squid / 2, y1 - self.length_pad_up_to_squid)
                   ]

        # x2 = x1 - 3* H_a + L_a/4
        # y2 = y1 - H_b

        '''Add vertical part of JJs in SQUID with П-shape, where | | - vertical parts of JJs'''
        x2 = x1
        y2 = y1 - self.length_pad_up_to_squid

        # L_a = self.squid_shift_right or self.squid_shift_left
        # L_b = self.base_width
        # h = self.jj_gap

        # connection between jj1 and jj2 vertical parts
        points2 = [(x2 - self.width_pad_up_to_squid/2 -self.squid_shift_left , y2),
                   (x2 + self.width_pad_up_to_squid/2 +self.squid_shift_right , y2),
                   (x2 + self.width_pad_up_to_squid/2 +self.squid_shift_right , y2 - self.base_width),
                   (x2 - self.width_pad_up_to_squid/2 -self.squid_shift_left , y2 - self.base_width)
                   ]

        # H1_a = self.jj1_width (left)
        # H1_b = self.jj_vertical_length
        # H2_a = self.jj2_width (right)
        # H2_b = self.jj_vertical_length

        x3 = x2 - self.width_pad_up_to_squid/2 -self.squid_shift_left
        y3 = y2 - self.base_width

        x4 = x2 + self.width_pad_up_to_squid/2 +self.squid_shift_right
        y4 = y2 - self.base_width

        ## left jj vertical part
        points3 = [(x3, y3),
                   (x3 + self.jj1_width, y3),
                   (x3 + self.jj1_width, y3 - self.jj_vertical_length),
                   (x3, y3 - self.jj_vertical_length)
                   ]

        ## right jj vertical part
        points4 = [(x4, y4),
                   (x4 - self.jj2_width, y4),
                   (x4 - self.jj2_width, y4 - self.jj_vertical_length),
                   (x4, y4 - self.jj_vertical_length)
                   ]

        '''Add horizontal part of JJs in SQUID  and left-right contact pads'''


        x5 = x3 + self.jj1_width
        y5 = y3 - self.jj_vertical_length

        x6 = x4 - self.jj2_width
        y6 = y4 - self.jj_vertical_length


        ### left jj horizontal part
        points5 = [(x5 + self.base_shift, y5 - self.jj_gap),
                   (x5 + self.base_shift, y5 - self.jj_gap - self.jj1_height),
                   (x5 - self.jj1_width - self.base_shift, y5 - self.jj_gap - self.jj1_height),
                   (x5 - self.jj1_width - self.base_shift, y5 - self.jj_gap)
        ]


        ### right jj horizontal part
        points6 = [(x6 - self.base_shift, y6 - self.jj_gap),
                   (x6 - self.base_shift, y6 - self.jj_gap - self.jj2_height),
                   (x6 + self.jj2_width + self.base_shift, y6 - self.jj_gap - self.jj2_height),
                   (x6 + self.jj2_width + self.base_shift, y6 - self.jj_gap)
        ]

        #warning for horizontal parts overlap
        if x5 + self.base_shift >= x6 - self.base_shift:
            raise Warning('There is horizontal parts JJs overlap')

        x7 = self._x0 - self.side_pads_distance/2
        y7 = y5 - self.jj_gap

        x8 = self._x0 + self.side_pads_distance/2
        y8 = y6 - self.jj_gap

        self.rect1 = (x7, y7)
        self.rect2 = (x8, y8)

        ## variables for bandages
        self.contact_pad_a_outer = self.pad_up_length
        self.contact_pad_b_outer = self.pad_up_width
        self.rect_size_a = self.side_pad_length
        self.rect_size_b = self.side_pad_width




        # left side pad
        points7 = [(x7 - self.side_pad_length/2, y7),
                   (x7 - self.side_pad_length/2, y7 - self.side_pad_width),
                   (x7 + self.side_pad_length/2, y7 - self.side_pad_width),
                   (x7 + self.side_pad_length/2, y7)
        ]

        ## right side pad
        points8 = [(x8 + self.side_pad_length/2, y8),
                   (x8 + self.side_pad_length/2, y8 - self.side_pad_width),
                   (x8 - self.side_pad_length/2, y8 - self.side_pad_width),
                   (x8 - self.side_pad_length/2, y8)
        ]

        # connection between contact pads and horizontal jj parts

        points9 = [(x5 - self.jj1_width - self.base_shift, y5 - self.jj_gap),
                   (x5 - self.jj1_width - self.base_shift, y5 - self.jj_gap - self.base_width),
                   (x7 + self.side_pad_length / 2, y5 - self.jj_gap - self.base_width),
                   (x7 + self.side_pad_length / 2, y7)
        ]

        points10 = [(x6 + self.jj2_width + self.base_shift, y6 - self.jj_gap),
                    (x6 + self.jj2_width + self.base_shift, y6 - self.jj_gap - self.base_width),
                    (x8 - self.side_pad_length / 2, y6 - self.jj_gap - self.base_width),
                    (x8 - self.side_pad_length / 2, y8)

        ]

        #warning for the overlap of the side contact pad and the corresponding horizontal JJ part
        if x7 + self.side_pad_length / 2 > x5 - self.jj1_width - self.base_shift or\
                x6 + self.jj2_width + self.base_shift > x8 - self.side_pad_length / 2:
            raise Warning('There is the overlap of the side contact pad and the corresponding horizontal JJ part')

        squid = gdspy.PolygonSet([points0, points1, points2, points3, points4, points5, points6,
                                points7, points8, points9, points10])

        return squid





class JJ_2:
    def __init__(self, x0, y0, Hb, jj1_width, jj1_height, jj2_width, jj2_height, jj3_width, jj3_height, parametr6,
                 add_JJ=False, hole_in_squid_pad=False):
        self._x0 = x0
        self._y0 = y0

        self._parametr1 = Hb
        self.jj1_width = jj1_width
        self.jj1_height = jj1_height
        self.jj2_width = jj2_width
        self.jj2_height = jj2_height
        self.jj3_width = jj3_width
        self.jj3_height = jj3_height
        self._parametr6 = parametr6
        self.add_JJ = add_JJ
        self.hole_in_squid_pad = hole_in_squid_pad

    def generate_jj(self):
        contact_pad_a_outer = 10.5
        contact_pad_b_outer = 6
        self.contact_pad_b_outer = contact_pad_b_outer
        self.contact_pad_a_outer = contact_pad_a_outer
        if self.hole_in_squid_pad==True:
            contact_pad_a_inner = 7.5
            contact_pad_b_inner = 1
        else:
            contact_pad_a_inner = 0
            contact_pad_b_inner = 0


        # Add contact pad1
        points0 = [(self._x0 - contact_pad_a_outer / 2, self._y0 - contact_pad_b_outer),
                   (self._x0 - contact_pad_a_outer / 2, self._y0),
                   (self._x0 + contact_pad_a_outer / 2, self._y0),
                   (self._x0 + contact_pad_a_outer / 2, self._y0 - contact_pad_b_outer),

                   (self._x0 - contact_pad_a_outer / 2, self._y0 - contact_pad_b_outer),

                   (self._x0 - contact_pad_a_inner / 2,
                    self._y0 - (contact_pad_b_outer - contact_pad_b_inner) / 2 - contact_pad_b_inner),
                   (self._x0 + contact_pad_a_inner / 2,
                    self._y0 - (contact_pad_b_outer - contact_pad_b_inner) / 2 - contact_pad_b_inner),

                   (self._x0 + contact_pad_a_inner / 2, self._y0 - (contact_pad_b_outer - contact_pad_b_inner) / 2),
                   (self._x0 - contact_pad_a_inner / 2, self._y0 - (contact_pad_b_outer - contact_pad_b_inner) / 2),

                   (self._x0 - contact_pad_a_inner / 2,
                    self._y0 - (contact_pad_b_outer - contact_pad_b_inner) / 2 - contact_pad_b_inner),
                   ]

        x1 = self._x0
        y1 = self._y0 - contact_pad_b_outer

        # parametr1=H_b

        H_a = 0.5
        H_b = self._parametr1

        L_a = 4#10.7
        L_b = 0.4#0.75

        h = 0.16#0.5

        if self.add_JJ == False:
            points1 = [(x1 - H_a / 2, y1),
                       (x1 + H_a / 2, y1),
                       (x1 + H_a / 2, y1 - H_b),
                       (x1 - H_a / 2, y1 - H_b)
                       ]
        else:
            points1 = [(x1 - 3 * H_a, y1 - H_b / 3),
                       (x1 + H_a, y1 - H_b / 3),
                       (x1 + H_a, y1 - H_b / 3 - self.jj3_height),
                       (x1 - 2 * H_a, y1 - H_b / 3 - self.jj3_height),
                       (x1 - 2 * H_a, y1 - H_b),
                       (x1 - 3 * H_a, y1 - H_b)
                       ]

            points1_1 = [(x1 - H_a, y1),
                         (x1 - H_a/4, y1),
                         (x1 - H_a/4, y1 - H_b / 4),
                         (x1 - H_a + self.jj3_width, y1 - H_b / 4),
                         (x1 - H_a + self.jj3_width, y1 - H_b / 3 + h),
                         (x1 - H_a, y1 - H_b / 3 + h)
                         ]

        x2 = x1 - 3* H_a + L_a/4
        y2 = y1 - H_b



        points2 = [(x2 - L_a / 2, y2),
                   (x2 + L_a / 2, y2),
                   (x2 + L_a / 2, y2 - L_b),
                   (x2 - L_a / 2, y2 - L_b)
                   ]

        H1_a = self.jj2_width#0.8
        H1_b = 1#2
        H2_a = self.jj1_width#0.8
        H2_b = 1#2

        x3 = x2 - L_a / 2 + H1_a / 2
        y3 = y2 - L_b


        x4 = x2 + L_a / 2 - H1_a / 2
        y4 = y2 - L_b


        points3 = [(x3 - H1_a / 2, y3),
                   (x3 + H1_a / 2, y3),
                   (x3 + H1_a / 2, y3 - H1_b),
                   (x3 - H1_a / 2, y3 - H1_b)
                   ]


        points4 = [(x4 - H2_a / 2, y4),
                   (x4 + H2_a / 2, y4),
                   (x4 + H2_a / 2, y4 - H2_b),
                   (x4 - H2_a / 2, y4 - H2_b)
                   ]


        x5 = x3 + H1_a / 2
        y5 = y3 - H1_b
        x6 = x4 - H2_a / 2
        y6 = y4 - H2_b


        # parametr2=pad1_a
        # parametr3=pad2_a

        pad1_a = self.jj2_width
        pad1_b = 1#3
        pad2_a = self.jj1_width
        pad2_b = 1#3

        points5_for_pad1 = [(x5, y5),
                            (x5, y5 - pad1_b),
                            (x5 - pad1_a, y5 - pad1_b),
                            (x5 - pad1_a, y5)
                            ]


        points6_for_pad2 = [(x6, y6),
                            (x6 + pad2_a, y6),
                            (x6 + pad2_a, y6 - pad2_b),
                            (x6, y6 - pad2_b)
                            ]


        contact_pad1_a_outer = L_a+2#6#13
        contact_pad1_b_outer = 2.5#6.4
        contact_pad1_a_inner = L_a+1
        contact_pad1_b_inner = 2

        x7 = x2#self._x0
        y7 = self._y0 - contact_pad_b_outer - H_b - L_b - H1_b - pad1_b - h - contact_pad1_b_outer
        x7_ = x7
        y7_ = y7 + (contact_pad1_b_outer - contact_pad1_b_inner)


        points7 = [(x7 - contact_pad1_a_outer / 2, y7 + contact_pad1_b_outer),

                   (x7 - contact_pad1_a_outer / 2, y7),

                   (x7 + contact_pad1_a_outer / 2, y7),

                   (x7 + contact_pad1_a_outer / 2, y7 + contact_pad1_b_outer),

                   (x7_ + contact_pad1_a_inner / 2, y7_ + contact_pad1_b_inner),

                   (x7_ + contact_pad1_a_inner / 2, y7_),

                   (x7_ - contact_pad1_a_inner / 2, y7_),

                   (x7_ - contact_pad1_a_inner / 2, y7_ + contact_pad1_b_inner)]

        x8 = x7_ - contact_pad1_a_inner / 2
        y8 = y7_ + contact_pad1_b_inner

        x9 = x7_ + contact_pad1_a_inner / 2
        y9 = y7_ + contact_pad1_b_inner

        # parametr4=pad3_b
        # parametr5=pad4_b

        pad3_a = 1.5#2.5
        pad3_b = self.jj2_height

        pad4_a = 1.5#2.5
        pad4_b = self.jj1_height

        points8_for_pad3 = [(x8, y8),

                            (x8 + pad3_a, y8),

                            (x8 + pad3_a, y8 - pad3_b),

                            (x8, y8 - pad3_b)]

        points9_for_pad4 = [(x9 - pad4_a, y9),

                            (x9, y9),

                            (x9, y9 - pad4_b),

                            (x9 - pad4_a, y9 - pad4_b)]

        delta = contact_pad1_b_outer

        L1_a = 3.4
        L1_b = 0.5#1

        L2_a = 3.4
        L2_b = 0.5#1

        x10 = x7 - contact_pad1_a_outer / 2
        y10 = y7 + L1_b

        x11 = x7 + contact_pad1_a_outer / 2
        y11 = y7 + L2_b

        rec1_a_outer = 5
        rec1_b_outer = 5

        if self.hole_in_squid_pad == True:
            rec1_a_inner = 2
            rec1_b_inner = 1
        else:
            rec1_a_inner = 0
            rec1_b_inner = 0

        rec2_a_outer = rec1_a_outer
        rec2_b_outer = rec1_b_outer

        rec2_a_inner = rec1_a_inner
        rec2_b_inner = rec1_b_inner

        self.rect_size_a = rec1_a_outer
        self.rect_size_b = rec1_b_outer

        points10 = [(x10 - L1_a, y10),
                    (x10, y10),
                    (x10, y10 - L1_b),
                    (x10 - L1_a, y10 - L1_b)]

        points11 = [(x11, y11),
                    (x11 + L2_a, y11),
                    (x11 + L2_a, y11 - L2_b),
                    (x11, y11 - L2_b)]

        x12 = x10 - L1_a - (rec1_a_outer / 2)
        y12 = y10 - L1_b / 2 + (rec1_b_outer / 2)

        x13 = x11 + L2_a + (rec2_a_outer / 2)
        y13 = y11 - L2_b / 2 + (rec2_b_outer / 2)
        self.rect1 = (x12, y12)
        self.rect2 = (x13, y13)
        points12 = [(x12 - rec1_a_outer / 2, y12 - rec1_b_outer),
                    (x12 - rec1_a_outer / 2, y12),
                    (x12 + rec1_a_outer / 2, y12),
                    (x12 + rec1_a_outer / 2, y12 - rec1_b_outer),

                    (x12 - rec1_a_outer / 2, y12 - rec1_b_outer),

                    (x12 - rec1_a_inner / 2, y12 - (rec1_b_outer - rec1_b_inner) / 2 - rec1_b_inner),
                    (x12 + rec1_a_inner / 2, y12 - (rec1_b_outer - rec1_b_inner) / 2 - rec1_b_inner),

                    (x12 + rec1_a_inner / 2, y12 - (rec1_b_outer - rec1_b_inner) / 2),
                    (x12 - rec1_a_inner / 2, y12 - (rec1_b_outer - rec1_b_inner) / 2),

                    (x12 - rec1_a_inner / 2, y12 - (rec1_b_outer - rec1_b_inner) / 2 - rec1_b_inner),
                    ]

        points13 = [(x13 - rec2_a_outer / 2, y13 - rec2_b_outer),
                    (x13 - rec2_a_outer / 2, y13),
                    (x13 + rec2_a_outer / 2, y13),
                    (x13 + rec2_a_outer / 2, y13 - rec2_b_outer),

                    (x13 - rec2_a_outer / 2, y13 - rec2_b_outer),

                    (x13 - rec2_a_inner / 2, y13 - (rec2_b_outer - rec2_b_inner) / 2 - rec2_b_inner),
                    (x13 + rec2_a_inner / 2, y13 - (rec2_b_outer - rec2_b_inner) / 2 - rec2_b_inner),

                    (x13 + rec2_a_inner / 2, y13 - (rec2_b_outer - rec2_b_inner) / 2),
                    (x13 - rec2_a_inner / 2, y13 - (rec2_b_outer - rec2_b_inner) / 2),

                    (x13 - rec2_a_inner / 2, y13 - (rec2_b_outer - rec2_b_inner) / 2 - rec2_b_inner),
                    ]
        if self.add_JJ == False:
            squid = gdspy.PolygonSet([points0, points1, points2, points3, points4, points5_for_pad1, points6_for_pad2,
                                    points7, points8_for_pad3, points9_for_pad4, points10, points11, points12, points13])
        else:
            squid = gdspy.PolygonSet([points0, points1, points1_1, points2, points3, points4, points5_for_pad1, points6_for_pad2,
                                      points7, points8_for_pad3, points9_for_pad4,points10, points11, points12, points13])
        return squid
