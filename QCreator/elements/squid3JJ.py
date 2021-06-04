import gdspy


class JJ_2_small:
    def __init__(self, x0, y0, Hb, parametr2, parametr3, parametr4, parametr5, parametr6, add_JJ=False,
                 hole_in_squid_pad=True):
        self._x0 = x0
        self._y0 = y0

        self._parametr1 = Hb
        self._parametr2 = parametr2
        self._parametr3 = parametr3
        self._parametr4 = parametr4
        self._parametr5 = parametr5
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

        H_a = 0.5#1
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
                       (x1 + H_a, y1 - H_b / 3 - self._parametr4),
                       (x1 - 2 * H_a, y1 - H_b / 3 - self._parametr4),
                       (x1 - 2 * H_a, y1 - H_b),
                       (x1 - 3 * H_a, y1 - H_b)
                       ]

            points1_1 = [(x1 - H_a, y1),
                         (x1 - H_a/4, y1),
                         (x1 - H_a/4, y1 - H_b / 4),
                         (x1 - H_a + self._parametr3, y1 - H_b / 4),
                         (x1 - H_a + self._parametr3, y1 - H_b / 3 + h),
                         (x1 - H_a, y1 - H_b / 3 + h)
                         ]

        x2 = x1 - 3* H_a + L_a/4
        y2 = y1 - H_b

        points2 = [(x2 - L_a / 2, y2),
                   (x2 + L_a / 2, y2),
                   (x2 + L_a / 2, y2 - L_b),
                   (x2 - L_a / 2, y2 - L_b)
                   ]

        H1_a = self._parametr2#0.8
        H1_b = 1#2
        H2_a = self._parametr2#0.8
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

        pad1_a = self._parametr3
        pad1_b = 1#3
        pad2_a = self._parametr2
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
        pad3_b = self._parametr4

        pad4_a = 1.5#2.5
        pad4_b = self._parametr5

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
