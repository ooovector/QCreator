import gdspy

class JJ_2:
    def __init__(self, x0, y0, Hb, parametr2, parametr3, parametr4, parametr5 ,parametr6):
        self._x0=x0
        self._y0=y0

        self._parametr1 = Hb
        self._parametr2 = parametr2
        self._parametr3 = parametr3
        self._parametr4 = parametr4
        self._parametr5 = parametr5
        self._parametr6 = parametr6


    def generate_jj(self):
        contact_pad_a_outer = 10.5
        contact_pad_b_outer=3
        self.contact_pad_b_outer = contact_pad_b_outer
        self.contact_pad_a_outer = contact_pad_a_outer
        contact_pad_a_inner=7.5
        contact_pad_b_inner=1


        #Add contact pad1
        points0=[(self._x0-contact_pad_a_outer/2,self._y0-contact_pad_b_outer),
        (self._x0-contact_pad_a_outer/2,self._y0),
        (self._x0+contact_pad_a_outer/2,self._y0),
        (self._x0+contact_pad_a_outer/2,self._y0-contact_pad_b_outer),

        (self._x0-contact_pad_a_outer/2,self._y0-contact_pad_b_outer),

        (self._x0-contact_pad_a_inner/2,self._y0-(contact_pad_b_outer-contact_pad_b_inner)/2-contact_pad_b_inner),
        (self._x0+contact_pad_a_inner/2,self._y0-(contact_pad_b_outer-contact_pad_b_inner)/2-contact_pad_b_inner),

        (self._x0+contact_pad_a_inner/2,self._y0-(contact_pad_b_outer-contact_pad_b_inner)/2),
        (self._x0-contact_pad_a_inner/2,self._y0-(contact_pad_b_outer-contact_pad_b_inner)/2),

        (self._x0-contact_pad_a_inner/2,self._y0-(contact_pad_b_outer-contact_pad_b_inner)/2-contact_pad_b_inner),
        ]

        x1=self._x0
        y1=self._y0-contact_pad_b_outer

        #parametr1=H_b

        H_a=1
        H_b=self._parametr1

        L_a=10.7
        L_b=0.75

        points1=[(x1-H_a/2, y1),
        (x1+H_a/2, y1),
        (x1+H_a/2, y1-H_b),
        (x1-H_a/2, y1-H_b)
        ]

        x2=x1
        y2=y1-H_b

        points2=[(x2-L_a/2, y2),
        (x2+L_a/2, y2),
        (x2+L_a/2, y2-L_b),
        (x2-L_a/2, y2-L_b)
        ]

        H1_a=0.8
        H1_b=2
        H2_a=0.8
        H2_b=2

        x3=x2-L_a/2+H1_a/2
        y3=y2-L_b

        x4=x2+L_a/2-H1_a/2
        y4=y2-L_b

        points3=[(x3-H1_a/2, y3),
        (x3+H1_a/2, y3),
        (x3+H1_a/2, y3-H1_b),
        (x3-H1_a/2, y3-H1_b)
        ]

        points4=[(x4-H2_a/2, y4),
        (x4+H2_a/2, y4),
        (x4+H2_a/2, y4-H2_b),
        (x4-H2_a/2, y4-H2_b)
        ]

        x5=x3+H1_a/2
        y5=y3-H1_b
        x6=x4-H2_a/2
        y6=y4-H2_b

        #parametr2=pad1_a
        #parametr3=pad2_a

        pad1_a=self._parametr2
        pad1_b=3
        pad2_a=self._parametr2
        pad2_b=3

        points5_for_pad1=[(x5, y5),
                  (x5, y5-pad1_b),
                  (x5-pad1_a, y5-pad1_b),
                  (x5-pad1_a, y5)
                 ]

        points6_for_pad2=[(x6, y6),
                  (x6+pad2_a, y6),
                  (x6+pad2_a, y6-pad2_b),
                  (x6, y6-pad2_b)
        ]

        contact_pad1_a_outer=13
        contact_pad1_b_outer=6.4
        contact_pad1_a_inner=12
        contact_pad1_b_inner=5.8
        h=0.5
        x7=self._x0
        y7=self._y0-contact_pad_b_outer-H_b-L_b-H1_b-pad1_b-h-contact_pad1_b_outer
        x7_=x7
        y7_=y7+(contact_pad1_b_outer-contact_pad1_b_inner)

        points7=[(x7-contact_pad1_a_outer/2,y7+contact_pad1_b_outer),

         (x7-contact_pad1_a_outer/2,y7),

         (x7+contact_pad1_a_outer/2,y7),

         (x7+contact_pad1_a_outer/2,y7+contact_pad1_b_outer),

         (x7_+contact_pad1_a_inner/2,y7_+contact_pad1_b_inner),

         (x7_+contact_pad1_a_inner/2,y7_),

         (x7_-contact_pad1_a_inner/2,y7_),

         (x7_-contact_pad1_a_inner/2,y7_+contact_pad1_b_inner)]

        x8=x7_-contact_pad1_a_inner/2
        y8=y7_+contact_pad1_b_inner

        x9=x7_+contact_pad1_a_inner/2
        y9=y7_+contact_pad1_b_inner

         #parametr4=pad3_b
         #parametr5=pad4_b

        pad3_a=2.5
        pad3_b=self._parametr4

        pad4_a=2.5
        pad4_b=self._parametr5


        points8_for_pad3=[(x8,y8),

                  (x8+pad3_a,y8),

                  (x8+pad3_a,y8-pad3_b),

                  (x8,y8-pad3_b)]


        points9_for_pad4=[(x9-pad4_a,y9),

                  (x9,y9),

                  (x9,y9-pad4_b),

                  (x9-pad4_a,y9-pad4_b)]

        delta=6

        x10=x7-contact_pad1_a_outer/2
        y10=y7+delta

        x11=x7+contact_pad1_a_outer/2
        y11=y7+delta



        L1_a=2.1
        L1_b=1

        L2_a=2.1
        L2_b=1


        rec1_a_outer=4.8
        rec1_b_outer=2.8

        rec1_a_inner=2
        rec1_b_inner=1

        rec2_a_outer=rec1_a_outer
        rec2_b_outer=rec1_b_outer

        rec2_a_inner=rec1_a_inner
        rec2_b_inner=rec1_b_inner

        self.rect_size_a = rec1_a_outer
        self.rect_size_b = rec1_b_outer

        points10=[(x10-L1_a,y10),
          (x10,y10),
          (x10,y10-L1_b),
          (x10-L1_a,y10-L1_b)]

        points11=[(x11,y11),
          (x11+L2_a,y11),
          (x11+L2_a,y11-L2_b),
          (x11,y11-L2_b)]

        x12=x10-L1_a-(rec1_a_outer/2)
        y12=y10-L1_b/2+(rec1_b_outer/2)

        x13=x11+L2_a+(rec2_a_outer/2)
        y13=y11-L2_b/2+(rec2_b_outer/2)
        self.rect1 = (x12,y12)
        self.rect2 = (x13, y13)
        points12=[(x12-rec1_a_outer/2,y12-rec1_b_outer),
        (x12-rec1_a_outer/2,y12),
        (x12+rec1_a_outer/2,y12),
        (x12+rec1_a_outer/2,y12-rec1_b_outer),

        (x12-rec1_a_outer/2,y12-rec1_b_outer),

        (x12-rec1_a_inner/2,y12-(rec1_b_outer-rec1_b_inner)/2-rec1_b_inner),
        (x12+rec1_a_inner/2,y12-(rec1_b_outer-rec1_b_inner)/2-rec1_b_inner),

        (x12+rec1_a_inner/2,y12-(rec1_b_outer-rec1_b_inner)/2),
        (x12-rec1_a_inner/2,y12-(rec1_b_outer-rec1_b_inner)/2),

        (x12-rec1_a_inner/2,y12-(rec1_b_outer-rec1_b_inner)/2-rec1_b_inner),
        ]

        points13=[(x13-rec2_a_outer/2,y13-rec2_b_outer),
        (x13-rec2_a_outer/2,y13),
        (x13+rec2_a_outer/2,y13),
        (x13+rec2_a_outer/2,y13-rec2_b_outer),

        (x13-rec2_a_outer/2,y13-rec2_b_outer),

        (x13-rec2_a_inner/2,y13-(rec2_b_outer-rec2_b_inner)/2-rec2_b_inner),
        (x13+rec2_a_inner/2,y13-(rec2_b_outer-rec2_b_inner)/2-rec2_b_inner),

        (x13+rec2_a_inner/2,y13-(rec2_b_outer-rec2_b_inner)/2),
        (x13-rec2_a_inner/2,y13-(rec2_b_outer-rec2_b_inner)/2),

        (x13-rec2_a_inner/2,y13-(rec2_b_outer-rec2_b_inner)/2-rec2_b_inner),
        ]


        p0 = gdspy.Polygon(points0)
        p1 = gdspy.Polygon(points1)
        p2 = gdspy.Polygon(points2)
        p3 = gdspy.Polygon(points3)
        p4 = gdspy.Polygon(points4)
        p5 = gdspy.Polygon(points5_for_pad1)
        p6 = gdspy.Polygon(points6_for_pad2)
        p7 = gdspy.Polygon(points7)
        p8 = gdspy.Polygon(points8_for_pad3)
        p9 = gdspy.Polygon(points9_for_pad4)
        p10 = gdspy.Polygon(points10)
        p11 = gdspy.Polygon(points11)
        p12 = gdspy.Polygon(points12)
        p13 = gdspy.Polygon(points13)


        return  p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13
