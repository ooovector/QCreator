import gdspy

class JJ_1:
    def __init__(self, x0, y0, parametr1, parametr2):
        self._x0=x0
        self._y0=y0

        self._parametr1=parametr1
        self._parametr2=parametr2






    def generate_jj(self):

        #parametrs

        contact_pad1_a=16
        contact_pad1_b=10
        self.contact_pad_a = contact_pad1_a
        self.contact_pad_b = contact_pad1_b

        arm1_a=1
        arm1_b=5

        delta1=arm1_a*3

        hand1_a=2
        hand1_b=self._parametr1

        h=2

        contact_pad2_a=16
        contact_pad2_b=10

        arm2_a=1
        arm2_b=5

        hand2_a=self._parametr2
        hand2_b=7




        x1=self._x0-delta1
        y1=self._y0-contact_pad1_b

        x2=x1+arm1_a/2
        y2=y1-arm1_b

        x3=self._x0

        y3=self._y0-contact_pad1_b-arm1_b-arm2_b-hand2_b-h

        x4=x3
        y4=y3+arm2_b






        # contact pad 1
        points0=[(self._x0-contact_pad1_a/2,self._y0),(self._x0+contact_pad1_a/2,self._y0),(self._x0+contact_pad1_a/2,self._y0-contact_pad1_b),(self._x0-contact_pad1_a/2,self._y0-contact_pad1_b)]
        points1=[(x1-arm1_a/2,y1),(x1+arm1_a/2,y1),(x1+arm1_a/2,y1-arm1_b),(x1-arm1_a/2,y1-arm1_b)]
        points2=[(x2,y2),(x2,y2+hand1_b),(x2+hand1_a,y2+hand1_b),(x2+hand1_a,y2)]

        # contact pad 2
        # print(x3, y3)
        points3=[(x3-contact_pad1_a/2, y3) , (x3+contact_pad1_a/2, y3), (x3+contact_pad1_a/2, y3-contact_pad2_b), (x3-contact_pad1_a/2, y3-contact_pad2_b)]
        points4=[(x3-arm2_a/2,y3),(x3-arm2_a/2,y3+arm2_b), (x3+arm2_a/2,y3+arm2_b), (x3+arm2_a/2,y3)]
        points5=[(x4-hand2_a/2,y4),(x4-hand2_a/2,y4+hand2_b),(x4+hand2_a/2,y4+hand2_b),(x4+hand2_a/2,y4)]


        p0 = gdspy.Polygon(points0)
        p1 = gdspy.Polygon(points1)
        p2 = gdspy.Polygon(points2)
        p3 = gdspy.Polygon(points3)
        p4 = gdspy.Polygon(points4)
        p5 = gdspy.Polygon(points5)

        self.x_end = x3
        self.y_end = y3
        indent=1
        self.points_to_remove = [
            (self._x0-contact_pad1_a-indent,self._y0-contact_pad1_b/2-indent),(self._x0+contact_pad1_a+indent,self._y0-contact_pad1_b/2-indent),
            (x3 + contact_pad1_a+indent , y3 -indent), (x3 - contact_pad1_a-indent , y3-indent)
                             ]
        return  [p0, p1, p2, p3, p4, p4, p5]


#manhatten style junctions
class JJ_2:#hw,hd is the hole width, hole depth of the hole in the bridge between the two Plates of the ParallelPlate Transmon
    def __init__(self, x0, y0, parameter1, parameter2,hw,hd,l=20):
        self._x0=x0
        self._y0=y0
        self.hw = hw
        self.hd = hd
        self.l  = l

        self._parameter1=parameter1
        self._parameter2=parameter2

    def generate_jj(self):
        #parameters
        x0 = self._x0
        y0 = self._y0
        hw = self.hw
        hd = self.hd

        #bridge_gap = 10 um, bridge_width = 16
        #junction material and bridge gap is set to 1nm
        #l is the length of the thin materia piece extending beynd the initial block
        bg = 10
        bw = 16
        jg = 0.5
        l  = self.l #standard 20
        #left arm
        p0 = gdspy.Rectangle((x0-bg/2-bw/2-hw/2,y0+hd-jg),(x0-bg/2-bw/2-jg,y0-hd/2-jg))
        p1 = gdspy.Rectangle((x0-bg/2-bw/2-hw/4-jg/2-self._parameter1/2,y0-hd/2-jg),(x0-bg/2-bw/2-hw/4-jg/2+self._parameter1/2,y0-hd/2-jg-l))
        #right arm
        p2 = gdspy.Rectangle((x0+ bg/2 + hd-jg,y0-hw-bw/2+jg),(x0+ bg/2 -jg-hd/2,y0-hw-bw/2+hw/2))
        p3 = gdspy.Rectangle((x0+ bg/2 -jg-hd/2,y0-3/4*hw-bw/2+jg/2-self._parameter2/2),(x0+ bg/2 -jg-hd/2-l,y0-3/4*hw-bw/2+jg/2+self._parameter2/2))
        return [p0,p1,p2,p3]