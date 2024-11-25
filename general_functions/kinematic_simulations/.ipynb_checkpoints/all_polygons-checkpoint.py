'''
Library to build icosaehdra in a 3D space for Kinematic STEM HAADF simulations
'''
import numpy as np
from .volumes import Point, Face, Volume

SQRT_2 = np.sqrt(2)
SQRT_3 = np.sqrt(3)
SQRT_5 = np.sqrt(5)
SQRT_6 = np.sqrt(6)

def edge_dist(pointa, pointb):
    '''
    Calculate the distance between 2 points to get the initial base distance of our volumes
    '''
    return np.sqrt((pointa[0]-pointb[0])**2 + (pointa[1]-pointb[1])**2 + (pointa[2]-pointb[2])**2)

class RegularTetrahedron(Volume):
    '''
    Build a tetrahedron with a given middiameter, midradius or edge length
    '''

    def __init__(self, middiameter=None, midradius=None, edges_length=1):
        base_edges_length = edge_dist((0.0, 0.0, 1.732051), (1.632993, 0.0, -0.5773503))

        if middiameter is not None:
            self.middiameter = float(middiameter)
            self.midradius = self.middiameter / 2
            self.edges_length = self.middiameter / (2*(1/(2*SQRT_2)))

        elif midradius is not None:
            self.midradius = float(midradius)
            self.middiameter = 2 * self.midradius
            self.edges_length = self.midradius / ((1/(2*SQRT_2)))

        elif edges_length is not None:
            self.edges_length = float(edges_length)
            self.middiameter = 2 * (1/(2*SQRT_2)) * self.edges_length
            self.midradius = (1/(2*SQRT_2)) * self.edges_length

        scaler = self.edges_length/base_edges_length

        vert = [Point(np.array((0.0, 0.0, 1.732051)) * scaler),
                    Point(np.array((1.632993, 0.0, -0.5773503)) * scaler),
                    Point(np.array((-0.8164966, 1.414214, -0.5773503)) * scaler),
                    Point(np.array((-0.8164966, -1.414214, -0.5773503)) * scaler)]

        faces = [Face(points=(vert[0], vert[1], vert[2])),
                 Face(points=(vert[0], vert[2], vert[3])),
                 Face(points=(vert[0], vert[3], vert[1])),
                 Face(points=(vert[1], vert[3], vert[2]))]

        super().__init__(faces=faces)
        
class Pyramid(Volume):
    '''
    Build a pyramid with a given midiameter, midradius or edge length
    '''

    def __init__(self, middiameter=None, midradius=None, edges_length=1):
        base_edges_length = edge_dist((0.0, 0.0, 1.414214), (1.414214, 0.0, 0.0))

        if middiameter is not None:
            self.middiameter = float(middiameter)
            self.midradius = self.middiameter / 2
            self.edges_length = self.middiameter / (2*(1/2))

        elif midradius is not None:
            self.midradius = float(midradius)
            self.middiameter = 2 * self.midradius
            self.edges_length = self.midradius / ((1/2))

        elif edges_length is not None:
            self.edges_length = float(edges_length)
            self.middiameter = 2 * (1/2) * self.edges_length
            self.midradius = (1/2) * self.edges_length

        scaler = self.edges_length/base_edges_length

        vert = [Point(np.array((0.0, 0.0, 1.414214)) * scaler),
                    Point(np.array((1.414214, 0.0, 0.0)) * scaler),
                    Point(np.array((0.0, 1.414214, 0.0)) * scaler),
                    Point(np.array((-1.414214, 0.0, 0.0)) * scaler),
                    Point(np.array((0.0, -1.414214, 0.0)) * scaler)]

        faces = [Face(points=(vert[0], vert[1], vert[2])),
                 Face(points=(vert[0], vert[2], vert[3])),
                 Face(points=(vert[0], vert[3], vert[4])),
                 Face(points=(vert[0], vert[4], vert[1])),
                 Face(points=(vert[4], vert[3], vert[2], vert[1]))]

        super().__init__(faces=faces)
    

class RegularCube(Volume):
    '''
    Build a cube with a given middiameter, midradius or edge length
    '''

    def __init__(self, middiameter=None, midradius=None, edges_length=1):
        base_edges_length = edge_dist((0.0, 0.0, 1.224745), (1.154701, 0.0, 0.4082483))

        if middiameter is not None:
            self.middiameter = float(middiameter)
            self.midradius = self.middiameter / 2
            self.edges_length = self.middiameter / (2*(1/(SQRT_2)))

        elif midradius is not None:
            self.midradius = float(midradius)
            self.middiameter = 2 * self.midradius
            self.edges_length = self.midradius / ((1/(SQRT_2)))

        elif edges_length is not None:
            self.edges_length = float(edges_length)
            self.middiameter = 2 * (1/(SQRT_2)) * self.edges_length
            self.midradius = (1/(SQRT_2)) * self.edges_length

        scaler = self.edges_length/base_edges_length

        vert = [Point(np.array((0.0, 0.0, 1.224745)) * scaler),
                    Point(np.array((1.154701, 0.0, 0.4082483)) * scaler),
                    Point(np.array((-0.5773503, 1.0, 0.4082483)) * scaler),
                    Point(np.array((-0.5773503, -1.0, 0.4082483)) * scaler),
                    Point(np.array((0.5773503, 1.0, -0.4082483)) * scaler),
                    Point(np.array((0.5773503, -1.0, -0.4082483)) * scaler),
                    Point(np.array((-1.154701, 0.0, -0.4082483)) * scaler),
                    Point(np.array((0.0, 0.0, -1.224745)) * scaler)]

        faces = [Face(points=(vert[0], vert[1], vert[4], vert[2])),
                 Face(points=(vert[0], vert[2], vert[6], vert[3])),
                 Face(points=(vert[0], vert[3], vert[5], vert[1])),
                 Face(points=(vert[1], vert[5], vert[7], vert[4])),
                 Face(points=(vert[2], vert[4], vert[7], vert[6])),
                 Face(points=(vert[3], vert[6], vert[7], vert[5]))]

        super().__init__(faces=faces)

class RegularOctahedron(Volume):
    '''
    Build an octahedron with a given middiameter, midradius or edge length
    '''

    def __init__(self, middiameter=None, midradius=None, edges_length=1):
        base_edges_length = edge_dist((0.0, 0.0, 1.414214), (1.414214, 0.0, 0.0))

        if middiameter is not None:
            self.middiameter = float(middiameter)
            self.midradius = self.middiameter / 2
            self.edges_length = self.middiameter / (2*(1/2))

        elif midradius is not None:
            self.midradius = float(midradius)
            self.middiameter = 2 * self.midradius
            self.edges_length = self.midradius / ((1/2))

        elif edges_length is not None:
            self.edges_length = float(edges_length)
            self.middiameter = 2 * (1/2) * self.edges_length
            self.midradius = (1/2) * self.edges_length

        scaler = self.edges_length/base_edges_length

        vert = [Point(np.array((0.0, 0.0, 1.414214)) * scaler),
                    Point(np.array((1.414214, 0.0, 0.0)) * scaler),
                    Point(np.array((0.0, 1.414214, 0.0)) * scaler),
                    Point(np.array((-1.414214, 0.0, 0.0)) * scaler),
                    Point(np.array((0.0, -1.414214, 0.0)) * scaler),
                    Point(np.array((0.0, 0.0, -1.414214)) * scaler)]

        faces = [Face(points=(vert[0], vert[1], vert[2])),
                 Face(points=(vert[0], vert[2], vert[3])),
                 Face(points=(vert[0], vert[3], vert[4])),
                 Face(points=(vert[0], vert[4], vert[1])),
                 Face(points=(vert[1], vert[4], vert[5])),
                 Face(points=(vert[1], vert[5], vert[2])),
                 Face(points=(vert[2], vert[5], vert[3])),
                 Face(points=(vert[3], vert[5], vert[4]))]

        super().__init__(faces=faces)

class RegularDodecahedron(Volume):
    '''
    Build a dodecahedron with a given middiameter, midradius or edge length
    '''

    def __init__(self, middiameter=None, midradius=None, edges_length=1):
        base_edges_length = edge_dist((0.0, 0.0, 1.070466), (0.7136442, 0.0, 0.7978784))

        if middiameter is not None:
            self.middiameter = float(middiameter)
            self.midradius = self.middiameter / 2
            self.edges_length = self.middiameter / (2*((1/4)*(3+SQRT_5)))

        elif midradius is not None:
            self.midradius = float(midradius)
            self.middiameter = 2 * self.midradius
            self.edges_length = self.midradius / (((1/4)*(3+SQRT_5)))

        elif edges_length is not None:
            self.edges_length = float(edges_length)
            self.middiameter = 2 * ((1/4)*(3+SQRT_5)) * self.edges_length
            self.midradius = ((1/4)*(3+SQRT_5)) * self.edges_length

        scaler = self.edges_length/base_edges_length

        vert = [Point(np.array((0.0, 0.0, 1.070466)) * scaler),
                Point(np.array((0.7136442, 0.0, 0.7978784)) * scaler),
                Point(np.array((-0.3568221, 0.618034, 0.7978784)) * scaler),
                Point(np.array((-0.3568221, -0.618034, 0.7978784)) * scaler),
                Point(np.array((0.7978784, 0.618034, 0.3568221)) * scaler),
                Point(np.array((0.7978784, -0.618034, 0.3568221)) * scaler),
                Point(np.array((-0.9341724, 0.381966, 0.3568221)) * scaler),
                Point(np.array((0.1362939, 1.0, 0.3568221)) * scaler),
                Point(np.array((0.1362939, -1.0, 0.3568221)) * scaler),
                Point(np.array((-0.9341724, -0.381966, 0.3568221)) * scaler),
                Point(np.array((0.9341724, 0.381966, -0.3568221)) * scaler),
                Point(np.array((0.9341724, -0.381966, -0.3568221)) * scaler),
                Point(np.array((-0.7978784, 0.618034, -0.3568221)) * scaler),
                Point(np.array((-0.1362939, 1.0, -0.3568221)) * scaler),
                Point(np.array((-0.1362939, -1.0, -0.3568221)) * scaler),
                Point(np.array((-0.7978784, -0.618034, -0.3568221)) * scaler),
                Point(np.array((0.3568221, 0.618034, -0.7978784)) * scaler),
                Point(np.array((0.3568221, -0.618034, -0.7978784)) * scaler),
                Point(np.array((-0.7136442, 0.0, -0.7978784)) * scaler),
                Point(np.array((0.0, 0.0, -1.070466)) * scaler)]

        faces = [Face(points=(vert[0], vert[1], vert[4], vert[7], vert[2])),
                 Face(points=(vert[0], vert[2], vert[6], vert[9], vert[3])),
                 Face(points=(vert[0], vert[3], vert[8], vert[5], vert[1])),
                 Face(points=(vert[1], vert[5], vert[11], vert[10], vert[4])),
                 Face(points=(vert[2], vert[7], vert[13], vert[12], vert[6])),
                 Face(points=(vert[3], vert[9], vert[15], vert[14], vert[8])),
                 Face(points=(vert[4], vert[10], vert[16], vert[13], vert[7])),
                 Face(points=(vert[5], vert[8], vert[14], vert[17], vert[11])),
                 Face(points=(vert[6], vert[12], vert[18], vert[15], vert[9])),
                 Face(points=(vert[10], vert[11], vert[17], vert[19], vert[16])),
                 Face(points=(vert[12], vert[13], vert[16], vert[19], vert[18])),
                 Face(points=(vert[14], vert[15], vert[18], vert[19], vert[17]))]

        super().__init__(faces=faces)

class RegularIcosahedron(Volume):
    '''
    Build an icosahedron with a given middiameter, midradius or edge length
    '''

    def __init__(self, middiameter=None, midradius=None, edges_length=1):
        base_edges_length = edge_dist((0.0, 0.0, 1.175571), (1.051462, 0.0, 0.5257311))

        if middiameter is not None:
            self.middiameter = float(middiameter)
            self.midradius = self.middiameter / 2
            self.edges_length = self.middiameter / (2*((1 + 5 ** 0.5) / 4))

        elif midradius is not None:
            self.midradius = float(midradius)
            self.middiameter = 2 * self.midradius
            self.edges_length = self.midradius / (((1 + 5 ** 0.5) / 4))

        elif edges_length is not None:
            self.edges_length = float(edges_length)
            self.middiameter = 2 * ((1 + 5 ** 0.5) / 4) * self.edges_length
            self.midradius = ((1 + 5 ** 0.5) / 4) * self.edges_length

        scaler = self.edges_length/base_edges_length

        vert = [Point(np.array((0.0, 0.0, 1.175571)) * scaler),
                Point(np.array((1.051462, 0.0, 0.5257311)) * scaler),
                Point(np.array((0.3249197, 1.0, 0.5257311)) * scaler),
                Point(np.array((-0.8506508, 0.618034, 0.5257311)) * scaler),
                Point(np.array((-0.8506508, -0.618034, 0.5257311)) * scaler),
                Point(np.array((0.3249197, -1.0, 0.5257311)) * scaler),
                Point(np.array((0.8506508, 0.618034, -0.5257311)) * scaler),
                Point(np.array((0.8506508, -0.618034, -0.5257311)) * scaler),
                Point(np.array((-0.3249197, 1.0, -0.5257311)) * scaler),
                Point(np.array((-1.051462, 0.0, -0.5257311)) * scaler),
                Point(np.array((-0.3249197, -1.0, -0.5257311)) * scaler),
                Point(np.array((0.0, 0.0, -1.175571)) * scaler)]

        faces = [Face(points=(vert[0], vert[1], vert[2])),
                 Face(points=(vert[0], vert[2], vert[3])),
                 Face(points=(vert[0], vert[3], vert[4])),
                 Face(points=(vert[0], vert[4], vert[5])),
                 Face(points=(vert[0], vert[5], vert[1])),
                 Face(points=(vert[1], vert[5], vert[7])),
                 Face(points=(vert[1], vert[7], vert[6])),
                 Face(points=(vert[1], vert[6], vert[2])),
                 Face(points=(vert[2], vert[6], vert[8])),
                 Face(points=(vert[2], vert[8], vert[3])),
                 Face(points=(vert[3], vert[8], vert[9])),
                 Face(points=(vert[3], vert[9], vert[4])),
                 Face(points=(vert[4], vert[9], vert[10])),
                 Face(points=(vert[4], vert[10], vert[5])),
                 Face(points=(vert[5], vert[10], vert[7])),
                 Face(points=(vert[6], vert[7], vert[11])),
                 Face(points=(vert[6], vert[11], vert[8])),
                 Face(points=(vert[7], vert[10], vert[11])),
                 Face(points=(vert[8], vert[11], vert[9])),
                 Face(points=(vert[9], vert[11], vert[10]))]

        super().__init__(faces=faces)

class Cuboctahedron(Volume):
    '''
    Build a cuboctahedron with a given middiameter, midradius or edge length
    '''

    def __init__(self, middiameter=None, midradius=None, edges_length=1):
        base_edges_length = edge_dist((0.0, 0.0, 1.154701), (1.0, 0.0, 0.5773503))

        if middiameter is not None:
            self.middiameter = float(middiameter)
            self.midradius = self.middiameter / 2
            self.edges_length = self.middiameter / (2*(SQRT_3/2))

        elif midradius is not None:
            self.midradius = float(midradius)
            self.middiameter = 2 * self.midradius
            self.edges_length = self.midradius / ((SQRT_3/2))

        elif edges_length is not None:
            self.edges_length = float(edges_length)
            self.middiameter = 2 * (SQRT_3/2) * self.edges_length
            self.midradius = (SQRT_3/2) * self.edges_length

        scaler = self.edges_length/base_edges_length

        vert = [Point(np.array((0.0, 0.816, -0.816)) * scaler),  #00
                Point(np.array((-0.816, 0.0, -0.816)) * scaler), #01
                Point(np.array((-0.816, 0.816, 0.0)) * scaler),  #02
                Point(np.array((0.816, 0.816, 0.0)) * scaler),   #03
                Point(np.array((0.816, 0.0, -0.816)) * scaler),  #04
                Point(np.array((0.0, -0.816, -0.816)) * scaler), #05
                Point(np.array((-0.816, -0.816, 0.0)) * scaler), #06
                Point(np.array((-0.816, 0.0, 0.816)) * scaler),  #07
                Point(np.array((0.0, 0.816, 0.816)) * scaler),   #08
                Point(np.array((0.816, 0.0, 0.816)) * scaler),   #09
                Point(np.array((0.816,-0.816,0.0)) * scaler),    #10
                Point(np.array((0.0, -0.816, 0.816)) * scaler)]  #11

        faces = [Face(points=(vert[0], vert[1], vert[2])),
                 Face(points=(vert[0], vert[3], vert[4])),
                 Face(points=(vert[1], vert[5], vert[6])),
                 Face(points=(vert[2], vert[7], vert[8])),
                 Face(points=(vert[3], vert[8], vert[9])),
                 Face(points=(vert[4], vert[10], vert[5])),
                 Face(points=(vert[6], vert[11], vert[7])),
                 Face(points=(vert[9], vert[11], vert[10])),
                 Face(points=(vert[0], vert[2], vert[8], vert[3])),
                 Face(points=(vert[0], vert[4], vert[5], vert[1])),
                 Face(points=(vert[1], vert[6], vert[7], vert[2])),
                 Face(points=(vert[3], vert[9], vert[10], vert[4])),
                 Face(points=(vert[6], vert[5], vert[10], vert[11])),
                 Face(points=(vert[8], vert[7], vert[11], vert[9]))]

        super().__init__(faces=faces)

class TruncatedCube(Volume):
    '''
    Build a truncated cube with a given middiameter, midradius or edge length
    '''

    def __init__(self, middiameter=None, midradius=None, edges_length=1):
        base_edges_length = edge_dist((0.0, 0.0, 1.042011), (0.5621693, 0.0, 0.8773552))

        if middiameter is not None:
            self.middiameter = float(middiameter)
            self.midradius = self.middiameter / 2
            self.edges_length = self.middiameter / (2*((1/2)*(2+SQRT_2)))

        elif midradius is not None:
            self.midradius = float(midradius)
            self.middiameter = 2 * self.midradius
            self.edges_length = self.midradius / (((1/2)*(2+SQRT_2)))

        elif edges_length is not None:
            self.edges_length = float(edges_length)
            self.middiameter = 2 * ((1/2)*(2+SQRT_2)) * self.edges_length
            self.midradius = ((1/2)*(2+SQRT_2)) * self.edges_length

        scaler = self.edges_length/base_edges_length

        vert = [Point(np.array((0.0, 0.0, 1.042011)) * scaler),
                Point(np.array((0.5621693, 0.0, 0.8773552)) * scaler),
                Point(np.array((-0.4798415, 0.2928932, 0.8773552)) * scaler),
                Point(np.array((0.2569714, -0.5, 0.8773552)) * scaler),
                Point(np.array((0.8773552, 0.2928932, 0.4798415)) * scaler),
                Point(np.array((-0.9014684, 0.2071068, 0.4798415)) * scaler),
                Point(np.array((-0.5962706, 0.7071068, 0.4798415)) * scaler),
                Point(np.array((0.1405423, -0.9142136, 0.4798415)) * scaler),
                Point(np.array((1.017898, 0.2071068, -0.08232778)) * scaler),
                Point(np.array((0.7609261, 0.7071068, 0.08232778)) * scaler),
                Point(np.array((-1.017898, -0.2071068, 0.08232778)) * scaler),
                Point(np.array((-0.2810846, 1.0, 0.08232778)) * scaler),
                Point(np.array((-0.2810846, -1.0, 0.08232778)) * scaler),
                Point(np.array((0.2810846, -1.0, -0.08232778)) * scaler),
                Point(np.array((0.9014684, -0.2071068, -0.4798415)) * scaler),
                Point(np.array((0.2810846, 1.0, -0.08232778)) * scaler),
                Point(np.array((-0.7609261, -0.7071068, -0.08232778)) * scaler),
                Point(np.array((-0.8773552, -0.2928932, -0.4798415)) * scaler),
                Point(np.array((-0.1405423, 0.9142136, -0.4798415)) * scaler),
                Point(np.array((0.5962706, -0.7071068, -0.4798415)) * scaler),
                Point(np.array((0.4798415, -0.2928932, -0.8773552)) * scaler),
                Point(np.array((-0.5621693, 0.0, -0.8773552)) * scaler),
                Point(np.array((-0.2569714, 0.5, -0.8773552)) * scaler),
                Point(np.array((0.0, 0.0, -1.042011)) * scaler)]

        faces = [Face(points=(vert[0], vert[3], vert[1])),
                 Face(points=(vert[2], vert[6], vert[5])),
                 Face(points=(vert[4], vert[8], vert[9])),
                 Face(points=(vert[7], vert[12], vert[13])),
                 Face(points=(vert[10], vert[17], vert[16])),
                 Face(points=(vert[11], vert[15], vert[18])),
                 Face(points=(vert[14], vert[19], vert[20])),
                 Face(points=(vert[21], vert[22], vert[23])),
                 Face(points=(vert[0], vert[1], vert[4], vert[9],
                              vert[15], vert[11], vert[6], vert[2])),
                 Face(points=(vert[0], vert[2], vert[5], vert[10],
                              vert[16], vert[12], vert[7], vert[3])),
                 Face(points=(vert[1], vert[3], vert[7], vert[13],
                              vert[19], vert[14], vert[8], vert[4])),
                 Face(points=(vert[5], vert[6], vert[11], vert[18],
                              vert[22], vert[21], vert[17], vert[10])),
                 Face(points=(vert[8], vert[14], vert[20], vert[23],
                              vert[22], vert[18], vert[15], vert[9])),
                 Face(points=(vert[12], vert[16], vert[17], vert[21],
                              vert[23], vert[20], vert[19], vert[13]))]

        super().__init__(faces=faces)

class TruncatedOctahedron(Volume):
    '''
    Build a truncated octahedron with a given middiameter, midradius or edge length
    '''

    def __init__(self, middiameter=None, midradius=None, edges_length=1):
        base_edges_length = edge_dist((0.0, 0.0, 1.054093), (0.6324555, 0.0, 0.843274))

        if middiameter is not None:
            self.middiameter = float(middiameter)
            self.midradius = self.middiameter / 2
            self.edges_length = self.middiameter / (2*(3/2))

        elif midradius is not None:
            self.midradius = float(midradius)
            self.middiameter = 2 * self.midradius
            self.edges_length = self.midradius / ((3/2))

        elif edges_length is not None:
            self.edges_length = float(edges_length)
            self.middiameter = 2 * (3/2) * self.edges_length
            self.midradius = (3/2) * self.edges_length

        scaler = self.edges_length/base_edges_length

        vert = [Point(np.array((0.0, 0.0, 1.054093)) * scaler),
                Point(np.array((0.6324555, 0.0, 0.843274)) * scaler),
                Point(np.array((-0.421637, 0.4714045, 0.843274)) * scaler),
                Point(np.array((-0.07027284, -0.6285394, 0.843274)) * scaler),
                Point(np.array((0.843274, 0.4714045, 0.421637)) * scaler),
                Point(np.array((0.5621827, -0.6285394, 0.6324555)) * scaler),
                Point(np.array((-0.9135469, 0.3142697, 0.421637)) * scaler),
                Point(np.array((-0.2108185, 0.942809, 0.421637)) * scaler),
                Point(np.array((-0.5621827, -0.7856742, 0.421637)) * scaler),
                Point(np.array((0.9838197, 0.3142697, -0.2108185)) * scaler),
                Point(np.array((0.421637, 0.942809, 0.2108185)) * scaler),
                Point(np.array((0.7027284, -0.7856742, 0.0)) * scaler),
                Point(np.array((-0.7027284, 0.7856742, 0.0)) * scaler),
                Point(np.array((-0.9838197, -0.3142697, 0.2108185)) * scaler),
                Point(np.array((-0.421637, -0.942809, -0.2108185)) * scaler),
                Point(np.array((0.5621827, 0.7856742, -0.421637)) * scaler),
                Point(np.array((0.9135469, -0.3142697, -0.421637)) * scaler),
                Point(np.array((0.2108185, -0.942809, -0.421637)) * scaler),
                Point(np.array((-0.5621827, 0.6285394, -0.6324555)) * scaler),
                Point(np.array((-0.843274, -0.4714045, -0.421637)) * scaler),
                Point(np.array((0.07027284, 0.6285394, -0.843274)) * scaler),
                Point(np.array((0.421637, -0.4714045, -0.843274)) * scaler),
                Point(np.array((-0.6324555, 0.0, -0.843274)) * scaler),
                Point(np.array((0.0, 0.0, -1.054093)) * scaler)]

        faces = [Face(points=(vert[0], vert[3], vert[5], vert[1])),
                 Face(points=(vert[2], vert[7], vert[12], vert[6])),
                 Face(points=(vert[4], vert[9], vert[15], vert[10])),
                 Face(points=(vert[8], vert[13], vert[19], vert[14])),
                 Face(points=(vert[11], vert[17], vert[21], vert[16])),
                 Face(points=(vert[18], vert[20], vert[23], vert[22])),
                 Face(points=(vert[0], vert[1], vert[4], vert[10], vert[7], vert[2])),
                 Face(points=(vert[0], vert[2], vert[6], vert[13], vert[8], vert[3])),
                 Face(points=(vert[1], vert[5], vert[11], vert[16], vert[9], vert[4])),
                 Face(points=(vert[3], vert[8], vert[14], vert[17], vert[11], vert[5])),
                 Face(points=(vert[6], vert[12], vert[18], vert[22], vert[19], vert[13])),
                 Face(points=(vert[7], vert[10], vert[15], vert[20], vert[18], vert[12])),
                 Face(points=(vert[9], vert[16], vert[21], vert[23], vert[20], vert[15])),
                 Face(points=(vert[14], vert[19], vert[22], vert[23], vert[21], vert[17]))]

        super().__init__(faces=faces)

class TriangularOrthobicupola(Volume):
    '''
    Build a triangular bicupola (J27) with a given middiameter, midradius or edge length
    '''

    def __init__(self, middiameter=None, midradius=None, edges_length=1):
        base_edges_length = edge_dist((-0.288675, 0.500000, 0.816497),
                                      (0.577350, -0.000000, 0.816497))

        if middiameter is not None:
            self.middiameter = float(middiameter)
            self.midradius = self.middiameter / 2
            self.edges_length = self.middiameter / (2*(SQRT_3/2))

        elif midradius is not None:
            self.midradius = float(midradius)
            self.middiameter = 2 * self.midradius
            self.edges_length = self.midradius / ((SQRT_3/2))

        elif edges_length is not None:
            self.edges_length = float(edges_length)
            self.middiameter = 2 * (SQRT_3/2) * self.edges_length
            self.midradius = (SQRT_3/2) * self.edges_length

        scaler = self.edges_length/base_edges_length

        vert = [Point(np.array((-0.288675, 0.500000, 0.816497)) * scaler),
                Point(np.array((0.577350, -0.000000, 0.816497)) * scaler),
                Point(np.array((0.866025, 0.500000, 0.000000)) * scaler),
                Point(np.array((-0.866025, 0.500000, -0.000000)) * scaler),
                Point(np.array((-0.769800, -0.333334, 0.544331)) * scaler),
                Point(np.array((0.866025, -0.500000, -0.000000)) * scaler),
                Point(np.array((0.096225, -0.833333, 0.544331)) * scaler),
                Point(np.array((-0.000000, 1.000000, 0.000000)) * scaler),
                Point(np.array((0.288675, 0.500000, -0.816496)) * scaler),
                Point(np.array((-0.577350, 0.000000, -0.816497)) * scaler),
                Point(np.array((-0.481125, -0.833333, -0.272166)) * scaler),
                Point(np.array((0.288675, -0.500000, -0.816497)) * scaler)]

        faces = [Face(points=(vert[1], vert[5], vert[2])),
                 Face(points=(vert[0], vert[3], vert[4])),
                 Face(points=(vert[1], vert[6], vert[5])),
                 Face(points=(vert[2], vert[8], vert[7])),
                 Face(points=(vert[0], vert[7], vert[3])),
                 Face(points=(vert[4], vert[10], vert[6])),
                 Face(points=(vert[11], vert[9], vert[8])),
                 Face(points=(vert[9], vert[11], vert[10])),
                 Face(points=(vert[1], vert[2], vert[7], vert[0])),
                 Face(points=(vert[0], vert[4], vert[6], vert[1])),
                 Face(points=(vert[5], vert[11], vert[8], vert[2])),
                 Face(points=(vert[3], vert[9], vert[10], vert[4])),
                 Face(points=(vert[5], vert[6], vert[10], vert[11])),
                 Face(points=(vert[7], vert[8], vert[9], vert[3]))]

        super().__init__(faces=faces)

class TriangularBipyramid(Volume):
    '''
    Build a triangular bipyramid (J12) with a given middiameter (twice the tetrahedron middiameter),
    midradius (twice the tetrahedron midradius) or edge length
    '''

    def __init__(self, middiameter=None, midradius=None, edges_length=1):
        base_edges_length = edge_dist((-0.610389, 0.243975, 0.531213),
                                      (-0.187812, -0.48795, -0.664016))

        if middiameter is not None:
            self.middiameter = float(middiameter)
            self.midradius = self.middiameter / 2
            self.edges_length = self.middiameter / (2*(2/(2*SQRT_2)))

        elif midradius is not None:
            self.midradius = float(midradius)
            self.middiameter = 2 * self.midradius
            self.edges_length = self.midradius / ((2/(2*SQRT_2)))

        elif edges_length is not None:
            self.edges_length = float(edges_length)
            self.middiameter = 2 * (2/(2*SQRT_2)) * self.edges_length
            self.midradius = (2/(2*SQRT_2)) * self.edges_length

        scaler = self.edges_length/base_edges_length

        vert = [Point(np.array((-0.610389, 0.243975, 0.531213)) * scaler),
                Point(np.array((-0.187812, -0.487950, -0.664016)) * scaler),
                Point(np.array((-0.187812, 0.975900, -0.664016)) * scaler),
                Point(np.array((0.187812, -0.975900, 0.664016)) * scaler),
                Point(np.array((0.798201, 0.243975, 0.132803)) * scaler)]

        faces = [Face(points=(vert[1], vert[3], vert[0])),
                 Face(points=(vert[3], vert[4], vert[0])),
                 Face(points=(vert[3], vert[1], vert[4])),
                 Face(points=(vert[0], vert[2], vert[1])),
                 Face(points=(vert[0], vert[4], vert[2])),
                 Face(points=(vert[2], vert[4], vert[1]))]

        super().__init__(faces=faces)

class HexagonalPlate(Volume):
    '''
    Build an hexagonal plate with a given hexagonal edges length and thick
    '''

    def __init__(self, edges_length, thick=None):
        self.edges_length = edges_length

        if thick is None:
            self.thick = self.edges_length
        else:
            self.thick = thick

        vert = [Point((self.edges_length, 0.000000, self.thick/2)),
                Point((self.edges_length/2, self.edges_length*SQRT_3/2, self.thick/2)),
                Point((-self.edges_length/2, self.edges_length*SQRT_3/2, self.thick/2)),
                Point((-self.edges_length, 0.000000, self.thick/2)),
                Point((-self.edges_length/2, -self.edges_length*SQRT_3/2, self.thick/2)),
                Point((self.edges_length/2, -self.edges_length*SQRT_3/2, self.thick/2)),
                Point((self.edges_length, 0.000000, -self.thick/2)),
                Point((self.edges_length/2, self.edges_length*SQRT_3/2, -self.thick/2)),
                Point((-self.edges_length/2, self.edges_length*SQRT_3/2, -self.thick/2)),
                Point((-self.edges_length, 0.000000, -self.thick/2)),
                Point((-self.edges_length/2, -self.edges_length*SQRT_3/2, -self.thick/2)),
                Point((self.edges_length/2, -self.edges_length*SQRT_3/2, -self.thick/2))]

        faces = [Face(points=(vert[0], vert[6], vert[7], vert[1])),
                 Face(points=(vert[1], vert[7], vert[8], vert[2])),
                 Face(points=(vert[2], vert[8], vert[9], vert[3])),
                 Face(points=(vert[3], vert[9], vert[10], vert[4])),
                 Face(points=(vert[4], vert[10], vert[11], vert[5])),
                 Face(points=(vert[5], vert[11], vert[6], vert[0])),
                 Face(points=(vert[0], vert[1], vert[2], vert[3], vert[4], vert[5])),
                 Face(points=(vert[11], vert[10], vert[9], vert[8], vert[7], vert[6]))]

        super().__init__(faces=faces)

class TriangularPlate(Volume):
    '''
    Build a triangular plate with a given middiameter, midradius or edge length
    '''

    def __init__(self, edges_length, thick=None, ratio=0.85):

        if thick is not None:
            self.edges_length = float(edges_length)
            self.thick = float(thick)
            self.ratio = 1 - ((self.thick*SQRT_3)/self.edges_length)
            self.ratio = max(self.ratio, 0)

        elif ratio is not None:
            self.edges_length = float(edges_length)
            if ratio >= 1:
                raise ValueError("The ratio must be inferior to 1 so that the volume is valid")
            self.ratio = ratio
            self.thick = (self.edges_length/SQRT_3)*(1-self.ratio)

        if self.ratio != 0:
            vert = [Point((self.edges_length/SQRT_3, 0, 0)),
                    Point((-self.edges_length/(2*SQRT_3), self.edges_length/2, 0)),
                    Point((-self.edges_length/(2*SQRT_3), -self.edges_length/2, 0)),
                    Point((self.edges_length*ratio/SQRT_3, 0,
                           self.edges_length*(1-ratio)/(SQRT_6))),
                    Point((-self.edges_length*ratio/(2*SQRT_3), self.edges_length*ratio/2,
                        self.edges_length*(1-ratio)/(SQRT_6))),
                    Point((-self.edges_length*ratio/(2*SQRT_3), -self.edges_length*ratio/2,
                        self.edges_length*(1-ratio)/(SQRT_6))),
                    Point((self.edges_length*ratio/SQRT_3, 0,
                           -self.edges_length*(1-ratio)/(SQRT_6))),
                    Point((-self.edges_length*ratio/(2*SQRT_3), self.edges_length*ratio/2,
                        -self.edges_length*(1-ratio)/(SQRT_6))),
                    Point((-self.edges_length*ratio/(2*SQRT_3), -self.edges_length*ratio/2,
                        -self.edges_length*(1-ratio)/(SQRT_6)))]

            faces = [Face(points=(vert[3], vert[4], vert[5])),
                    Face(points=(vert[6], vert[8], vert[7])),
                    Face(points=(vert[0], vert[1], vert[4], vert[3])),
                    Face(points=(vert[1], vert[2], vert[5], vert[4])),
                    Face(points=(vert[2], vert[0], vert[3], vert[5])),
                    Face(points=(vert[0], vert[6], vert[7], vert[1])),
                    Face(points=(vert[1], vert[7], vert[8], vert[2])),
                    Face(points=(vert[2], vert[8], vert[6], vert[0]))]
        else:
            vert = [Point((self.edges_length/SQRT_3, 0, 0)),
                    Point((-self.edges_length/(2*SQRT_3), self.edges_length/2, 0)),
                    Point((-self.edges_length/(2*SQRT_3), -self.edges_length/2, 0)),
                    Point((0, 0, self.edges_length*(1-ratio)/(SQRT_6))),
                    Point((0, 0, -self.edges_length*(1-ratio)/(SQRT_6)))]

            faces = [Face(points=(vert[0], vert[1], vert[3])),
                     Face(points=(vert[1], vert[2], vert[3])),
                     Face(points=(vert[2], vert[0], vert[3])),
                     Face(points=(vert[1], vert[0], vert[4])),
                     Face(points=(vert[2], vert[1], vert[4])),
                     Face(points=(vert[0], vert[2], vert[4])),]

        super().__init__(faces=faces)

class PentagonalRod(Volume):
    '''
    Build a pentagonal rod with a given pentagonal edges length and thick
    '''
    def __init__(self, edges_length, thick=None):
        self.edges_length = edges_length

        if thick is None:
            self.thick = self.edges_length
        else:
            self.thick = thick

        self.pyramid_height = self.edges_length * (np.sqrt(1-(2/(5-SQRT_5))))
        self.p_radius = self.edges_length * np.sqrt(2/(5-SQRT_5))

        if self.thick == 0:
            vertices = [Point((0, 0, self.pyramid_height)),
                        Point((self.p_radius, 0, 0)),
                        Point((self.p_radius*0.309017, self.p_radius*0.951057, 0)),
                        Point((self.p_radius*-0.809017, self.p_radius*0.587785, 0)),
                        Point((self.p_radius*-0.809017, -self.p_radius*0.587785, 0)),
                        Point((self.p_radius*0.309017, -self.p_radius*0.951057, 0)),
                        Point((0, 0, -self.pyramid_height))]

            faces = [Face(points=(vertices[0], vertices[1], vertices[2])),
                     Face(points=(vertices[0], vertices[2], vertices[3])),
                     Face(points=(vertices[0], vertices[3], vertices[4])),
                     Face(points=(vertices[0], vertices[4], vertices[5])),
                     Face(points=(vertices[0], vertices[5], vertices[1])),
                     Face(points=(vertices[2], vertices[1], vertices[6])),
                     Face(points=(vertices[3], vertices[2], vertices[6])),
                     Face(points=(vertices[4], vertices[3], vertices[6])),
                     Face(points=(vertices[5], vertices[4], vertices[6])),
                     Face(points=(vertices[1], vertices[5], vertices[6])),]

        else:
            vertices = [Point((0, 0, self.pyramid_height+(self.thick/2))),
                        Point((self.p_radius, 0, (self.thick/2))),
                        Point((self.p_radius*0.309017, self.p_radius*0.951057,
                               (self.thick/2))),
                        Point((self.p_radius*-0.809017, self.p_radius*0.587785,
                               (self.thick/2))),
                        Point((self.p_radius*-0.809017, -self.p_radius*0.587785,
                               (self.thick/2))),
                        Point((self.p_radius*0.309017, -self.p_radius*0.951057,
                               (self.thick/2))),
                        Point((self.p_radius, 0, -(self.thick/2))),
                        Point((self.p_radius*0.309017, self.p_radius*0.951057,
                               -(self.thick/2))),
                        Point((self.p_radius*-0.809017, self.p_radius*0.587785,
                               -(self.thick/2))),
                        Point((self.p_radius*-0.809017, -self.p_radius*0.587785,
                               -(self.thick/2))),
                        Point((self.p_radius*0.309017, -self.p_radius*0.951057,
                               -(self.thick/2))),
                        Point((0, 0, -self.pyramid_height-(self.thick/2)))]

            faces = [Face(points=(vertices[0], vertices[1], vertices[2])),
                    Face(points=(vertices[0], vertices[2], vertices[3])),
                    Face(points=(vertices[0], vertices[3], vertices[4])),
                    Face(points=(vertices[0], vertices[4], vertices[5])),
                    Face(points=(vertices[0], vertices[5], vertices[1])),
                    Face(points=(vertices[7], vertices[6], vertices[11])),
                    Face(points=(vertices[8], vertices[7], vertices[11])),
                    Face(points=(vertices[9], vertices[8], vertices[11])),
                    Face(points=(vertices[10], vertices[9], vertices[11])),
                    Face(points=(vertices[6], vertices[10], vertices[11])),
                    Face(points=(vertices[1], vertices[6], vertices[7], vertices[2])),
                    Face(points=(vertices[2], vertices[7], vertices[8], vertices[3])),
                    Face(points=(vertices[3], vertices[8], vertices[9], vertices[4])),
                    Face(points=(vertices[4], vertices[9], vertices[10], vertices[5])),
                    Face(points=(vertices[5], vertices[10], vertices[6], vertices[1]))]

        super().__init__(faces=faces)

class RandomDistortedEllipsePlate(Volume):
    '''
    Generate a prism with a randomly distorted ellipse to create the background shapes in insitu
    liquid experiments
    '''
    def __init__(self, size, thick, seed=None):
        nb_point = 200
        self.thick = thick

        try:
            self.size_x = size[0]
            self.size_y = size[1]
        except TypeError:
            self.size_x = self.size_y = size

        rng_gen = np.random.default_rng(seed)
        number_sin = rng_gen.integers(10,16)
        frequency = rng_gen.random(number_sin) * np.logspace(-0.5, -2.5, number_sin)
        phase = rng_gen.random(number_sin) * 2 * np.pi

        angle = np.linspace(0, 2*np.pi, nb_point)
        amplitude = np.ones(nb_point)

        for step in range(number_sin):
            amplitude += frequency[step]*np.sin(step*angle + phase[step])
            pos_x = self.size_x * amplitude * np.cos(angle)
            pos_y = self.size_y * amplitude * np.sin(angle)

        self.vertices = []
        for point in range(nb_point):
            self.vertices.append(Point((pos_x[point], pos_y[point], 0)))

        super().__init__(faces=Face(points=self.vertices))

    def thickness(self, point):
        '''
        Evaluate the thickness of the volume at the point along z
        '''
        # all_z = []
        # _thickness = 0

        if self.faces[0].point_in_face(point):
            return self.thick
        return 0
        