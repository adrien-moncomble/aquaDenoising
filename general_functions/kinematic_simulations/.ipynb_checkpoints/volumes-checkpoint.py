'''
In this librairy, we want to define numerically differents kind of volumes. \n
With those we will be able to make them interact with our transmission electron microscope probe to
simulate the samples we are interested in.
'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class Face:
    '''
    Create a face in a 3D space
    '''
    def __init__(self, points=None):
        self.normal = None
        self.plan_cst = None
        self.points = []
        if isinstance(points, Point):
            self.points.append(points)
        else:
            try:
                for point in points:
                    if not isinstance(point, Point):
                        raise TypeError("Points must be an array of \"Point\" type objects")
                    self.points.append(point)
            except TypeError:
                raise TypeError("Points must be None, a \"Point\" "
                                "or a list of \"Point \" objects") from None

        # self.points = np.unique(self.points, axis=0)

        if not self.is_valid:
            print("This face is not valid, the plane formed by those points doesn't exist")

    def __repr__(self):
        if self.is_valid:
            return (f"Face ({self.normal[0]:.2f}x + {self.normal[1]:.2f}y + {self.normal[2]:.2f}z"
                    f" = {self.plan_cst:.2f})")
        return "Invalid Face"

    def reorder_points(self, new_order):
        '''
        Change the sequence of index to create another polygon for the face
        '''
        new_order = np.array(new_order)
        if not np.all(np.sort(new_order) == np.arange(self.nb_points)):
            raise ValueError("New_order must contain all the indexes of the points of the face")
        points = self.points
        self.points = []
        for idx in new_order:
            self.points.append(points[idx])

    def z_proj_point_in_face(self, point, return_z=False):
        '''
        Return True if the projection along z of the point is in the face
        '''
        positions = self.positions.tolist()
        positions.append(positions[0])
        positions = np.array(positions)
        count = 0

        for position_1, position_2 in zip(positions[:-1], positions[1:]):
            if ((point.position[1] < position_1[1]) != (point.position[1] < position_2[1]) and
                point.position[0] < position_1[0] + (position_2[0]-position_1[0])*
                ((point.position[1]-position_1[1])/(position_2[1]-position_1[1]))):
                count += 1

        if return_z:
            good_z = None
            if count%2 == 1 and self.is_valid:
                if self.normal[2] == 0:
                    good_z = 0
                else:
                    good_z = ((self.plan_cst - point.position[0]*self.normal[0]
                            - point.position[1]*self.normal[1]) / (self.normal[2]))
            return count%2 == 1, good_z
        return count%2 == 1


    def point_in_face(self, point):
        '''
        Return True if the point is in the face
        '''
        is_in_face, good_z = self.z_proj_point_in_face(point, return_z=True)

        return (is_in_face and point.position[2]==good_z)

    @property
    def is_valid(self):
        '''
        Verify if a plane containing all points exists
        If it exists then a normal vector and its equation are calculated
        '''
        if len(self.points) < 3:
            return False

        positions = self.positions
        vectors = positions[1:] - positions[0]
        idx=0
        for vector in vectors[1:]:
            self.normal = np.cross(vectors[0], vector)
            if not np.all(self.normal == np.zeros(3)):
                break
            idx += 1
        self.normal = self.normal/np.linalg.norm(self.normal)
        self.plan_cst = np.around(np.sum(self.normal*positions[0]), decimals=1)

        if len(self.points) > 3+idx:
            test_cst = np.around(np.sum(self.normal*positions[3+idx:], axis=1), decimals=1)
            if not np.all(test_cst == self.plan_cst):
                return False
        return True

    @property
    def plane_equation(self):
        '''
        Show the equation of the plane containing this face
        '''
        if self.is_valid:
            print("The equation of the plane containing this face is "
                  f"{self.normal[0]:.2f}x + {self.normal[1]:.2f}y + {self.normal[2]:.2f}z"
                  f" = {self.plan_cst:.2f}")
        else:
            print("This face is not valid, the plane formed by those points doesn't exist")

    @property
    def positions(self):
        '''
        Return all the positions of all vertices in the volume
        '''
        positions = [point.position for point in self.points]
        return np.array(positions)

    @property
    def nb_points(self):
        '''
        Return the number of vertices in the volume
        '''
        return len(self.points)

class Volume:
    '''
    Create a volume in a 3D space
    '''

    def __init__(self, points=None, faces=None):
        self.points = []
        if points is None:
            pass
        elif isinstance(points, Point):
            self.points.append(points)
        else:
            try:
                for point in points:
                    if not isinstance(point, Point):
                        raise TypeError("Points must be an array of \"Point\" type objects")
                    self.points.append(point)
            except TypeError:
                raise TypeError("Points must be None, a \"Point\" "
                                "or a list of \"Point \" objects") from None

        self.faces = []
        if isinstance(faces, Face):
            self.points += faces.points
            self.faces.append(faces)
        else:
            try:
                for face in faces:
                    if not isinstance(face, Face):
                        raise TypeError("Faces must be an array of \"Face\" type objects")
                    self.points += face.points
                    self.faces.append(face)
            except TypeError:
                raise TypeError("Faces must be None, a \"Face\" "
                                "or a list of \"Face \" objects") from None

        points_idx = np.unique(self.positions, return_index=True, axis=0)[1]
        self.points = [self.points[idx] for idx in np.sort(points_idx)]

    def add_points(self, coordinates):
        '''
        Associates points with given coordinates to the volume built

        Parameters :
        coordinates : array_like
            The 3D position of the points to add to the volume.
        '''

        coordinates = np.array(coordinates, dtype=float)
        shape = coordinates.shape

        if len(shape) == 1:
            if shape[0] == 2:
                coordinates = np.insert(coordinates, 2, 0)
            elif shape[0] != 3:
                raise ValueError("coordinates must have 3 components "
                                 "to add the point to the volume")
            self.points.append(Point(coordinates))

        elif len(shape) == 2:
            if shape[0] == 2:
                coordinates = np.append(coordinates, np.zeros((shape[0],1)), axis=1)
            elif shape[0] != 3:
                raise ValueError("Coordinates must have 3 components for "
                                 "each of the points to add to the volume")
            for idx in range(shape[0]):
                self.points.append(Point(coordinates[idx]))

        points_idx = np.unique(self.positions, return_index=True, axis=0)[1]
        self.points = [self.points[idx] for idx in np.sort(points_idx)]

    def translation(self, displacement):
        '''
        Translate the whole volume
        '''
        for point in self.points:
            point.move(displacement)

    def rotation(self, angles, origin_rotation=None, unit="deg", rotation_sequence="xyz"):
        '''
        Rotate the volume around its center of mass
        '''
        positions = self.positions
        if origin_rotation is None:
            origin_rotation = self.center_mass
        positions -= origin_rotation

        angles = np.array(angles)
        if angles.shape != (len(rotation_sequence),):
            raise ValueError(f"Angles must have {len(rotation_sequence)} components"
                             f" to do the {rotation_sequence} rotation sequence")

        if unit == "deg":
            angles = angles*np.pi/180
        elif unit != "rad":
            raise ValueError("Unit must be either \"deg\" for angles in"
                             " degrees or \"rad\" for angles in radians")

        for step, angle in zip(rotation_sequence, angles):
            if step == "x":
                rot_x = np.array(((             1,              0,              0),
                                  (             0,  np.cos(angle), -np.sin(angle)),
                                  (             0,  np.sin(angle),  np.cos(angle))))
                positions = np.matmul(positions, rot_x)
            elif step == "y":
                rot_y = np.array((( np.cos(angle),              0,  np.sin(angle)),
                                  (             0,              1,              0),
                                  (-np.sin(angle),              0,  np.cos(angle))))
                positions = np.matmul(positions, rot_y)
            elif step == "z":
                rot_z = np.array((( np.cos(angle), -np.sin(angle),              0),
                                  ( np.sin(angle),  np.cos(angle),              0),
                                  (             0,              0,              1)))
                positions = np.matmul(positions, rot_z)
            else:
                raise ValueError("Rotation_sequence must be a combination of x, y and z")

        positions += origin_rotation
        for point, idx in zip(self.points, range(self.nb_points)):
            point.position = positions[idx]
            
            
    def hkl_rotation(self, h, k, l, origin_rotation=None):
        
        positions = self.positions
        if origin_rotation is None:
            origin_rotation = self.center_mass
        positions -= origin_rotation

        Vect=np.array((h,k,l))/np.linalg.norm((h,k,l))

        if (Vect[0],Vect[1],Vect[2])==(0,0,1):
            rotation_matrix = np.array([[1,0,0],[0,1,0],[0,0,1]])
        elif (Vect[0],Vect[1],Vect[2])==(0,0,-1):
            rotation_matrix = np.array([[1,0,0],[0,1,0],[0,0,-1]])
        else:
            Oz = np.array((0,0,1))
            c = np.dot(Oz,Vect)
            Axis = np.cross(Oz,Vect)
            s = np.linalg.norm(Axis)
            Axis = Axis / s            
            rotation_matrix = np.array([[(Axis[0]*Axis[0])*(1-c)+c,
                                         (Axis[0]*Axis[1])*(1-c)-Axis[2]*s,
                                         (Axis[0]*Axis[2])*(1-c)+Axis[1]*s],
                                        [(Axis[0]*Axis[1])*(1-c)+Axis[2]*s,
                                         (Axis[1]*Axis[1])*(1-c)+c,
                                         (Axis[2]*Axis[1])*(1-c)-Axis[0]*s],
                                        [(Axis[0]*Axis[2])*(1-c)-Axis[1]*s,
                                         (Axis[2]*Axis[1])*(1-c)+Axis[0]*s,
                                         (Axis[2]*Axis[2])*(1-c)+c]])

        positions = np.matmul(positions, rotation_matrix)
        for point, idx in zip(self.points, range(self.nb_points)):
            point.position = positions[idx]
        

    def rescale(self, scaling_factor):
        '''
        Rescale the whole volume
        '''
        print(f"Scaling Factor = {scaling_factor}\nNot implemented yet... :(")

    def thickness(self, point, return_all_z=False):
        '''
        Evaluate the thickness of the volume at the point along z
        '''
        all_z = []
        _thickness = 0

        for face in self.faces:
            test_in_face, return_z = face.z_proj_point_in_face(point, return_z=True)
            if test_in_face:
                all_z.append(return_z)
        if len(all_z)%2 == 0 and len(all_z) != 0 :
            try:
                all_z = np.sort(all_z)
            except TypeError:
                print("Error Volume")
                return 0.0
            for i in range(len(all_z)//2):
                _thickness += all_z[1+i*2] - all_z[i*2]

        if return_all_z:
            return _thickness, all_z
        return _thickness

    def show_volume(self, view_x=None, view_y=None, view_z=None):
        '''
        Display the volume with Matplotlib
        '''
        fig = plt.figure()
        axe = fig.add_subplot(projection='3d')

        count = 0
        for face in self.faces:
            axe.add_collection3d(Poly3DCollection([face.positions], alpha=0.2,
                                                  facecolor=f"C{count}", edgecolor="k"))
            count += 1

        axe.scatter(self.positions[:,0],self.positions[:,1],self.positions[:,2], color="0.5")
        # axe.scatter(0, 0, 0)

        all_lim = np.max(np.abs([axe.get_xlim(), axe.get_ylim(), axe.get_zlim()]))
        axe.set_xlim(-all_lim, all_lim)
        axe.set_ylim(-all_lim, all_lim)
        axe.set_zlim(-all_lim, all_lim)

        axe.set_xlabel("X (nm)")
        axe.set_ylabel("Y (nm)")
        axe.set_zlabel("Z (nm)")

        axe.set_aspect('equal')
        axe.view_init(view_x, view_y, view_z)

        return fig, axe


    @property
    def nb_points(self):
        '''
        Return the number of vertices in the volume
        '''
        return len(self.points)

    @property
    def positions(self):
        '''
        Return all the positions of all vertices in the volume
        '''
        positions = [point.position for point in self.points]
        return np.array(positions)

    @property
    def center_mass(self):
        '''
        Return the center of mass of the volume
        '''
        if len(self.points) == 0:
            return np.zeros((3))
        else:
            return np.mean(self.positions, axis=0)

class Point:
    '''
    Create a point in a 3D space
    '''
    def __init__(self, coordinate):
        self.position = coordinate
        self.pos_x = self.position[0]
        self.pos_y = self.position[1]
        self.pos_z = self.position[2]

    def __repr__(self):
        return f"Point (x:{self.pos_x:.2f}, y:{self.pos_y:.2f}, z:{self.pos_z:.2f})"

    def move(self, displacement):
        '''
        Move the point by the displacement quantity
        '''
        self.position += displacement

    @property
    def position(self):
        '''
        Return the position of the Point
        '''
        return self._position

    @position.setter
    def position(self, coordinate):
        coordinate = np.array(coordinate)
        if coordinate.shape == (2,):
            coordinate = np.insert(coordinate, 2, 0)
        elif coordinate.shape != (3,):
            raise ValueError("Coordinate must have 3 components "
                             f"to define a point. Here shape is {coordinate.shape}")
        self._position = coordinate

    @property
    def pos_x(self):
        '''
        Return the position of the Point on the x axis
        '''
        return self._position[0]

    @pos_x.setter
    def pos_x(self, new_x):
        self.position = (new_x, self.pos_y, self.pos_z)

    @property
    def pos_y(self):
        '''
        Return the position of the Point on the y axis
        '''
        return self._position[1]

    @pos_y.setter
    def pos_y(self, new_y):
        self.position = (self.pos_x, new_y, self.pos_z)

    @property
    def pos_z(self):
        '''
        Return the position of the Point on the z axis
        '''
        return self._position[2]

    @pos_z.setter
    def pos_z(self, new_z):
        self.position = (self.pos_x, self.pos_y, new_z)
