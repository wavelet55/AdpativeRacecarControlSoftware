"""The model of the box used for drawing in 3D

Randy Direen
7/21/2018

This model knows how to rotate and project itself onto a 2D plane. You just
need to supply it with an orientation quaternion.

"""
import numpy as np
from operator import itemgetter
import quots


class Point3D:
    def __init__(self, point):
        """ The box is made of a set of these points. Each of these points
        knows how to rotate itself with a quaternion.

        @param point: a vector containing the x, y, and z coordinates of the
                      point
        """
        self.point = np.array([point[0], point[1], point[2]], np.float64)

    def rotate(self, quaternion):
        """ Rotates the point using the quaternion"""
        v = quots.rotate(self.point, quaternion)
        return Point3D(v)

    def project(self, win_width, win_height, fov, viewer_distance):
        """ Transforms this 3D point to 2D using a perspective projection. """
        factor = fov / (viewer_distance + self.point[2])
        x = self.point[0] * factor*.75 + win_width / 2
        y = -self.point[1] * factor + win_height / 2
        return Point3D([x, y, self.point[2]])


class HeadModel:
    def __init__(self, screen_width, screen_height, scale=1.0, location=(0, 0)):
        """ The points below are the points of a box

        @param screen_width: get these from pygame
        @param screen_height: get these from pygame
        """

        self.screen_width = screen_width
        self.screen_height = screen_height
        self.scale = scale
        self.location = location

        self.vertices = [
            Point3D([-scale,  scale, -.5*scale]),
            Point3D([ scale,  scale, -.5*scale]),
            Point3D([ scale, -scale, -.5*scale]),
            Point3D([-scale, -scale, -.5*scale]),
            Point3D([-scale,  scale,  .5*scale]),
            Point3D([ scale,  scale,  .5*scale]),
            Point3D([ scale, -scale,  .5*scale]),
            Point3D([-scale, -scale,  .5*scale])
        ]

        # Define the vertices that compose each of the 6 faces. These numbers are
        # indices to the vertices list defined above.
        self.faces = [(0, 1, 2, 3), (1, 5, 6, 2), (5, 4, 7, 6), (4, 0, 3, 7), (0, 4, 5, 1), (3, 2, 6, 7)]

        # Define colors for each face
        self.colors = [(255, 145, 46), (123, 232, 107), (7, 130, 192),(255, 145, 46), (109, 185, 217),  (181, 227, 246)]

    def oriented_vertices(self, quaternion):
        """Returns the date used for drawing the box given the orientation
        quaternion
        """

        t = []

        for v in self.vertices:
            # Use orientation quaternion to put vertices in place
            r = v.rotate(quaternion)
            # Transform the point from 3D to 2D
            p = r.project(self.screen_width, self.screen_height, 256, 4)

            p.point[0] += self.location[0]
            p.point[1] += self.location[1]

            # Put the point in the list of transformed vertices
            t.append(p)

        # Calculate the average Z values of each face.
        avg_z = []
        i = 0
        for f in self.faces:
            z = (t[f[0]].point[2] + t[f[1]].point[2] + t[f[2]].point[2] + t[f[3]].point[2]) / 4.0
            avg_z.append([i, z])
            i = i + 1

        draw_data = []
        # Draw the faces using the Painter's algorithm:
        # Distant faces are drawn before the closer ones.
        for tmp in sorted(avg_z, key=itemgetter(1), reverse=True):
            face_index = tmp[0]
            f = self.faces[face_index]
            pointlist = [(t[f[0]].point[0], t[f[0]].point[1]), (t[f[1]].point[0], t[f[1]].point[1]),
                         (t[f[1]].point[0], t[f[1]].point[1]), (t[f[2]].point[0], t[f[2]].point[1]),
                         (t[f[2]].point[0], t[f[2]].point[1]), (t[f[3]].point[0], t[f[3]].point[1]),
                         (t[f[3]].point[0], t[f[3]].point[1]), (t[f[0]].point[0], t[f[0]].point[1])]

            draw_data.append([self.colors[face_index], pointlist])

        return draw_data
