"""Some quaternion helper functions

Randy Direen
7/21/2018

Functions:
    * norm:                 for normalizing a quaternion
    * rotate:               for rotating vectors
    * q_to_left_matrix      for generating a left matrix from q
    * q_to_right_matrix     for generating a right matrix from q

"""

import numpy as np
import quaternion as qt


def norm(q):
    """
    Returns a normalized quaternion

    @param q: quaternion being normalized
    @return: normalize quaternion
    """
    mag = np.absolute(q)
    return q/mag


def rotate(v, q):
    """
    Rotates the vector v using the unit quaternion q

    @param v: vector to be rotated
    @param q: orientation quaternion
    @return: rotated vector
    """
    qv = qt.quaternion(0, v[0], v[1], v[2])
    qi = q*qv*q.conj()
    return np.array([qi.x, qi.y, qi.z])


def q_to_left_matrix(q):
    """
    Turns quaternion into its matrix form. This is the form it takes if we
    multiply q*p. In this case we have q on the left side, so it has the
    form of a left matrix.

    @param q: quaternion being used as a linear operator
    @return: matrix
    """
    m = np.array([[q.w, -q.x, -q.y, -q.z],
                 [q.x,  q.w, -q.z,  q.y],
                 [q.y,  q.z,  q.w, -q.x],
                 [q.z, -q.y,  q.x,  q.w]], np.float64)
    return m


def q_to_right_matrix(q):
    """
    Turns quaternion into its matrix form. This is the form it takes if we
    multiply p*q. In this case we have q on the right side, so it has the
    form of a right matrix.

    @param q: quaternion being used as a linear operator
    @return: matrix
    """

    m = np.array([[q.w, -q.x, -q.y, -q.z],
                  [q.x,  q.w,  q.z, -q.y],
                  [q.y, -q.z,  q.w,  q.x],
                  [q.z,  q.y, -q.x,  q.w]], np.float64)
    return m


