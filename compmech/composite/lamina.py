"""
Composite Lamina Module (:mod:`compmech.composite.lamina`)
==========================================================

.. currentmodule:: compmech.composite.lamina

"""
from __future__ import division, absolute_import

import numpy as np
from numpy import cos, sin
from numpy.linalg import inv

from compmech.constants import DOUBLE
from .matlamina import MatLamina


class Lamina(object):
    """
    =========  ===========================================================
    attribute  description
    =========  ===========================================================
    plyid      id of the composite lamina
    matobj     a pointer to a MatLamina object
    t          ply thickness
    theta      ply angle in degrees
    L          transformation matrix for displacements to laminate csys
    R          transformation matrix for stresses to laminate csys
    T          transformation matrix for stresses to lamina csys
    QL         constitutive matrix for plane-stress in laminate csys
    laminates  laminates that contain this lamina
    =========  ===========================================================

    References:
    -----------
    .. [1] Reddy, J. N., Mechanics of Laminated Composite Plates and
       Shells - Theory and Analysys. Second Edition. CRC PRESS, 2004.

    """

    def __init__(self):
        self.plyid = None
        self.matobj = None
        self.t = None
        self.theta = None
        self.L = None
        self.R = None
        self.T = None
        self.QL = None
        self.laminates = []

    def rebuild(self):
        thetarad = np.deg2rad(self.theta)
        cost = cos(thetarad)
        sint = sin(thetarad)
        sin2t = sin(2 * thetarad)
        #
        cos2 = cost**2
        cos3 = cost**3
        cos4 = cost**4
        sin2 = sint**2
        sin3 = sint**3
        sin4 = sint**4
        sincos = sint * cost
        self.L = np.array([[cost, sint, 0],
                           [-sint, cost, 0],
                           [0,     0, 1]], dtype=DOUBLE)
        # STRESS
        # to lamina  Reddy Eq. 2.3.10
        self.R = np.array(
            [[cos2,   sin2, 0,   0,    0,     sin2t],
             [sin2,   cos2, 0,   0,    0,    -sin2t],
             [0,      0, 1,   0,    0,         0],
             [0,      0, 0, cost, -sint,         0],
             [0,      0, 0, sint,  cost,         0],
             [-sincos, sincos, 0,   0,    0, cos2 - sin2]], dtype=DOUBLE)
        # to laminate (VDI 2014 eq. 35, 36) # Reddy Eq. 2.3.8
        self.T = np.array(
            [[cos2,    sin2, 0,    0,   0,    -sin2t],
             [sin2,    cos2, 0,    0,   0,     sin2t],
             [0,       0, 1,    0,   0,         0],
             [0,       0, 0,  cost, sint,         0],
             [0,       0, 0, -sint, cost,         0],
             [sincos, -sincos, 0,    0,   0, cos2 - sin2]], dtype=DOUBLE)
        # STRAINS
        self.Te = np.array(
            [[cos2,    sin2, 0,    0,   0,    -2 * sin2t],
             [sin2,    cos2, 0,    0,   0,     2 * sin2t],
             [0,       0, 1,    0,   0,         0],
             [0,       0, 0,  cost, sint,         0],
             [0,       0, 0, -sint, cost,         0],
             [2 * sincos, -2 * sincos, 0,    0,   0, cos2 - sin2]], dtype=DOUBLE)

        # different from stress due to:
        #     2*e12 = e6    2*e13 = e5    2*e23 = e4
        # to laminate
        # self.Rstrain = np.transpose(self.Tstress)
        # to lamina
        # self.Tstrain = np.transpose(self.Rstress)

        if isinstance(self.matobj, MatLamina):
            e1 = self.matobj.e1
            e2 = self.matobj.e2
            nu12 = self.matobj.nu12
            nu21 = self.matobj.nu21
            g12 = self.matobj.g12
            g13 = self.matobj.g13
            g23 = self.matobj.g23
        else:
            e1 = self.matobj.e
            e2 = self.matobj.e
            nu12 = self.matobj.nu
            nu21 = self.matobj.nu
            g12 = self.matobj.g
            g = self.matobj.g

        # plane stress
        q11 = e1 / (1 - nu12 * nu21)
        q12 = nu12 * e2 / (1 - nu12 * nu21)
        q22 = e2 / (1 - nu12 * nu21)
        q44 = g23
        q55 = g13
        q16 = 0
        q26 = 0
        q66 = g12

        self.Q = np.array([[q11, q12, q16,    0,    0],
                           [q12, q22, q26,    0,    0],
                           [q16, q26, q66,    0,    0],
                           [0,    0,    0,  q44,    0],
                           [0,    0,    0,    0, q55]], dtype=DOUBLE)
        # Reddy Eq. 2.4.8
        q11L = q11 * cos4 + 2 * (q12 + 2 * q66) * sin2 * cos2 + q22 * sin4
        q12L = (q11 + q22 - 4 * q66) * sin2 * cos2 + q12 * (sin4 + cos4)
        q22L = q11 * sin4 + 2 * (q12 + 2 * q66) * sin2 * cos2 + q22 * cos4
        q16L = (q11 - q12 - 2 * q66) * sint * cos3 + \
            (q12 - q22 + 2 * q66) * sin3 * cost
        q26L = (q11 - q12 - 2 * q66) * sin3 * cost + \
            (q12 - q22 + 2 * q66) * sint * cos3
        q66L = (q11 + q22 - 2 * q12 - 2 * q66) * \
            sin2 * cos2 + q66 * (sin4 + cos4)
        q44L = q44 * cos2 + q55 * sin2
        q45L = (q55 - q44) * sincos
        q55L = q55 * cos2 + q44 * sin2

        self.QL = np.array([[q11L, q12L, q16L,    0,    0],
                            [q12L, q22L, q26L,    0,    0],
                            [q16L, q26L, q66L,    0,    0],
                            [0,    0,    0, q44L, q45L],
                            [0,    0,    0, q45L, q55L]], dtype=DOUBLE)

        # Reddy Eq. 2.3.17
        C = self.matobj.c
        self.CL = np.dot(np.dot(self.T, C), np.transpose(
            self.T))

        # Bogetti Eq. 28
        self.delta_CL45 = self.CL[3, 3] * \
            self.CL[4, 4] - self.CL[3, 4] * self.CL[4, 3]

        a1 = self.matobj.a1
        a2 = self.matobj.a2
        a3 = self.matobj.a3

        if not a1:
            a1 = 0.
        if not a2:
            a2 = 0.
        if not a3:
            a3 = 0.

        self.A = np.array([a1, a2, 0,    0,   0], dtype=DOUBLE)

        self.A3D = np.array([a1, a2, a3,    0,   0, 0], dtype=DOUBLE)

        # Reddy Eq 2.3.23
        a11L = a1 * cos2 + a2 * sin2
        a22L = a1 * sin2 + a2 * cos2
        a12L = (a1 - a2) * sint * cost
        a13L = 0.
        a23L = 0.
        a33L = a3

        self.AL = np.array([a11L, a22L, a12L,    0,   0], dtype=DOUBLE)

        self.AL3D = np.array(
            [a11L, a22L, a33L,    a23L,   a13L, a12L], dtype=DOUBLE)

    def calc_loading(self, eps_laminate, dT):
        ''' laminate strain needs to come in the following notation
        TODO: extend model to handle 3D stresses
        [eps_x, eps_y, eps_z, gamma_yz, gamma_xz, gamma_xy]
        and output is:
        [sigma_1, sigma_2, sigma_3, tau_23, tau_13, tau_12]
        '''
        # transform from engineering strain
        #     2*e12 = e6    2*e13 = e5    2*e23 = e4
        # self.rebuild()
        # self.theta
        '''
        # calculate thermal loads
        _eps_therm = self.AL * dT
        eps_therm = np.zeros_like(eps_laminate)
        eps_therm[0] = _eps_therm[0]
        eps_therm[1] = _eps_therm[1]
        eps_therm[5] = _eps_therm[2]
        '''
        T = self.Te

        # transform strain to lamina coordinate sysytem

        # Rstrain = np.transpose(T)  # np.transpose(T)
        Rstrain = inv(T)

        #Rstrain = T

        eps = np.dot(Rstrain, eps_laminate)

        # recover stress
        # reorder vector to plane
        # [eps_1, eps_2, gamma_21, gamma_23, gamma_13]
        eps_plane = np.zeros(len(eps) - 1)
        eps_plane[0] = eps[0]
        eps_plane[1] = eps[1]
        eps_plane[2] = eps[5]
        eps_plane[3] = eps[3]
        eps_plane[4] = eps[4]
        sig_plane = np.dot(self.Q, (eps_plane - self.AL * dT))
        # reorder back to 3D COS
        # [sigma_1, sigma_2, sigma_3, tau_23, tau_13, tau_12]
        sig = np.zeros_like(eps)
        sig[0] = sig_plane[0]
        sig[1] = sig_plane[1]
        sig[2] = 0.  # sigma_3
        sig[3] = sig_plane[3]
        sig[4] = sig_plane[4]
        sig[5] = sig_plane[2]

        return eps, sig
