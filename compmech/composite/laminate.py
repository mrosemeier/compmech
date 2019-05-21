"""
Composite Laminate Module (:mod:`compmech.composite.laminate`)
==============================================================

.. currentmodule:: compmech.composite.laminate

"""
from __future__ import division, absolute_import

import numpy as np

from .lamina import Lamina
from .matlamina import read_laminaprop
from compmech.constants import DOUBLE
from compmech.logger import *
from numpy.linalg.linalg import inv


def read_stack(stack, plyt=None, laminaprop=None, plyts=[], laminaprops=[],
               offset=0., lam3D=False):
    """Read a laminate stacking sequence data.

    An ``Laminate`` object is returned based on the inputs given.

    Parameters
    ----------
    stack : list
        Angles of the stacking sequence in degrees.
    plyt : float, optional
        When all plies have the same thickness, ``plyt`` can be supplied.
    laminaprop : tuple, optional
        When all plies have the same material properties, ``laminaprop``
        can be supplied.
    plyts : list, optional
        A list of floats with the thickness of each ply.
    laminaprops : list, optional
        A list of tuples with a laminaprop for each ply.
    offset : float, optional
        Offset along the normal axis about the mid-surface, which influences
        the laminate properties.
    lam3D : bool
        Use 3D model by Chou 1971, requires 3D material properties

    Notes
    -----
    ``plyt`` or ``plyts`` must be supplied
    ``laminaprop`` or ``laminaprops`` must be supplied

    For orthotropic plies, the ``laminaprop`` should be::

        laminaprop = (E11, E22, nu12, G12, G13, G23)

    For isotropic plies, the ``laminaprop`` should be::

        laminaprop = (E, E, nu)

    For lam3D, the ``laminaprop`` should be::

        laminaprop = (e1, e2, nu12, g12, g13, g23, e3, nu13, nu23, a1, a2, a3)

    """
    lam = Laminate()
    lam.offset = offset
    lam.stack = stack
    lam.lam3D = lam3D

    if not plyts:
        if not plyt:
            error('plyt or plyts must be supplied')
            raise ValueError
        else:
            plyts = [plyt for i in stack]

    if not laminaprops:
        if not laminaprop:
            error('laminaprop or laminaprops must be supplied')
            raise ValueError
        else:
            laminaprops = [laminaprop for i in stack]

    lam.plies = []
    for plyt, laminaprop, theta in zip(plyts, laminaprops, stack):
        laminaprop = laminaprop
        ply = Lamina()
        ply.theta = float(theta)
        ply.t = plyt
        ply.matobj = read_laminaprop(laminaprop)
        lam.plies.append(ply)

    lam.rebuild()
    lam.calc_constitutive_matrix()

    return lam


def read_lamination_parameters(thickness, laminaprop,
                               xiA1, xiA2, xiA3, xiA4,
                               xiB1, xiB2, xiB3, xiB4,
                               xiD1, xiD2, xiD3, xiD4,
                               xiE1, xiE2, xiE3, xiE4):
    r"""Calculates a laminate based on the lamination parameters.

    The lamination parameters:
    `\xi_{A1} \cdots \xi_{A4}`,  `\xi_{B1} \cdots \xi_{B4}`,
    `\xi_{C1} \cdots \xi_{C4}`,  `\xi_{D1} \cdots \xi_{D4}`,
    `\xi_{E1} \cdots \xi_{E4}`

    are used to calculate the laminate constitutive matrix.

    Parameters
    ----------
    thickness : float
        The total thickness of the laminate
    laminaprop : tuple
        The laminaprop tuple used to define the laminate material.
    xiA1 to xiD4 : float
        The 16 lamination parameters used to define the laminate.

    Returns
    -------
    lam : Laminate
        laminate with the ABD and ABDE matrices already calculated

    """
    lam = Laminate()
    lam.t = thickness
    lam.matobj = read_laminaprop(laminaprop)
    lam.xiA = np.array([1, xiA1, xiA2, xiA3, xiA4], dtype=DOUBLE)
    lam.xiB = np.array([0, xiB1, xiB2, xiB3, xiB4], dtype=DOUBLE)
    lam.xiD = np.array([1, xiD1, xiD2, xiD3, xiD4], dtype=DOUBLE)
    lam.xiE = np.array([1, xiE1, xiE2, xiE3, xiE4], dtype=DOUBLE)

    lam.calc_ABDE_from_lamination_parameters()
    return lam


class Laminate(object):
    r"""
    =========  ===========================================================
    attribute  description
    =========  ===========================================================
    plies      list of plies
    t          total thickness of the laminate
    offset     offset at the normal direction
    e1         equivalent laminate modulus in 1 direction
    e2         equivalent laminate modulus in 2 direction
    g12        equivalent laminate shear modulus in 12 direction
    nu12       equivalent laminate Poisson ratio in 12 direction
    nu21       equivalent laminate Poisson ratio in 21 direction
    xiA        laminate parameters for extensional matrix A
    xiB        laminate parameters for extension-bending matrix B
    xiD        laminate parameters for bending matrix D
    A          laminate extension matrix
    B          laminate extension-bending matrix
    D          laminate bending matrix
    E          laminate transferse shear matrix
    ABD        laminate ABD matrix
    ABDE       laminate ABD matrix with transverse shear terms
    =========  ===========================================================

    """

    def __init__(self):
        self.plies = []
        self.matobj = None
        self.t = None
        self.offset = 0.
        self.e1 = None
        self.e2 = None
        self.e3 = None
        self.nu12 = None
        self.g12 = None
        self.g13 = None
        self.g23 = None
        self.a1 = None
        self.a2 = None
        self.xiA = None
        self.xiB = None
        self.xiD = None
        self.A = None
        self.B = None
        self.D = None
        self.E = None
        self.ABD = None
        self.ABDE = None
        self.lam3D = False

    def rebuild(self):
        lam_thick = 0
        for ply in self.plies:
            ply.rebuild()
            lam_thick += ply.t
        self.t = lam_thick

    def calc_equivalent_modulus(self):
        """Calculates the equivalent laminate properties.

        The following attributes are calculated:
            e1, e2, g12, nu12, nu21

        """
        if not self.lam3D:
            AI = np.matrix(self.ABD, dtype=DOUBLE).I
            a11, a12, a22, a33 = AI[0, 0], AI[0, 1], AI[1, 1], AI[2, 2]
            self.e1 = 1. / (self.t * a11)
            self.e2 = 1. / (self.t * a22)
            self.g12 = 1. / (self.t * a33)
            self.nu12 = - a12 / a11
            self.nu21 = - a12 / a22

            # Eq. 5.110 Ganesh/Rana Lecture19 Hygrothermal laminate theory
            # or Eq. 4.72 into Eg.4.64 with delta_T=1 (Kaw 2006)
            a = np.squeeze(np.array(np.dot(AI, self.QLAL)))
            self.a1 = a[0]
            self.a2 = a[1]
            self.a12 = a[2]

        else:
            H = inv(self.C_general)  # Bogetti 1995 Eq. 29

            self.e1 = 1. / H[0, 0]  # Bogetti 1995 Eq. 30
            self.e2 = 1. / H[1, 1]  # Bogetti 1995 Eq. 31
            self.e3 = 1. / H[2, 2]  # Bogetti 1995 Eq. 32
            self.g23 = 1. / H[3, 3]  # Bogetti 1995 Eq. 33
            self.g13 = 1. / H[4, 4]  # Bogetti 1995 Eq. 34
            self.g12 = 1. / H[5, 5]  # Bogetti 1995 Eq. 35
            self.nu23 = - H[1, 2] / H[1, 1]  # Bogetti 1995 Eq. 36
            self.nu13 = - H[0, 2] / H[0, 0]  # Bogetti 1995 Eq. 37
            self.nu12 = - H[0, 1] / H[0, 0]  # Bogetti 1995 Eq. 38
            self.nu32 = - H[1, 2] / H[2, 2]  # Bogetti 1995 Eq. 39
            self.nu31 = - H[0, 2] / H[2, 2]  # Bogetti 1995 Eq. 40
            self.nu21 = - H[0, 1] / H[1, 1]  # Bogetti 1995 Eq. 41

            N = self.N
            self.a1 = np.dot(H[0, :], N)  # Bogetti Eq. 44
            self.a2 = np.dot(H[1, :], N)  # Bogetti Eq. 45
            self.a3 = np.dot(H[2, :], N)  # Bogetti Eq. 46
            self.a23 = np.dot(H[3, :], N)  # Bogetti Eq. 47
            self.a13 = np.dot(H[4, :], N)  # Bogetti Eq. 48
            self.a12 = np.dot(H[5, :], N)  # Bogetti Eq. 49

    def calc_lamination_parameters(self):
        """Calculate the lamination parameters.

        The following attributes are calculated:
            xiA, xiB, xiD, xiE

        """
        xiA1, xiA2, xiA3, xiA4 = 0, 0, 0, 0
        xiB1, xiB2, xiB3, xiB4 = 0, 0, 0, 0
        xiD1, xiD2, xiD3, xiD4 = 0, 0, 0, 0
        xiE1, xiE2, xiE3, xiE4 = 0, 0, 0, 0

        lam_thick = sum([ply.t for ply in self.plies])
        self.t = lam_thick

        h0 = -lam_thick / 2. + self.offset
        for ply in self.plies:
            hk_1 = h0
            h0 += ply.t
            hk = h0

            Afac = ply.t / lam_thick
            Bfac = (2. / lam_thick**2) * (hk**2 - hk_1**2)
            Dfac = (4. / lam_thick**3) * (hk**3 - hk_1**3)
            Efac = (1. / lam_thick) * (hk - hk_1)  # * (5./6) * (5./6)

            cos2t = ply.cos2t
            cos4t = ply.cos4t
            sin2t = ply.sin2t
            sin4t = ply.sin4t

            xiA1 += Afac * cos2t
            xiA2 += Afac * sin2t
            xiA3 += Afac * cos4t
            xiA4 += Afac * sin4t

            xiB1 += Bfac * cos2t
            xiB2 += Bfac * sin2t
            xiB3 += Bfac * cos4t
            xiB4 += Bfac * sin4t

            xiD1 += Dfac * cos2t
            xiD2 += Dfac * sin2t
            xiD3 += Dfac * cos4t
            xiD4 += Dfac * sin4t

            xiE1 += Efac * cos2t
            xiE2 += Efac * sin2t
            xiE3 += Efac * cos4t
            xiE4 += Efac * sin4t

        self.xiA = np.array([1, xiA1, xiA2, xiA3, xiA4], dtype=DOUBLE)
        self.xiB = np.array([0, xiB1, xiB2, xiB3, xiB4], dtype=DOUBLE)
        self.xiD = np.array([1, xiD1, xiD2, xiD3, xiD4], dtype=DOUBLE)
        self.xiE = np.array([1, xiE1, xiE2, xiE3, xiE4], dtype=DOUBLE)

    def calc_ABDE_from_lamination_parameters(self):
        """Use the ABDE matrix based on lamination parameters.

        Given the lamination parameters ``xiA``, ``xiB``, ``xiC`` and ``xiD``,
        the ABD matrix is calculated.

        """
        # dummies used to unpack vector results
        du1, du2, du3, du4, du5, du6 = 0, 0, 0, 0, 0, 0
        # A matrix terms
        A11, A22, A12, du1, du2, du3, A66, A16, A26 =\
            (self.t) * np.dot(self.matobj.u, self.xiA)
        # B matrix terms
        B11, B22, B12, du1, du2, du3, B66, B16, B26 =\
            (self.t**2 / 4.) * np.dot(self.matobj.u, self.xiB)
        # D matrix terms
        D11, D22, D12, du1, du2, du3, D66, D16, D26 =\
            (self.t**3 / 12.) * np.dot(self.matobj.u, self.xiD)
        # E matrix terms
        du1, du2, du3, E44, E55, E45, du4, du5, du6 =\
            (self.t) * np.dot(self.matobj.u, self.xiE)

        self.A = np.array([[A11, A12, A16],
                           [A12, A22, A26],
                           [A16, A26, A66]], dtype=DOUBLE)

        self.B = np.array([[B11, B12, B16],
                           [B12, B22, B26],
                           [B16, B26, B66]], dtype=DOUBLE)

        self.D = np.array([[D11, D12, D16],
                           [D12, D22, D26],
                           [D16, D26, D66]], dtype=DOUBLE)

        # printing E acoordingly to Reddy definition for E44, E45 and E55
        self.E = np.array([[E55, E45],
                           [E45, E44]], dtype=DOUBLE)

        self.ABD = np.array([[A11, A12, A16, B11, B12, B16],
                             [A12, A22, A26, B12, B22, B26],
                             [A16, A26, A66, B16, B26, B66],
                             [B11, B12, B16, D11, D12, D16],
                             [B12, B22, B26, D12, D22, D26],
                             [B16, B26, B66, D16, D26, D66]], dtype=DOUBLE)

        # printing ABDE acoordingly to Reddy definition for E44, E45 and E55
        self.ABDE = np.array([[A11, A12, A16, B11, B12, B16, 0, 0],
                              [A12, A22, A26, B12, B22, B26, 0, 0],
                              [A16, A26, A66, B16, B26, B66, 0, 0],
                              [B11, B12, B16, D11, D12, D16, 0, 0],
                              [B12, B22, B26, D12, D22, D26, 0, 0],
                              [B16, B26, B66, D16, D26, D66, 0, 0],
                              [0, 0, 0, 0, 0, 0, E55, E45],
                              [0, 0, 0, 0, 0, 0, E45, E44]],
                             dtype=DOUBLE)

    def calc_constitutive_matrix(self):
        """Calculates the laminate constitutive matrix

        This is the commonly called ``ABD`` matrix with ``shape=(6, 6)`` when
        the classical laminated plate theory is used, or the ``ABDE`` matrix
        when the first-order shear deformation theory is used, containing the
        transverse shear terms.

        """
        self.A_general = np.zeros([5, 5], dtype=DOUBLE)
        self.B_general = np.zeros([5, 5], dtype=DOUBLE)
        self.D_general = np.zeros([5, 5], dtype=DOUBLE)
        self.QLALN_general = np.zeros([5], dtype=DOUBLE)
        self.QLALM_general = np.zeros([5], dtype=DOUBLE)

        lam_thick = sum([ply.t for ply in self.plies])
        self.t = lam_thick

        h0 = -lam_thick / 2 + self.offset
        for ply in self.plies:
            hk_1 = h0
            h0 += ply.t
            hk = h0
            self.A_general += ply.QL * (hk - hk_1)
            self.B_general += 1 / 2. * ply.QL * (hk**2 - hk_1**2)
            self.D_general += 1 / 3. * ply.QL * (hk**3 - hk_1**3)
            # TODO add CTE laminate matrix
            # Reddy Eq. 3.3.41
            QLAL_dot = np.dot(ply.QL, ply.AL)
            self.QLALN_general += QLAL_dot * (hk - hk_1)
            self.QLALM_general += 1 / 2. * QLAL_dot * (hk**2 - hk_1**2)
        self.A = self.A_general[0:3, 0:3]
        self.B = self.B_general[0:3, 0:3]
        self.D = self.D_general[0:3, 0:3]
        self.E = self.A_general[3:5, 3:5]

        conc1 = np.concatenate([self.A, self.B], axis=1)
        conc2 = np.concatenate([self.B, self.D], axis=1)

        self.ABD = np.concatenate([conc1, conc2], axis=0)
        self.ABDE = np.zeros((8, 8), dtype=DOUBLE)
        self.ABDE[0:6, 0:6] = self.ABD
        self.ABDE[6:8, 6:8] = self.E

        self.QLALN = self.QLALN_general[0:3]
        self.QLALM = self.QLALM_general[0:3]

        self.QLAL = np.concatenate([self.QLALN, self.QLALM], axis=0)

        self._calc_stiffness_matrix_3D()

    def _calc_stiffness_matrix_3D(self):
        ''' Calculates the laminate stiffness matrix
        Chou, Carleone and Hsu, 1971, Elastic Constants of Layered Media
        Theory assumes symmetric laminate
        '''

        # general laminate stiffness matrix
        self.C_general = np.zeros([6, 6], dtype=DOUBLE)

        lam_thick = self.t

        # Chou 1971 Eq. 8

        def _sum_l_up(j, lam_thick):
            sum_l_up = 0.
            for plyl in self.plies:
                tl = plyl.t
                vl = tl / lam_thick
                CLl = plyl.CL
                sum_l_up += vl * CLl[2, j] / CLl[2, 2]
            return sum_l_up

        def _sum_l_low(lam_thick):
            sum_l_low = 0.
            for plyl in self.plies:
                tl = plyl.t
                vl = tl / lam_thick
                CLl = plyl.CL
                sum_l_low += vl / CLl[2, 2]
            return sum_l_low

        for i in [0, 1, 2, 5]:
            for j in [0, 1, 2, 5]:
                for plyk in self.plies:
                    tk = plyk.t
                    vk = tk / lam_thick
                    CLk = plyk.CL

                    self.C_general[i, j] += vk * (CLk[i, j] -
                                                  (CLk[i, 2] * CLk[2, j]) /
                                                  (CLk[2, 2]) +
                                                  (CLk[i, 2] * _sum_l_up(j, lam_thick)) /
                                                  (CLk[2, 2] * _sum_l_low(lam_thick)))
        # Chou 1971 Eq. 9

        def _sum_k_up_34(i, j, lam_thick):
            sum_k_up_34 = 0.
            for plyk in self.plies:
                tk = plyk.t
                vk = tk / lam_thick
                CLk = plyk.CL
                deltak = plyk.delta_CL45
                sum_k_up_34 += vk / deltak * CLk[i, j]
            return sum_k_up_34

        def _sum_kl_low_34(lam_thick):
            sum_kl_low_34 = 0.
            for plyk in self.plies:
                tk = plyk.t
                vk = tk / lam_thick
                CLk = plyk.CL
                deltak = plyk.delta_CL45
                for plyl in self.plies:
                    tl = plyl.t
                    vl = tl / lam_thick
                    CLl = plyl.CL
                    deltal = plyk.delta_CL45

                    sum_kl_low_34 += (vk * vl) /\
                        (deltak * deltal) * \
                        (CLk[3, 3] * CLl[4, 4] - CLk[3, 4] * CLl[4, 3])

            return sum_kl_low_34

        for i in [3, 4]:
            for j in [3, 4]:
                self.C_general[i, j] = _sum_k_up_34(
                    i, j, lam_thick) / _sum_kl_low_34(lam_thick)

        # Bogetti Eq. 43
        self.N = np.zeros([6], dtype=DOUBLE)

        for i in range(6):
            for j in range(6):
                for plyk in self.plies:
                    tk = plyk.t
                    vk = tk / lam_thick
                    CLk = plyk.CL
                    AL3D = plyk.AL3D
                    self.N[i] += CLk[i, j] * AL3D[j] * vk

    def force_balanced_LP(self):
        r"""Force balanced lamination parameters

        The lamination parameters `\xi_{A2}` and `\xi_{A4}` are set to null
        to force a balanced laminate.

        """
        dummy, xiA1, xiA2, xiA3, xiA4 = self.xiA
        self.xiA = np.array([1, xiA1, 0, xiA3, 0], dtype=DOUBLE)
        self.calc_ABDE_from_lamination_parameters()

    def force_symmetric_LP(self):
        r"""Force symmetric lamination parameters

        The lamination parameters `\xi_{Bi}` are set to null
        to force a symmetric laminate.

        """
        self.xiB = np.zeros(5)
        self.calc_ABDE_from_lamination_parameters()

    def force_orthotropic(self):
        r"""Force an orthotropic laminate

        The terms
        `A_{13}`, `A_{23}`, `A_{31}`, `A_{32}`,
        `B_{13}`, `B_{23}`, `B_{31}`, `B_{32}`,
        `D_{13}`, `D_{23}`, `D_{31}`, `D_{32}` are set to zero to force an
        orthotropic laminate.

        """
        if self.offset != 0.:
            raise RuntimeError(
                'Laminates with offset cannot be forced orthotropic!')
        self.A[0, 2] = 0.
        self.A[1, 2] = 0.
        self.A[2, 0] = 0.
        self.A[2, 1] = 0.

        self.B[0, 2] = 0.
        self.B[1, 2] = 0.
        self.B[2, 0] = 0.
        self.B[2, 1] = 0.

        self.D[0, 2] = 0.
        self.D[1, 2] = 0.
        self.D[2, 0] = 0.
        self.D[2, 1] = 0.

        self.ABD[0, 2] = 0.  # A16
        self.ABD[1, 2] = 0.  # A26
        self.ABD[2, 0] = 0.  # A61
        self.ABD[2, 1] = 0.  # A62

        self.ABD[0, 5] = 0.  # B16
        self.ABD[5, 0] = 0.  # B61
        self.ABD[1, 5] = 0.  # B26
        self.ABD[5, 1] = 0.  # B62

        self.ABD[3, 2] = 0.  # B16
        self.ABD[2, 3] = 0.  # B61
        self.ABD[4, 2] = 0.  # B26
        self.ABD[2, 4] = 0.  # B62

        self.ABD[3, 5] = 0.  # D16
        self.ABD[4, 5] = 0.  # D26
        self.ABD[5, 3] = 0.  # D61
        self.ABD[5, 4] = 0.  # D62

        self.ABDE[0, 2] = 0.  # A16
        self.ABDE[1, 2] = 0.  # A26
        self.ABDE[2, 0] = 0.  # A61
        self.ABDE[2, 1] = 0.  # A62

        self.ABDE[0, 5] = 0.  # B16
        self.ABDE[5, 0] = 0.  # B61
        self.ABDE[1, 5] = 0.  # B26
        self.ABDE[5, 1] = 0.  # B62

        self.ABDE[3, 2] = 0.  # B16
        self.ABDE[2, 3] = 0.  # B61
        self.ABDE[4, 2] = 0.  # B26
        self.ABDE[2, 4] = 0.  # B62

        self.ABDE[3, 5] = 0.  # D16
        self.ABDE[4, 5] = 0.  # D26
        self.ABDE[5, 3] = 0.  # D61
        self.ABDE[5, 4] = 0.  # D62

    def force_symmetric(self):
        """Force a symmetric laminate

        The `B` terms of the constitutive matrix are set to zero.

        """
        if self.offset != 0.:
            raise RuntimeError(
                'Laminates with offset cannot be forced symmetric!')
        self.B = np.zeros((3, 3))
        self.ABD[0:3, 3:6] = 0
        self.ABD[3:6, 0:3] = 0

        self.ABDE[0:3, 3:6] = 0
        self.ABDE[3:6, 0:3] = 0

    def apply_load(self, F, dT):
        ''' Obtain strains of stacking due to loading
        F = [n_x, n_y, n_xy, m_x, m_y, m_xy] in N/m
        :param: F: force vector
        :param: dT: delta temperature
        :return: eps0: vector of strains due to normal loads eps0=[eps_x, eps_y, eps_xy]
        :return: eps1: vector of strains due to bending loads eps1=[eps_x, eps_y, eps_xy]
        '''
        # Reddy, Eq. 3.3.40
        eps = np.dot(inv(self.ABD), (F + self.QLAL * dT))
        eps0 = eps[0:3]
        eps1 = eps[3:6]
        return eps0, eps1
