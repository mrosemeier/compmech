.. _theory-conecyl-clpt:

=======================================
CLPT - Classical Laminated Plate Theory
=======================================

Description
===========

For the CLPT the displacement field components are:

.. math::
    u, v, w

And approximated as:

.. math::
    u = u_0(x, \theta) + z \phi_x(x, \theta) \\
    v = v_0(x, \theta) + z \phi_\theta(x, \theta) \\
    w = w_0(x, \theta)

where `u_0, v_0, w_0` are the displacements of the shell mid-surface and
`\phi_x` and `\phi_\theta` the shell rotations along `x` and `\theta`
following the right-hand rule. For the CLPT the rotations are defined as:

.. math::
    \phi_x = - \frac{\partial w}{\partial x} = -w_{,x} \\
    \phi_\theta = - \frac 1 r \frac{\partial w}{\partial \theta}
           = - \frac 1 r w_{,\theta}

For the ``ConeCyl`` implementations the displacement field is approximated
and the approximated functions can be separated as:

.. math::
    u = u_0 + u_1 + u_2\\
    v = v_0 + v_1 + v_2\\
    w = w_0 + w_1 + w_2\\

where `u_0` contains the approximation functions corresponding to the
prescribed degrees of freedom, `u_1` contains the functions independent
of `\theta` and `u_2` the functions that depend on both `x`
and `\theta`.

The aim is to have models capable of simulating the displacement field of
cones and cylinders. The approximation functions are the same
for both the Donnell's and the Sanders' models.

Models
======

Below it follows a more detailed description of each of the implementations:

- clpt_donnell_bc1_
- clpt_donnell_bc2_
- clpt_donnell_bc3_
- clpt_donnell_bc4_
- clpt_donnell_bcn_

- clpt_sanders_bc1_
- clpt_sanders_bc4_
- clpt_sanders_bc2_


Each model can be accessed using the ``linear_kinematics`` parameter of the
``ConeCyl`` object. For linear static analysis the most general model is the
clpt_donnell_bcn_.

For linear buckling analysis the following models should be used for
each type of boundary conditions:

- SS1- or CC1-type: clpt_donnell_bc1_ or clpt_sanders_bc1_

- SS2- or CC2-type: clpt_donnell_bc2_ or clpt_sanders_bc2_

- SS3- or CC3-type: clpt_donnell_bc3_ or clpt_sanders_bc3_

- SS4- or CC4-type: clpt_donnell_bc4_ or clpt_sanders_bc4_

- Free edges: use the :ref:`fsdt_donnell_bcn` (CLPT not implemented)


.. _clpt_donnell_bc1:

clpt_donnell_bc1
----------------

SS1- and CC1-types of boundary conditions, or anything
in between by using elastic restrained edges in `w_{,x}` and `w_{,\theta}`.
The approximation functions are:

.. _clpt_approx_functions:

.. math::
    u = u_0 + \sum_{i_1=0}^{m_1} {c_{i_1}}^{u} \sin{{b_x}_1}
            + \sum_{i_2=0}^{m_2} \sum_{j_2=1}^{n_2} \left(
                     {c_{i_2 j_2}}_a^{u} \sin{{b_x}_2} \sin{j_2 \theta}
                    +{c_{i_2 j_2}}_b^{u} \sin{{b_x}_2} \cos{j_2 \theta}
                  \right)
    \\
    v = v_0 + \sum_{i_1=0}^{m_1} {c_{i_1}}^{v}\sin{{b_x}_1}
            + \sum_{i_2=0}^{m_2} \sum_{j_2=1}^{n_2} \left(
                     {c_{i_2 j_2}}_a^{v} \sin{{b_x}_2} \sin{j_2 \theta}
                    +{c_{i_2 j_2}}_b^{v} \sin{{b_x}_2} \cos{j_2 \theta}
                  \right)
    \\
    w = w_0 + \sum_{i_1=0}^{m_1} {c_{i_1}}^{w}\sin{{b_x}_1}
            + \sum_{i_2=0}^{m_2} \sum_{j_2=1}^{n_2} \left(
                     {c_{i_2 j_2}}_a^{w} \sin{{b_x}_2} \sin{j_2 \theta}
                    +{c_{i_2 j_2}}_b^{w} \sin{{b_x}_2} \cos{j_2 \theta}
                \right)

with:

.. math::
    {b_x}_1 = i_1 \pi \frac x L \\
    {b_x}_2 = i_2 \pi \frac x L

The following general form of elastic constraints at the edges is used:

.. math::
    U_{springs} = \int_\theta r_1 \left(
                      K_{Bot}^u u_{x=L}^2
                    + K_{Bot}^v v_{x=L}^2
                    + K_{Bot}^w w_{x=L}^2
                    + K_{Bot}^{\phi_x} {\phi_x}_{x=L}^2
                    + K_{Bot}^{\phi_\theta} {\phi_\theta}_{x=L}^2
                  \right)
                  \\
                + \int_\theta r_2 \left(
                      K_{Top}^u u_{x=0}^2
                    + K_{Top}^v v_{x=0}^2
                    + K_{Top}^w w_{x=0}^2
                    + K_{Top}^{\phi_x} {\phi_x}_{x=0}^2
                    + K_{Top}^{\phi_\theta} {\phi_\theta}_{x=0}^2
                  \right)

Note that the stiffnesses: `K_{Top}^u`, `K_{Top}^v` and `K_{Top}^w` are not
used in clpt_donnell_bc1_, but since they are required in other
implementations, it is convenient to present the general form using
all the elastic terms.

The equation for `U_{springs}` can be written in matrix form, and it will
result in an additional term `[K_e]` to the linear stiffness matrix
`[K_0]`. The new stiffness matrix with the elastic constraints at the edges
(`[{K_0}_e]`) becomes:

.. _elastic_constraints:

.. math::
    [{K_0}_e] = [K_0] + [K_e]

    [K_e] = \int_{\theta} { \left(
                r_1 [g_{new}]_{x=L}^T [K]_{Bot} [g_{new}]_{x=L}^.
              + r_2 [g_{new}]_{x=0}^T [K]_{Top} [g_{new}]_{x=0}^.
             \right) d\theta
            }

with :

.. math::
    [K_{Bot}] = \begin{bmatrix}
          K_{Bot}^u &       0 &       0 &              0 &             0 \\
                0 & K_{Bot}^v &       0 &              0 &             0 \\
                0 &       0 & K_{Bot}^w &              0 &             0 \\
                0 &       0 &       0 & K_{Bot}^{\phi_x} &             0 \\
                0 &       0 &       0 &              0 &K_{Bot}^{\phi_\theta}
                    \end{bmatrix}

and:

.. math::
    [K_{Top}] = \begin{bmatrix}
          K_{Top}^u &       0 &       0 &              0 &             0 \\
                0 & K_{Top}^v &       0 &              0 &             0 \\
                0 &       0 & K_{Top}^w &              0 &             0 \\
                0 &       0 &       0 & K_{Top}^{\phi_x} &             0 \\
                0 &       0 &       0 &              0 &K_{Top}^{\phi_\theta}
                    \end{bmatrix}


and the shape functions `[g_{new}]` contains two extra rows that are built
from the relations:

.. math::
    \phi_x = - \frac{\partial w}{\partial x} = -w_{,x} \\
    \phi_\theta = - \frac 1 r \frac{\partial w}{\partial \theta}
                = - \frac 1 r w_{,\theta}

and therefore:

.. math::
    [g^{\phi_x}] = - \frac {\partial [g^w]} {\partial x} \\
    [g^{\phi_\theta}] = - \frac 1 r \frac {\partial [g^w]} {\partial \theta} \\
    [g_{new}]^T = \left[ [g^u], [g^v], [g^w],
                          [g^{\phi_x}], [g^{\phi_\theta}] \right]


Observations:

    `\checkmark` linear buckling implemented

    `\checkmark` linear static implemented

    `\checkmark` non-linear analysis implemented


.. _clpt_donnell_bc2:

clpt_donnell_bc2
----------------

Planned to simulate the SS2- and CC2-types of boundary conditions (or anything
in between). The flexibily in `v` is removed if compared to the
clpt_donnell_bc4_. Giving:

.. math::
    u = u_0 + \sum_{i_1=0}^{m_1} {c_{i_1}}^{u} \sin{{b_x}_1}
            + \sum_{i_2=0}^{m_2} \sum_{j_2=1}^{n_2} \left(
                     {c_{i_2 j_2}}_a^{u} \cos{{b_x}_2} \sin{j_2 \theta}
                    +{c_{i_2 j_2}}_b^{u} \cos{{b_x}_2} \cos{j_2 \theta}
                  \right)
    \\
    v = v_0 + \sum_{i_1=0}^{m_1} {c_{i_1}}^{v}\sin{{b_x}_1}
            + \sum_{i_2=0}^{m_2} \sum_{j_2=1}^{n_2} \left(
                     {c_{i_2 j_2}}_a^{v} \sin{{b_x}_2} \sin{j_2 \theta}
                    +{c_{i_2 j_2}}_b^{v} \sin{{b_x}_2} \cos{j_2 \theta}
                  \right)
    \\
    w = w_0 + \sum_{i_1=0}^{m_1} {c_{i_1}}^{w}\sin{{b_x}_1}
            + \sum_{i_2=0}^{m_2} \sum_{j_2=1}^{n_2} \left(
                     {c_{i_2 j_2}}_a^{w} \sin{{b_x}_2} \sin{j_2 \theta}
                    +{c_{i_2 j_2}}_b^{w} \sin{{b_x}_2} \cos{j_2 \theta}
                \right)
    \\

The linear stiffness matrix `[K_0]` is changed using the same
:ref:`elastic contraints used for the clpt_donnell_bc1 <elastic_constraints>`.

Observations:

    `\checkmark` linear buckling implemented

    `\checkmark` linear static implemented

    `\checkmark` non-linear analysis implemented


.. _clpt_donnell_bc3:

clpt_donnell_bc3
----------------

Planned for SS3- and CC3-types of boundary conditions (or anything
in between). The approximation functions are:

.. math::
    u = u_0 + \sum_{i_1=0}^{m_1} {c_{i_1}}^{u} \sin{{b_x}_1}
            + \sum_{i_2=0}^{m_2} \sum_{j_2=1}^{n_2} \left(
                     {c_{i_2 j_2}}_a^{u} \sin{{b_x}_2} \sin{j_2 \theta}
                    +{c_{i_2 j_2}}_b^{u} \sin{{b_x}_2} \cos{j_2 \theta}
                  \right)
    \\
    v = v_0 + \sum_{i_1=0}^{m_1} {c_{i_1}}^{v}\sin{{b_x}_1}
            + \sum_{i_2=0}^{m_2} \sum_{j_2=1}^{n_2} \left(
                     {c_{i_2 j_2}}_a^{v} \cos{{b_x}_2} \sin{j_2 \theta}
                    +{c_{i_2 j_2}}_b^{v} \cos{{b_x}_2} \cos{j_2 \theta}
                  \right)
    \\
    w = w_0 + \sum_{i_1=0}^{m_1} {c_{i_1}}^{w}\sin{{b_x}_1}
            + \sum_{i_2=0}^{m_2} \sum_{j_2=1}^{n_2} \left(
                     {c_{i_2 j_2}}_a^{w} \sin{{b_x}_2} \sin{j_2 \theta}
                    +{c_{i_2 j_2}}_b^{w} \sin{{b_x}_2} \cos{j_2 \theta}
                \right)
    \\

The linear stiffness matrix `[K_0]` is changed using the same
:ref:`elastic contraints used for the clpt_donnell_bc1 <elastic_constraints>`.

Observations:

    `\checkmark` linear buckling implemented

    `\checkmark` linear static implemented

    `\checkmark` non-linear analysis implemented


.. _clpt_donnell_bc4:

clpt_donnell_bc4
----------------

SS4- or CC4-types of boundary conditions (or anything in between).

.. math::
    u = u_0 + \sum_{i_1=0}^{m_1} {c_{i_1}}^{u} \sin{{b_x}_1}
            + \sum_{i_2=0}^{m_2} \sum_{j_2=1}^{n_2} \left(
                     {c_{i_2 j_2}}_a^{u} \cos{{b_x}_2} \sin{j_2 \theta}
                    +{c_{i_2 j_2}}_b^{u} \cos{{b_x}_2} \cos{j_2 \theta}
                  \right)
    \\
    v = v_0 + \sum_{i_1=0}^{m_1} {c_{i_1}}^{v}\sin{{b_x}_1}
            + \sum_{i_2=0}^{m_2} \sum_{j_2=1}^{n_2} \left(
                     {c_{i_2 j_2}}_a^{v} \cos{{b_x}_2} \sin{j_2 \theta}
                    +{c_{i_2 j_2}}_b^{v} \cos{{b_x}_2} \cos{j_2 \theta}
                  \right)
    \\
    w = w_0 + \sum_{i_1=0}^{m_1} {c_{i_1}}^{w}\sin{{b_x}_1}
            + \sum_{i_2=0}^{m_2} \sum_{j_2=1}^{n_2} \left(
                     {c_{i_2 j_2}}_a^{w} \sin{{b_x}_2} \sin{j_2 \theta}
                    +{c_{i_2 j_2}}_b^{w} \sin{{b_x}_2} \cos{j_2 \theta}
                \right)
    \\

The linear stiffness matrix `[K_0]` is changed using the same
:ref:`elastic contraints used for the clpt_donnell_bc1 <elastic_constraints>`.

Observations:

    `\checkmark` linear buckling implemented

    `\checkmark` linear static implemented

    `\checkmark` non-linear analysis implemented


.. _clpt_donnell_bcn:

clpt_donnell_bcn
----------------

General approximation function for the CLPT. It allows any type of
boundary condition by setting the proper values for the elastic
constants.

.. math::
    u = u_0 + \sum_{i_1=0}^{m_1} {c_{i_1}}^{u} \sin{{b_x}_1}
            + \sum_{i_2=0}^{m_2} \sum_{j_2=1}^{n_2} \left(
                     {c_{i_2 j_2}}_a^{u} \cos{{b_x}_2} \sin{j_2 \theta}
                    +{c_{i_2 j_2}}_b^{u} \cos{{b_x}_2} \cos{j_2 \theta}
                  \right)
    \\
    v = v_0 + \sum_{i_1=0}^{m_1} {c_{i_1}}^{v}\sin{{b_x}_1}
            + \sum_{i_2=0}^{m_2} \sum_{j_2=1}^{n_2} \left(
                     {c_{i_2 j_2}}_a^{v} \cos{{b_x}_2} \sin{j_2 \theta}
                    +{c_{i_2 j_2}}_b^{v} \cos{{b_x}_2} \cos{j_2 \theta}
                  \right)
    \\
    w = w_0 + \sum_{i_1=0}^{m_1} {c_{i_1}}^{w}\sin{{b_x}_1}
            + \sum_{i_2=0}^{m_2} \sum_{j_2=1}^{n_2} \left(
                     {c_{i_2 j_2}}_a^{w} \sin{{b_x}_2} \sin{j_2 \theta}
                    +{c_{i_2 j_2}}_b^{w} \sin{{b_x}_2} \cos{j_2 \theta}
                    +{c_{i_2 j_2}}_c^{w} \cos{{b_x}_2} \sin{j_2 \theta}
                    +{c_{i_2 j_2}}_d^{w} \cos{{b_x}_2} \cos{j_2 \theta}
                \right)

The linear stiffness matrix `[K_0]` is changed using the same
:ref:`elastic contraints used for the clpt_donnell_bc1 <elastic_constraints>`.

Observations:

    `\checkmark` linear static implemented

    `\times` not working for linear buckling

    `\rightarrow` non-linear analysis not implemented


.. _clpt_sanders_bc1:

clpt_sanders_bc1
----------------

Counterpart of :ref:`clpt_donnell_bc1` using the Sanders non-linear equations.

Observations:

    `\checkmark` linear static implemented

    `\checkmark` linear buckling implemented

    `\rightarrow` non-linear analysis not implemented


.. _clpt_sanders_bc2:

clpt_sanders_bc2
----------------

Counterpart of :ref:`clpt_donnell_bc2` using the Sanders non-linear equations.

Observations:

    `\checkmark` linear static implemented

    `\checkmark` linear buckling implemented

    `\rightarrow` non-linear analysis not implemented


.. _clpt_sanders_bc3:

clpt_sanders_bc3
----------------

Counterpart of :ref:`clpt_donnell_bc3` using the Sanders non-linear equations.

Observations:

    `\checkmark` linear static implemented

    `\checkmark` linear buckling implemented

    `\rightarrow` non-linear analysis not implemented


.. _clpt_sanders_bc4:

clpt_sanders_bc4
----------------

Counterpart of :ref:`clpt_donnell_bc4` using the Sanders non-linear equations.

Observations:

    `\checkmark` linear static implemented

    `\checkmark` linear buckling implemented

    `\rightarrow` non-linear analysis not implemented

.. _clpt_geier1997_bc2:

clpt_geier1997_bc2
------------------

.. note:: NOT RECOMMENDED, implemented for comparative purposes only.

Analogous to the model published by Geier and Singh (1997)
(see [geier1997]_ for more details) for the SS2- and CC2-types of
boundary condition. Originally proposed by Khdeir et al. (1989)
(see [khdeir1989]_). Uses the Donnell's equations and the
approximation functions are:

.. math::
    u = \sum_{i_2=0}^{m_2} \sum_{j_2=0}^{n_2} \left(
                     {c_{i_2 j_2}}^{u} \cos{{b_x}_2} \cos{j_2 \theta}
                  \right)
    \\
    v = \sum_{i_2=0}^{m_2} \sum_{j_2=0}^{n_2} \left(
                     {c_{i_2 j_2}}^{v} \sin{{b_x}_2} \sin{j_2 \theta}
                  \right)
    \\
    w = \sum_{i_2=0}^{m_2} \sum_{j_2=0}^{n_2} \left(
                     {c_{i_2 j_2}}^{w} \sin{{b_x}_2} \cos{j_2 \theta}
                  \right)

Observations:

    `\checkmark` linear buckling implemented

    `\rightarrow` linear static not implemented

    `\rightarrow` non-linear analysis not implemented

