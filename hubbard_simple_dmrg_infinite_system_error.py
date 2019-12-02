#!/usr/bin/env python
#
# Simple DMRG tutorial.  This code contains a basic implementation of the
# infinite system algorithm
#
# Copyright 2013 James R. Garrison and Ryan V. Mishmash.
# Open source under the MIT license.  Source code at
# <https://github.com/simple-dmrg/simple-dmrg/>

# This code will run under any version of Python >= 2.6.  The following line
# provides consistency between python2 and python3.
from __future__ import print_function, division  # requires Python >= 2.6

# numpy and scipy imports
import numpy as np
import scipy
from scipy.sparse import kron, identity
from scipy.sparse.linalg import eigsh  # Lanczos routine from ARPACK
from matplotlib import pyplot as plt

# We will use python's "namedtuple" to represent the Block and EnlargedBlock
# objects
from collections import namedtuple

Block = namedtuple("Block", ["length", "basis_size", "operator_dict"])
EnlargedBlock = namedtuple("EnlargedBlock", ["length", "basis_size", "operator_dict"])

def is_valid_block(block):
    for op in block.operator_dict.values():
        if op.shape[0] != block.basis_size or op.shape[1] != block.basis_size:
            return False
    return True

# This function should test the same exact things, so there is no need to
# repeat its definition.
is_valid_enlarged_block = is_valid_block

# Model-specific code for the Heisenberg XXZ chain
model_d = 4  # single-site basis size

# Sz1 = np.array([[0.5, 0], [0, -0.5]], dtype='d')  # single-site S^z
# Sp1 = np.array([[0, 1], [0, 0]], dtype='d')  # single-site S^+

c_up_one = np.zeros((4, 4), dtype='d')
c_down_one = np.zeros((4, 4), dtype='d')

c_up_one[0, 1] = 1.0
c_up_one[2, 3] = 1.0

c_down_one[0, 2] = 1.0
c_down_one[1, 3] = -1.0  # changed this value from -1.

phase_one = np.eye(4)
phase_one[1,1] = -1
phase_one[2,2] = -1
#phase_one = np.eye(4)  #turn off phase
N_up = c_up_one.T.conj().dot(c_up_one)
N_down = c_down_one.T.conj().dot(c_down_one)
N_U = (N_up - 0.5 * np.eye(4)).dot(N_down - 0.5 * np.eye(4))



H1 = np.zeros((4,4), dtype='d')  # single-site portion of H is zero

U = 10.0

#H1 = U*N_U

# def H2(Sz1, Sp1, Sz2, Sp2):  # two-site part of H
#     """Given the operators S^z and S^+ on two sites in different Hilbert spaces
#     (e.g. two blocks), returns a Kronecker product representing the
#     corresponding two-site term in the Hamiltonian that joins the two sites.
#     """
#     J = Jz = 1.
#     return (
#         (J / 2) * (kron(Sp1, Sp2.conjugate().transpose()) + kron(Sp1.conjugate().transpose(), Sp2)) +
#         Jz * kron(Sz1, Sz2)
#     )

def H2_huckel(c_up1, c_down1, c_up2, c_down2, phase):  # two-site part of H
    """Given the operators S^z and S^+ on two sites in different Hilbert spaces
    (e.g. two blocks), returns a Kronecker product representing the
    corresponding two-site term in the Hamiltonian that joins the two sites.
    """

    t = -2.0
    H2= \
        t * (kron(c_up1.conjugate().transpose().dot(phase), c_up2) + kron((1 * phase).dot(c_up1), c_up2.conjugate().transpose() )
        + kron(c_down1.conjugate().transpose().dot(phase), c_down2) + kron((1 * phase).dot(c_down1), c_down2.conjugate().transpose()))

        # changed the sign from + to - in the spirit c_1c_2 = c_2c_1
    return H2

def H2_huckel_intersite(m_sys,phase_env):  # two-site part of H
    """Given the operators S^z and S^+ on two sites in different Hilbert spaces
    (e.g. two blocks), returns a Kronecker product representing the
    corresponding two-site term in the Hamiltonian that joins the two sites.
    """
    negative_phase_one = 1 * phase_one



    sys_en1_up = kron(identity(m_sys),c_up_one.conjugate().transpose().dot(phase_one))
    sys_en2_up = kron(identity(m_sys), negative_phase_one.dot(c_up_one))
    env_en1_up = kron(phase_env,c_up_one)
    env_en2_up = kron(phase_env, c_up_one.conjugate().transpose())

    sys_en1_down = kron(identity(m_sys), c_down_one.conjugate().transpose().dot(phase_one))
    sys_en2_down = kron(identity(m_sys), negative_phase_one.dot(c_down_one))
    env_en1_down = kron(phase_env, c_down_one)
    env_en2_down = kron(phase_env, c_down_one.conjugate().transpose())
    t = -2.0
    H2= t * (kron(sys_en1_up,env_en1_up)+kron(sys_en2_up,env_en2_up)
             +kron(sys_en1_down,env_en1_down)+kron(sys_en2_down,env_en2_down))

        # changed the sign from + to - in the spirit c_1c_2 = c_2c_1
    return H2



# conn refers to the connection operator, that is, the operator on the edge of
# the block, on the interior of the chain.  We need to be able to represent S^z
# and S^+ on that site in the current basis in order to grow the chain.
# initial_block = Block(length=1, basis_size=model_d, operator_dict={
#     "H": H1,
#     "conn_Sz": Sz1,
#     "conn_Sp": Sp1,
# })

initial_block = Block(length=1, basis_size=model_d, operator_dict={
    "H": H1,
    "conn_c_up": c_up_one,
    "conn_c_down": c_down_one,
    "phase": phase_one,
    "phase_site": phase_one
})

def enlarge_block(block):
    """This function enlarges the provided Block by a single site, returning an
    EnlargedBlock.
    """
    mblock = block.basis_size
    o = block.operator_dict

    # Create the new operators for the enlarged block.  Our basis becomes a
    # Kronecker product of the Block basis and the single-site basis.  NOTE:
    # `kron` uses the tensor product convention making blocks of the second
    # array scaled by the first.  As such, we adopt this convention for
    # Kronecker products throughout the code.
    # enlarged_operator_dict = {
    #     "H": kron(o["H"], identity(model_d)) + kron(identity(mblock), H1) + H2(o["conn_Sz"], o["conn_Sp"], Sz1, Sp1),
    #     "conn_Sz": kron(identity(mblock), Sz1),
    #     "conn_Sp": kron(identity(mblock), Sp1),
    # }

    #print(kron(identity(mblock-model_d),phase_one).shape)
    enlarged_operator_dict = {

        # not passing phase matrix in this step
        #"H": kron(o["H"], identity(model_d)) + kron(identity(mblock), H1) + H2_huckel(o["conn_c_up"],
         #                                                                             o["conn_c_down"], c_up_one, c_down_one, np.eye(mblock)),
        "H" : kron(o["H"], identity(model_d)) + kron(identity(mblock), H1) + H2_huckel(o["conn_c_up"],
                                                                                       o["conn_c_down"], c_up_one,
                                                                                       c_down_one, o["phase_site"]),
        "conn_c_up": kron(identity(mblock), c_up_one),
        "conn_c_down": kron(identity(mblock), c_down_one),
        "phase": kron(o["phase"], phase_one),
        "phase_site" : kron(identity(mblock), phase_one)

        # phase contains information about number of electrons in previous states of the matrix
    }




    return EnlargedBlock(length=(block.length + 1),
                         basis_size=(block.basis_size * model_d),
                         operator_dict=enlarged_operator_dict)


def rotate_and_truncate(operator, transformation_matrix):
    """Transforms the operator to the new (possibly truncated) basis given by
    `transformation_matrix`.
    """
    return transformation_matrix.conjugate().transpose().dot(operator.dot(transformation_matrix))

def single_dmrg_step(sys, env, m, n):
    """Performs a single DMRG step using `sys` as the system and `env` as the
    environment, keeping a maximum of `m` states in the new basis.
    """
    assert is_valid_block(sys)
    assert is_valid_block(env)

    # Enlarge each block by a single site.
    sys_enl = enlarge_block(sys)
    #env_enl = enlarge_block_env(sys)

    if sys is env:  # no need to recalculate a second time
        env_enl = sys_enl
    else:
        env_enl = enlarge_block(env)

    assert is_valid_enlarged_block(sys_enl)
    assert is_valid_enlarged_block(env_enl)

    #sys and env operators

    m_sys = sys.basis_size

    env_op = env.operator_dict

    # Construct the full superblock Hamiltonian.
    m_sys_enl = sys_enl.basis_size
    m_env_enl = env_enl.basis_size
    sys_enl_op = sys_enl.operator_dict
    env_enl_op = env_enl.operator_dict

    H2_intersite = H2_huckel_intersite(m_sys,env_op["phase"])

    superblock_hamiltonian = kron(sys_enl_op["H"], identity(m_env_enl)) + kron(identity(m_sys_enl), env_enl_op["H"]) + H2_intersite



    #print(superblock_hamiltonian)
    # Call ARPACK to find the superblock ground state.  ("SA" means find the
    # "smallest in amplitude" eigenvalue.)
    energy, psi = eigsh(superblock_hamiltonian, k=n, which="SA")

    #print(energy)

    # Construct the reduced density matrix of the system by tracing out the
    # environment
    #
    # We want to make the (sys, env) indices correspond to (row, column) of a
    # matrix, respectively.  Since the environment (column) index updates most
    # quickly in our Kronecker product structure, psi0 is thus row-major ("C
    # style").
    rho = np.zeros(psi[:, 0].reshape([sys_enl.basis_size, -1], order="C").shape)

    for i in range(n):
        psi_temp = psi[:,i].reshape([sys_enl.basis_size, -1], order="C")
        rho += (1./n) * np.dot(psi_temp, psi_temp.conjugate().transpose())


    # Diagonalize the reduced density matrix and sort the eigenvectors by
    # eigenvalue.
    evals, evecs = np.linalg.eigh(rho)
    possible_eigenstates = []
    for eval, evec in zip(evals, evecs.transpose()):
        possible_eigenstates.append((eval, evec))
    possible_eigenstates.sort(reverse=True, key=lambda x: x[0])  # largest eigenvalue first

    # Build the transformation matrix from the `m` overall most significant
    # eigenvectors.
    my_m = min(len(possible_eigenstates), m)
    transformation_matrix = np.zeros((sys_enl.basis_size, my_m), dtype='d', order='F')
    for i, (eval, evec) in enumerate(possible_eigenstates[:my_m]):
        transformation_matrix[:, i] = evec

    #print(transformation_matrix)
    #transformation_matrix=np.eye(my_m)

    truncation_error = 1 - sum([x[0] for x in possible_eigenstates[:my_m]])
    print("truncation error:", truncation_error)

    # Rotate and truncate each operator.
    new_operator_dict = {}
    for name, op in sys_enl.operator_dict.items():
        new_operator_dict[name] = rotate_and_truncate(op, transformation_matrix)

    #print(new_operator_dict["phase"])

    newblock = Block(length=sys_enl.length,
                     basis_size=my_m,
                     operator_dict=new_operator_dict)

    return newblock, energy, truncation_error

def infinite_system_algorithm(L, m, n):
    block = initial_block
    # Repeatedly enlarge the system by performing a single DMRG step, using a
    # reflection of the current block as the environment.
    while 2 * block.length < L:
        print("L =", block.length * 2 )
        block, energy, error = single_dmrg_step(block, block, m=m, n=n)
        print("E =", energy)
        #print("delta E", energy[1]-energy[0])
        print("E/L =", energy / (block.length * 2))

    return error, energy[0]/L

if __name__ == "__main__":
    np.set_printoptions(precision=20, suppress=True, threshold=10000, linewidth=300)

    errorAr = np.zeros(10)
    energyAr = np.zeros(10)
    counter = 0
    for m in range(10, 30, 2) :
        error, energy = infinite_system_algorithm(L=40, m=m, n=1)  # n is the number of states
        errorAr[counter] = error
        energyAr[counter] = energy
        counter += 1

    errorAr = pow(10, 7) * errorAr

    plt.figure("Error", figsize=(8, 4))
    plt.title("Infinite algorithm length of superblock : " + str(40))
    plt.xlabel("1/m")
    plt.ylabel("Truncation error ( $ x  10^7 $)")
    plt.plot(1. / np.array(range(10, 30, 2)), errorAr, 'kx')

    # print(np.array(Egs) / L)
    # Egs = pow(10, 0) * (np.array(Egs) / L)

    plt.figure("Ground state energy per unit length", figsize=(8, 4))
    plt.title("Infinite algorithm length of superblock : " + str(40))
    plt.xlabel("1/m")
    plt.ylabel("Ground state energy/L")
    plt.plot(1. / np.array(range(10, 30, 2)), energyAr, 'kx')
    # plt.autoscale(False)
    plt.ylim(-2.5, -1.5)
    plt.show()
