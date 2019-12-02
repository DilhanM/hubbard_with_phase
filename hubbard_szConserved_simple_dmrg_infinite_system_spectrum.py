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
from scipy.sparse import kron, identity, lil_matrix
from scipy.sparse.linalg import eigsh  # Lanczos routine from ARPACK
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

# We will use python's "namedtuple" to represent the Block and EnlargedBlock
# objects
from collections import namedtuple

Block = namedtuple("Block", ["length", "basis_size", "operator_dict", "basis_sector_array"])
EnlargedBlock = namedtuple("EnlargedBlock", ["length", "basis_size", "operator_dict", "basis_sector_array"])

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
single_site_sectors = np.array([0.0, 0.5, -0.5, 0.0])  # S^z sectors corresponding to the
                                             # single site basis elements

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

U = 100.0

H1 = U*N_U

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
}, basis_sector_array=single_site_sectors)

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

    # This array keeps track of which sector each element of the new basis is
    # in.  `np.add.outer()` creates a matrix that adds each element of the
    # first vector with each element of the second, which when flattened
    # contains the sector of each basis element in the above Kronecker product.
    enlarged_basis_sector_array = np.add.outer(block.basis_sector_array, single_site_sectors).flatten()




    return EnlargedBlock(length=(block.length + 1),
                         basis_size=(block.basis_size * model_d),
                         operator_dict=enlarged_operator_dict,
                         basis_sector_array=enlarged_basis_sector_array)


def rotate_and_truncate(operator, transformation_matrix):
    """Transforms the operator to the new (possibly truncated) basis given by
    `transformation_matrix`.
    """
    return transformation_matrix.conjugate().transpose().dot(operator.dot(transformation_matrix))

def index_map(array):
    """Given an array, returns a dictionary that allows quick access to the
    indices at which a given value occurs.

    Example usage:

    >>> by_index = index_map([3, 5, 5, 7, 3])
    >>> by_index[3]
    [0, 4]
    >>> by_index[5]
    [1, 2]
    >>> by_index[7]
    [3]
    """
    d = {}
    for index, value in enumerate(array):
        d.setdefault(value, []).append(index)
    return d


def single_dmrg_step(sys, env, m, n, target_Sz):
    """Performs a single DMRG step using `sys` as the system and `env` as the
    environment, keeping a maximum of `m` states in the new basis.
    """
    assert is_valid_block(sys)
    assert is_valid_block(env)

    # Enlarge each block by a single site.
    sys_enl = enlarge_block(sys)
    sys_enl_basis_by_sector = index_map(sys_enl.basis_sector_array)
    #env_enl = enlarge_block_env(sys)

    if sys is env:  # no need to recalculate a second time
        env_enl = sys_enl
        env_enl_basis_by_sector = sys_enl_basis_by_sector
    else:
        env_enl = enlarge_block(env)
        env_enl_basis_by_sector = index_map(env_enl.basis_sector_array)

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

    # Build up a "restricted" basis of states in the target sector and
    # reconstruct the superblock Hamiltonian in that sector.
    sector_indices = {}  # will contain indices of the new (restricted) basis
    # for which the enlarged system is in a given sector
    restricted_basis_indices = []  # will contain indices of the old (full) basis, which we are mapping to
    for sys_enl_Sz, sys_enl_basis_states in sys_enl_basis_by_sector.items() :
        sector_indices[sys_enl_Sz] = []
        env_enl_Sz = target_Sz - sys_enl_Sz
        if env_enl_Sz in env_enl_basis_by_sector :
            for i in sys_enl_basis_states :
                i_offset = m_env_enl * i  # considers the tensor product structure of the superblock basis
                for j in env_enl_basis_by_sector[env_enl_Sz] :
                    current_index = len(restricted_basis_indices)  # about-to-be-added index of restricted_basis_indices
                    sector_indices[sys_enl_Sz].append(current_index)
                    restricted_basis_indices.append(i_offset + j)

    restricted_superblock_hamiltonian = superblock_hamiltonian[:, restricted_basis_indices][restricted_basis_indices, :]

    #print(superblock_hamiltonian)
    # Call ARPACK to find the superblock ground state.  ("SA" means find the
    # "smallest in amplitude" eigenvalue.)
    energy, restricted_psi = eigsh(restricted_superblock_hamiltonian, k=n, which="SA")

    #print(energy)

    # Construct the reduced density matrix of the system by tracing out the
    # environment
    #
    # We want to make the (sys, env) indices correspond to (row, column) of a
    # matrix, respectively.  Since the environment (column) index updates most
    # quickly in our Kronecker product structure, psi0 is thus row-major ("C
    # style").
    #rho = np.zeros(psi[:, 0].reshape([sys_enl.basis_size, -1], order="C").shape)

    #for i in range(n):
     #   psi_temp = psi[:,i].reshape([sys_enl.basis_size, -1], order="C")
      #  rho += (1./n) * np.dot(psi_temp, psi_temp.conjugate().transpose())
    rho_block_dict = {}
    for i in range(n):

        for sys_enl_Sz, indices in sector_indices.items() :
            if indices :  # if indices is nonempty
                psi_sector = restricted_psi[indices, :]
                # We want to make the (sys, env) indices correspond to (row,
                # column) of a matrix, respectively.  Since the environment
                # (column) index updates most quickly in our Kronecker product
                # structure, psi0_sector is thus row-major ("C style").
                psi_sector = psi_sector.reshape([len(sys_enl_basis_by_sector[sys_enl_Sz]), -1], order="C")
                #print(rho_block_dict )
                #exit(1)

                #print(rho_block_dict.get(sys_enl_Sz))
                if (sys_enl_Sz in rho_block_dict) == False:
                    rho_block_dict[sys_enl_Sz] = (1. / pow(n,2)) * np.dot(psi_sector, psi_sector.conjugate().transpose())
                    #print(rho_block_dict.get(sys_enl_Sz))
                    #print(sys_enl_Sz,i)
                else :
                    rho_block_dict[sys_enl_Sz] += (1. / pow(n,2)) * np.dot(psi_sector, psi_sector.conjugate().transpose())

    # Diagonalize each block of the reduced density matrix and sort the
    # eigenvectors by eigenvalue.
    #print(rho_block_dict.keys())
    # Diagonalize the reduced density matrix and sort the eigenvectors by
    # eigenvalue.
    #evals, evecs = np.linalg.eigh(rho)
    #possible_eigenstates = []
    #for eval, evec in zip(evals, evecs.transpose()):
    #    possible_eigenstates.append((eval, evec))
    #possible_eigenstates.sort(reverse=True, key=lambda x: x[0])  # largest eigenvalue first

    possible_eigenstates = []
    for Sz_sector, rho_block in rho_block_dict.items() :
        evals, evecs = np.linalg.eigh(rho_block)
        current_sector_basis = sys_enl_basis_by_sector[Sz_sector]
        for eval, evec in zip(evals, evecs.transpose()) :
            possible_eigenstates.append((eval, evec, Sz_sector, current_sector_basis))
    possible_eigenstates.sort(reverse=True, key=lambda x : x[0])  # largest eigenvalue first

    # Build the transformation matrix from the `m` overall most significant
    # eigenvectors.
    #my_m = min(len(possible_eigenstates), m)
    #transformation_matrix = np.zeros((sys_enl.basis_size, my_m), dtype='d', order='F')
    #for i, (eval, evec) in enumerate(possible_eigenstates[:my_m]):
     #   transformation_matrix[:, i] = evec

    #print(transformation_matrix)
    #transformation_matrix=np.eye(my_m)

        # Build the transformation matrix from the `m` overall most significant
        # eigenvectors.  It will have sparse structure due to the conserved quantum
        # number.
    my_m = min(len(possible_eigenstates), m)
    transformation_matrix = lil_matrix((sys_enl.basis_size, my_m), dtype='d')
    new_sector_array = np.zeros((my_m,), dtype='d')  # lists the sector of each
    # element of the new/truncated basis
    for i, (eval, evec, Sz_sector, current_sector_basis) in enumerate(possible_eigenstates[:my_m]) :
        for j, v in zip(current_sector_basis, evec) :
            transformation_matrix[j, i] = v
        new_sector_array[i] = Sz_sector
    # Convert the transformation matrix to a more efficient internal
    # representation.  `lil_matrix` is good for constructing a sparse matrix
    # efficiently, but `csr_matrix` is better for performing quick
    # multiplications.
    transformation_matrix = transformation_matrix.tocsr()
    #transformation_matrix=np.eye(transformation_matrix.shape[0])
    #print([x[0] for x in possible_eigenstates[:]])
    truncation_error = 1 - sum([x[0] for x in possible_eigenstates[:my_m]])
    print("truncation error:", truncation_error)

    # Rotate and truncate each operator.
    new_operator_dict = {}
    for name, op in sys_enl.operator_dict.items():
        new_operator_dict[name] = rotate_and_truncate(op, transformation_matrix)

    #print(new_operator_dict["phase"])

    newblock = Block(length=sys_enl.length,
                     basis_size=my_m,
                     operator_dict=new_operator_dict,
                     basis_sector_array=new_sector_array)
    #print(new_sector_array,len(new_sector_array))
    return newblock, energy, truncation_error

def infinite_system_algorithm(L, m, n, target_Sz):
    block = initial_block
    # Repeatedly enlarge the system by performing a single DMRG step, using a
    # reflection of the current block as the environment.
    while 2 * block.length < L:
        current_L = 2 * block.length + 2  # current superblock length
        if current_L == L:
            current_target_Sz = target_Sz
        else:
            current_target_Sz = 0
        #current_target_Sz = int(target_Sz) * current_L // L  #CHECK HERE
        #current_target_Sz = target_Sz
        print("L =", current_L)
        print("Current target Sz : ",current_target_Sz)
        block, energy, error = single_dmrg_step(block, block, m=m, n=n, target_Sz=current_target_Sz)
        print("energy :",energy)
        print("E/L =", energy / current_L)

        plt.figure(2 * block.length )
        plt.title("Infinite algorithm length of superblock : " + str(block.length * 2) + " Target $S_Z$ :" +str(current_target_Sz))
        plt.xlabel("Energy level")
        plt.ylabel("Energy/L")
        plt.plot(range(n), energy/L, 'kx')

    plt.show()

if __name__ == "__main__":
    np.set_printoptions(precision=20, suppress=True, threshold=10000, linewidth=300)

    infinite_system_algorithm(L=8, m=100, n = 5, target_Sz=-1)

