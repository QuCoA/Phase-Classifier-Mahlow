import numpy as np
from scipy.sparse import csr_matrix

def SpatialSignScaling(X: np.array) -> np.array:
    X = X - np.mean(X, axis=1, keepdims=True)
    for i in range(len(X[0,:])):
        X[i,:] = X[i,:]/np.linalg.norm(X[i,:])

    return X    

spin_states = { 
    '1': 3,
    '1/2': 2,
}

spin_operators = {
    '1': {
        'Sz' : csr_matrix( np.array( [[1,0,0],
                                     [0,0,0],
                                     [0,0,-1]]), dtype=np.complex128),
        
        'Sz2' : csr_matrix(np.array([[1,0,0],
                                     [0,0,0],
                                     [0,0,1]]), dtype=np.complex128),
        
        'Sx' : csr_matrix(1/np.sqrt(2)*np.array( [[0,1,0],
                                                 [1,0,1],
                                                 [0,1,0]]), dtype=np.complex128),
        
        'Sx2' : csr_matrix(np.array( [[0.5,0,0.5],
                                     [0 , 1 , 0],
                                     [0.5,0,0.5]]), dtype=np.complex128),
        
        'Sy' : csr_matrix(1j/np.sqrt(2)*np.array( [[0,-1,0],
                                                  [1,0,-1],
                                                  [0, 1,0]]), dtype=np.complex128),
        
        'Sy2' : csr_matrix(np.array( [[.5,0,-.5],
                                     [0 , 1 ,0],
                                     [-.5,0,.5]]), dtype=np.complex128),
    },

    '1/2': {
        'Sz' : csr_matrix(np.array([[1,0],[0,-1]])/2),
        'Sx' : csr_matrix( np.array([[0,1],[1,0]])/2),
        'Sy' : csr_matrix(np.array([[0,-1j],[1j,0]])/2),
    }
}


def is_hermitian(matrix: csr_matrix):
    # Check if the matrix is equal to its conjugate transpose
    return (matrix != matrix.getH()).nnz == 0