import numpy as np
from scipy.sparse import csr_array

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
        'Sz' : csr_array( np.array( [[1,0,0],
                                     [0,0,0],
                                     [0,0,-1]])),
        
        'Sz2' : csr_array(np.array([[1,0,0],
                                     [0,0,0],
                                     [0,0,1]])),
        
        'Sx' : csr_array(1/np.sqrt(2)*np.array( [[0,1,0],
                                                 [1,0,1],
                                                 [0,1,0]])),
        
        'Sx2' : csr_array(np.array( [[0.5,0,0.5],
                                     [0 , 1 , 0],
                                     [0.5,0,0.5]])),
        
        'Sy' : csr_array(1j/np.sqrt(2)*np.array( [[0,-1,0],
                                                  [1,0,-1],
                                                  [0, 1,0]])),
        
        'Sy2' : csr_array(np.array( [[.5,0,-.5],
                                     [0 , 1 ,0],
                                     [-.5,0,.5]])),
    },

    '1/2': {
        'Sz' : csr_array(np.array([[1,0],[0,-1]])/2),
        'Sx' : csr_array( np.array([[0,1],[1,0]])/2),
        'Sy' : csr_array(np.array([[0,-1j],[1j,0]])/2),
    }
}


def is_hermitian(matrix: csr_array):
    # Check if the matrix is equal to its conjugate transpose
    return (matrix != matrix.getH()).nnz == 0