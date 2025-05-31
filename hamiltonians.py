import utils
import numpy as np
from scipy.sparse import csr_matrix, kron, eye, linalg

class Hamiltonian:
    def __init__(self, n, spin='1'):
        """
        Parameters
        ----------
        n : int
            Number of spins in the system
        spin: str
            Spin value of the system
        """

        self._n = n
        self._spin = spin
        self._matrix = csr_matrix((utils.spin_states[spin]**self._n, 
            utils.spin_states[spin]**self._n), dtype=np.complex128)
        self._kroned_identities = [ 
                eye(utils.spin_states[spin]**i, format="csr") for i in range(0,n)
            ]
        self._gstate = None # ground state

    @property
    def n(self) -> int:
        return self._n
    @property
    def spin(self) -> str:
        return self._spin
    @property
    def matrix_dim(self) -> int:
        return self._matrix.shape[0] # type: ignore
    @property
    def gstate(self) -> np.ndarray:
        if self._gstate is None:
            self._gstate = linalg.eigsh(self._matrix, k=1, which='SA')[1][:,0] 
        return self._gstate
    
    def kroned_identity(self, index) -> csr_matrix:        
        return self._kroned_identities[index]

    def _build_term(self, i, operator) -> csr_matrix:
        return csr_matrix(kron(
        kron(self.kroned_identity(i), operator),
        kron(operator, self.kroned_identity(self.n - 2 - i))
    ))

    def _cyclical_term(self, operator) -> csr_matrix:
        return csr_matrix(kron(
            operator, kron(
            self.kroned_identity(self.n - 2),
            operator)
        ))

    def __str__(self):
        return f"Hamiltonian of {self.n} spins of value {self.spin}"


class BondAlternatingXXZ(Hamiltonian):
    def __init__(self, n, Delta, delta, spin='1'):
        """
        Parameters
        ----------
        n : int
            Number of spins in the system
        Delta : float
            Anisotropy parameter
        delta : float
            Alternation parameter
        spin: str
            Spin value of the system.
        """
        super().__init__(n, spin)
        self._Delta = Delta 
        self._delta = delta

        # Build the Hamiltonian matrix for the non-cyclic terms
        for l in range(n-1):
            coeff = 1 - delta * (-1)**(l)
            self._matrix += coeff*self._build_term(l, utils.spin_operators[spin]['Sx'])   # S_l^x S_{l+1}^x term
            self._matrix += coeff*self._build_term(l, utils.spin_operators[spin]['Sy'])   # S_l^y S_{l+1}^y term
            self._matrix += Delta*coeff*self._build_term(l, utils.spin_operators[spin]['Sz']) # Δ(S_l^z S_{l+1}^z) term

        # Cyclical terms
        coeff = 1 - delta * (-1)**(n-1)
        self._matrix += coeff*self._cyclical_term(utils.spin_operators[spin]['Sx'])  # S_N^x S_1^x term
        self._matrix += coeff*self._cyclical_term(utils.spin_operators[spin]['Sy'])  # S_N^y S_1^y term  
        self._matrix += Delta*coeff*self._cyclical_term(utils.spin_operators[spin]['Sz']) # Δ (S_N^z S_1^z) term

    
    @property
    def Delta(self):
        return self._Delta
    @property
    def delta(self):
        return self._delta
    
    def __str__(self):
        return super().__str__()  +  \
        f"""\nBond-alternating XXZ chain\n\nHamiltonian properties:
        δ = {self.delta} 
        Δ = {self.Delta}.\n"""
        

class XXZUniaxialSingleIonAnisotropy(Hamiltonian):
    def __init__(self, n, Jz, D, spin='1', J=1):
        """
        Parameters
        ----------
        n : int
            Number of spins in the system.
        J : float
            Spin coupling parameter in the xy-axis. Defaults to 1.
        Jz : float
            Spin coupling parameter in the z-axis.
        D : float
            Uniaxial single ion anisotropy parameter.
        spin: str
            Spin value of the system.
        """
        super().__init__(n, spin)
        self._Jz = Jz 
        self._D = D

        # Build the Hamiltonian matrix for the non-cyclic terms
        for l in range(n-1):
            self._matrix += J*self._build_term(l, utils.spin_operators[spin]['Sx'])   # S_l^x S_{l+1}^x term
            self._matrix += J*self._build_term(l, utils.spin_operators[spin]['Sy'])   # S_l^y S_{l+1}^y term
            self._matrix += Jz*self._build_term(l, utils.spin_operators[spin]['Sz'])   # S_l^z S_{l+1}^z term
            self._matrix += D*self._build_anisotropy(l, utils.spin_operators[spin]['Sz2'])   # S_l^z2 term
        # Cyclical terms
        self._matrix += J*self._cyclical_term(utils.spin_operators[spin]['Sx'])  # S_N^x S_1^x term
        self._matrix += J*self._cyclical_term(utils.spin_operators[spin]['Sy'])  # S_N^y S_1^y term  
        self._matrix += Jz*self._cyclical_term(utils.spin_operators[spin]['Sz']) # S_N^z S_1^z term
        self._matrix += D*self._build_anisotropy(n-1, utils.spin_operators[spin]['Sz2'])   # S_l^z2 term
    
    def _build_anisotropy(self, i, operator):
        return kron(
        kron(self.kroned_identity(i), operator),
        self.kroned_identity(self._n - 1 - i)
    )
    
    @property
    def Jz(self):
        return self._Jz
    @property
    def D(self):
        return self._D
    def __str__(self):
        return super().__str__()  +  \
        f"""XXZ chains with uniaxial single-ion-type anisotropy,\n\nHamiltonian properties:
        Jz = {self.Jz} 
        D = {self.D}.\n"""



class BilinearBiquadratic(Hamiltonian):
    def __init__(self, n, theta, spin='1'):
        """
        Parameters
        ----------
        n : int
            Number of spins in the system.
        theta : float
            Spin coupling parameter.
        spin: str
            Spin value of the system.
        """
        super().__init__(n, spin)
        self._theta = theta 

        # Build the Hamiltonian matrix for the non-cyclic terms
        arg1 = np.cos(theta)
        arg2 = np.sin(theta)

        for l in range(n-1):
            X_term =  self._build_term(l, utils.spin_operators[spin]['Sx'])
            Y_term = self._build_term(l, utils.spin_operators[spin]['Sy'])
            Z_term = self._build_term(l, utils.spin_operators[spin]['Sz'])

            # linear and pure quadratic terms
            self._matrix += arg1*X_term + arg2*(X_term @ X_term)    # S_l^x S_{l+1}^x term
            self._matrix += arg1*Y_term + arg2*(Y_term @ Y_term)   # S_l^y S_{l+1}^y term
            self._matrix += arg1*Z_term + arg2*(Z_term @ Z_term)   # S_l^z S_{l+1}^z term
            
            # Mixed quadratic terms
            self._matrix += arg2*(X_term @ Y_term + Y_term @ X_term) 
            self._matrix += arg2*(X_term @ Z_term + Z_term @ X_term)
            self._matrix += arg2*(Y_term @ Z_term + Z_term @ Y_term)



        # Cyclical terms
        X_term = self._cyclical_term(utils.spin_operators[spin]['Sx'])
        Y_term = self._cyclical_term(utils.spin_operators[spin]['Sy'])
        Z_term = self._cyclical_term(utils.spin_operators[spin]['Sz'])

        self._matrix += arg1*X_term + arg2*(X_term @ X_term) # S_N^x S_1^x term
        self._matrix += arg1*Y_term + arg2*(Y_term @ Y_term) # S_N^y S_1^y term
        self._matrix += arg1*Z_term + arg2*(Z_term @ Z_term) # S_N^z S_1^z term

        self._matrix += arg2*(X_term @ Y_term + Y_term @ X_term) 
        self._matrix += arg2*(X_term @ Z_term + Z_term @ X_term)
        self._matrix += arg2*(Y_term @ Z_term + Z_term @ Y_term)

    @property
    def theta(self):
        return self._theta
    
    def __str__(self):
        return super().__str__()  +  \
        f"""Bilinear Biquadratic Chain.,\n\nHamiltonian properties:
        θ = {self.theta}\n"""