import utils
from scipy.sparse import kron, eye

class Correlators:
    def __init__(self, n, spin='1'):
        self._n = n
        self._spin = spin
        
        i_sup = int(n/2 + 1) 

        k = kron(utils.spin_operators[spin]['Sx'], eye(utils.spin_states[spin]**(n-1), format="csr"))
        self._S1Six_array =   [self._build_S1Si(i, utils.spin_operators[spin]['Sx']) for i in range(1, i_sup)] +\
            [self._build_S1Si(i_sup, utils.spin_operators[spin]['Sx'])]
        
        k = kron(utils.spin_operators[spin]['Sy'], eye(utils.spin_states[spin]**(n-1), format="csr"))
        self._S1Siy_array =  [self._build_S1Si(i, utils.spin_operators[spin]['Sy']) for i in range(1, i_sup)] +\
            [self._build_S1Si(i_sup, utils.spin_operators[spin]['Sy'])]
        
        k = kron(utils.spin_operators[spin]['Sz'], eye(utils.spin_states[spin]**(n-1), format="csr"))
        self._S1Siz_array =  [self._build_S1Si(i, utils.spin_operators[spin]['Sz']) for i in range(1, i_sup)] +\
            [self._build_S1Si(i_sup, utils.spin_operators[spin]['Sz'])] 
        
        self._prodSix = self._build_prodSi(utils.spin_operators[spin]['Sx'])
        self._prodSiy = self._build_prodSi(utils.spin_operators[spin]['Sy'])
        self._prodSiz = self._build_prodSi(utils.spin_operators[spin]['Sz'])

    def _build_prodSi(self, operator):
        prodSi = eye(1, format="csr") 
        for j in range(self._n):
            prodSi = kron(prodSi, operator)
        return prodSi
    
    def _build_S1Si(self, i, operator):
        S1Si = operator
        for j in range(1, self._n):
            if i == j:
                S1Si = kron(S1Si, operator) 
            else: 
                S1Si = kron(S1Si, eye(utils.spin_states[self._spin], format="csr"));
        return S1Si
    
    def S1Six(self, i): return self._S1Six_array[i]
    def S1Siy(self, i): return self._S1Siy_array[i]
    def S1Siz(self, i ): return self._S1Siz_array[i]

    @property
    def prodSix(self): return self._prodSix
    @property
    def prodSiy(self): return self._prodSiy
    @property
    def prodSiz(self): return self._prodSiz
