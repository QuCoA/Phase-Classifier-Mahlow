import utils
from scipy.sparse import kron, eye


class Correlators:
    """
    Correlators class for constructing spin correlation operators in quantum spin systems.

    This class provides methods to build and access two-site and multi-site spin correlation operators
    for a chain of spins of arbitrary length and spin value.

    Attributes:
        _n (int): Number of spins in the chain.
        _spin (str): Spin value as a string (e.g., "1" for spin-1).
        _S1Six_array (list): List of S1·Si correlation operators for Sx.
        _S1 Siy_array (list): List of S1·Si correlation operators for Sy.
        _S1Siz_array (list): List of S1·Si correlation operators for Sz.
        _prodSix (csr_matrix): Product operator for Sx on all sites.
        _prodSiy (csr_matrix): Product operator for Sy on all sites.
        _prodSiz (csr_matrix): Product operator for Sz on all sites.

    Methods:
        S1Six(i): Returns the S1·Si correlation operator for Sx at site i.
        S1Siy(i): Returns the S1·Si correlation operator for Sy at site i.
        S1Siz(i): Returns the S1·Si correlation operator for Sz at site i.

    Properties:
        prodSix: Returns the product operator for Sx on all sites.
        prodSiy: Returns the product operator for Sy on all sites.
        prodSiz: Returns the product operator for Sz on all sites.

    Private Methods:
        _build_prodSi(operator): Constructs the product operator for a given spin component.
        _build_S1Si(i, operator): Constructs the S1·Si correlation operator for a given spin component and site.
    """

    def __init__(self, n, spin="1"):
        self._n = n
        self._spin = spin

        i_sup = int(n / 2 + 1)

        self._S1Six_array = [
            kron(
                utils.spin_operators[spin]["Sx2"],
                eye(utils.spin_states[spin] ** (n - 1), format="csr"),
            )
        ] + [
            self._build_S1Si(i, utils.spin_operators[spin]["Sx"])
            for i in range(1, i_sup)
        ]

        self._S1Siy_array = [
            kron(
                utils.spin_operators[spin]["Sy2"],
                eye(utils.spin_states[spin] ** (n - 1), format="csr"),
            )
        ] + [
            self._build_S1Si(i, utils.spin_operators[spin]["Sy"])
            for i in range(1, i_sup)
        ]

        self._S1Siz_array = [
            kron(
                utils.spin_operators[spin]["Sz2"],
                eye(utils.spin_states[spin] ** (n - 1), format="csr"),
            )
        ] + [
            self._build_S1Si(i, utils.spin_operators[spin]["Sz"])
            for i in range(1, i_sup)
        ]

        self._prodSix = self._build_prodSi(utils.spin_operators[spin]["Sx"])
        self._prodSiy = self._build_prodSi(utils.spin_operators[spin]["Sy"])
        self._prodSiz = self._build_prodSi(utils.spin_operators[spin]["Sz"])

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
                S1Si = kron(S1Si, eye(utils.spin_states[self._spin], format="csr"))
        return S1Si

    def S1Six(self, i):
        return self._S1Six_array[i]

    def S1Siy(self, i):
        return self._S1Siy_array[i]

    def S1Siz(self, i):
        return self._S1Siz_array[i]

    @property
    def prodSix(self):
        return self._prodSix

    @property
    def prodSiy(self):
        return self._prodSiy

    @property
    def prodSiz(self):
        return self._prodSiz
