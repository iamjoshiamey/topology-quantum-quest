import numpy as np


class Vector:
    """
    Finite-dimensional real or complex vector with an inner product.
    """

    def __init__(self, data):
        self.data = np.asarray(data, dtype=np.complex128)

    def __len__(self):
        return self.data.shape[0]

    def __add__(self, other):
        self._check_dim(other)
        return Vector(self.data + other.data)

    def __mul__(self, scalar):
        return Vector(scalar * self.data)

    __rmul__ = __mul__

    def inner(self, other):
        """
        Inner product ⟨self, other⟩, conjugate-linear in the first argument.
        """
        self._check_dim(other)
        return np.vdot(self.data, other.data)

    def norm(self):
        return np.sqrt(self.inner(self).real)

    def normalize(self):
        n = self.norm()
        if np.isclose(n, 0):
            raise ValueError("Cannot normalize the zero vector.")
        return (1 / n) * self

    def _check_dim(self, other):
        if len(self) != len(other):
            raise ValueError("Dimension mismatch.")


