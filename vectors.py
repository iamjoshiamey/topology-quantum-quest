import numpy as np

class Vector:
    def __init__(self, data):
        self.data = np.asarray(data, dtype=np.complex128)

    def inner(self, other):
        return np.vdot(self.data, other.data)

    def norm(self):
        return np.sqrt(self.inner(self))

    def __mul__(self, scalar):
        return Vector(self.data * scalar)

    def normalize(self):
        return self * (1 / self.norm())
    def __repr__(self):
    return f"Vector({self.data})"



v = Vector([1, 2j, 3])

print(v.norm())
print(v.normalize().norm())

