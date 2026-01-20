import numpy as np

class Vector:
    def __init__(self, data):
        self.data = np.asarray(data, dtype=np.complex128)

    def inner(self, other):
        return np.vdot(self.data, other.data)

v = Vector([1, 2j, 3])
w = Vector([0, -1j, 4])

print(v.inner(w))

