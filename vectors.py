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
    def is_orthogonal(self, other, tol=1e-10):
        return abs(self.inner(other)) < tol
    def project_onto(self,other):
        return other*(other.inner(self)*(1/(other.inner(other))))
    def __sub__(self, other):
        return Vector(self.data- other.data)



v = Vector([1, 1])
w = Vector([0, 1])


print(v.project_onto(w))
print(v-w)
