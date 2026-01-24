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
class Operator:
    def __init__(self, matrix):
        self.matrix = np.asarray(matrix, dtype=np.complex128)
    def act_on(self,v):
        return Vector(self.matrix @ v.data)
    def __call__(self, v):
        return self.act_on(v)
    def adjoint(self):
        return Operator(np.conjugate(np.transpose(self.matrix)))
    @staticmethod #
    def identity(n):
        return Operator(np.eye(n))
    def compose(self, other):
        return Operator(self.matrix @ other.matrix)
    def is_unitary(self, tol=1e-10):
        n = self.matrix.shape[0]
        I = Operator.identity(n).matrix
        return np.allclose(self.adjoint().matrix @ self.matrix, I, atol=tol)

H = (1/np.sqrt(2)) * np.array([[1, 1],
                               [1, -1]])
Y= Operator(np.array([[1,0],[0,0]]))                              

A = Operator(H)

v = Vector([1, 0])

print(A(v))
print(A.adjoint().matrix)
print(A.compose(A.adjoint()).matrix)
print(Y.is_unitary())





