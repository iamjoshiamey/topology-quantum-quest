import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
# way to access vectors.py so we can get the stuff

from vectors import Vector, Operator


class CodeSpace:
    def __init__(self, basis_vectors):
        self.basis = basis_vectors

    def projector(self):
        n = len(self.basis[0].data)
        P = np.zeros((n, n), dtype=np.complex128)

        for v in self.basis:
            P += np.outer(v.data, v.data.conj())

        return Operator(P)

    def detects(self, errors, tol=1e-10):
        k = len(self.basis)

        for E in errors:
            for F in errors:
                A = E.adjoint().compose(F)
                ref = self.basis[0].inner(A(self.basis[0]))

                for i in range(k):
                    for j in range(k):
                        val = self.basis[i].inner(A(self.basis[j]))

                        if i != j:
                            if abs(val) > tol:
                                return False
                        else:
                            if abs(val - ref) > tol:
                                return False

        return True


if __name__ == "__main__":

    psi0 = Vector([1, 0])
    C = CodeSpace([psi0])

    P = C.projector()
    print(P.matrix)

    I = Operator.identity(2)
    X = Operator([[0, 1], [1, 0]])
    Z = Operator([[1, 0], [0, -1]])

    print(C.detects([I]))
    print(C.detects([I, Z]))
    print(C.detects([I, X]))
