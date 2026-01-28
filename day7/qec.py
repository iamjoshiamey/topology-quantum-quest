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
    def compress(self, A):
        a=[]
        k=len(self.basis)
        for i in range(k):
            b=[]
            for j in range(k):
               b.append(self.basis[i].inner(A(self.basis[j])))
            a.append(b)
        return np.array(a, dtype=np.complex128)


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



    e00 = Vector([1,0,0,0])
    e01 = Vector([0,1,0,0])
    e10 = Vector([0,0,1,0])
    e11 = Vector([0,0,0,1])
    zeroL = (e00 + e11) * (1/np.sqrt(2))
    oneL  = (e01 + e10) * (1/np.sqrt(2))
    I4 = Operator.identity(4)

    print("||0_L|| =", zeroL.norm())
    print("||1_L|| =", oneL.norm())
    print("<0_L|1_L> =", zeroL.inner(oneL))

    I2 = np.array([[1,0],[0,1]], dtype=np.complex128)   #already exit in thefile but rename and restart for day8
    X2 = np.array([[0,1],[1,0]], dtype=np.complex128)
    Z2 = np.array([[1,0],[0,-1]], dtype=np.complex128)

    X1 = Operator(np.kron(X2, I2)) #kron is tensor prodct of matrices
    X2op = Operator(np.kron(I2, X2))
    Z1 = Operator(np.kron(Z2, I2))
    Z2op = Operator(np.kron(I2, Z2))
    print("X1|0_L> =", X1(zeroL))
    print("X2|0_L> =", X2op(zeroL))
    print("Z1|0_L> =", Z1(zeroL))
    print("Z2|0_L> =", Z2op(zeroL))
    #santiy check done here

    Cbell_code = CodeSpace([zeroL, oneL])
    print("Bell code Detects {I}:", Cbell_code.detects([I4]))
    print("Bell code Detects {I, X1}:", Cbell_code.detects([I4, X1]))
    print("Bell code Detects {I, Z1}:", Cbell_code.detects([I4, Z1]))
    print("Bell code Detects {I, X1, X2}:", Cbell_code.detects([I4, X1, X2op]))
    print("Bell code Detects {I, Z1, Z2}:", Cbell_code.detects([I4, Z1, Z2op]))

    print("len(Cbell_code.basis) =", len(Cbell_code.basis))

    print("\nLogical X from X1:")
    print(Cbell_code.compress(X1))

    print("\nLogical X from X2:")
    print(Cbell_code.compress(X2op))

    print("\nLogical action of Z1:")
    print(Cbell_code.compress(Z1))

    print("\nLogical action of Z2:")
    print(Cbell_code.compress(Z2op))

    print("\nLogical Z from Z1Z2:")
    print(Cbell_code.compress(Z1.compose(Z2op)))


