import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from vectors import Vector, Operator
from qec import CodeSpace
ket0=Vector([1,0])
ket1=Vector([0,1])
def kron_vec(v: Vector, w: Vector) -> Vector:
    return Vector(np.kron(v.data, w.data))

def kron_op(X: Operator, Y: Operator) -> Operator:
    return Operator(np.kron(X.matrix, Y.matrix))
def sign_pm_one(z, tol=1e-6):   #only want to see 1 or -1 instead of complex numbers.
    return 1 if z.real > 0 else -1 




ket000 = kron_vec(kron_vec(ket0, ket0), ket0)
ket111 = kron_vec(kron_vec(ket1, ket1), ket1)

zeroL = ket000
oneL = ket111
print("||0L|| =", zeroL.norm())
print("||1L|| =", oneL.norm())
print("<0L|1L> =", zeroL.inner(oneL))

I = Operator([[1, 0], [0, 1]])
X = Operator([[0, 1], [1, 0]])
Z = Operator([[1, 0], [0, -1]])
I3 = kron_op(kron_op(I, I), I)

X1 = kron_op(kron_op(X, I), I)
X2 = kron_op(kron_op(I, X), I)
X3 = kron_op(kron_op(I, I), X)

Z1 = kron_op(kron_op(Z, I), I)
Z2 = kron_op(kron_op(I, Z), I)
Z3 = kron_op(kron_op(I, I), Z)
I3 = kron_op(kron_op(I, I), I)

print("X1 |000> =", X1(zeroL))
print("X2 |000> =", X2(zeroL))
print("X3 |000> =", X3(zeroL))
S1 = Z1.compose(Z2)   
S2 = Z2.compose(Z3)   
def syndrome(state: Vector):
    s1 = state.inner(S1(state))
    s2 = state.inner(S2(state))
    return (s1, s2)
print("syn(000) =", syndrome(zeroL))
print("syn(X1 000) =", syndrome(X1(zeroL)))
print("syn(X2 000) =", syndrome(X2(zeroL)))
print("syn(X3 000) =", syndrome(X3(zeroL)))


def correct_bitflip(state: Vector) -> Vector:
    s1, s2 = syndrome(state)
    s = (sign_pm_one(s1), sign_pm_one(s2))

    if s == (1, 1):
        return state
    if s == (-1, 1):
        return X1(state)
    if s == (-1, -1):
        return X2(state)
    if s == (1, -1):
        return X3(state)

    raise ValueError(f"Unexpected syndrome {s}")



# random logical state
alpha = np.random.randn() + 1j*np.random.randn()
beta  = np.random.randn() + 1j*np.random.randn()

psi = zeroL * alpha + oneL * beta
psi = psi * (1 / psi.norm())

# inject random single-qubit error
error = np.random.choice([X1, X2, X3])
psi_err = error(psi)

# correct
psi_corr = correct_bitflip(psi_err)

print("|<psi | psi_corrected>| =", abs(psi.inner(psi_corr)))

XL = X1.compose(X2).compose(X3)
ZL = Z1
print("XL |0L> =", XL(zeroL))
print("XL |1L> =", XL(oneL))

print("ZL |0L> =", ZL(zeroL))
print("ZL |1L> =", ZL(oneL))

