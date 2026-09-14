import numpy as np
import sympy as sp

v1 = np.array([-1, 1, 0])
v2 = np.array([1, 1, -2])
v3 = np.array([1, 1, 0])

x = 0.5*v1 - v2 + 2.5*v3

print(x)

"""
E.6: Verify linear independence and find the dimension of the span
for four sets of vectors/functions.
"""

print("=" * 60)
print("i. Vectors in R^3: [1,2,3], [1,0,1], [1,2,1]")
print("=" * 60)

A = np.array([[1, 1, 1],
              [2, 0, 2],
              [3, 1, 1]], dtype=float)

det_A = np.linalg.det(A)
rank_A = np.linalg.matrix_rank(A)

print(f"Matrix (vectors as columns):\n{A}")
print(f"Determinant: {det_A:.4f}")
print(f"Rank: {rank_A}")
print(f"Independent? {'Yes' if abs(det_A) > 1e-9 else 'No'}")
print(f"Dimension of span: {rank_A}\n")


print("=" * 60)
print("ii. Functions: sin(t), cos(t), cos(2t)")
print("=" * 60)

t = sp.symbols('t')
funcs = [sp.sin(t), sp.cos(t), sp.cos(2 * t)]
c1, c2, c3 = sp.symbols('c1 c2 c3')

# Method 1: force the linear combination to be zero at several points
combo = c1 * funcs[0] + c2 * funcs[1] + c3 * funcs[2]
sample_points = [0, sp.pi / 2, sp.pi]
equations = [sp.Eq(combo.subs(t, val), 0) for val in sample_points]
solution = sp.solve(equations, [c1, c2, c3])
print(f"Solving c1*sin(t)+c2*cos(t)+c3*cos(2t)=0 at sample points: {solution}")

# Method 2: Wronskian (nonzero at some point => independent)
W = sp.Matrix([[f, sp.diff(f, t), sp.diff(f, t, 2)] for f in funcs]).T
W_det = sp.simplify(W.det())
W_at_0 = W_det.subs(t, 0)
print(f"Wronskian determinant: {W_det}")
print(f"Wronskian at t=0: {W_at_0}")
independent_ii = (solution == {c1: 0, c2: 0, c3: 0}) and (W_at_0 != 0)
print(f"Independent? {'Yes' if independent_ii else 'No'}")
print(f"Dimension of span: {3 if independent_ii else 'less than 3'}\n")


print("=" * 60)
print("iii. Polynomials: 1+t, 1-t")
print("=" * 60)

g1, g2 = 1 + t, 1 - t
a, b = sp.symbols('a b')
combo3 = a * g1 + b * g2
poly = sp.Poly(combo3, t)
coeffs = poly.all_coeffs()
solution3 = sp.solve(coeffs, [a, b])

print(f"a*(1+t) + b*(1-t) = 0  =>  coefficient equations: {coeffs}")
print(f"Solving: {solution3}")
independent_iii = (solution3 == {a: 0, b: 0})
print(f"Independent? {'Yes' if independent_iii else 'No'}")
# 2 independent vectors inside the 2-dimensional space P1 span all of P1
print("Dimension of span: 2 (spans all of P1, since dim(P1) = 2)\n")


print("=" * 60)
print("iv. Vectors in R^4: [1,2,2,1], [1,0,0,1], [3,4,4,3]")
print("=" * 60)

B = np.array([[1, 1, 3],
              [2, 0, 4],
              [2, 0, 4],
              [1, 1, 3]], dtype=float)

rank_B = np.linalg.matrix_rank(B)
print(f"Matrix (vectors as columns):\n{B}")
print(f"Rank: {rank_B}  (3 vectors, but rank < 3 => dependent)")

v1 = sp.Matrix([1, 2, 2, 1])
v2 = sp.Matrix([1, 0, 0, 1])
v3 = sp.Matrix([3, 4, 4, 3])
a_, b_ = sp.symbols('a_ b_')
dependency = sp.solve(sp.Eq(a_ * v1 + b_ * v2, v3), [a_, b_])
print(f"Checking v3 = a*v1 + b*v2: solution = {dependency}")
print(f"Independent? {'No' if rank_B < 3 else 'Yes'}")
print(f"Dimension of span: {rank_B}\n")


print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"{'Set':<6}{'Independent?':<15}{'Dim of span'}")
print(f"{'i.':<6}{'Yes':<15}{3}")
print(f"{'ii.':<6}{'Yes':<15}{3}")
print(f"{'iii.':<6}{'Yes':<15}{2}")
print(f"{'iv.':<6}{'No':<15}{2}")