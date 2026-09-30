import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

## Eigenvalues and Eigenvectors
m1=np.array([[2, 0],[0, 5]])
m2=np.array([[5, 1],[4, 5]])
m3=np.array([[3, 5],[3, 1]])

eigenvalues, eigenvectors = np.linalg.eig(m3)
print("Eigenvalues:")
print(np.real_if_close(eigenvalues))
print("\nEigenvectors:")
print(np.real_if_close(eigenvectors))

### E.1
B=np.array([[1, 1],[1, -1]], dtype=complex)
A_std = np.array([[1, -1], [1,  1]], dtype=complex)
A_prime = np.linalg.inv(B) @ A_std @ B
print(np.round(A_prime, 10))

#Eigenvalues of A prime
eigvals, Z = np.linalg.eig(A_prime)
print("eigenvalues =", eigvals, "\n")
print("eigenvectors =", Z, "\n")

## A_prime_z
A_prime_z = np.linalg.inv(Z) @ A_prime @ Z
print(np.round(A_prime_z, 10))

### Part 1
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

# ----------------------------------------------------------------------
# Part i: symbolic gradient
# ----------------------------------------------------------------------
x, y = sp.symbols('x y')
f_sym = 2 * x * y + x ** 2 + y

df_dx = sp.diff(f_sym, x)
df_dy = sp.diff(f_sym, y)

print("f(x, y)       =", f_sym)
print("df/dx         =", df_dx)
print("df/dy         =", df_dy)
print("grad f(x, y)  = (%s, %s)" % (df_dx, df_dy))
print()

# Turn the symbolic derivatives into fast numeric functions
grad_x_func = sp.lambdify((x, y), df_dx, 'numpy')
grad_y_func = sp.lambdify((x, y), df_dy, 'numpy')
f_func = sp.lambdify((x, y), f_sym, 'numpy')

# ----------------------------------------------------------------------
# Part ii: evaluate the gradient at the three points
# ----------------------------------------------------------------------
points = [(1, 1), (0, -1), (0, 0)]

print("Gradient at requested points:")
gradients = {}
for (px, py) in points:
    gx = grad_x_func(px, py)
    gy = grad_y_func(px, py)
    gradients[(px, py)] = (gx, gy)
    print(f"  grad f({px:>2}, {py:>2}) = ({gx}, {gy})")
print()

# ----------------------------------------------------------------------
# Plot the surface
# ----------------------------------------------------------------------
X = np.linspace(-2, 2, 100)
Y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(X, Y)
Z = f_func(X, Y)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6, edgecolor='none')

# Plot each point on the surface and its gradient as a 3D arrow.
# We show the gradient (which lives in the x-y plane) as a horizontal
# arrow starting at the point on the surface, scaled for visibility.
scale = 0.3
for (px, py) in points:
    pz = f_func(px, py)
    gx, gy = gradients[(px, py)]

    # point on surface
    ax.scatter(px, py, pz, color='red', s=50)

    # gradient arrow (projected in x-y directions), drawn at surface height
    ax.quiver(px, py, pz,
              gx * scale, gy * scale, 0,
              color='blue', linewidth=2, arrow_length_ratio=0.2)

    ax.text(px, py, pz, f'  ({px},{py})\n  grad=({gx},{gy})', color='black', fontsize=9)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')
ax.set_title('f(x, y) = 2xy + x^2 + y  with gradient vectors')

plt.tight_layout()
plt.savefig('gradient_plot.png', dpi=150)
plt.show()