import numpy as np
import jax.numpy as jnp
import jax

def solve_periodic_spline_derivatives(y):
    """
    Solve for the first derivatives (k) of a periodic cubic spline.
    
    For a periodic spline with points 0..N-1 (and N=0), the continuity equations
    lead to a system A * k = 3 * (y_{i+1} - y_{i-1}) (roughly).
    
    Actually, let's use the standard relation for derivatives k_i at knots:
    k_{i-1} + 4*k_i + k_{i+1} = 3*(y_{i+1} - y_{i-1})
    (assuming unit spacing h=1).
    
    System is:
    [4 1 ... 1] [k_0]   [3(y_1 - y_{N-1})]
    [1 4 1 ...] [k_1] = [3(y_2 - y_0)]
    [...      ] [...]   [...]
    
    Args:
        y: (N,) values
    Returns:
        k: (N,) derivatives dy/dx at knots
    """
    N = len(y)
    # RHS vector
    # y_{i+1} - y_{i-1} with wrapping
    y_next = np.roll(y, -1)
    y_prev = np.roll(y, 1)
    rhs = 3.0 * (y_next - y_prev)
    
    # Matrix A
    # Diagonals are 4
    # Off-diagonals are 1
    # Corners are 1 (periodic)
    
    A = np.zeros((N, N))
    for i in range(N):
        A[i, i] = 4.0
        A[i, (i-1)%N] = 1.0
        A[i, (i+1)%N] = 1.0
        
    # Solve
    k = np.linalg.solve(A, rhs)
    return k

def compute_bicubic_params(grid):
    """
    Compute f, fx, fy, fxy at each grid point.
    
    Args:
        grid: (N, N) array of values
    Returns:
        params: (N, N, 4) array where last dim is [f, fx, fy, fxy]
    """
    N = grid.shape[0]
    f = grid
    
    # 1. fx: Solve along cols (derivative w.r.t row index i)
    fx = np.zeros_like(grid)
    for j in range(N):
        fx[:, j] = solve_periodic_spline_derivatives(grid[:, j])
        
    # 2. fy: Solve along rows (derivative w.r.t col index j)
    fy = np.zeros_like(grid)
    for i in range(N):
        fy[i, :] = solve_periodic_spline_derivatives(grid[i, :])
        
    # 3. fxy: Solve spline on fx along rows (d/dy of df/dx)
    fxy = np.zeros_like(grid)
    for i in range(N):
        fxy[i, :] = solve_periodic_spline_derivatives(fx[i, :])
        
    # Stack
    return np.stack([f, fx, fy, fxy], axis=-1)

def eval_bicubic(params, x, y):
    """
    Evaluate bicubic spline at (x, y) using precomputed params.
    x, y are in grid units [0, N).
    """
    N = params.shape[0]
    
    # Grid indices
    i = int(np.floor(x)) % N
    j = int(np.floor(y)) % N
    
    # Next indices
    i1 = (i + 1) % N
    j1 = (j + 1) % N
    
    # Local coordinates
    u = x - np.floor(x)
    v = y - np.floor(y)
    
    # Get params at 4 corners
    # p00 = params[i, j]
    # p10 = params[i1, j]
    # p01 = params[i, j1]
    # p11 = params[i1, j1]
    
    # We need to construct the 16 coefficients for the cell.
    # Or use the matrix form:
    # f(u,v) = [1 u u^2 u^3] * M * [F] * M^T * [1 v v^2 v^3]^T
    # Where F is the 4x4 matrix of values/derivatives at corners.
    # Actually, standard form uses values and derivatives.
    
    # Let's use the standard Hermite basis form.
    # Vector C = [f(0,0), f(1,0), f(0,1), f(1,1), 
    #             fx(0,0), fx(1,0), fx(0,1), fx(1,1),
    #             fy(0,0), fy(1,0), fy(0,1), fy(1,1),
    #             fxy(0,0), fxy(1,0), fxy(0,1), fxy(1,1)]
    
    # Construct C vector
    p00 = params[i, j]
    p10 = params[i1, j]
    p01 = params[i, j1]
    p11 = params[i1, j1]
    
    # Order: f, fx, fy, fxy
    # We need to map these to the 16 coeffs.
    # A common way is to solve A * x = b where x are coeffs.
    # But A^-1 is constant.
    
    # Matrix A_inv for unit square [0,1]x[0,1]
    # Reference: https://en.wikipedia.org/wiki/Bicubic_interpolation
    
    A_inv = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-3, 3, 0, 0, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, -2, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, -3, 3, 0, 0, -2, -1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 1, 1, 0, 0],
        [-3, 0, 3, 0, 0, 0, 0, 0, -2, 0, -1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, -3, 0, 3, 0, 0, 0, 0, 0, -2, 0, -1, 0],
        [9, -9, -9, 9, 6, 3, -6, -3, 6, -6, 3, -3, 4, 2, 2, 1],
        [-6, 6, 6, -6, -3, -3, 3, 3, -4, 4, -2, 2, -2, -2, -1, -1],
        [2, 0, -2, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 2, 0, -2, 0, 0, 0, 0, 0, 1, 0, 1, 0],
        [-6, 6, 6, -6, -4, -2, 4, 2, -3, 3, -3, 3, -2, -1, -2, -1],
        [4, -4, -4, 4, 2, 2, -2, -2, 2, -2, 2, -2, 1, 1, 1, 1]
    ])
    
    # Vector x (input values)
    # Order: 
    # f(0,0), f(1,0), f(0,1), f(1,1)
    # fx(0,0), fx(1,0), fx(0,1), fx(1,1)
    # fy(0,0), fy(1,0), fy(0,1), fy(1,1)
    # fxy(0,0), fxy(1,0), fxy(0,1), fxy(1,1)
    
    x_vec = np.array([
        p00[0], p10[0], p01[0], p11[0],
        p00[1], p10[1], p01[1], p11[1],
        p00[2], p10[2], p01[2], p11[2],
        p00[3], p10[3], p01[3], p11[3]
    ])
    
    coeffs = A_inv @ x_vec
    
    # Evaluate polynomial
    # sum a_ij * u^i * v^j
    # coeffs order: 
    # a00, a10, a20, a30,
    # a01, a11, a21, a31,
    # ...
    
    val = 0.0
    idx = 0
    for j_pow in range(4):
        for i_pow in range(4):
            val += coeffs[idx] * (u**i_pow) * (v**j_pow)
            idx += 1
            
    return val

# Test
def test_1d():
    print("Testing 1D Spline...")
    N = 10
    x = np.linspace(0, 2*np.pi, N, endpoint=False)
    y = np.sin(x)
    
    # True derivatives: cos(x)
    k_true = np.cos(x)
    
    # Computed derivatives
    k_computed = solve_periodic_spline_derivatives(y)
    
    # Scale: The derivative is dy/d(index).
    # x = index * (2pi/N).
    # dy/dx = cos(x).
    # dy/d(index) = dy/dx * dx/d(index) = cos(x) * (2pi/N).
    
    scale = 2*np.pi/N
    k_scaled = k_computed / scale
    
    print(f"Max Diff 1D Deriv: {np.max(np.abs(k_true - k_scaled)):.6f}")
    
    # Interpolation check
    # Evaluate at x=1.5 (index units)
    # u = 0.5, i=1
    # y(1.5) approx sin(1.5 * 2pi/N)
    
    # Hermite interpolation in 1D
    # p(t) = h00(t)p0 + h10(t)m0 + h01(t)p1 + h11(t)m1
    # where m0, m1 are derivatives dy/dt (so k_computed)
    
    def hermite_1d(p0, p1, m0, m1, t):
        h00 = 2*t**3 - 3*t**2 + 1
        h10 = t**3 - 2*t**2 + t
        h01 = -2*t**3 + 3*t**2
        h11 = t**3 - t**2
        return h00*p0 + h10*m0 + h01*p1 + h11*m1
        
    idx = 1
    t = 0.5
    val = hermite_1d(y[idx], y[idx+1], k_computed[idx], k_computed[idx+1], t)
    
    true_x = (idx + t) * scale
    true_val = np.sin(true_x)
    
    print(f"1D Interp Diff: {abs(val - true_val):.6f}")

def test():
    test_1d()
    
    print("\nTesting 2D Spline...")
    N = 24
    x = np.linspace(0, 2*np.pi, N, endpoint=False)
    y = np.linspace(0, 2*np.pi, N, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Test function
    Z = np.sin(X) * np.cos(Y)
    
    # Compute params
    params = compute_bicubic_params(Z)
    
    # Evaluate at random point
    test_x = 1.5
    test_y = 2.7
    
    # True value
    true_x = test_x * (2*np.pi/N)
    true_y = test_y * (2*np.pi/N)
    true_val = np.sin(true_x) * np.cos(true_y)
    
    # Interp value
    interp_val = eval_bicubic(params, test_x, test_y)
    
    print(f"True: {true_val:.6f}")
    print(f"Interp: {interp_val:.6f}")
    print(f"Diff: {abs(true_val - interp_val):.6f}")
    
    if abs(true_val - interp_val) < 1e-3:
        print("PASS")
    else:
        print("FAIL")

if __name__ == "__main__":
    test()
