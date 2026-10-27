import numpy as np
from scipy.optimize import linprog

class GameSolution:
    def __init__(self, value, x, y, formulation="ferguson"):
        self.value = float(value)
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.formulation = formulation

def solve_matrix_game(A):
    """
    Solves a general m x n zero-sum matrix game using Linear Programming (SciPy HiGHS).
    Supports all dimensions (2x2, 2x3, 3x2, 4x4, 6x6, etc.).
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or min(A.shape) == 0:
        raise ValueError("Matrix A must be a non-empty 2D array (m x n).")

    m, n = A.shape

    # 1. Shift matrix elements to guarantee strict positivity (B > 0)
    min_val = np.min(A)
    shift = 0.0
    if min_val <= 0:
        shift = abs(min_val) + 1.0
    B = A + shift

    # 2. Primal Problem for Player 1 (Maximizer)
    # Minimize sum(p_i) subject to -B^T * p <= -1, p >= 0
    c_p = np.ones(m)
    A_ub_p = -B.T
    b_p = -np.ones(n)
    res_p = linprog(c_p, A_ub=A_ub_p, b_ub=b_p, bounds=[(0, None)] * m, method='highs')

    # 3. Dual Problem for Player 2 (Minimizer)
    # Maximize sum(q_j) ==> Minimize -sum(q_j) subject to B * q <= 1, q >= 0
    c_q = -np.ones(n)
    A_ub_q = B
    b_q = np.ones(m)
    res_q = linprog(c_q, A_ub=A_ub_q, b_ub=b_q, bounds=[(0, None)] * n, method='highs')

    if not res_p.success or not res_q.success:
        raise RuntimeError("Linear Programming solver failed to converge.")

    # 4. Extract Game Value v and Strategy Vectors x*, y*
    v_B = 1.0 / res_p.fun
    x_opt = res_p.x * v_B
    y_opt = res_q.x * v_B
    game_value = v_B - shift

    # Normalize probability vectors to ensure exact sum = 1
    x_opt = np.clip(x_opt, 0, None)
    y_opt = np.clip(y_opt, 0, None)
    if x_opt.sum() > 0: x_opt = x_opt / x_opt.sum()
    if y_opt.sum() > 0: y_opt = y_opt / y_opt.sum()

    return GameSolution(game_value, x_opt, y_opt)
