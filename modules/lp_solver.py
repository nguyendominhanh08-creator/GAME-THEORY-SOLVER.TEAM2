import numpy as np
from scipy.optimize import linprog

class LPSolution:
    def __init__(self, x, y, value):
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.value = float(value)

def solve_matrix_game(A):
    """
    Solves a general m x n zero-sum matrix game using Linear Programming (SciPy HiGHS).
    """
    A = np.asarray(A, dtype=float)
    m, n = A.shape

    # 1. Shift matrix elements to guarantee strict positivity (A' > 0)
    min_val = np.min(A)
    shift = 0.0
    if min_val <= 0:
        shift = abs(min_val) + 1.0
    A_shifted = A + shift

    # 2. Primal problem for Player 1 (Maximize v' -> Minimize sum(p_i))
    # Constraints: A_shifted^T * p >= 1  ==>  -A_shifted^T * p <= -1
    c_p = np.ones(m)
    A_ub_p = -A_shifted.T
    b_p = -np.ones(n)
    
    res_p = linprog(c_p, A_ub=A_ub_p, b_ub=b_p, bounds=(0, None), method='highs')

    # 3. Dual problem for Player 2 (Minimize v' -> Maximize sum(q_j))
    # Constraints: A_shifted * q <= 1
    c_q = -np.ones(n)
    A_ub_q = A_shifted
    b_q = np.ones(m)
    
    res_q = linprog(c_q, A_ub=A_ub_q, b_ub=b_q, bounds=(0, None), method='highs')

    if not res_p.success or not res_q.success:
        raise RuntimeError("Linear Programming solver failed to converge.")

    # 4. Recover Game Value v and Probability Vectors x*, y*
    v_shifted = 1.0 / res_p.fun
    x_opt = res_p.x * v_shifted
    y_opt = res_q.x * v_shifted
    game_value = v_shifted - shift

    # Normalize vectors to ensure sum(x) = 1, sum(y) = 1
    x_opt = np.clip(x_opt, 0, None)
    y_opt = np.clip(y_opt, 0, None)
    if x_opt.sum() > 0: x_opt /= x_opt.sum()
    if y_opt.sum() > 0: y_opt /= y_opt.sum()

    return LPSolution(x_opt, y_opt, game_value)
