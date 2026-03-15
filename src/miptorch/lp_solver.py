from collections import namedtuple
from itertools import count
from dataclasses import dataclass

import torch
from torch import Tensor

# Define tolerances
tol_opt = 1e-5
tol_piv = 1e-5
tol_bound = 1e-5
tol_dual = 1e-6
tol_primal_feas = 1e-4

# number of iterations allowed before a full recompute of the basis inverse is performed
full_update_check_interval = 15

# Config
Scale = namedtuple('Scale', ['r', 's'])

@dataclass
class LPState:
    A_s: torch.Tensor
    b_s: torch.Tensor
    c_s: torch.Tensor
    B_inv: torch.Tensor
    x_B: torch.Tensor
    basics: torch.Tensor
    reduced_costs_full: torch.Tensor | None
    y: torch.Tensor | None
    scale: Scale

def compute_scale(A: torch.Tensor, slack_cols: torch.Tensor = None, omit_col_scale: bool|torch.Tensor = False,
                  eps: float = 1e-12):
    """
    Return r (row scales) and s (column scales) so that
      A_s = (r[:,None] * A) / s[None,:]
    has O(1) rows/cols, and slack columns are identity if slack_cols is given.
    """

    # Sanity check. Make sure if omit_col_scale is specified, it does not include slack columns
    if type(omit_col_scale) is not bool and slack_cols is not None:
        unique_values, counts = torch.unique(torch.cat([omit_col_scale, slack_cols]), return_counts=True)
        if (counts > 1).any():
            raise ValueError("omit_col_scale cannot include slack columns")

    # Row scales: r_i = 1 / max_j |A_ij|
    row_max = A.abs().amax(dim=1)
    r = (row_max.clamp_min(eps)).reciprocal()          # shape (m,)

    A_r = r[:, None] * A
    # Column scales AFTER row scaling: s_j = 1 / max_i |(D_r A)_ij|
    col_max = A_r.abs().amax(dim=0)
    s = (col_max.clamp_min(eps)).reciprocal()          # shape (n,)

    # Omit column scaling (e.g. for integer variables)
    if omit_col_scale is not False:
        if omit_col_scale is True:
            omit_col_scale = ~torch.isin(s, slack_cols)

        s[omit_col_scale] = 1.0

    # Slack fix: for slack of row i at column j, force s_j = r_i
    if slack_cols is not None:
        s = s.clone()
        s[slack_cols] = r

    return r, s


def rescale_lp(A: torch.Tensor, b: torch.Tensor, c: torch.Tensor, r: torch.Tensor, s: torch.Tensor):
    A_s = (r[:, None] * A) / s[None, :]
    b_s = r * b

    # since we're performing row scaling first in the scale computation, all max() col value _reciprocals_
    # are > 1, so we want a divide here to shrink the column scale to 0..1
    c_s = c / s

    return A_s, b_s, c_s


def index_to_mask(indices: torch.Tensor, n: int) -> torch.Tensor:
    mask = torch.zeros(n, dtype=torch.bool, device=indices.device)
    mask[indices] = True
    return mask


def make_nonbasic(basic: torch.Tensor, n: int, *, check=True):
    if check:
        if basic.numel() == 0 or basic.min() < 0 or basic.max() >= n:
            raise ValueError("basic indices out of range")
        # ensure no duplicates (a valid basis needs unique columns)
        if basic.unique().numel() != basic.numel():
            raise ValueError("basic contains duplicate indices")

    # boolean membership mask
    is_basic = index_to_mask(basic, n)

    # complement gives nonbasic indices; sorted ascending by construction
    nonbasic = torch.arange(n, device=basic.device, dtype=torch.long)[~is_basic]
    return nonbasic


def make_nonbasic_mask(basic: torch.Tensor, n: int, *, check=True):
    return index_to_mask(make_nonbasic(basic, n, check=check), n)


def get_reduced_costs(A :torch.Tensor, c_B :torch.Tensor, B_inv :torch.Tensor, c_s :torch.Tensor):
    """
    Calculates reduced costs and the dual vector for a given linear programming problem.

    The function computes the reduced costs of non-basic variables and the vector of simplex
    multipliers (dual variables) using the provided inputs. Reduced costs help in identifying
    entering variables for the simplex algorithm.

    Parameters:
        A (torch.Tensor): The coefficient matrix of the constraints.
        c_B (torch.Tensor): The cost vector corresponding to the basic variables.
        B_inv (torch.Tensor): The inverse of the basis matrix.
        c_s (torch.Tensor): The scaled cost vector.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple where the first element is the reduced
        costs (torch.Tensor) and the second element is the dual vector (torch.Tensor).
    """

    y = c_B @ B_inv
    wc_full = A.t() @ y
    return c_s - wc_full, y


def find_entering_candidate(reduced_costs_full :torch.Tensor, basic :torch.Tensor, minimize :bool):
    """
    Pick an entering column from full reduced costs.
    - For minimization (min=True): choose most negative rc (< -tol).
    - For maximization (min=False): choose largest positive rc (> tol).
    Returns the column index (int) or None if no improving candidate exists.
    """
    rc = reduced_costs_full
    n = rc.numel()

    # mark basics
    is_basic = basic_mask(basic, n)

    # use only nonbasic entries to set scale; fall back to 0 if none
    nb_mask = ~is_basic
    if not nb_mask.any():
        return None

    # relative tolerance (fp32-friendly)
    scale = rc[nb_mask].abs().max()
    tol = tol_opt * (1.0 + scale)  # tensor scalar on same device

    # candidate mask for nonbasics that pass a tolerance test
    if minimize:
        # if the reduced cost is negative, we want to enter the column as it will improve the min objective
        cand_mask = nb_mask & (rc < -tol)
        op = torch.argmin
    else:
        # if the reduced cost is positive, we want to enter the column as it will improve the max objective
        cand_mask = nb_mask & (rc > tol)
        op = torch.argmax

    if not cand_mask.any():
        # we are optimal
        return None

    cand_idx = torch.nonzero(cand_mask, as_tuple=False).squeeze(1)
    j_local = op(rc[cand_idx])
    j = int(cand_idx[j_local].item())

    return j


def basic_mask(basic: Tensor, n: int) -> Tensor:
    is_basic = torch.zeros(n, dtype=torch.bool, device=basic.device)
    if basic.numel() > 0:
        is_basic[basic] = True
    return is_basic


def package_solution(c :torch.Tensor, state :LPState):
    """
    Packages the solution from the simplex method into the standard form for further analysis or processing.

    Args:
        n (int): The total number of variables in the problem.
        basic (torch.Tensor): A tensor indicating the indices of basic variables.
        x_B (torch.Tensor): Solution values corresponding to the basic variables.
        c (torch.Tensor): Cost coefficients for all variables.
        s (torch.Tensor): Variable scaling factors.

    Returns:
        tuple: A tuple containing:
            - x_out (torch.Tensor): The solution vector with values for all variables.
            - c_out (torch.Tensor): The cost contribution vector for each variable.
            - obj (float): The total objective function value at the solution.
    """

    # init with zeros, since all nonbasics are 0
    x_out = torch.zeros(state.A_s.shape[1], dtype=state.A_s.dtype, device=state.A_s.device)
    x_out[state.basics] = state.x_B
    x_out = x_out / state.scale.s

    # extend c with zeros for any extra columns (slacks/cut slacks/etc.)
    n = state.A_s.shape[1]
    if c.numel() < n:
        c_eff = torch.cat([c, torch.zeros(n - c.numel(), dtype=c.dtype, device=c.device)])
    else:
        c_eff = c[:n]

    #x_out = torch.where(x_out.abs() <= tol_bound, torch.zeros((), dtype=c.dtype, device=c.device), x_out)
    #x_out = torch.clamp_min(x_out, 0.0)

    c_out = x_out * c_eff
    obj = c_out.sum()

    return x_out, c_out, obj, state


def find_leaving_basic(x_B :torch.Tensor, u :torch.Tensor):
    """
    Finds the index of the exiting basic variable based on the minimum ratio test.

    Args:
        x_B: Solution values corresponding to the basic variables.
        u: Upper bounds for the basic variables.

    Returns:
        tuple: A tuple containing:
            - t (float): The value of minimum ratio
            - l (int): The index of the basic variable with the minimum ratio
    """

    t = torch.full_like(x_B, float('inf'))
    pos = u > tol_piv
    t[pos] = (x_B[pos] + tol_bound) / u[pos]
    l = torch.argmin(t)
    return t[l], l


def get_objective_parts(c_s: torch.Tensor, basic: torch.Tensor, nonbasic: torch.Tensor) \
        -> tuple[torch.Tensor, torch.Tensor]:
    c_B = c_s[basic]
    c_N = c_s[nonbasic]
    return c_B, c_N


def sm_pivot(B_inv :torch.Tensor, u :torch.Tensor, l :int):
    """
    Performs a Sherman-Morrison update on the inverse of the basis matrix.

    Args:
        B_inv: Inverse of the basis matrix.
        u: Upper bounds for the basic variables.
        l: Index of the basic variable with the minimum ratio.

    Returns:
        torch.Tensor: Updated inverse of the basis matrix.
    """

    p = u[l]
    row_old = B_inv[l, :].clone()
    row_new = row_old / p # B^-1_l / u_l
    B_inv -= u[:, None] * row_new[None, :]  # u * B^-1_l / u_l
    B_inv[l, :] = row_new  # + e_l(B^-1_l / u_l)  since e_l is identity

    return B_inv


def full_recompute_inverse(A, b, basic):
    """
    Performs a full recomputation of the inverse matrix, basic variables, dual prices,
    and reduced costs in a simplex-like optimization algorithm. Ensures numerical
    stability by utilizing direct linear solving instead of matrix inversion.

    Parameters:
        A (torch.Tensor): The coefficient matrix of the linear program with shape (m, n).
        b (torch.Tensor): The right-hand side vector of the constraints with shape (m,).
        basic (torch.Tensor): Indices of the currently chosen basic variables with shape (m,).

    Returns:
        tuple: Tuple containing the following:
            B_inv (torch.Tensor): The inverse of the current basis matrix with shape (m, m).
            x_B (torch.Tensor): Values of the basic variables with shape (m,).
    """
    m, n = A.shape

    if basic.numel() != m:
        raise RuntimeError("basics length mismatch")
    if (basic < 0).any() or (basic >= n).any():
        raise RuntimeError("basics out of range")
    if basic.unique().numel() != m:
        raise RuntimeError("duplicate basics -> singular basis")

    # Gather basis
    B = A.index_select(1, basic)                    # (m, m)
    m = B.size(0)
    I = torch.eye(m, device=A.device, dtype=A.dtype)

    # Prefer solve-over-inv for stability
    B_inv = torch.linalg.solve(B, I)                # columns of B^{-1}

    # Recompute basics, duals, reduced costs
    x_B = B_inv @ b                                # (m,)

    return B_inv, x_B


def primal_pivot(state :LPState, reduced_costs_full :torch.Tensor, basic :torch.Tensor, minimize :bool):
    """
    Performs a primal pivot operation in the simplex algorithm for linear programming. This function updates the state
    of the linear program by determining and applying an entering and exiting variable, and resolving potential
    unbounded behavior in the feasible region.

    Parameters:
        state (LPState): The current state of the linear program.
        reduced_costs_full (torch.Tensor): The full reduced costs vector.
        basic (torch.Tensor): Basic variable indices.
        minimize (bool): Indicates whether the objective is to minimize or maximize.

    Returns:
        Tuple: The updated state of the linear program after the pivot operation.
        l (int): The index of the basic variable that will leave the basis.
        bool: A flag indicating if the problem is optimal (True).

    Raises:
        ValueError: If the feasible region is unbounded in the direction of the entering column.
    """

    ec_j = find_entering_candidate(reduced_costs_full, basic, minimize)

    if ec_j is None:
        return state, True, None, None, None

    # we have our entering candidate, now we need to find our exiting basic l
    u = state.B_inv @ state.A_s[:, ec_j]
    if not (u > tol_piv).any():
        raise ValueError("Unbounded in direction of entering column")

    t, l = find_leaving_basic(state.x_B, u)

    # take the step and update our state. move to the boundary point (all basics updated; row l hits 0)
    state.x_B = state.x_B - (u * t)
    # walk the entering column into x_B
    state.x_B[l] = t

    return state, False, l, ec_j, u


def find_dual_leaving_basic(x_B: torch.Tensor):
    # remember we want to get back to primal feasibility this means x >= 0
    # so we find the most negative x_B value to exit the corresponding basic

    masked = x_B.masked_fill(x_B >= -tol_primal_feas, float('inf'))
    if not torch.isfinite(masked).any():
        # no primal infeasibilities
        return None
    return torch.argmin(masked)


def find_dual_entering_basic(l_i :torch.Tensor, A_s :torch.Tensor, reduced_costs_full :torch.Tensor,
                             B_inv :torch.Tensor, x_B :torch.Tensor, basic :torch.Tensor, primal_minimize :bool):
    """
    Finds the entering variable for the dual simplex algorithm.

    This function calculates the entering variable (non-basic variable) for the
    dual simplex algorithm based on the current tableau information. It identifies
    the most eligible column to enter the basis for the optimization problem.

    Args:
        l_i (torch.Tensor): A scalar index of the leaving basic variable row.
        A_s (torch.Tensor): The constraint matrix of the optimization problem.
        reduced_costs_full (torch.Tensor): The reduced costs vector (c - A^T y).
            Must have the same shape as the number of columns in `A_s`.
        B_inv (torch.Tensor): The current inverse of the basis matrix B.
        basic (torch.Tensor): The indices of the currently basic variables.
        primal_minimize (bool): A flag indicating if the problem is in primal
            minimization form. If True, the dual max problem is solved.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The index of the entering variable
        and the associated step length.

    Raises:
        RuntimeError: If no eligible column is found to enter the basis, indicating
            that the problem is infeasible or unbounded with respect to the current
            row.
    """

    a_s_row = B_inv[l_i, :] @ A_s  # (1, m) @ (m, n) -> (n,)
    t = torch.full_like(a_s_row, float('inf'))

    # reduced_costs_full must already be rc = c - A^T y (shape (n,))
    if primal_minimize:
        # dual max, A^T y <= c -> rc >= 0 is feasible slack
        slack = reduced_costs_full
    else:
        # dual min, A^T y >= c -> rc <= 0, so feasible slack is -rc
        slack = -reduced_costs_full

    is_basic = basic_mask(basic, A_s.shape[1])

    num = torch.clamp(slack, min=0.0)
    eligible = (~is_basic) & (a_s_row < -tol_piv) & (slack >= -tol_dual)
    t[eligible] = num[eligible] / (-a_s_row[eligible])
    j_entering = torch.argmin(t)

    if not torch.isfinite(t[j_entering]):
        # cand = (~is_basic) & (a_s_row < -tol_piv)
        # idx = torch.nonzero(cand, as_tuple=True)[0]
        # print(torch.max(A_s))
        # print("a_s_row[cand] =", a_s_row[idx])
        # print("slack[cand]   =", slack[idx])
        # print("min slack cand:", slack[idx].min().item() if idx.numel() else None)

        raise RuntimeError("Dual simplex: no eligible entering column (infeasible/unbounded wrt this row).")

    return j_entering, t[j_entering]


def dual_pivot(state :LPState, reduced_costs_full :torch.Tensor, basics :torch.Tensor, minimize :bool):
    l_i = find_dual_leaving_basic(state.x_B)

    if l_i is None:
        return state, True, None, None, None

    # we need to find the entering basic based on our current reduced costs
    j_entering, t_dual = find_dual_entering_basic(l_i, state.A_s, reduced_costs_full, state.B_inv, state.x_B, basics,
                                                  minimize)

    # j_entering is the column of the entering var. we need to compute the tableau value of the entering
    # var (column) for each constraint
    tableau_new_basic_coeff = state.B_inv @ state.A_s[:, j_entering]

    # since we are entering basic j, we need to compute the ratio of basic j in the leaving row
    # to the current value in the leaving row. This will leave the coefficient of our
    # new basic at 1 for row l_i
    t = state.x_B[l_i] / tableau_new_basic_coeff[l_i]  # scalar

    # take the step and update our state. move to the boundary point (all basics updated; row l hits 0)
    state.x_B = state.x_B - (tableau_new_basic_coeff * t)

    # walk the entering column into x_B
    state.x_B[l_i] = t

    return state, False, l_i, j_entering, tableau_new_basic_coeff


def solve_for_state(c :torch.Tensor, state :LPState, minimize :bool = True,
                    return_state :bool = False, dual_simplex :bool = False, last_it :int = 0):
    m, n = state.A_s.shape

    nonbasic = make_nonbasic(state.basics, n)

    # current objective values
    c_B, c_N = get_objective_parts(state.c_s, state.basics, nonbasic)

    for opt_it in count(last_it):
        state.reduced_costs_full, state.y = get_reduced_costs(state.A_s, c_B, state.B_inv, state.c_s)

        if dual_simplex:
            state, is_optimal, l, ec_j, u = dual_pivot(state, state.reduced_costs_full, state.basics, minimize)
        else:
            state, is_optimal, l, ec_j, u = primal_pivot(state, state.reduced_costs_full, state.basics, minimize)

        if is_optimal:
            if return_state:
                return state, opt_it
            else:
                return package_solution(c, state)

        # update basic and nonbasic indexes
        leaving_col = state.basics[l].item()
        state.basics[l] = ec_j
        nonbasic[nonbasic == ec_j] = leaving_col

        # update the objective
        c_B, c_N = get_objective_parts(state.c_s, state.basics, nonbasic)

        # finally, update the inverse
        # Always try SM update
        state.B_inv = sm_pivot(state.B_inv, u, l)

        # periodic sanity check
        if opt_it % full_update_check_interval == 0:
            xB_check = state.B_inv @ state.b_s
            if (xB_check - state.x_B).abs().max() > 1e-3:
                state.B_inv, state.x_B = full_recompute_inverse(state.A_s, state.b_s, state.basics)


def prepare_state(A :torch.Tensor, b :torch.Tensor, c :torch.Tensor, initial_basics :torch.Tensor, scale :Scale = None,
          minimize :bool = True, omit_col_scale :bool|torch.Tensor = False):

    m, n = A.shape
    dev = A.device

    if not scale:
        # scale the LP so we can use fp32
        r, s = compute_scale(A, initial_basics, omit_col_scale)
        scale = Scale(r, s)
        A_s, b_s, c_s = rescale_lp(A, b, c, r, s)

    else:
        A_s, b_s, c_s = A, b, c

    # To be compact, rather than the tableau, we will store:
    # ---------------------------------------------------------
    # The basic and nonbasic variable indexes
    # remember that basic[i] is the index for the basic variable for row i
    basic = initial_basics

    # The basis inverse matrix.
    # This is an identity since we're starting with a basis as the slacks
    B_inv = torch.eye(m, dtype=A.dtype, device=dev)

    # Our current basic values
    # Since our inverse is identity, these are the same as the RHS
    x_B = b_s

    state = LPState(A_s, b_s, c_s, B_inv, x_B, basic, None, None, scale)

    return state


def solve(A :torch.Tensor, b :torch.Tensor, c :torch.Tensor, initial_basics :torch.Tensor, scale :Scale = None,
          minimize :bool = True, return_state :bool = False, omit_col_scale :bool|torch.Tensor = False):
    """
    Solves a linear program. Assumes that the slacks are already added and are
    the initial basic variables.

    Args:
        A: Constraint matrix with slacks
        b: Constraint RHS
        c: Objective function coefficients with slacks
        initial_basics: The indexes from A belonging to the initial basic variables
        scale: None or Scale object representing the scaling factors for the LP. If None, the LP will be automatically
            rescaled.
        minimize: Whether to minimize or maximize the objective function.
        return_state: Whether to return the state of the simplex algorithm at the end of the computation or the result.
        omit_col_scale: Whether to omit column scaling. If True, column scaling will be entirely omitted.
            If a tensor, columns at these indices will be omitted.
    Returns:
        An optimal solution
    """
    state = prepare_state(A, b, c, initial_basics, scale, minimize, omit_col_scale)
    return solve_for_state(c, state, minimize, return_state, dual_simplex=False)


def check_solution(A, b, x, *, eps_primal=1e-6, eps_primal_abs=1e-8, int_cols=None, eps_int=1e-4):
    x = x[:A.shape[1]]
    r = A @ x - b

    # scaled/relative row-wise tolerance
    row_scale = A.abs().max(dim=1).values * x.abs().max().clamp_min(1.0) + b.abs().clamp_min(1.0)
    rel = (r.abs() / row_scale).max()

    primal_feasible = (rel <= eps_primal) or (r.abs().max() <= eps_primal_abs)

    # nonnegativity: allow tiny negatives
    primal_feasible = primal_feasible and (x.min() >= -1e-7)

    # integrality
    int_feasible = True
    int_residual = None
    if int_cols is not None:
        x_int = x[int_cols]
        int_residual = (x_int - x_int.round()).abs()
        int_feasible = torch.isfinite(int_residual).all() and (int_residual.max() <= eps_int if int_residual.numel() else True)

    return primal_feasible, r, int_feasible, int_residual, rel
