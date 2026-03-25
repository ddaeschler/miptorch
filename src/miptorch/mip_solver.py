import torch
from . import lp_solver
import math
from dataclasses import dataclass
from typing import List, Tuple

from itertools import count
import heapq


tol_int = 1e-5
tol_zero = 1e-5
tol_cont_abs = 1e-7
tol_viol = 1e-5
tol_opt = 1e-5

max_cut_iters = 20
max_candidate_cuts = 20

# (col_index, sense, value)
# sense: "ub" for x_j <= value
#        "lb" for x_j >= value   (implemented as -x_j <= -value)
Bound = Tuple[int, str, float]

@dataclass
class Node:
    lb: torch.Tensor  # shape (n_base,), float
    ub: torch.Tensor  # shape (n_base,), float

def append_bounds_scaled_batch(A_s: torch.Tensor, b_s: torch.Tensor, c_s: torch.Tensor,
                               basics: torch.Tensor, B_inv: torch.Tensor, x_B: torch.Tensor,
                               scale: lp_solver.Scale,
                               node,  # Node with .lb and .ub tensors
                               n_base: int,
                               inf: float = 1e30):
    """
    Apply Node(lb, ub) bounds in one batch:
      - for each finite ub[j], add  x_j <= ub[j]
      - for each finite lb[j], add  x_j >= lb[j]  (as -x_j <= -lb[j])

    n_base: number of original columns (only apply bounds to 0..n_base-1)
    """
    dev = A_s.device
    dtA = A_s.dtype
    dtb = b_s.dtype

    lb = node.lb[:n_base]
    ub = node.ub[:n_base]

    # infeasible node early
    if torch.any(lb > ub):
        return None  # caller should prune

    lb_idx = torch.nonzero(torch.isfinite(lb) & (lb > -inf), as_tuple=True)[0]
    ub_idx = torch.nonzero(torch.isfinite(ub) & (ub <  inf), as_tuple=True)[0]

    k = lb_idx.numel() + ub_idx.numel()
    if k == 0:
        return A_s, b_s, c_s, basics, B_inv, x_B, scale

    m, n = A_s.shape

    rows = torch.zeros((k, n), dtype=dtA, device=dev)
    rhs = torch.empty((k,), dtype=dtb, device=dev)

    # Lower bounds: x_j >= L  =>  -x_s,j <= -(L*s_j)
    t = 0
    if lb_idx.numel():
        s_j = scale.s[lb_idx]
        v = lb[lb_idx].to(dtb) * s_j
        rows[t:t+lb_idx.numel(), lb_idx] = -1.0
        rhs[t:t+lb_idx.numel()] = -v
        t += lb_idx.numel()

    # Upper bounds: x_j <= U  =>  x_s,j <= U*s_j
    if ub_idx.numel():
        s_j = scale.s[ub_idx]
        v = ub[ub_idx].to(dtb) * s_j
        rows[t:t+ub_idx.numel(), ub_idx] = 1.0
        rhs[t:t+ub_idx.numel()] = v

    # append k slack columns to existing rows (m x k zeros)
    A_s_ext = torch.cat([A_s, torch.zeros((m, k), dtype=dtA, device=dev)], dim=1)  # (m, n+k)

    # new rows with slack identity (k x (n+k))
    new_rows = torch.cat([rows, torch.eye(k, dtype=dtA, device=dev)], dim=1)       # (k, n+k)

    A_s_new = torch.cat([A_s_ext, new_rows], dim=0)                                 # (m+k, n+k)
    b_s_new = torch.cat([b_s, rhs], dim=0)
    c_s_new = torch.cat([c_s, torch.zeros((k,), dtype=c_s.dtype, device=dev)], dim=0)

    # scales: slack cols scale=1, new rows scale=1
    s_new = torch.cat([scale.s, torch.ones((k,), dtype=scale.s.dtype, device=dev)], dim=0)
    r_new = torch.cat([scale.r, torch.ones((k,), dtype=scale.r.dtype, device=dev)], dim=0)
    scale_new = lp_solver.Scale(r_new, s_new)

    # basics: new slacks are basic for the new rows
    new_slack_cols = torch.arange(n, n + k, dtype=torch.long, device=dev)
    basics_new = torch.cat([basics, new_slack_cols], dim=0)

    # B_inv: block-diagonal extend with I_k
    B_inv_new = torch.zeros((m + k, m + k), dtype=B_inv.dtype, device=dev)
    B_inv_new[:m, :m] = B_inv
    B_inv_new[m:, m:] = torch.eye(k, dtype=B_inv.dtype, device=dev)

    # x_B: append new slack basic values = rhs
    x_B_new = torch.cat([x_B, rhs], dim=0)

    return A_s_new, b_s_new, c_s_new, basics_new, B_inv_new, x_B_new, scale_new



def choose_branch_var_from_x(
    x_full: torch.Tensor,
    int_dec_cols_idx: torch.Tensor,
    node,
    tol_fix: float = 1e-9,
):
    """
    Pick an integer decision column to branch on.
    Returns a column index j (in full space) or None if integer-feasible (w.r.t tol_int)
    or all candidates are fixed by bounds.
    """
    dev = x_full.device

    cols = int_dec_cols_idx
    x = x_full[cols]

    lb = node.lb[cols]
    ub = node.ub[cols]

    # clamp x to bounds to avoid tiny numerical bound violations affecting frac
    x = torch.minimum(torch.maximum(x, lb), ub)

    # ignore fixed vars
    not_fixed = (ub - lb) > tol_fix

    # fractional distance to nearest integer
    frac = torch.abs(x - torch.round(x))

    # eligible vars are fractional enough AND not fixed
    eligible = not_fixed & (frac > tol_int)

    if not torch.any(eligible):
        return None

    # choose the most fractional eligible var
    frac_masked = torch.where(eligible, frac, torch.tensor(-1.0, device=dev))
    k = torch.argmax(frac_masked).item()
    return int(cols[k].item())



def is_integer_like(x :torch.Tensor):
    dist = (x - x.round()).abs()
    int_rows = torch.nonzero(torch.isfinite(dist) & (dist <= tol_int)).squeeze(-1)
    return dist, int_rows


def find_top_gmi_cuts(dist: torch.Tensor, k: int = max_candidate_cuts):
    return torch.topk(dist, min(k, math.ceil(dist.numel()*0.2)), largest=True, sorted=False)


def frac01(x: torch.Tensor) -> torch.Tensor:
    return x - torch.floor(x)  # in [0,1)


def create_gmi_cuts(target_tableau_rows: torch.Tensor, x_B_cuts: torch.Tensor, int_cols: bool|torch.Tensor,
                    basic: torch.Tensor, n: int, x_B_full: torch.Tensor):
    """
    Creates Gomory Mixed-Integer (GMI) cuts from a given tableau of rows and fractional basic variables.

    The function computes GMI cuts for both integer and continuous variables, based on their contributions in the
    tableau. The contributions are determined by their fractional and residual coefficients with respect to the
    basic fractional parts. For integer variables, it calculates fractional residuals directly, and for continuous
    variables, it evaluates contributions by considering values greater or smaller than predefined tolerances.

    Parameters:
        tableau_rows: torch.Tensor
            The coefficient tableau containing the rows corresponding to the constraints in the optimization
            problem.
        x_B_cuts: torch.Tensor
            The basic variables vector from which fractional parts are derived, used to create cuts.
        int_cols: bool | torch.Tensor
            Boolean or tensor indicating the columns corresponding to integer variables in the tableau.

    Returns:
        torch.Tensor
            The computed GMI cuts as a tensor of the same shape as the input coefficient tableau.
    """
    # get the fractional part of each basic variable
    f_0 = frac01(x_B_cuts)

    # reshape f0 for broadcasting (k_good, 1)
    f_0 = frac01(x_B_cuts).unsqueeze(1)

    # we only use non-basics to form the cut
    nb_mask = lp_solver.make_nonbasic_mask(basic, n)

    # set up for integer and continuous variables
    int_mask = lp_solver.index_to_mask(int_cols, n)
    cont_mask = ~int_mask
    basic_is_int = int_mask[basic]

    int_and_nb_mask = int_mask & nb_mask
    cont_and_nb_mask = cont_mask & nb_mask

    cut = torch.zeros_like(target_tableau_rows)

    # work with the integer variables first
    if int_and_nb_mask.any():
        f_j_int = frac01(target_tableau_rows[:, int_and_nb_mask])

        int_coeffs = torch.zeros_like(f_j_int)

        frac_lt_mask = f_j_int <= f_0
        frac_gt_mask = f_j_int > f_0

        int_coeffs[frac_lt_mask] = (f_j_int / f_0)[frac_lt_mask]
        int_coeffs[frac_gt_mask] = ((1 - f_j_int) / (1 - f_0))[frac_gt_mask]

        cut[:, int_and_nb_mask] = int_coeffs

    #now continuous variables
    if cont_and_nb_mask.any():
        cont_nb = target_tableau_rows[:, cont_and_nb_mask]
        cont_lt_mask = cont_nb < -tol_zero
        cont_gt_mask = cont_nb > tol_zero

        cont_coeffs = torch.zeros_like(cont_nb)

        cont_coeffs[cont_lt_mask] = (-cont_nb / (1 - f_0))[cont_lt_mask]
        cont_coeffs[cont_gt_mask] = (cont_nb / f_0)[cont_gt_mask]

        cut[:, cont_and_nb_mask] = cont_coeffs

    # flip the sign of the cuts for <= constraints
    cut *= -1.0

    cut = trim_cuts_by_violation(cut, x_B_full, basic)

    return cut


def trim_cuts_by_violation(cut: torch.Tensor, x_B_full: torch.Tensor, basic :torch.Tensor):
    """
    Filters GMI cuts based on their violation of the current solution.

    This function evaluates which cuts are significantly violated by the current fractional solution
    and returns only those cuts that exceed a violation threshold. Cuts with insufficient violation
    are not useful for tightening the feasible region.

    Parameters:
        cut: torch.Tensor
            The GMI cut coefficients computed from the tableau rows.
        x_B_full: torch.Tensor
            The basic variables vector for the current fractional solution.

    Returns:
        torch.Tensor
            A filtered tensor containing only the cuts that have sufficient violation.
    """
    # Assumes the cut is already in <= form: cut @ x <= -1.

    # assert cut.shape[1] == x_B_full.numel()

    x_full = torch.zeros(cut.shape[1], device=cut.device, dtype=x_B_full.dtype)
    x_full[basic] = x_B_full

    rhs = -torch.ones((cut.shape[0], 1), dtype=cut.dtype, device=cut.device)  # (k,1)

    # If you have x as (n,) -> make it (n,1)
    x_col = x_full.unsqueeze(1)  # (n,1)

    lhs = cut @ x_col  # (k,1)
    viol = (lhs - rhs).squeeze(1)  # (k,)

    # scale-aware violation (prevents big-coefficient rows from dominating)
    row_scale = cut.abs().amax(dim=1).clamp_min(1.0)  # (k,)
    scaled_viol = viol / row_scale

    # Keep only meaningfully violated cuts
    keep = scaled_viol > tol_viol

    # Option 1 (recommended): filter cuts here
    cut = cut[keep]

    return cut

def solution_is_integer(x :torch.Tensor, int_cols :torch.Tensor):
    xI = x[int_cols]  # int_cols is index list of integer decision columns
    int_ok = (xI - xI.round()).abs().max() <= tol_int
    return int_ok


def do_cut_loop(c :torch.Tensor, dec_cols :torch.Tensor, int_cols: bool|torch.Tensor, minimize :bool,
                state :lp_solver.LPState, last_it :int):
    """
    Executes the main loop for solving a linear programming problem with integer-constrained
    variables using cut-and-branch techniques.

    This function repeatedly applies Gomory Mixed-Integer (GMI) cuts to enforce integrality on
    integer-constrained variables. If no cuts can be added, it transitions to a branching step
    to address fractional solutions.

    Parameters:
        dec_cols (torch.Tensor): The decision columns for the LP
        int_cols (bool | torch.Tensor): Specifies which columns correspond to integer-constrained
            variables. If True, all columns are integer-constrained. If False, no columns are
            constrained. If a tensor of column indices, only those indices are constrained.
        minimize (bool): Whether the objective function is to be minimized. True for minimization,
            False for maximization.
        state (lp_solver.LPState): The current state of the linear programming solver.
        last_it (int): The last iteration number used by the solver for bookkeeping.

    Returns:
        lp_solver.LPState | Tuple[lp_solver.LPState, bool]: If the solution is integer-feasible or
            if integer constraints do not apply, returns a packaged solution state directly.
            Otherwise, returns a tuple where the first element is the solver state, and the
            second element indicates whether branching is required.

    Raises:
        No specific exceptions raised. Refer to external function dependencies for details on
        possible errors.
    """
    dev = state.A_s.device

    for cut_it in count(0):
        # only require integrality for rows whose basic variable corresponds to
        # an integer-constrained column
        if int_cols is True:
            integer_basics = torch.arange(state.x_B.numel(), device=dev)
        elif int_cols is False:
            integer_basics = torch.tensor([], device=dev)
        else:
            integer_basics = torch.nonzero(torch.isin(state.basics, int_cols), as_tuple=True)[0]

        # check if our solution is already integer
        x_out, c_out, obj, state = lp_solver.package_solution(c, state)

        if solution_is_integer(x_out, int_cols):
            # integer (on integer-constrained columns)
            return (x_out, c_out, obj, state), True
        elif cut_it == max_cut_iters:
            return state, False
        else:
            dist, int_rows = is_integer_like(state.x_B)

            # what we got back was fractional. we need to apply some cuts
            cut_state, cuts_added = add_gmi_cuts(dist, int_cols, int_rows, state)

            if cuts_added:
                state, last_it = lp_solver.solve_for_state(c, cut_state, minimize=minimize, return_state=True,
                                              dual_simplex=True, last_it=last_it)
            else:
                # no cuts were added, we need to branch
                return state, False


def solve(A: torch.Tensor, b: torch.Tensor, c: torch.Tensor, initial_basics: torch.Tensor,
          int_cols: bool | torch.Tensor = True, minimize: bool = True, meta: dict = {},
          branch_groups: list[torch.Tensor] | None = None):

    if int_cols is False:
        raise ValueError("MIP requested without integer constraints")

    dev = A.device

    basics0 = initial_basics.clone()
    dec_cols_idx = lp_solver.make_nonbasic(initial_basics, A.shape[1])

    # integer decision columns to branch on:
    if int_cols is True:
        int_cols_idx = torch.arange(A.shape[1], device=dev, dtype=torch.long)
    else:
        int_cols_idx = int_cols.clone().to(device=dev)

    # Normalize/validate branch_groups (optional)
    if branch_groups is not None:
        norm_groups: list[torch.Tensor] = []
        for g in branch_groups:
            if g is None:
                continue
            gg = g.to(device=dev, dtype=torch.long)
            if gg.numel() == 0:
                continue
            # Keep only indices that are actually integer-constrained
            # (prevents accidental branching on slacks/cuts)
            mask = torch.isin(gg, int_cols_idx)
            gg = gg[mask]
            if gg.numel() > 0:
                norm_groups.append(gg)
        branch_groups = norm_groups if norm_groups else None

    # best incumbent
    best_sol = None
    best_obj = float("inf") if minimize else -float("inf")

    state0 = lp_solver.prepare_state(A, b, c, initial_basics, minimize=minimize, omit_col_scale=int_cols_idx)

    A_s0, b_s0, c_s0 = state0.A_s, state0.b_s, state0.c_s
    scale0 = state0.scale
    m0, n0 = A_s0.shape

    n_base = A_s0.shape[1]  # base column count
    root = Node(
        lb=torch.full((n_base,), -float("inf"), device=dev),
        ub=torch.full((n_base,), float("inf"), device=dev),
    )

    heap = []
    push_id = 0  # tie-breaker

    def heap_key(obj_lp_val: float) -> float:
        # heap pops smallest key
        return obj_lp_val if minimize else -obj_lp_val

    # root priority: best possible so it’s processed first
    root_key = heap_key(-float("inf") if minimize else float("inf"))
    heapq.heappush(heap, (root_key, push_id, root))
    push_id += 1

    node_cap = 2000
    for _ in range(node_cap):
        if not heap:
            break

        _, _, node = heapq.heappop(heap)

        # start node from base scaled model (no rescaling)
        A_s_n = A_s0
        b_s_n = b_s0
        c_s_n = c_s0
        basics_n = basics0.clone()
        scale_n = scale0

        # fresh basis inverse for this node (slack-start)
        B_inv_n = torch.eye(m0, dtype=A.dtype, device=dev)
        x_B_n = b_s0.clone()

        # apply bounds in SCALED space (each adds one row + one slack col)
        A_s_n, b_s_n, c_s_n, basics_n, B_inv_n, x_B_n, scale_n = append_bounds_scaled_batch(
            A_s_n, b_s_n, c_s_n, basics_n, B_inv_n, x_B_n, scale_n, node, n_base
        )

        st = lp_solver.LPState(A_s_n, b_s_n, c_s_n, B_inv_n, x_B_n, basics_n, None, None, scale_n)

        # solve node relaxation
        try:
            needs_dual = (x_B_n.min() < -lp_solver.tol_primal_feas)
            st, last_it = lp_solver.solve_for_state(
                c, st,
                minimize=minimize,
                return_state=True,
                dual_simplex=bool(needs_dual),
                last_it=0
            )
        except RuntimeError as e:
            msg = str(e)
            if "Dual simplex: no eligible entering column" in msg:
                continue
            raise

        # cut loop (may add cuts locally)
        out, solved = do_cut_loop(c, dec_cols_idx, int_cols, minimize, st, last_it)

        if solved:
            x, c_out, obj, _ = out
            if (minimize and obj < best_obj - tol_opt) or ((not minimize) and obj > best_obj + tol_opt):
                best_obj = obj
                best_sol = out
                # print(f"New incumbent: {best_obj:.3f}")
            continue

        # Not solved => branch. Note: out is LPState here.
        x_full, c_out, obj_lp, _ = lp_solver.package_solution(c, out)

        # Bound prune: if this relaxation can’t beat incumbent, skip branching
        if best_sol is not None:
            if minimize and float(obj_lp) >= best_obj - tol_opt:
                continue
            if (not minimize) and float(obj_lp) <= best_obj + tol_opt:
                continue

        # choose branch var with optional priority groups
        j = None
        if branch_groups is not None:
            for cols in branch_groups:
                j = choose_branch_var_from_x(x_full, cols, node)
                if j is not None:
                    break
        if j is None:
            j = choose_branch_var_from_x(x_full, int_cols_idx, node)

        # debug: how fractional are declared integer vars (optional)
        # frac = torch.abs(x_full[int_cols_idx] - torch.round(x_full[int_cols_idx]))
        # print("max_frac(int):", float(frac.max().item()), "obj_lp:", float(obj_lp))

        if j is None:
            # integer-feasible w.r.t integer columns
            if (minimize and obj_lp < best_obj - tol_opt) or ((not minimize) and obj_lp > best_obj + tol_opt):
                best_obj = obj_lp
                best_sol = (x_full, c_out, obj_lp)
                # print(f"New incumbent: {best_obj:.3f}")
            continue

        v = float(x_full[j].item())
        lo = math.floor(v)
        hi = lo + 1

        ub_j = float(node.ub[j].item())
        lb_j = float(node.lb[j].item())

        left_ok = (ub_j > lo + tol_int)
        right_ok = (lb_j < hi - tol_int)

        prio = heap_key(float(obj_lp))

        if left_ok:
            left_lb = node.lb.clone()
            left_ub = node.ub.clone()
            left_ub[j] = min(float(left_ub[j].item()), float(lo))
            if not torch.any(left_lb > left_ub):
                heapq.heappush(heap, (prio, push_id, Node(lb=left_lb, ub=left_ub)))
                push_id += 1

        if right_ok:
            right_lb = node.lb.clone()
            right_ub = node.ub.clone()
            right_lb[j] = max(float(right_lb[j].item()), float(hi))
            if not torch.any(right_lb > right_ub):
                heapq.heappush(heap, (prio, push_id, Node(lb=right_lb, ub=right_ub)))
                push_id += 1

    return best_sol


def block_diag_append_identity(old_B_inv: torch.Tensor, k: int) -> torch.Tensor:
    """
    Replace torch.block_diag(old_B_inv, I_k) without block_diag.
    Returns a new (m+k, m+k) tensor with old_B_inv in the top-left and I_k in bottom-right.
    """
    if k <= 0:
        return old_B_inv

    m = old_B_inv.shape[0]
    dev = old_B_inv.device
    dt = old_B_inv.dtype

    new_B_inv = torch.zeros((m + k, m + k), device=dev, dtype=dt)
    new_B_inv[:m, :m] = old_B_inv
    new_B_inv[m:, m:] = torch.eye(k, device=dev, dtype=dt)
    return new_B_inv



def add_gmi_cuts(dist: torch.Tensor, int_cols: bool | torch.Tensor,
                 int_rows: torch.Tensor, state: None | lp_solver.LPState):
    m, n = state.A_s.shape
    dev = state.A_s.device
    dtype = state.A_s.dtype

    # rows whose BASIC variable is integer
    int_col_mask = lp_solver.index_to_mask(int_cols, n)  # (n,)
    basic_is_int = int_col_mask[state.basics]  # (m,)

    # rows with safe fractional RHS
    f0 = frac01(state.x_B)  # (m,)
    f0_min = 1e-4 if dtype == torch.float32 else 1e-8
    rhs_good = (f0 > f0_min) & (f0 < 1.0 - f0_min)

    cand_mask = basic_is_int & rhs_good
    cand_rows = torch.nonzero(cand_mask, as_tuple=True)[0]
    if cand_rows.numel() == 0:
        return state, False

    # pick most fractional among eligible rows
    dist_cand = dist[cand_rows]
    top = find_top_gmi_cuts(dist_cand)
    cut_rows = cand_rows[top.indices]

    # tableau rows for those basics
    u_t = torch.index_select(state.B_inv, dim=0, index=cut_rows)
    target_tableau_rows = u_t @ state.A_s

    cuts = create_gmi_cuts(target_tableau_rows, state.x_B[cut_rows], int_cols, state.basics, n, state.x_B)
    if cuts.numel() == 0:
        # no good cuts remain
        return state, False

    gmi_constraint_count = cuts.shape[0]

    # add identity columns for the new slacks in scaled space
    cuts_with_slacks = torch.cat([cuts, torch.eye(gmi_constraint_count, dtype=dtype, device=dev)],
                                 dim=1)

    eps = 1e-12
    cut_row_max = cuts_with_slacks.abs().amax(dim=1).clamp_min(eps)  # (k,)
    r_cut = cut_row_max.reciprocal().to(dtype=dtype)  # (k,)

    # apply row scaling to the scaled-space rows we append
    cuts_with_slacks = cuts_with_slacks * r_cut[:, None]  # (k, n+k)

    # each existing constraint will need additional zeroed slacks. add these to A and A_s
    A_s_gmi = torch.cat([state.A_s, torch.zeros(m, gmi_constraint_count, dtype=dtype, device=dev)],
                        dim=1)
    A_s_new = torch.cat([A_s_gmi, cuts_with_slacks], dim=0)

    # extend the column and row scales. in this case we keep scale 1 for the column cuts
    col_scale_new = torch.cat([state.scale.s, torch.ones(gmi_constraint_count, dtype=dtype, device=dev)])
    row_scale_new = torch.cat([state.scale.r, r_cut])

    # extend b
    b_s_cuts = -torch.ones(gmi_constraint_count, dtype=dtype, device=dev)
    b_s_cuts = b_s_cuts * r_cut
    b_s_new = torch.cat([state.b_s,  b_s_cuts])

    # extend c, all new slacks costs are 0
    c_s_new = torch.cat([state.c_s, torch.zeros(gmi_constraint_count, dtype=dtype, device=dev)])

    # add the slacks to the basis and basis inverses
    new_basics = torch.cat([state.basics, torch.arange(n, n + gmi_constraint_count, dtype=torch.long, device=dev)])
    new_B_inv = block_diag_append_identity(state.B_inv, gmi_constraint_count)

    x_B_new = torch.cat([state.x_B, b_s_cuts])

    gmi_state = lp_solver.LPState(A_s_new, b_s_new, c_s_new,
                                  new_B_inv, x_B_new, new_basics,
                                  None, None, lp_solver.Scale(row_scale_new, col_scale_new))

    return gmi_state, True


def add_bound_constraint(A: torch.Tensor, b: torch.Tensor, c: torch.Tensor,
                         state: lp_solver.LPState,
                         j: int, sense: str, value: float,
                         last_it: int,
                         minimize: bool):
    """
    Adds one branching bound as a new <= row with a new slack column.
      sense: "ub" for x_j <= value
             "lb" for x_j >= value  (implemented as -x_j <= -value)
    """
    dev = A.device
    m, n = A.shape

    # --- build new row in ORIGINAL space (A,b) ---
    # We will build ORIGINAL row from the SCALED row later to keep basic-variable branching correct.
    # (Original x = x_s / s, so original coefficients are row_s * s and RHS is rhs_s.)
    # For now only keep rhs in original units for nonbasic branching if needed.
    rhs_orig = torch.tensor([value], dtype=b.dtype, device=dev)
    if sense == "lb":
        rhs_orig = -rhs_orig

    # --- scaled space version (A_s, b_s, c_s) ---
    s_old = state.scale.s
    r_old = state.scale.r
    s_j = s_old[j]

    n_s = state.A_s.shape[1]
    is_basic = lp_solver.basic_mask(state.basics, n_s)
    j_is_basic = bool(is_basic[j].item())
    nb_mask = lp_solver.make_nonbasic_mask(state.basics, n_s)

    # Build scaled row_s and scaled rhs_s
    row_s = torch.zeros((1, n_s), dtype=state.A_s.dtype, device=dev)

    if not j_is_basic:
        # Nonbasic branching: simple bound on x_s,j
        if sense == "ub":
            row_s[0, j] = 1.0
            rhs_s = (torch.tensor([value], dtype=state.b_s.dtype, device=dev) * s_j)
        elif sense == "lb":
            row_s[0, j] = -1.0
            rhs_s = (torch.tensor([-value], dtype=state.b_s.dtype, device=dev) * s_j)
        else:
            raise ValueError("sense must be 'ub' or 'lb'")
    else:
        # Basic branching: substitute x_j using the tableau row for the basic variable
        l = int(torch.nonzero(state.basics == j, as_tuple=True)[0].item())

        # tableau row: a_row has 1 at basic j, 0 at other basics; b_l is state.x_B[l]
        a_row = state.B_inv[l, :] @ state.A_s          # (n_s,)
        b_l = state.x_B[l]                             # scalar (scaled)

        U_s = value * s_j
        L_s = value * s_j

        if sense == "ub":
            # x_j = b_l - sum_{N} a_row[N] x_N  <= U_s
            # => sum_{N} (-a_row[N]) x_N <= U_s - b_l
            row_s[0, nb_mask] = -a_row[nb_mask]
            rhs_s = torch.tensor([U_s - b_l], dtype=state.b_s.dtype, device=dev)
        elif sense == "lb":
            # x_j = b_l - sum_{N} a_row[N] x_N  >= L_s
            # => sum_{N} a_row[N] x_N <= b_l - L_s
            row_s[0, nb_mask] = a_row[nb_mask]
            rhs_s = torch.tensor([b_l - L_s], dtype=state.b_s.dtype, device=dev)
        else:
            raise ValueError("sense must be 'ub' or 'lb'")

    # --- now build ORIGINAL-space row from scaled row ---
    # original inequality: (row_s * s_old) x <= rhs_s
    row = (row_s * s_old.unsqueeze(0)).to(dtype=A.dtype)
    rhs = rhs_s.to(dtype=b.dtype)

    # add new slack column for this new row (single slack)
    slack_col = torch.zeros((m, 1), dtype=A.dtype, device=dev)
    A_ext = torch.cat([A, slack_col], dim=1)  # (m, n+1)

    new_row_with_slack = torch.cat([row, torch.ones((1, 1), dtype=A.dtype, device=dev)], dim=1)  # (1, n+1)
    A_new = torch.cat([A_ext, new_row_with_slack], dim=0)  # (m+1, n+1)
    b_new = torch.cat([b, rhs], dim=0)

    # extend c (new slack cost 0)
    c_new = torch.cat([c, torch.zeros((1,), dtype=c.dtype, device=dev)], dim=0)

    # --- scaled space matrix update ---
    slack_col_s = torch.zeros((state.A_s.shape[0], 1), dtype=state.A_s.dtype, device=dev)
    A_s_ext = torch.cat([state.A_s, slack_col_s], dim=1)

    new_row_s_with_slack = torch.cat([row_s, torch.ones((1, 1), dtype=state.A_s.dtype, device=dev)], dim=1)
    A_s_new = torch.cat([A_s_ext, new_row_s_with_slack], dim=0)

    b_s_new = torch.cat([state.b_s, rhs_s], dim=0)
    c_s_new = torch.cat([state.c_s, torch.zeros((1,), dtype=state.c_s.dtype, device=dev)], dim=0)

    # extend scales: new slack column scale = 1; new row scale = 1
    s_new = torch.cat([s_old, torch.ones((1,), dtype=s_old.dtype, device=dev)], dim=0)
    r_new = torch.cat([r_old, torch.ones((1,), dtype=r_old.dtype, device=dev)], dim=0)

    # basis: new slack column index is old n (before extension) in both A and A_s
    new_slack_idx = n  # because A got one new column appended
    basics_new = torch.cat([state.basics, torch.tensor([new_slack_idx], dtype=torch.long, device=dev)], dim=0)

    # B_inv block diagonal extension (new slack basic)
    B_inv_new = block_diag_append_identity(state.B_inv, 1)

    # x_B: append new slack basic value (scaled RHS)
    x_B_new = torch.cat([state.x_B, rhs_s], dim=0)

    child_state = lp_solver.LPState(
        A_s_new, b_s_new, c_s_new,
        B_inv_new, x_B_new, basics_new,
        None, None, lp_solver.Scale(r_new, s_new)
    )

    try:
        # reoptimize (dual simplex is usually appropriate after adding a violated bound)
        child_state, last_it = lp_solver.solve_for_state(
            c_new, child_state, minimize=minimize,
            return_state=True, dual_simplex=True, last_it=last_it
        )
    except RuntimeError:
        return None, None, last_it

    return (A_new, b_new, c_new), child_state, last_it


def choose_branch_var(state: lp_solver.LPState, c: torch.Tensor,
                      int_cols: bool | torch.Tensor):
    m, n = state.A_s.shape
    x_full, _, obj, _ = lp_solver.package_solution(c, state)

    if int_cols is True:
        int_idx = torch.arange(n, device=x_full.device)
    else:
        int_idx = int_cols.clone()
        int_idx = int_idx[int_idx < n]  # ignore newly added slacks/cuts beyond original int list

    x_int = x_full[int_idx]
    dist = (x_int - x_int.round()).abs()

    # pick the most fractional above tolerance
    mask = dist > tol_int
    if not mask.any():
        return None, x_full, obj

    j_local = torch.argmax(dist * mask)
    j = int(int_idx[j_local].item())
    return j, x_full, obj


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


if __name__ == "__main__":
    device = get_device()
    dtype = torch.float32
    print(f"Using device: {device}")

    # m = 8 constraints, n_dec = 12 decision vars, plus 8 slacks → n = 20
    A_dec = torch.tensor([
        [5., 3., 0., 2., 1., 4., 0., 3., 2., 0., 1., 2.],
        [2., 5., 3., 0., 4., 1., 2., 0., 3., 1., 0., 2.],
        [0., 4., 6., 2., 0., 3., 1., 2., 0., 4., 2., 1.],
        [3., 0., 2., 5., 4., 0., 2., 3., 1., 2., 5., 0.],
        [1., 2., 0., 3., 6., 4., 0., 2., 3., 0., 1., 2.],
        [4., 1., 3., 0., 2., 5., 3., 0., 2., 4., 0., 1.],
        [0., 3., 1., 2., 0., 4., 6., 2., 0., 3., 1., 2.],
        [2., 0., 4., 1., 3., 0., 2., 5., 4., 0., 2., 3.],
    ], dtype=dtype, device=device)

    m = A_dec.size(0)
    n = A_dec.size(1)

    A = torch.cat([A_dec, torch.eye(m, dtype=dtype, device=device)], dim=1)  # add slacks
    b = torch.tensor([60., 65., 70., 75., 68., 72., 66., 80.], dtype=dtype, device=device)

    # maximize profit on the 12 decision vars; slacks have zero cost
    c = torch.tensor([8., 7., 6., 9., 5., 8., 7., 6., 5., 9., 4., 7.,
                      0., 0., 0., 0., 0., 0., 0., 0.], dtype=dtype, device=device)

    # slack start (identity at columns 12..19, 0-based)
    basics = torch.arange(n, n + m, dtype=torch.long, device=device)
    int_cols = torch.arange(0, n - 3, dtype=torch.long, device=device)

    sol = solve(A, b, c, basics, int_cols=int_cols, minimize=False)
    print('solution:', sol)
    pf, pres, intf, ires, rel = lp_solver.check_solution(A, b, sol[0], int_cols=int_cols)
    print(pres)
    print("primal feasible:", bool(pf), "max |Ax-b|:", float(pres.abs().max()))
    print("integer feasible:", bool(intf), "max |x-round(x)|:",
          float(ires.max()) if ires is not None and ires.numel() else 0.0)
