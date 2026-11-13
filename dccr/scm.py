"""Linear latent structure learning and differentiable counterfactual rollout.

The matrix convention is W[source, target], so row-vector observations obey
Z = Z @ W + U.  NOTEARS estimates a directed dependency model under its stated
assumptions; fitting it does not establish causal identification.  Callers must
standardize the latent coordinates before both fitting and using this module.
"""

import heapq
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
import torch


def _numpy_matrix(value: Any, name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != array.shape[1] or not array.shape[0]:
        raise ValueError("{} must be a nonempty square matrix".format(name))
    if not np.isfinite(array).all():
        raise FloatingPointError("{} contains nonfinite values".format(name))
    return array


def h_acyclic(W: np.ndarray) -> Tuple[float, np.ndarray]:
    """Return tr(expm(W * W)) - d and its exact analytic gradient.

    The product W * W is elementwise and expm is the matrix exponential.
    In particular, h(0) is zero for every dimension, including d = 1.
    """
    weights = _numpy_matrix(W, "W")
    with np.errstate(over="raise", invalid="raise"):
        exponential = expm(weights * weights)
        value = float(np.trace(exponential) - weights.shape[0])
        gradient = exponential.T * (2.0 * weights)
    if not np.isfinite(value) or not np.isfinite(gradient).all():
        raise FloatingPointError("nonfinite NOTEARS acyclicity value or gradient")
    return value, gradient


def topological_order(W: Any) -> List[int]:
    """Return a deterministic node order, raising ValueError on any cycle.

    Every nonzero entry is an edge; no implicit threshold is applied here.
    Both NumPy matrices and Torch tensors are accepted.
    """
    weights = _numpy_matrix(W, "W")
    adjacency = weights != 0.0
    indegree = adjacency.sum(axis=0).astype(np.int64)
    ready = [int(j) for j in np.flatnonzero(indegree == 0)]
    heapq.heapify(ready)
    result = []  # type: List[int]
    while ready:
        source = heapq.heappop(ready)
        result.append(source)
        for target in np.flatnonzero(adjacency[source]):
            indegree[target] -= 1
            if indegree[target] == 0:
                heapq.heappush(ready, int(target))
    if len(result) != weights.shape[0]:
        raise ValueError("W contains a directed cycle (including possible self-loops)")
    return result


def _prune_to_dag(W: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Insert strongest edges first; skip edges that would close a cycle."""
    weights = _numpy_matrix(W, "W")
    dimension = weights.shape[0]
    dag = np.zeros_like(weights)
    children = [set() for _ in range(dimension)]
    removed = []  # type: List[Dict[str, Any]]
    candidates = [(int(i), int(j)) for i, j in np.argwhere(weights != 0.0)]
    candidates.sort(key=lambda edge: (-abs(weights[edge]), edge[0], edge[1]))
    for source, target in candidates:
        # Adding source -> target is legal iff target cannot reach source.
        pending = [target]
        visited = set()
        makes_cycle = False
        while pending:
            node = pending.pop()
            if node == source:
                makes_cycle = True
                break
            if node not in visited:
                visited.add(node)
                pending.extend(children[node])
        if makes_cycle:
            removed.append({"source": source, "target": target,
                            "weight": float(weights[source, target])})
        else:
            children[source].add(target)
            dag[source, target] = weights[source, target]
    topological_order(dag)
    return dag, removed


def fit_dag(
    X_standardized: np.ndarray,
    lambda1: float = 0.01,
    max_iter: int = 20,
    h_tol: float = 1e-8,
    w_threshold: float = 0.1,
    max_weight: float = 2.0,
    init: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Fit linear NOTEARS, then threshold and deterministically enforce a DAG.

    X_standardized is an N-by-D finite matrix, standardized by the caller with
    statistics that must also be used for counterfactuals.  max_iter bounds
    augmented-Lagrangian outer iterations.  Each positive/negative split
    L-BFGS-B solve has at most 100 iterations.  The returned information keeps
    continuous-solver convergence separate from postprocessed DAG validity.
    """
    data = np.asarray(X_standardized, dtype=np.float64)
    if data.ndim != 2 or not data.shape[0] or not data.shape[1]:
        raise ValueError("X_standardized must be a nonempty N-by-D matrix")
    if not np.isfinite(data).all():
        raise FloatingPointError("X_standardized contains nonfinite values")
    for name, value in (("lambda1", lambda1), ("h_tol", h_tol),
                        ("w_threshold", w_threshold), ("max_weight", max_weight)):
        if not np.isfinite(value):
            raise ValueError("{} must be finite".format(name))
    if lambda1 < 0 or h_tol < 0 or w_threshold < 0 or max_weight <= 0:
        raise ValueError("penalties/tolerances must be nonnegative and max_weight positive")
    if isinstance(max_iter, bool) or not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")

    dimension = data.shape[1]
    size = dimension * dimension
    initial = np.zeros((dimension, dimension), dtype=np.float64)
    if init is not None:
        initial = _numpy_matrix(init, "init").copy()
        if initial.shape != (dimension, dimension):
            raise ValueError("init shape does not match X_standardized")
        if np.any(np.diag(initial) != 0.0):
            raise ValueError("init must have a zero diagonal")
        if np.any(np.abs(initial) > max_weight):
            raise ValueError("init exceeds max_weight")

    with np.errstate(over="raise", invalid="raise"):
        gram = data.T.dot(data) / data.shape[0]
    if not np.isfinite(gram).all():
        raise FloatingPointError("nonfinite sample second-moment matrix")
    identity = np.eye(dimension)
    split = np.concatenate((np.maximum(initial, 0).ravel(),
                            np.maximum(-initial, 0).ravel()))
    bounds = [(0.0, 0.0) if i == j else (0.0, float(max_weight))
              for _ in range(2) for i in range(dimension) for j in range(dimension)]
    rho, alpha, previous_h = 1.0, 0.0, np.inf
    rho_max = 1e16
    optimization = []  # type: List[Dict[str, Any]]
    outer_history = []  # type: List[Dict[str, Any]]
    weights = initial
    raw_h, _ = h_acyclic(weights)
    initial_h = raw_h

    for outer in range(max_iter):
        while True:
            def objective(candidate: np.ndarray) -> Tuple[float, np.ndarray]:
                candidate_weights = (candidate[:size] - candidate[size:]).reshape(dimension, dimension)
                h_value, h_gradient = h_acyclic(candidate_weights)
                difference = identity - candidate_weights
                # This is ||X - XW||_F^2 / (2N), computed without repeated NxD work.
                with np.errstate(over="raise", invalid="raise"):
                    loss = 0.5 * float(np.sum(difference * gram.dot(difference)))
                    smooth_gradient = -gram.dot(difference) + (rho * h_value + alpha) * h_gradient
                    value = loss + 0.5 * rho * h_value * h_value + alpha * h_value + lambda1 * candidate.sum()
                    gradient = np.concatenate((smooth_gradient.ravel(), -smooth_gradient.ravel())) + lambda1
                if not np.isfinite(value) or not np.isfinite(gradient).all():
                    raise FloatingPointError("nonfinite NOTEARS objective or gradient")
                return float(value), gradient

            result = minimize(objective, split, method="L-BFGS-B", jac=True,
                              bounds=bounds, options={"maxiter": 100, "maxls": 40,
                                                      "ftol": 1e-12, "gtol": 1e-8})
            if not np.isfinite(result.x).all() or not np.isfinite(result.fun):
                raise FloatingPointError("NOTEARS optimizer returned nonfinite values")
            candidate_weights = (result.x[:size] - result.x[size:]).reshape(dimension, dimension)
            candidate_h, _ = h_acyclic(candidate_weights)
            # SciPy's all-variables-fixed path (D=1) omits status and nit.
            optimization.append({"outer_iteration": outer + 1, "rho": float(rho),
                                 "success": bool(result.success),
                                 "status": int(result.get("status", 0 if result.success else -1)),
                                 "message": str(result.message), "iterations": int(result.get("nit", 0)),
                                 "objective": float(result.fun), "h": float(candidate_h)})
            if candidate_h > 0.25 * previous_h and rho < rho_max:
                rho = min(rho * 10.0, rho_max)
                continue
            split = result.x
            weights = candidate_weights
            raw_h = candidate_h
            break
        alpha += rho * raw_h
        previous_h = raw_h
        outer_history.append({"iteration": outer + 1, "h": float(raw_h),
                              "rho": float(rho), "alpha": float(alpha)})
        if raw_h <= h_tol or rho >= rho_max:
            break

    thresholded = weights.copy()
    threshold_removed = int(np.count_nonzero((thresholded != 0.0) &
                                             (np.abs(thresholded) < w_threshold)))
    thresholded[np.abs(thresholded) < w_threshold] = 0.0
    h_thresholded, _ = h_acyclic(thresholded)
    dag, cycle_removed = _prune_to_dag(thresholded)
    h_after, _ = h_acyclic(dag)
    final_order = topological_order(dag)
    info = {
        "method": "linear_notears_with_deterministic_dag_projection",
        "n_samples": int(data.shape[0]), "n_dimensions": int(dimension),
        "h_initial": float(initial_h), "h_before": float(raw_h),
        "h_after_threshold": float(h_thresholded), "h_after": float(h_after),
        "acyclicity_converged": bool(raw_h <= h_tol),
        "optimizer_success": bool(optimization[-1]["success"]),
        "optimizer_status": int(optimization[-1]["status"]),
        "optimizer_message": optimization[-1]["message"],
        "converged": bool(raw_h <= h_tol and optimization[-1]["success"]),
        "outer_iterations": len(outer_history), "optimizer_calls": len(optimization),
        "rho": float(rho), "alpha": float(alpha), "edge_threshold": float(w_threshold),
        "max_weight": float(max_weight), "threshold_removed_edges": threshold_removed,
        "cycle_removed_edges": cycle_removed, "removed_edges": cycle_removed,
        "n_cycle_removed_edges": len(cycle_removed), "n_edges": int(np.count_nonzero(dag)),
        "is_dag": True, "topological_order": final_order,
        "optimization": optimization, "outer_history": outer_history,
    }
    return dag, info


def scm_counterfactual(
    z: torch.Tensor,
    W: torch.Tensor,
    mask: torch.Tensor,
    values: torch.Tensor,
    order: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    """Apply coordinate interventions and propagate through a fixed linear DAG.

    All tensors share device; z, W and values share a floating-point dtype.
    z, mask and values are N-by-D, W is D-by-D, and mask has boolean dtype.
    Residuals ``z - z @ W.detach()`` stay in the autograd graph.  Intervened
    coordinates are set to values; all other mechanisms retain their factual
    residual. Teacher/donor tensors should be detached by the caller, while
    a factual-dependent action such as (1-alpha)*z + alpha*donor retains its
    factual derivative. Detaching that whole action would erase intervention
    directions from the Jacobian. This
    function neither detaches z nor modifies any input tensor in place.
    """
    if not all(isinstance(tensor, torch.Tensor) for tensor in (z, W, mask, values)):
        raise TypeError("z, W, mask and values must be torch tensors")
    if z.ndim != 2 or not z.shape[0] or not z.shape[1]:
        raise ValueError("z must be a nonempty N-by-D tensor")
    dimension = z.shape[1]
    if W.shape != (dimension, dimension) or mask.shape != z.shape or values.shape != z.shape:
        raise ValueError("incompatible shapes for z, W, mask or values")
    if not z.is_floating_point() or W.dtype != z.dtype or values.dtype != z.dtype:
        raise TypeError("z, W and values must share a floating-point dtype")
    if mask.dtype != torch.bool:
        raise TypeError("mask must have boolean dtype")
    if any(tensor.device != z.device for tensor in (W, mask, values)):
        raise ValueError("all inputs must be on the same device")
    if any(not bool(torch.isfinite(tensor).all()) for tensor in (z, W, values)):
        raise FloatingPointError("counterfactual inputs contain nonfinite values")
    fixed_weights = W.detach()
    if order is None:
        node_order = topological_order(fixed_weights)
    else:
        node_order = list(order)
        if (len(node_order) != dimension or
                any(isinstance(node, bool) or not isinstance(node, (int, np.integer)) for node in node_order) or
                sorted(node_order) != list(range(dimension))):
            raise ValueError("order must be a permutation of all node indices")
        position = [0] * dimension
        for rank, node in enumerate(node_order):
            position[node] = rank
        positions = torch.tensor(position, device=z.device)
        invalid_edges = (fixed_weights != 0) & (positions[:, None] >= positions[None, :])
        if bool(invalid_edges.any()):
            raise ValueError("order is not topological for W, or W contains a cycle")

    residual = z - z.matmul(fixed_weights)
    columns = [None] * dimension  # type: List[Optional[torch.Tensor]]
    processed = []  # type: List[int]
    for node in node_order:
        natural_value = residual[:, node]
        if processed:
            previous = torch.stack([columns[parent] for parent in processed], dim=1)
            natural_value = natural_value + previous.matmul(fixed_weights[processed, node])
        columns[node] = torch.where(mask[:, node], values[:, node], natural_value)
        processed.append(node)
    counterfactual = torch.stack(columns, dim=1)
    if not bool(torch.isfinite(counterfactual).all()):
        raise FloatingPointError("nonfinite counterfactual rollout result")
    return counterfactual
