"""SCM augmentations with neighborhood consistency, without a classifier.

Positive targets are the current factual soft affinities (stop-gradient), not
uniform binary-neighbor labels. Identity augmentation therefore adds no loss.
Frozen spectral labels/anchors select candidates; they do not certify genuine
causal labels or a crossing of the spectral clustering decision boundary.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist
import torch
from torch.nn import functional as F

from .scm import scm_counterfactual


DEFAULT_CF_CONFIG = {
    "cf_neighbors": 10, "cf_intervention_dims": 2,
    "cf_positive_alpha": 0.15, "cf_negative_alphas": [0.5, 0.75, 0.9],
    "cf_source_purity": 0.6, "cf_positive_purity": 0.6,
    "cf_positive_overlap": 0.5, "cf_negative_target_purity": 0.6,
    "cf_negative_source_purity": 0.25, "cf_positive_budget": 0.5,
    "cf_negative_budget": 3.0, "cf_support_multiplier": 2.0,
    "cf_radius_floor": 0.001, "cf_tau_floor": 1e-6,
    "cf_nonzero_tol": 1e-6, "cf_negative_margin": 0.2,
    "cf_ridge": 0.01,
}


def _config(config):
    result = dict(DEFAULT_CF_CONFIG)
    result.update({key: value for key, value in config.items() if key in result})
    aliases = {"positive_alpha": "cf_positive_alpha", "negative_alphas": "cf_negative_alphas",
               "intervention_dims": "cf_intervention_dims", "ridge": "cf_ridge"}
    for old, new in aliases.items():
        if old in config:
            if new in config and config[new] != config[old]:
                raise ValueError("Conflicting CF configuration: {} / {}".format(old, new))
            result[new] = config[old]
    if not 0 <= result["cf_positive_alpha"] < 1:
        raise ValueError("cf_positive_alpha must be in [0,1), preserving the live query")
    if not result["cf_negative_alphas"] or any(not 0 < value < 1 for value in result["cf_negative_alphas"]):
        raise ValueError("cf_negative_alphas must be nonempty and in (0,1)")
    for name in ("cf_neighbors", "cf_intervention_dims", "cf_radius_floor", "cf_tau_floor", "cf_ridge"):
        if result[name] <= 0:
            raise ValueError(name + " must be positive")
    return result


def _finite(value, name):
    valid = bool(torch.isfinite(value).all()) if torch.is_tensor(value) else bool(np.isfinite(value).all())
    if not valid:
        raise FloatingPointError("Non-finite " + name)


def _squared_distances(query, keys):
    # Matrix form avoids an N x N x D intermediate for full-batch training.
    return (query.square().sum(1, keepdim=True) + keys.square().sum(1)[None, :]
            - 2.0 * query @ keys.T).clamp_min(0.0)


def neighborhood_kl(z, z_cf, tau, valid=None):
    """Current factual targets, live CF query AND current factual keys.

    Both rows exclude the original sample index and use identical temperatures.
    Teacher targets have no gradient; no trainable classifier is introduced.
    """
    if z.shape != z_cf.shape or z.ndim != 2 or len(z) < 2:
        raise ValueError("z and z_cf must be equal-shaped N x D matrices, N>=2")
    tau = torch.as_tensor(tau, dtype=z.dtype, device=z.device).reshape(-1, 1).detach()
    if len(tau) != len(z) or bool((tau <= 0).any()):
        raise ValueError("tau must contain one positive temperature per row")
    mask = torch.eye(len(z), dtype=torch.bool, device=z.device)
    with torch.no_grad():
        factual_logits = -_squared_distances(z.detach(), z.detach()) / tau
        factual_logits = factual_logits.masked_fill(mask, -torch.finfo(z.dtype).max)
        log_target = factual_logits.log_softmax(1)
        target = log_target.exp()
    cf_logits = -_squared_distances(z_cf, z) / tau
    cf_logits = cf_logits.masked_fill(mask, -torch.finfo(z.dtype).max)
    log_prediction = cf_logits.log_softmax(1)
    per_row = (target * (log_target - log_prediction)).sum(1)
    if valid is None:
        valid = torch.ones(len(z), dtype=torch.bool, device=z.device)
    if not bool(valid.any()):
        return z.sum() * 0.0
    result = per_row[valid].mean()
    _finite(result, "positive neighborhood KL")
    return result


def negative_ranking(z, positive_anchor, negative_anchor, rho_squared, valid, margin=0.2):
    """Smooth contrastive ranking: only the factual query receives gradients."""
    if not bool(valid.any()):
        return z.sum() * 0.0
    positive_anchor = positive_anchor.detach()
    negative_anchor = negative_anchor.detach()
    scale = rho_squared.detach().clamp_min(torch.finfo(z.dtype).eps)
    positive_distance = (z - positive_anchor).square().sum(1)
    negative_distance = (z - negative_anchor).square().sum(1)
    result = F.softplus(float(margin) + (positive_distance - negative_distance) / scale)[valid].mean()
    _finite(result, "negative contrastive ranking")
    return result


def _teacher_geometry(z, labels, config):
    n = len(z)
    k = min(int(config["cf_neighbors"]), n - 1)
    distances = cdist(z, z)
    np.fill_diagonal(distances, np.inf)
    neighbors = np.argsort(distances, axis=1, kind="mergesort")[:, :k]
    neighbor_distances = distances[np.arange(n)[:, None], neighbors]
    rho = np.maximum(neighbor_distances[:, -1], config["cf_radius_floor"])
    tau = np.maximum(rho ** 2, config["cf_tau_floor"])
    weights = np.exp(-neighbor_distances ** 2 / tau[:, None])
    purity = (weights * (labels[neighbors] == labels[:, None])).sum(1) / weights.sum(1)
    support = np.zeros(int(labels.max()) + 1, dtype=np.float64)
    fallback = max(float(np.quantile(distances.min(1), 0.95)), config["cf_radius_floor"])
    for cluster in np.unique(labels):
        indices = np.flatnonzero(labels == cluster)
        radius = (float(np.quantile(distances[np.ix_(indices, indices)].min(1), 0.95))
                  if len(indices) > 1 else fallback)
        support[cluster] = config["cf_support_multiplier"] * max(radius, config["cf_radius_floor"])
    return {"neighbors": neighbors, "rho": rho, "tau": tau, "purity": purity,
            "support": support, "distances": distances, "k": k}


def _gate_candidates(cf_raw, factual_raw, source_labels, targets, state, config, kind):
    """Frozen-anchor gates. A donor label alone never makes a CF valid."""
    anchors = state["anchors_np"]
    labels = state["labels_np"]
    n = len(anchors)
    cf_raw = np.asarray(cf_raw)
    valid = np.isfinite(cf_raw).all(1)
    if not valid.all():
        raise FloatingPointError("Non-finite counterfactual candidate")
    distances = cdist(cf_raw, anchors)
    np.fill_diagonal(distances, np.inf)  # never count the source itself as support
    neighbors = np.argsort(distances, axis=1, kind="mergesort")[:, :state["k"]]
    local_distances = distances[np.arange(n)[:, None], neighbors]
    weights = np.exp(-local_distances ** 2 / state["tau_np"][:, None])
    denominator = weights.sum(1)
    # Underflow gives zero support, not repaired or fabricated neighbors.
    usable = denominator > 0
    source_purity = np.zeros(n, dtype=np.float64)
    target_purity = np.zeros(n, dtype=np.float64)
    source_purity[usable] = ((weights * (labels[neighbors] == source_labels[:, None])).sum(1)[usable]
                             / denominator[usable])
    target_purity[usable] = ((weights * (labels[neighbors] == targets[:, None])).sum(1)[usable]
                             / denominator[usable])
    displacement = np.linalg.norm(cf_raw - factual_raw, axis=1)
    nonzero = displacement > config["cf_nonzero_tol"]
    budget_name = "cf_positive_budget" if kind == "positive" else "cf_negative_budget"
    within_budget = displacement <= config[budget_name] * state["rho_np"]
    support_ok = np.zeros(n, dtype=bool)
    for cluster in np.unique(targets):
        rows = np.flatnonzero(targets == cluster)
        members = np.flatnonzero(labels == cluster)
        if len(members):
            support_ok[rows] = distances[np.ix_(rows, members)].min(1) <= state["support_np"][cluster]
    valid &= usable & nonzero & within_budget & support_ok
    if kind == "positive":
        overlap = np.array([len(np.intersect1d(neighbors[i], state["neighbors_np"][i])) / state["k"]
                            for i in range(n)])
        semantic = (source_purity >= config["cf_positive_purity"]) & (overlap >= config["cf_positive_overlap"])
    elif kind == "negative":
        overlap = np.zeros(n)
        semantic = ((targets != source_labels)
                    & (target_purity >= config["cf_negative_target_purity"])
                    & (source_purity <= config["cf_negative_source_purity"]))
    else:
        raise ValueError("kind must be positive or negative")
    valid &= semantic
    return valid, {"support_ok": support_ok, "within_budget": within_budget,
                   "nonzero": nonzero, "semantic_ok": semantic,
                   "source_purity": source_purity, "target_purity": target_purity,
                   "neighbor_overlap": overlap}


def _source_agreement(factual_raw, state, config):
    """Do not trust a stale source pseudo label after student representation drift."""
    labels = state["labels_np"]
    distances = cdist(factual_raw, state["anchors_np"])
    np.fill_diagonal(distances, np.inf)
    neighbors = np.argsort(distances, axis=1, kind="mergesort")[:, :state["k"]]
    local_distances = distances[np.arange(len(labels))[:, None], neighbors]
    weights = np.exp(-local_distances ** 2 / state["tau_np"][:, None])
    denominator = weights.sum(1)
    valid = denominator > 0
    purity = np.zeros(len(labels))
    purity[valid] = ((weights * (labels[neighbors] == labels[:, None])).sum(1)[valid]
                     / denominator[valid])
    valid &= purity >= config["cf_source_purity"]
    for cluster in np.unique(labels):
        members = np.flatnonzero(labels == cluster)
        valid[members] &= distances[np.ix_(members, members)].min(1) <= state["support_np"][cluster]
    return valid


def _response_diagnostics(factual, counterfactual, mask, valid, prefix, tolerance):
    """Measure genuine propagated response separately from directly edited entries."""
    result = {prefix + "_delta_rms": 0.0, prefix + "_changed_nonintervened_dims": 0.0,
              prefix + "_descendant_delta_rms": 0.0, prefix + "_active_mask_entries": 0}
    if not bool(valid.any()):
        return result
    delta = (counterfactual.detach() - factual.detach())[valid]
    intervened = mask[valid]
    nonintervened = ~intervened
    result[prefix + "_delta_rms"] = float(delta.square().mean().sqrt())
    result[prefix + "_changed_nonintervened_dims"] = float(((delta.abs() > tolerance) & nonintervened).sum(1).float().mean())
    result[prefix + "_active_mask_entries"] = int(intervened.sum())
    if bool(nonintervened.any()):
        result[prefix + "_descendant_delta_rms"] = float(delta[nonintervened].square().mean().sqrt())
    return result


def _raw_rollout(z, phase, mask, donor, alpha):
    mean, std = phase["mean"].detach(), phase["std"].detach()
    standardized = (z - mean) / std
    if not torch.is_tensor(alpha):
        alpha = torch.as_tensor(alpha, dtype=z.dtype, device=z.device)
    if alpha.ndim == 1:
        alpha = alpha[:, None]
    # alpha<1 retains the action coordinate's own live dependency, unlike a
    # hard full replacement. The donor and structural parameters are frozen.
    values = (1.0 - alpha) * standardized + alpha * donor.detach()
    cf = scm_counterfactual(standardized, phase["W"].detach(), mask, values, phase["order"])
    return cf * std + mean


def build_phase(z_np, labels, cfg, rng, device, phase):
    """Refresh frozen candidate selection using pseudo labels only.

    Required phase fields: W, W_np, order, mean, std, mean_np, std_np.
    The model owns W fitting; this module recomputes the task readout in the
    retained W coordinate system, independently of a model's structural readout.
    """
    config = _config(cfg)
    z_np = np.asarray(z_np, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    if z_np.ndim != 2 or len(z_np) < 3 or labels.shape != (len(z_np),):
        raise ValueError("Expected N>=3 latent rows and one pseudo label per row")
    _finite(z_np, "phase embeddings")
    if np.any(labels < 0) or not np.array_equal(np.unique(labels), np.arange(labels.max() + 1)):
        raise ValueError("Pseudo labels must be contiguous nonnegative cluster IDs")
    n, d = z_np.shape
    cluster_count = int(labels.max()) + 1
    if cluster_count < 2:
        raise ValueError("Counterfactual contrast requires at least two pseudo clusters")
    standardized = (z_np - phase["mean_np"]) / phase["std_np"]
    centered = standardized - standardized.mean(0)
    onehot = np.eye(cluster_count, dtype=np.float64)[labels]
    prior = onehot.mean(0)
    readout = np.linalg.solve(centered.T @ centered + config["cf_ridge"] * np.eye(d),
                              centered.T @ (onehot - prior))
    # Single-coordinate total effects are used only to rank candidate masks.
    # Actual multi-coordinate interventions always use the full SCM rollout.
    total_effect = np.linalg.solve(np.eye(d) - np.asarray(phase["W_np"]), readout)
    low_impact = np.linalg.norm(total_effect - total_effect.mean(1, keepdims=True), axis=1)
    geometry = _teacher_geometry(z_np, labels, config)
    trusted = geometry["purity"] >= config["cf_source_purity"]
    state = {"anchors_np": z_np.copy(), "labels_np": labels.copy(),
             "neighbors_np": geometry["neighbors"], "rho_np": geometry["rho"],
             "tau_np": geometry["tau"], "support_np": geometry["support"],
             "k": geometry["k"], "trusted_np": trusted,
             "task_readout_np": readout, "total_effect_np": total_effect}
    count = min(int(config["cf_intervention_dims"]), d)
    pos_mask = np.zeros((n, d), dtype=bool)
    neg_mask = np.zeros((n, d), dtype=bool)
    pos_donor = standardized.copy()
    neg_donor = standardized.copy()
    pos_possible = np.zeros(n, dtype=bool)
    neg_possible = np.zeros(n, dtype=bool)
    negative_target = labels.copy()
    positive_anchor = z_np.copy()
    positive_anchor_available = np.zeros(n, dtype=bool)
    positive_coords = np.argsort(low_impact, kind="mergesort")[:count]
    for i in np.flatnonzero(trusted):
        local = geometry["neighbors"][i]
        same = local[(labels[local] == labels[i]) & trusted[local]]
        if len(same) == 0:
            continue
        # First is the nearest stable same-cluster reference for negative loss.
        positive_anchor[i] = z_np[same[0]]
        positive_anchor_available[i] = True
        donor = int(rng.choice(same))
        pos_mask[i, positive_coords] = True
        pos_donor[i] = standardized[donor]
        pos_possible[i] = True
        other = np.flatnonzero(trusted & (labels != labels[i]))
        if len(other) == 0:
            continue
        nearest_other = int(other[np.argmin(geometry["distances"][i, other])])
        target = labels[nearest_other]
        pair_score = np.abs(total_effect[:, target] - total_effect[:, labels[i]])
        coords = np.argsort(-pair_score, kind="mergesort")[:count]
        unaffected = np.ones(d, dtype=bool)
        unaffected[coords] = False
        pool = other[labels[other] == target]
        if unaffected.any():
            donor_distance = np.sum((standardized[pool][:, unaffected] - standardized[i, unaffected]) ** 2, axis=1)
        else:
            donor_distance = geometry["distances"][i, pool]
        donor = int(pool[np.argmin(donor_distance)])
        neg_mask[i, coords] = True
        neg_donor[i] = standardized[donor]
        negative_target[i] = target
        neg_possible[i] = True
    raw = torch.as_tensor(z_np, device=device)
    pos_mask_t = torch.as_tensor(pos_mask, device=device)
    neg_mask_t = torch.as_tensor(neg_mask, device=device)
    pos_donor_t = torch.as_tensor(pos_donor, dtype=raw.dtype, device=device).detach()
    neg_donor_t = torch.as_tensor(neg_donor, dtype=raw.dtype, device=device).detach()
    with torch.no_grad():
        pos_cf = _raw_rollout(raw, phase, pos_mask_t, pos_donor_t, config["cf_positive_alpha"])
        pos_gate, pos_details = _gate_candidates(pos_cf.cpu().numpy(), z_np, labels, labels, state, config, "positive")
        pos_active = pos_possible & pos_gate
        neg_active = np.zeros(n, dtype=bool)
        chosen_alpha = np.zeros(n, dtype=np.float32)
        last_neg_details = None
        for alpha in config["cf_negative_alphas"]:
            neg_cf = _raw_rollout(raw, phase, neg_mask_t, neg_donor_t, float(alpha))
            gate, last_neg_details = _gate_candidates(neg_cf.cpu().numpy(), z_np, labels, negative_target,
                                                       state, config, "negative")
            accept = neg_possible & positive_anchor_available & ~neg_active & gate
            chosen_alpha[accept] = float(alpha)
            neg_active |= accept
    state.update({"positive_mask": pos_mask_t, "negative_mask": neg_mask_t,
                  "positive_donor": pos_donor_t, "negative_donor": neg_donor_t,
                  "positive_active": torch.as_tensor(pos_active, device=device),
                  "negative_active": torch.as_tensor(neg_active, device=device),
                  "negative_alpha": torch.as_tensor(chosen_alpha, device=device).detach(),
                  "positive_anchor": torch.as_tensor(positive_anchor, device=device).detach(),
                  "negative_target_np": negative_target,
                  "rho_squared": torch.as_tensor(np.maximum(geometry["rho"] ** 2, config["cf_tau_floor"]),
                                                  dtype=raw.dtype, device=device).detach(),
                  "tau": torch.as_tensor(geometry["tau"], dtype=raw.dtype, device=device).detach()})
    state["generation"] = {
        "trusted_sources": int(trusted.sum()), "positive_proposed": int(pos_possible.sum()),
        "negative_proposed": int(neg_possible.sum()), "positive_generated": int(pos_active.sum()),
        "negative_generated": int(neg_active.sum()),
        "positive_mask_entries": int(pos_mask.sum()), "negative_mask_entries": int(neg_mask.sum()),
        "positive_rejected": int((pos_possible & ~pos_active).sum()),
        "negative_rejected": int((neg_possible & ~neg_active).sum()),
        "positive_support_rejected": int((pos_possible & ~pos_details["support_ok"]).sum()),
        "positive_semantic_rejected": int((pos_possible & ~pos_details["semantic_ok"]).sum()),
        "positive_budget_rejected": int((pos_possible & ~pos_details["within_budget"]).sum()),
        # Negative rejection reasons are evaluated at the largest configured
        # alpha, while overall acceptance is the union across attempted alphas.
        "negative_support_rejected_last_alpha": int((neg_possible & ~neg_active & ~last_neg_details["support_ok"]).sum()),
        "negative_semantic_rejected_last_alpha": int((neg_possible & ~neg_active & ~last_neg_details["semantic_ok"]).sum()),
        "negative_budget_rejected_last_alpha": int((neg_possible & ~neg_active & ~last_neg_details["within_budget"]).sum()),
    }
    phase["cf"] = state
    phase["generation"] = dict(state["generation"])
    return phase


def counterfactual_losses(z, phase, cfg):
    """Return losses and coverage. Negative candidates are recomputed from the
    current factual z each step, gated, and then detached as contrast examples;
    they are not immutable generated coordinates cached throughout the phase.
    """
    config = _config(cfg)
    zero = z.sum() * 0.0
    state = phase.get("cf") if phase else None
    if state is None:
        return zero, zero, {"positive_valid": 0, "negative_valid": 0,
                            "positive_coverage": 0.0, "negative_coverage": 0.0}
    diagnostics = dict(state["generation"])
    positive = negative = zero
    positive_count = negative_count = 0
    factual_np = z.detach().cpu().numpy()
    labels = state["labels_np"]
    source_agreement_np = _source_agreement(factual_np, state, config)
    source_agreement = torch.as_tensor(source_agreement_np, device=z.device)
    diagnostics["factual_source_agreement"] = int(source_agreement_np.sum())
    diagnostics["factual_source_drift_rejected"] = int((state["trusted_np"] & ~source_agreement_np).sum())
    for prefix in ("positive", "negative"):
        diagnostics.update({prefix + "_delta_rms": 0.0,
                            prefix + "_changed_nonintervened_dims": 0.0,
                            prefix + "_descendant_delta_rms": 0.0,
                            prefix + "_active_mask_entries": 0})
    if bool(state["positive_active"].any()):
        positive_cf = _raw_rollout(z, phase, state["positive_mask"], state["positive_donor"],
                                   config["cf_positive_alpha"])
        _finite(positive_cf, "positive live counterfactual")
        with torch.no_grad():
            gate, _ = _gate_candidates(positive_cf.detach().cpu().numpy(), factual_np, labels, labels,
                                       state, config, "positive")
            valid = state["positive_active"] & source_agreement & torch.as_tensor(gate, device=z.device)
        positive_count = int(valid.sum())
        positive = neighborhood_kl(z, positive_cf, state["tau"], valid)
        diagnostics.update(_response_diagnostics(z, positive_cf, state["positive_mask"], valid,
                                                  "positive", config["cf_nonzero_tol"]))
    if bool(state["negative_active"].any()):
        # Detach the current step's accepted negative example for contrast.
        with torch.no_grad():
            negative_cf = _raw_rollout(z.detach(), phase, state["negative_mask"], state["negative_donor"],
                                       state["negative_alpha"])
            gate, _ = _gate_candidates(negative_cf.cpu().numpy(), factual_np, labels,
                                       state["negative_target_np"], state, config, "negative")
            valid = state["negative_active"] & source_agreement & torch.as_tensor(gate, device=z.device)
        negative_count = int(valid.sum())
        negative = negative_ranking(z, state["positive_anchor"], negative_cf,
                                     state["rho_squared"], valid, config["cf_negative_margin"])
        diagnostics.update(_response_diagnostics(z, negative_cf, state["negative_mask"], valid,
                                                  "negative", config["cf_nonzero_tol"]))
    diagnostics.update({"positive_valid": positive_count, "negative_valid": negative_count,
                        "positive_coverage": positive_count / len(z), "negative_coverage": negative_count / len(z),
                        "positive_drift_rejected": diagnostics["positive_generated"] - positive_count,
                        "negative_drift_rejected": diagnostics["negative_generated"] - negative_count,
                        "positive_loss": float(positive.detach()), "negative_loss": float(negative.detach())})
    return positive, negative, diagnostics
