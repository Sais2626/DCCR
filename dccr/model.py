"""VGAE plus spectral clustering, with structure and counterfactual regularizers.

There is no prototype head or factual classification loss. Raw deterministic
latent means enter Euclidean nearest-neighbor SpectralClustering after every
optimizer update. True labels are not accepted. An optional callback observes the post-update
partitions for evaluation; its return value never enters training.

The structural task readout uses one-hot pseudo-label sinks. It is a regression
guide for the representation, not a claim that categories obey a Gaussian SCM.
"""
from __future__ import annotations

from contextlib import nullcontext
import copy
import math
import sys
import warnings

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity
from threadpoolctl import threadpool_limits
import torch
from torch import nn
from torch.nn import functional as F

from .counterfactual import DEFAULT_CF_CONFIG, build_phase, counterfactual_losses
from .scm import fit_dag, topological_order


DEFAULT_CONFIG = dict(DEFAULT_CF_CONFIG)
DEFAULT_CONFIG.update({
    "epochs": 1000, "warmup": 100, "hidden_dim": 64, "latent_dim": 16,
    "learning_rate": 0.01, "dropout": 0.01, "graph_neighbors": 8,
    "spectral_neighbors": 10, "spectral_random_state": None,
    "anchor_interval": 10, "structure_interval": 25,
    "structure_lambda": 0.1, "positive_lambda": 0.1, "negative_lambda": 0.1,
    "ridge": 0.01, "standardization_floor": 0.001,
    "logvar_min": -10.0, "logvar_max": 10.0, "gradient_clip": 0.5,
    "device": "cpu", "dag_kwargs": {},
})


def _finite(value, name):
    valid = bool(torch.isfinite(value).all()) if torch.is_tensor(value) else bool(np.isfinite(value).all())
    if not valid:
        raise FloatingPointError("Non-finite {}: no silent repair applied".format(name))


def build_graph(features, neighbors=8):
    """Original symmetric binary cosine top-k graph; self loops for GCN only."""
    n = len(features)
    if n < 2 or int(neighbors) < 1:
        raise ValueError("graph construction needs at least two samples and positive neighbors")
    similarity = cosine_similarity(features)
    np.fill_diagonal(similarity, -np.inf)
    nearest = np.argsort(-similarity, axis=1, kind="mergesort")[:, :min(int(neighbors), n - 1)]
    adjacency = np.zeros((n, n), dtype=np.float32)
    adjacency[np.arange(n)[:, None], nearest] = 1.0
    adjacency = np.maximum(adjacency, adjacency.T)
    with_loops = sp.coo_matrix(adjacency + np.eye(n, dtype=np.float32))
    degree = np.asarray(with_loops.sum(1)).ravel()
    inverse = sp.diags(np.power(degree, -0.5))
    normalized = (inverse @ with_loops @ inverse).tocoo()
    indices = torch.from_numpy(np.vstack([normalized.row, normalized.col]).astype(np.int64))
    values = torch.from_numpy(normalized.data.astype(np.float32))
    graph = torch.sparse_coo_tensor(indices, values, normalized.shape).coalesce()
    return adjacency, graph


class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(input_dim, output_dim))
        self.bias = nn.Parameter(torch.empty(output_dim)) if bias else None
        bound = 1.0 / math.sqrt(output_dim)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, features, graph):
        output = torch.sparse.mm(graph, features @ self.weight)
        return output + self.bias if self.bias is not None else output


class VGAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, latent_dim=16, dropout=0.01,
                 logvar_min=-10.0, logvar_max=10.0):
        super().__init__()
        self.gc1 = GraphConvolution(input_dim, hidden_dim)
        self.gc_mean = GraphConvolution(hidden_dim, latent_dim)
        self.gc_logvar = GraphConvolution(hidden_dim, latent_dim)
        self.dropout = float(dropout)
        self.logvar_min, self.logvar_max = float(logvar_min), float(logvar_max)

    def encode(self, features, graph, apply_dropout=False):
        hidden = F.relu(self.gc1(features, graph))
        hidden = F.dropout(hidden, p=self.dropout, training=apply_dropout)
        mean = self.gc_mean(hidden, graph)
        logvar = self.gc_logvar(hidden, graph).clamp(self.logvar_min, self.logvar_max)
        return mean, logvar

    def forward(self, features, graph):
        mean, logvar = self.encode(features, graph, apply_dropout=self.training)
        sample = mean + torch.randn_like(mean) * torch.exp(0.5 * logvar) if self.training else mean
        sample = F.dropout(sample, p=self.dropout, training=self.training)
        # Original decoder clips its inner-product logits before sigmoid/BCE.
        return (sample @ sample.T).clamp(-10.0, 10.0), mean, logvar


def aligned_spectral_labels(embedding, n_clusters, previous=None, neighbors=10, *, random_state=None):
    """SC operates on raw latent Euclidean distances, never normalized vectors.

    Hungarian matching only renames output cluster IDs for cross-epoch identity;
    it does not change the partition or use evaluation labels.
    """
    _finite(embedding, "spectral embedding")
    estimator = SpectralClustering(
        n_clusters=int(n_clusters), affinity="nearest_neighbors",
        n_neighbors=min(int(neighbors), len(embedding) - 1),
        assign_labels="kmeans", n_init=10, random_state=random_state,
    )
    omp_limit = nullcontext()
    if sys.platform == "win32" and len(embedding) < 256:
        # A single K-means chunk only needs one OpenMP thread on Windows/MKL.
        omp_limit = threadpool_limits(limits=1, user_api="openmp")
    with omp_limit, warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Graph is not fully connected")
        labels = estimator.fit_predict(embedding).astype(np.int64)
    if previous is not None:
        overlap = np.zeros((n_clusters, n_clusters), dtype=np.int64)
        np.add.at(overlap, (labels, previous), 1)
        rows, cols = linear_sum_assignment(-overlap)
        mapping = np.arange(n_clusters, dtype=np.int64)
        mapping[rows] = cols
        labels = mapping[labels]
    if len(np.unique(labels)) != n_clusters:
        raise RuntimeError("Spectral pseudo-labels contain an empty cluster")
    return labels


def standardize_live(z, floor=0.001):
    """Live centering and scaling; scale gradient is retained above the floor."""
    if not np.isfinite(floor) or floor <= 0:
        raise ValueError("standardization floor must be finite and positive")
    mean = z.mean(dim=0, keepdim=True)
    variance = (z - mean).square().mean(dim=0, keepdim=True)
    scale = variance.clamp_min(float(floor) ** 2).sqrt()
    return (z - mean) / scale


def fit_task_readout(z_np, labels, n_clusters, ridge=0.01, floor=0.001):
    """Fit a centered one-hot ridge readout on the current anchor snapshot."""
    labels = np.asarray(labels)
    if labels.ndim != 1 or len(labels) != len(z_np) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("pseudo labels must be an integer vector matching observations")
    if labels.min() < 0 or labels.max() >= n_clusters:
        raise ValueError("pseudo-label IDs must be in [0, n_clusters)")
    if not np.isfinite(ridge) or ridge <= 0:
        raise ValueError("ridge must be finite and positive")
    z64 = np.asarray(z_np, dtype=np.float64)
    _finite(z64, "task readout input")
    mean = z64.mean(0)
    scale = np.maximum(z64.std(0), floor)
    standardized = (z64 - mean) / scale
    onehot = np.eye(n_clusters, dtype=np.float64)[labels]
    intercept = onehot.mean(0)
    weights = np.linalg.solve(standardized.T @ standardized + ridge * np.eye(z64.shape[1]),
                              standardized.T @ (onehot - intercept))
    _finite(weights, "one-hot task readout")
    return {"pseudo_onehot_np": onehot.astype(np.float32),
            "readout_R_np": weights.astype(np.float32),
            "readout_pi_np": intercept.astype(np.float32),
            "readout_mean_np": mean.astype(np.float32),
            "readout_std_np": scale.astype(np.float32),
            "onehot_columns": list(range(int(n_clusters)))}


def structural_loss(z, phase, floor=0.001):
    """Separate per-entry residual means for the latent DAG and one-hot sinks."""
    standardized = standardize_live(z, floor)
    residual_latent = standardized - standardized @ phase["W"].detach()
    residual_labels = (standardized @ phase["readout_R"].detach()
                       + phase["readout_pi"].detach() - phase["pseudo_onehot"].detach())
    latent_loss = 0.5 * residual_latent.square().mean()
    readout_loss = 0.5 * residual_labels.square().mean()
    return latent_loss + readout_loss, latent_loss, readout_loss


def _configuration(config):
    supplied = dict(config)
    aliases = {"num_epochs": "epochs", "hidden_dim1": "hidden_dim",
               "hidden_dim2": "latent_dim", "causal_warmup_epochs": "warmup",
               "update_interval": "anchor_interval", "positive_alpha": "cf_positive_alpha",
               "negative_alphas": "cf_negative_alphas", "intervention_dims": "cf_intervention_dims",
               "negative_margin": "cf_negative_margin"}
    for old, canonical in aliases.items():
        if old in supplied:
            if canonical in supplied and supplied[canonical] != supplied[old]:
                raise ValueError("Conflicting configuration keys: {} and {}".format(old, canonical))
            supplied[canonical] = supplied.pop(old)
    # A single requested ridge strength applies to both one-hot readouts.
    # The CF module accepts the legacy alias, so keep canonical values equal.
    if "ridge" in supplied and "cf_ridge" in supplied and supplied["ridge"] != supplied["cf_ridge"]:
        raise ValueError("Conflicting configuration keys: ridge and cf_ridge")
    if "ridge" in supplied:
        supplied["cf_ridge"] = supplied["ridge"]
    elif "cf_ridge" in supplied:
        supplied["ridge"] = supplied["cf_ridge"]
    unknown = set(supplied) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError("Unknown model configuration keys: " + ", ".join(sorted(unknown)))
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg.update(supplied)
    for key, value in cfg.items():
        if isinstance(value, (int, float, np.number)) and not np.isfinite(value):
            raise ValueError(key + " must be finite")
    for key in ("epochs", "hidden_dim", "latent_dim", "anchor_interval", "structure_interval",
                "graph_neighbors", "spectral_neighbors"):
        if isinstance(cfg[key], bool) or not isinstance(cfg[key], (int, np.integer)) or cfg[key] < 1:
            raise ValueError(key + " must be a positive integer")
    if cfg["latent_dim"] < 2 or not isinstance(cfg["warmup"], (int, np.integer)) or cfg["warmup"] < 0:
        raise ValueError("latent_dim >= 2 and integer warmup >= 0 are required")
    for key in ("structure_lambda", "positive_lambda", "negative_lambda"):
        if float(cfg[key]) < 0:
            raise ValueError(key + " must be nonnegative")
    for key in ("learning_rate", "ridge", "standardization_floor", "gradient_clip"):
        if float(cfg[key]) <= 0:
            raise ValueError(key + " must be positive")
    if not 0 <= cfg["dropout"] < 1 or cfg["logvar_min"] >= cfg["logvar_max"]:
        raise ValueError("invalid dropout or log-variance bounds")
    return cfg


def train_model(X_scaled, n_clusters, config, seed, observe_partition=None):
    """Train DCCR and optionally pass each post-update partition to an evaluator.

    The evaluator receives a copy of the predicted labels. It supplies no input
    to the learner. All epochs, including warm-up, produce a partition.
    """
    cfg = _configuration(config)
    # The archived AD1 experiment fixed spectral clustering at 42 for all runs.
    # With no explicit override the run seed controls this step as well.
    if cfg["spectral_random_state"] is None:
        cfg["spectral_random_state"] = int(seed)
    spectral_seed = cfg["spectral_random_state"]
    if type(spectral_seed) is not int or not 0 <= spectral_seed < 2**32:
        raise ValueError("spectral_random_state must be null or an integer in [0, 2**32)")
    X = np.asarray(X_scaled, dtype=np.float32)
    if X.ndim != 2 or len(X) < 4 or not isinstance(n_clusters, (int, np.integer)) or not 1 < n_clusters < len(X):
        raise ValueError("Expected finite N x D features, N>=4, and integer 1<K<N")
    _finite(X, "input features")
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    rng = np.random.RandomState(int(seed))
    torch.use_deterministic_algorithms(True)
    device = torch.device(cfg["device"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    adjacency, normalized = build_graph(X, cfg["graph_neighbors"])
    graph = normalized.to(device)
    target_adjacency = torch.as_tensor(adjacency, device=device)
    features = torch.as_tensor(X, device=device)
    network = VGAE(X.shape[1], cfg["hidden_dim"], cfg["latent_dim"], cfg["dropout"],
                   cfg["logvar_min"], cfg["logvar_max"]).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=cfg["learning_rate"])
    labels = aligned_spectral_labels(X, n_clusters, neighbors=cfg["spectral_neighbors"],
                                     random_state=spectral_seed)
    phase = {}

    for step in range(int(cfg["epochs"])):
        auxiliary = step >= cfg["warmup"]
        structure_due = auxiliary and ((step - cfg["warmup"]) % cfg["structure_interval"] == 0)
        anchor_due = auxiliary and (structure_due or (step - cfg["warmup"]) % cfg["anchor_interval"] == 0)
        if anchor_due:
            network.eval()
            with torch.no_grad():
                snapshot, _ = network.encode(features, graph)
            z_np = snapshot.cpu().numpy().copy()
            _finite(z_np, "anchor representation")
            labels = aligned_spectral_labels(z_np, n_clusters, labels, cfg["spectral_neighbors"],
                                             random_state=spectral_seed)
            if structure_due:
                mean = z_np.mean(0).astype(np.float32)
                scale = np.maximum(z_np.std(0), cfg["standardization_floor"]).astype(np.float32)
                W_np, _ = fit_dag((z_np - mean) / scale, **cfg["dag_kwargs"])
                W_np = np.asarray(W_np, dtype=np.float32)
                _finite(W_np, "fitted latent DAG")
                phase = {"W": torch.as_tensor(W_np, device=device), "W_np": W_np,
                         "mean": torch.as_tensor(mean, device=device), "std": torch.as_tensor(scale, device=device),
                         "mean_np": mean, "std_np": scale, "order": topological_order(W_np)}
            readout = fit_task_readout(z_np, labels, n_clusters, cfg["ridge"], cfg["standardization_floor"])
            phase.update(readout)
            for key in ("pseudo_onehot", "readout_R", "readout_pi"):
                phase[key] = torch.as_tensor(readout[key + "_np"], device=device)
            phase = build_phase(z_np, labels.copy(), cfg, rng, device, phase)
            if not isinstance(phase, dict):
                raise TypeError("build_phase must return a phase dictionary")

        network.train()
        optimizer.zero_grad(set_to_none=True)
        reconstructed_logits, sampled_mean, logvar = network(features, graph)
        _finite(reconstructed_logits, "reconstruction logits")
        reconstruction = F.binary_cross_entropy_with_logits(reconstructed_logits, target_adjacency)
        kl = -0.1 * torch.mean(torch.sum(1.0 + logvar - sampled_mean.square() - logvar.exp(), dim=1))
        z, _ = network.encode(features, graph, apply_dropout=False)
        _finite(z, "deterministic representation")
        zero = z.sum() * 0.0
        structure = positive = negative = zero
        if auxiliary:
            if not phase:
                raise RuntimeError("auxiliary training requires a freshly initialized phase")
            structure, _, _ = structural_loss(z, phase, cfg["standardization_floor"])
            positive, negative, _ = counterfactual_losses(z, phase, cfg)
        loss = (reconstruction + kl + cfg["structure_lambda"] * structure
                + cfg["positive_lambda"] * positive + cfg["negative_lambda"] * negative)
        _finite(loss, "total loss")
        loss.backward()
        for name, parameter in network.named_parameters():
            if parameter.grad is not None:
                _finite(parameter.grad, "gradient " + name)
        torch.nn.utils.clip_grad_norm_(network.parameters(), cfg["gradient_clip"])
        optimizer.step()
        for name, parameter in network.named_parameters():
            _finite(parameter, "updated parameter " + name)

        # Evaluate the partition produced by the updated encoder.
        network.eval()
        with torch.no_grad():
            embedding, _ = network.encode(features, graph)
        embedding_np = embedding.cpu().numpy().astype(np.float32, copy=True)
        _finite(embedding_np, "post-update embedding")
        labels = aligned_spectral_labels(embedding_np, n_clusters, labels, cfg["spectral_neighbors"],
                                         random_state=spectral_seed)
        if observe_partition is not None:
            observe_partition(labels.copy())

    return {"labels": labels.copy(), "embedding": embedding_np.copy()}
