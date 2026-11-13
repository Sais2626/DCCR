"""Load the AD1 connectivity cohort and evaluate clustering against reference labels."""

import hashlib
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    adjusted_rand_score,
    f1_score,
    normalized_mutual_info_score,
)
from sklearn.preprocessing import StandardScaler


DEFAULT_DATA_DIR = Path("../data")
AD1_SHAPE = (96, 888)
AD1_CLUSTERS = 4


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_array(data_root, file_and_key):
    filename, matkey = file_and_key
    path = (data_root / filename).resolve()
    if not path.is_file():
        raise FileNotFoundError("Missing dataset file: {}".format(path))
    content = loadmat(str(path), variable_names=[matkey])
    if matkey not in content:
        raise KeyError("MAT key {!r} is absent from {}".format(matkey, path))
    return np.asarray(content[matkey]), {
        "path": str(path), "sha256": _sha256(path), "matkey": matkey,
    }


def _integer_labels(values, name):
    labels = np.asarray(values)
    if labels.ndim == 2 and 1 in labels.shape:
        labels = labels.reshape(-1)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("{} must be a nonempty label vector".format(name))
    if labels.dtype.kind not in "biuf":
        raise ValueError("{} must contain finite integer labels".format(name))
    if not np.isfinite(labels).all():
        raise ValueError("{} contains NaN or infinity".format(name))
    if labels.dtype.kind == "f" and not np.equal(labels, np.floor(labels)).all():
        raise ValueError("{} contains fractional labels".format(name))
    # The upper bound is exclusive because float(int64.max) rounds to 2**63.
    if labels.dtype.kind == "f":
        out_of_range = np.any(labels < -(2.0 ** 63)) or np.any(labels >= 2.0 ** 63)
    elif labels.dtype.kind == "u":
        out_of_range = np.any(labels > np.iinfo(np.int64).max)
    else:
        out_of_range = False
    if out_of_range:
        raise ValueError("{} contains labels outside int64 range".format(name))
    return labels.astype(np.int64, copy=False)


def load_ad1(data_dir=DEFAULT_DATA_DIR):
    """Return standardized float32 features, encoded labels and input provenance.

    Standardization uses all records, as in the transductive AD1 experiment.
    Reference labels are encoded for external evaluation and checked against the
    specified four-cluster setting.
    """
    data_root = Path(data_dir).expanduser().resolve()
    features, feature_source = _load_array(data_root, ("AD_connectivity.mat", "connectivities"))
    raw_labels, label_source = _load_array(data_root, ("AD_labels.mat", "VarName1"))
    source_files = [feature_source, label_source]

    if features.shape != AD1_SHAPE:
        raise ValueError("AD1 feature shape {} != expected {}".format(
            features.shape, AD1_SHAPE))
    if features.dtype.kind not in "biuf" or not np.isfinite(features).all():
        raise ValueError("AD1 features must be finite real numeric values")
    labels = _integer_labels(raw_labels, "AD1 labels")
    if labels.size != features.shape[0]:
        raise ValueError("AD1 has {} feature rows but {} labels".format(
            features.shape[0], labels.size))
    original_values, mapped_labels, counts = np.unique(
        labels, return_inverse=True, return_counts=True)
    if original_values.size != AD1_CLUSTERS:
        raise ValueError("AD1 has {} classes, expected known K={}".format(
            original_values.size, AD1_CLUSTERS))

    scaled = StandardScaler().fit_transform(features.astype(np.float64, copy=False))
    scaled = np.ascontiguousarray(scaled, dtype=np.float32)
    if not np.isfinite(scaled).all():
        raise ValueError("AD1 scaling produced non-finite values")
    mapped_labels = mapped_labels.astype(np.int64, copy=False)
    info = {
        "dataset": "AD1",
        "n_samples": int(scaled.shape[0]),
        "n_features": int(scaled.shape[1]),
        "n_clusters": int(original_values.size),
        "source_files": source_files,
        "original_label_values": original_values.tolist(),
        "class_counts": counts.tolist(),
        "label_mapping": [
            {"original": int(value), "encoded": index, "count": int(count)}
            for index, (value, count) in enumerate(zip(original_values, counts))
        ],
        "preprocessing": {
            "method": "StandardScaler(with_mean=True, with_std=True)",
            "fit_scope": "all_nodes",
            "setting": "transductive",
            "output_dtype": "float32",
            "missing_values": "fail; no imputation or row deletion",
            "label_use": "known K and external evaluation only",
            "n_clusters_convention": "known K from reference labels; checked against dataset specification",
        },
    }
    return scaled, mapped_labels, info


def _align_clusters(y, pred):
    """Match arbitrary cluster IDs to reference classes by maximum overlap."""
    y = _integer_labels(y, "reference labels")
    pred = _integer_labels(pred, "predicted labels")
    if y.size != pred.size:
        raise ValueError("Reference and predicted labels have different lengths")
    _, y_codes = np.unique(y, return_inverse=True)
    _, pred_codes = np.unique(pred, return_inverse=True)
    n_true, n_pred = int(y_codes.max()) + 1, int(pred_codes.max()) + 1
    contingency = np.zeros((n_pred, n_true), dtype=np.int64)
    np.add.at(contingency, (pred_codes, y_codes), 1)
    row_indices, column_indices = linear_sum_assignment(-contingency)
    mapping = np.full(n_pred, n_true, dtype=np.int64)
    mapping[row_indices] = column_indices
    aligned = mapping[pred_codes]
    return y_codes, pred_codes, aligned


def count_cluster_matches(y, pred):
    y_codes, _, aligned = _align_clusters(y, pred)
    return int(np.count_nonzero(y_codes == aligned))


def evaluate_clustering(y, pred):
    """Score one partition; ACC and macro-F1 use the same Hungarian matching."""
    y_codes, pred_codes, aligned = _align_clusters(y, pred)
    return {
        "ACC": float(np.mean(y_codes == aligned)),
        "NMI": float(normalized_mutual_info_score(y_codes, pred_codes)),
        "ARI": float(adjusted_rand_score(y_codes, pred_codes)),
        "F1": float(f1_score(y_codes, aligned, labels=np.unique(y_codes),
                             average="macro", zero_division=0)),
    }
