"""Run full DCCR on AD1 with the configuration in the main experiment."""

import argparse
import csv
import json
import os
import statistics
import time
from pathlib import Path
from tempfile import NamedTemporaryFile


ROOT = Path(__file__).resolve().parent
METRICS = ("ACC", "NMI", "ARI")
FIELDS = ("seed", *METRICS)


def parse_args(config):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / config["data_dir"],
                        help="directory containing both AD1 MAT files (default: %(default)s)")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs",
                        help="directory for results.csv (one row per seed in this invocation)")
    parser.add_argument("--seeds", type=int, nargs="+", default=config["seeds"],
                        help="override the ordered seeds in configs/ad1.json (default: %(default)s)")
    args = parser.parse_args()
    if not isinstance(args.seeds, list) or not args.seeds:
        parser.error("seeds must be a nonempty list, for example [442]")
    if any(type(s) is not int or not 0 <= s < 2**32 for s in args.seeds):
        parser.error("each seed must be an integer in [0, 2**32)")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be distinct integers in [0, 2**32)")
    return args


def write_results(path, rows):
    """Publish one complete CSV after all requested seeds have finished."""
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = NamedTemporaryFile(dir=path.parent, prefix=".results-", suffix=".tmp",
                                mode="w", encoding="utf-8", newline="", delete=False)
    staged = Path(stream.name)
    try:
        with stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        for attempt in range(6):
            try:
                os.replace(staged, path)
                return
            except PermissionError:
                if attempt == 5:
                    raise
                time.sleep(0.1 * (attempt + 1))
    finally:
        staged.unlink(missing_ok=True)


def main():
    config = json.loads((ROOT / "configs" / "ad1.json").read_text(encoding="utf-8"))
    args = parse_args(config)
    threads = config["threads"]
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(threads)

    import torch
    from threadpoolctl import threadpool_limits
    from dccr.data import load_ad1
    from dccr.evaluation import PartitionSelection
    from dccr.model import train_model

    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    settings = config["model"]
    rows = []
    with threadpool_limits(limits=threads):
        features, reference_labels, dataset = load_ad1(args.data_dir)
        for seed in args.seeds:
            selection = PartitionSelection(reference_labels)
            train_model(features, dataset["n_clusters"], settings, seed,
                        observe_partition=selection.observe)
            rows.append({"seed": seed, **selection.scores()})
    write_results(args.output_dir.expanduser().resolve() / "results.csv", rows)
    means = {metric: statistics.mean(row[metric] for row in rows) for metric in METRICS}
    print("  ".join(f"{metric}={means[metric]:.6f}" for metric in METRICS))


if __name__ == "__main__":
    main()
