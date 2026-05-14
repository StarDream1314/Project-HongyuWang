"""Regenerate ALFWorld anchors, OLE training data, and OLE particles."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.cluster import KMeans

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from experiments.aomp_mem.certificate.feature_pipeline import (  # noqa: E402
    FEATURE_DIM,
    FEATURE_NAMES,
    apply_stats,
    fit_stats,
)
from experiments.aomp_mem.certificate.ole_certificate import OLECertificate  # noqa: E402
from experiments.aomp_mem.certificate.reward_model import LinearRewardModel  # noqa: E402
from experiments.aomp_mem.certificate.sgld import ProjectedSGLDEnsemble  # noqa: E402
from experiments.aomp_mem.runtime import build_embedder, code_root, load_dataset  # noqa: E402

_SEED_RE = re.compile(r"seed_(\d+)")


def _collect_alfworld_texts(n_source_texts: int) -> list[str]:
    dataset = load_dataset("alfworld")
    tasks = dataset.tasks()
    if not tasks:
        raise SystemExit("ALFWorld dataset returned no tasks; cannot generate anchors.")
    selected = tasks[: max(1, int(n_source_texts))]
    return [str(task.input_text) for task in selected]


def _run_anchors_stage(
    *,
    embedder_version: str,
    model_name: str | None,
    cache_folder: str | None,
    device: str | None,
    output: Path,
    n_source_texts: int,
    k: int,
    random_state: int,
) -> None:
    embedder = build_embedder(
        embedder_version,
        model_name=model_name,
        cache_folder=cache_folder,
        device=device,
    )
    texts = _collect_alfworld_texts(n_source_texts=n_source_texts)
    matrix = embedder.encode_many(texts)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise SystemExit(f"Embedder returned unusable matrix shape {matrix.shape}")

    cluster_count = min(int(k), matrix.shape[0])
    model = KMeans(n_clusters=cluster_count, random_state=int(random_state), n_init="auto")
    model.fit(matrix)

    output.parent.mkdir(parents=True, exist_ok=True)
    resolved_model = getattr(embedder, "model_name", embedder_version)
    np.savez(
        output,
        anchors=model.cluster_centers_.astype(np.float32),
        embedder_version=np.asarray(str(embedder_version)),
        embedder_model=np.asarray(str(resolved_model)),
        source=np.asarray(f"alfworld_first_{matrix.shape[0]}"),
        random_state=int(random_state),
        k=int(cluster_count),
        n_source_texts=int(matrix.shape[0]),
        embedding_dim=int(matrix.shape[1]),
        pythonhashseed=np.asarray(_current_python_hash_seed()),
    )
    print(
        f"Wrote {cluster_count} ALFWorld coverage anchors "
        f"(dim={matrix.shape[1]}, source={matrix.shape[0]}, embedder={resolved_model}) to {output}"
    )


def _run_prepare_training_set_stage(
    *,
    input_dir: Path,
    output: Path,
    burst_window: int,
    label_source: str,
) -> None:
    trajectory_paths = sorted(input_dir.rglob("trajectory_*.csv"))
    if not trajectory_paths:
        raise SystemExit(f"No trajectory_*.csv files found under {input_dir}")

    features: list[np.ndarray] = []
    labels: list[float] = []
    seed_ids: list[int] = []
    scheduler_ids: list[str] = []
    source_paths: list[str] = []
    row_indices: list[int] = []

    for trajectory_path in trajectory_paths:
        seed_id = _seed_id_from_path(trajectory_path)
        scheduler_id = trajectory_path.parent.name
        last_burst_step: int | None = None
        burst_steps: list[int] = []
        with trajectory_path.open("r", encoding="utf-8", newline="") as handle:
            for row_index, row in enumerate(csv.DictReader(handle)):
                features.append(
                    _features_from_trajectory_row(
                        row,
                        task_step=row_index,
                        last_burst_step=last_burst_step,
                        burst_steps=burst_steps,
                        burst_window=burst_window,
                    )
                )
                labels.append(_label_from_trajectory_row(row, label_source=label_source))
                seed_ids.append(seed_id)
                scheduler_ids.append(scheduler_id)
                source_paths.append(str(trajectory_path.relative_to(input_dir)))
                row_indices.append(row_index)

                if _parse_bool(row.get("drift_onset_flag_t")) or _parse_bool(row.get("drift_detected")):
                    last_burst_step = row_index
                    burst_steps.append(row_index)

    feature_matrix = np.vstack(features).astype(np.float32)
    label_vector = np.asarray(labels, dtype=np.float32)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        features=feature_matrix,
        labels=label_vector,
        seed_ids=np.asarray(seed_ids, dtype=np.int64),
        scheduler_ids=np.asarray(scheduler_ids, dtype=str),
        source_paths=np.asarray(source_paths, dtype=str),
        row_indices=np.asarray(row_indices, dtype=np.int64),
        feature_names=np.asarray(FEATURE_NAMES, dtype=str),
        source_dir=np.asarray(str(input_dir)),
        burst_window=int(burst_window),
        label_source=np.asarray(str(label_source)),
    )
    print(
        f"Wrote ALFWorld OLE training set with {feature_matrix.shape[0]} rows "
        f"and {feature_matrix.shape[1]} features to {output}"
    )


def _run_train_ole_stage(
    *,
    training_set: Path,
    output: Path,
    n_iters: int,
    batch_size: int,
    n_particles: int,
    eta_0: float,
    beta: float,
    sigma_0: float,
    radius: float,
    seed: int,
    holdout_fraction: float,
    kappa: float | None,
    tau_percentile: float,
    coverage_uncertainty_bonus: float,
    coverage_uncertainty_mode: str,
) -> None:
    with np.load(training_set, allow_pickle=False) as data:
        raw_features = np.asarray(data["features"], dtype=float)
        labels = np.asarray(data["labels"], dtype=float).reshape(-1)
    if raw_features.ndim != 2 or raw_features.shape[1] != FEATURE_DIM:
        raise SystemExit(f"features must have shape (n, {FEATURE_DIM}); got {raw_features.shape}")
    if labels.shape[0] != raw_features.shape[0]:
        raise SystemExit("labels length does not match features rows")
    if raw_features.shape[0] < 2:
        raise SystemExit("train_ole requires at least two samples")
    if not 0.0 < holdout_fraction < 1.0:
        raise SystemExit("--holdout-fraction must be in (0, 1)")

    rng = np.random.default_rng(int(seed))
    indices = rng.permutation(raw_features.shape[0])
    holdout_size = max(1, int(round(raw_features.shape[0] * holdout_fraction)))
    holdout_size = min(holdout_size, raw_features.shape[0] - 1)
    holdout_idx = indices[:holdout_size]
    train_idx = indices[holdout_size:]

    train_raw = raw_features[train_idx]
    holdout_raw = raw_features[holdout_idx]
    train_labels = labels[train_idx]
    holdout_labels = labels[holdout_idx]

    stats = fit_stats(train_raw)
    train_features = apply_stats(train_raw, stats)

    model = LinearRewardModel(dim=FEATURE_DIM, radius=radius)
    sgld = ProjectedSGLDEnsemble(
        model,
        n_particles=n_particles,
        eta_0=eta_0,
        beta=beta,
        sigma_0=sigma_0,
        seed=seed,
    )
    trace = sgld.train(train_features, train_labels, n_iters=n_iters, batch_size=batch_size)

    normalized_holdout = apply_stats(holdout_raw, stats)
    cert_kappa = float(kappa) if kappa is not None else _calibrate_kappa(
        model=model,
        particles=sgld.particles,
        normalized_holdout=normalized_holdout,
        holdout_labels=holdout_labels,
    )
    cert = OLECertificate(
        model=model,
        sgld=sgld,
        feature_stats=stats,
        kappa=cert_kappa,
        tau=0.5,
        coverage_uncertainty_bonus=coverage_uncertainty_bonus,
        coverage_uncertainty_mode=coverage_uncertainty_mode,
    )
    cert.calibrate_threshold(holdout_raw, percentile=tau_percentile)
    coverage_rate = cert.empirical_coverage_rate(holdout_raw, holdout_labels)

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        particles=np.asarray(sgld.particles, dtype=np.float32),
        feature_stats_mean=np.asarray(stats.mean, dtype=np.float32),
        feature_stats_std=np.asarray(stats.std, dtype=np.float32),
        kappa=float(cert.kappa),
        tau=float(cert.tau),
        train_loss_curve=np.asarray(trace.loss_curve, dtype=np.float32),
        train_dispersion_curve=np.asarray(trace.dispersion_curve, dtype=np.float32),
        n_particles=int(sgld.n_particles),
        n_iters=int(n_iters),
        batch_size=int(batch_size),
        eta_0=float(eta_0),
        beta=float(beta),
        sigma_0=float(sigma_0),
        radius=float(radius),
        coverage_uncertainty_bonus=float(coverage_uncertainty_bonus),
        coverage_uncertainty_mode=np.asarray(str(coverage_uncertainty_mode)),
        seed=int(seed),
        train_size=int(train_idx.shape[0]),
        holdout_size=int(holdout_idx.shape[0]),
        holdout_coverage_rate=float(coverage_rate),
        tau_percentile=float(tau_percentile),
        label_source=np.asarray("alfworld"),
    )
    print(
        f"Wrote ALFWorld OLE particles to {output} "
        f"(train={train_idx.shape[0]}, holdout={holdout_idx.shape[0]}, coverage={coverage_rate:.3f})"
    )


def _features_from_trajectory_row(
    row: dict[str, str],
    *,
    task_step: int,
    last_burst_step: int | None,
    burst_steps: Sequence[int],
    burst_window: int,
) -> np.ndarray:
    precision = _parse_float(row.get("precision_t"), default=0.5)
    if not np.isfinite(precision):
        precision = 0.5
    window_start = int(task_step) - int(burst_window)
    raw = np.asarray(
        [
            _parse_float(row.get("coverage_t"), default=0.0),
            precision,
            _parse_float(row.get("redundancy_t"), default=0.0),
            _parse_float(row.get("freshness_t"), default=0.0),
            0.0,
            0.0,
            0.0,
            float(np.log1p(_parse_float(row.get("memory_size"), default=0.0))),
            0.0,
            float(task_step - last_burst_step) if last_burst_step is not None else 0.0,
            float(sum(1 for step in burst_steps if window_start <= step < int(task_step))),
            1.0,
        ],
        dtype=float,
    )
    if raw.shape != (FEATURE_DIM,):
        raise RuntimeError(f"feature vector shape mismatch: {raw.shape}")
    return raw


def _label_from_trajectory_row(row: dict[str, str], *, label_source: str) -> float:
    if label_source == "progress":
        return float(np.clip(_parse_float(row.get("progress"), default=0.0), 0.0, 1.0))
    if label_source != "success":
        raise ValueError(f"unsupported label_source: {label_source}")
    success = row.get("success")
    if success is not None and str(success).strip():
        return 1.0 if _parse_bool(success) else 0.0
    return 1.0 if _parse_float(row.get("progress"), default=0.0) >= 1.0 else 0.0


def _parse_float(value: str | None, *, default: float) -> float:
    if value is None:
        return float(default)
    text = str(value).strip()
    if not text:
        return float(default)
    try:
        return float(text)
    except ValueError:
        return float(default)


def _parse_bool(value: str | None) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _seed_id_from_path(path: Path) -> int:
    for part in path.parts:
        match = _SEED_RE.fullmatch(part)
        if match:
            return int(match.group(1))
    return -1


def _current_python_hash_seed() -> str:
    import os

    return str(os.environ.get("PYTHONHASHSEED", "")).strip() or "unset"


def _calibrate_kappa(
    *,
    model: LinearRewardModel,
    particles: np.ndarray,
    normalized_holdout: np.ndarray,
    holdout_labels: np.ndarray,
    min_kappa: float = 1.0,
) -> float:
    predictions = np.asarray(model.predict(particles, normalized_holdout), dtype=float)
    means = predictions.mean(axis=0)
    stds = np.sqrt(np.maximum(predictions.var(axis=0, ddof=0), 1e-12))
    required = np.abs(np.asarray(holdout_labels, dtype=float) - means) / stds
    if required.size == 0 or not np.all(np.isfinite(required)):
        return float(min_kappa)
    return float(max(min_kappa, np.percentile(required, 95.0)))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=["anchors", "prepare_training_set", "train_ole"])
    parser.add_argument("--embedder-version", default="hashing", choices=["hashing", "st-minilm"])
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--cache-folder", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, default=Path("data/coverage_anchors_v2.npz"))
    parser.add_argument("--input-dir", type=Path, default=Path("results/alfworld"))
    parser.add_argument("--training-set", type=Path, default=Path("data/ole_training_set_v2.npz"))
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-iters", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-particles", type=int, default=20)
    parser.add_argument("--eta-0", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--sigma-0", type=float, default=1.0)
    parser.add_argument("--radius", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--kappa", type=float, default=None)
    parser.add_argument("--tau-percentile", type=float, default=95.0)
    parser.add_argument("--coverage-uncertainty-bonus", type=float, default=1.0)
    parser.add_argument("--coverage-uncertainty-mode", choices=["multiplicative", "additive"], default="multiplicative")
    parser.add_argument("--burst-window", type=int, default=50)
    parser.add_argument("--label-source", choices=["success", "progress"], default="progress")
    parser.add_argument("--n-source-texts", type=int, default=134)
    args = parser.parse_args(argv)

    output = args.output if args.output.is_absolute() else (code_root() / args.output)
    input_dir = args.input_dir if args.input_dir.is_absolute() else (code_root() / args.input_dir)
    training_set = args.training_set if args.training_set.is_absolute() else (code_root() / args.training_set)

    if args.stage == "anchors":
        _run_anchors_stage(
            embedder_version=args.embedder_version,
            model_name=args.model_name,
            cache_folder=args.cache_folder,
            device=args.device,
            output=output,
            n_source_texts=args.n_source_texts,
            k=args.k,
            random_state=args.random_state,
        )
        return
    if args.stage == "prepare_training_set":
        _run_prepare_training_set_stage(
            input_dir=input_dir,
            output=output,
            burst_window=args.burst_window,
            label_source=args.label_source,
        )
        return
    _run_train_ole_stage(
        training_set=training_set,
        output=output,
        n_iters=args.n_iters,
        batch_size=args.batch_size,
        n_particles=args.n_particles,
        eta_0=args.eta_0,
        beta=args.beta,
        sigma_0=args.sigma_0,
        radius=args.radius,
        seed=args.seed,
        holdout_fraction=args.holdout_fraction,
        kappa=args.kappa,
        tau_percentile=args.tau_percentile,
        coverage_uncertainty_bonus=args.coverage_uncertainty_bonus,
        coverage_uncertainty_mode=args.coverage_uncertainty_mode,
    )


if __name__ == "__main__":
    main()
