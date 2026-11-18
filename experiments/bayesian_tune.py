"""Small Gaussian-process Bayesian search for a training-only Random Forest.

Uses trip-grouped validation and fixed weak labels. The student-test partition
is not read during this search.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.model_selection import GroupKFold

from train_compare import SEED, _load_sampled_candidates


def _decode(point: np.ndarray) -> dict[str, object]:
    return {
        "n_estimators": int(round(80 + point[0] * 120)),
        "max_depth": int(round(8 + point[1] * 20)),
        "min_samples_leaf": int(round(1 + point[2] * 4)),
        "max_features": round(float(0.5 + point[3] * 0.5), 3),
    }


def _top1_correct(labels: np.ndarray, keys: np.ndarray, scores: np.ndarray) -> int:
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    starts = np.r_[0, np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1]) + 1]
    ends = np.r_[starts[1:], len(order)]
    return sum(int(labels[group[np.argmax(scores[group])]] == 1)
               for start, end in zip(starts, ends) if len(group := order[start:end]))


def tune(dataset: Path, output_dir: Path, *, initial_trials: int = 4,
         bayesian_trials: int = 6, folds: int = 5) -> dict[str, object]:
    (x, y, groups, obs_keys, sample_ids, group_names, expected_validation,
     validation_misses, *_rest) = _load_sampled_candidates(dataset, 50.0, 8)
    splitter = GroupKFold(n_splits=folds)
    splits = list(splitter.split(np.arange(len(group_names)), groups=np.arange(len(group_names))))
    trial_rows: list[dict[str, object]] = []
    seen: set[tuple[object, ...]] = set()
    candidate_rng = np.random.default_rng(SEED + 41)
    started_all = time.perf_counter()

    def evaluate(params: dict[str, object], trial_number: int, method: str) -> dict[str, object]:
        signature = tuple(params.values())
        if signature in seen:
            return evaluate(_decode(candidate_rng.random(4)), trial_number, method)
        seen.add(signature)
        fold_scores: list[float] = []
        fold_seconds: list[float] = []
        for fold_index, (group_train, group_validation) in enumerate(splits, 1):
            train_rows = np.flatnonzero(np.isin(groups, group_train) & (sample_ids == 0))
            valid_rows = np.flatnonzero(np.isin(groups, group_validation) & (sample_ids == 1))
            model = RandomForestClassifier(
                **params, class_weight="balanced_subsample", n_jobs=-1,
                random_state=SEED + fold_index)
            start = time.perf_counter()
            model.fit(x[train_rows], y[train_rows])
            scores = model.predict_proba(x[valid_rows])[:, 1]
            correct = _top1_correct(y[valid_rows], obs_keys[valid_rows], scores)
            validation_groups = np.unique(groups[valid_rows])
            expected = sum(expected_validation.get(int(g), 0) for g in validation_groups)
            fold_scores.append(correct / max(1, expected))
            fold_seconds.append(time.perf_counter() - start)
        result = {
            "trial": trial_number, "method": method, **params,
            "top1_all_aligned_mean": float(np.mean(fold_scores)),
            "top1_all_aligned_std": float(np.std(fold_scores, ddof=1)),
            "fold_scores": fold_scores,
            "mean_fit_and_score_seconds_per_fold": float(np.mean(fold_seconds)),
        }
        trial_rows.append(result)
        print(f"trial {trial_number}: {method} {params} => {result['top1_all_aligned_mean']:.4f}", flush=True)
        return result

    for index in range(initial_trials):
        evaluate(_decode(candidate_rng.random(4)), index + 1, "seeded_random")

    for index in range(bayesian_trials):
        points = np.asarray([[
            (int(row["n_estimators"]) - 80) / 120,
            (int(row["max_depth"]) - 8) / 20,
            (int(row["min_samples_leaf"]) - 1) / 4,
            (float(row["max_features"]) - 0.5) / 0.5,
        ] for row in trial_rows], dtype=float)
        values = np.asarray([row["top1_all_aligned_mean"] for row in trial_rows], dtype=float)
        gp = GaussianProcessRegressor(
            kernel=ConstantKernel(1.0, (1e-2, 1e2)) * Matern(nu=2.5) + WhiteKernel(1e-5),
            normalize_y=True, random_state=SEED, n_restarts_optimizer=2,
        ).fit(points, values)
        pool = candidate_rng.random((512, 4))
        pool_params = [_decode(point) for point in pool]
        pool_signatures = [tuple(params.values()) for params in pool_params]
        keep = np.asarray([signature not in seen for signature in pool_signatures])
        pool = pool[keep]
        mean, std = gp.predict(pool, return_std=True)
        improvement = mean - np.max(values) - 0.005
        z = improvement / np.maximum(std, 1e-9)
        acquisition = improvement * norm.cdf(z) + std * norm.pdf(z)
        best = int(np.argmax(acquisition))
        evaluate(pool_params[np.flatnonzero(keep)[best]], initial_trials + index + 1, "gaussian_process_ei")

    elapsed = time.perf_counter() - started_all
    best_result = max(trial_rows, key=lambda row: float(row["top1_all_aligned_mean"]))
    report = {
        "partition": "training_grouped_cross_validation_only",
        "model": "random_forest",
        "method": "GaussianProcessRegressor with Matern 5/2 kernel and expected improvement",
        "initial_random_trials": initial_trials,
        "bayesian_trials": bayesian_trials,
        "folds": folds,
        "seed": SEED + 41,
        "features": "fixed 50m radius/top8; up to 100 sampled training observations per trip",
        "objective": "mean trip-grouped CV Top-1 accuracy over all weak-aligned observations; candidate misses score wrong",
        "best_trial": best_result,
        "elapsed_seconds": elapsed,
        "trials": trial_rows,
        "note": "Development and evaluator partitions were not read during hyperparameter search.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "bayesian_search_record.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "bayesian_search_trials.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[key for key in trial_rows[0] if key != "fold_scores"])
        writer.writeheader()
        writer.writerows({key: value for key, value in row.items() if key != "fold_scores"} for row in trial_rows)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("experiments/results"))
    parser.add_argument("--initial-trials", type=int, default=4)
    parser.add_argument("--bayesian-trials", type=int, default=6)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    report = tune(args.dataset, args.output, initial_trials=args.initial_trials,
                  bayesian_trials=args.bayesian_trials, folds=args.folds)
    print(json.dumps({"best_trial": report["best_trial"], "elapsed_seconds": report["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
