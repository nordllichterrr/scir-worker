import json
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

GT_PATH = Path("experiments") / "ground_truth.json"
RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("experiments") / "calibration"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_gt() -> Dict[str, List[str]]:
    if not GT_PATH.exists():
        return {}
    with open(GT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("data", data)


def load_reranker_scores(run_id: str) -> Tuple[np.ndarray, np.ndarray]:

    path = RESULTS_DIR / f"{run_id}.json"
    if not path.exists():
        raise FileNotFoundError(path)

    with open(path, "r", encoding="utf-8") as f:
        blob = json.load(f)

    reranker_scores = blob.get("reranker_scores", {})
    gt = load_gt()

    scores: List[float] = []
    labels: List[float] = []

    for uid, item_scores in reranker_scores.items():
        uid_str = str(uid)
        gt_items = set(str(x) for x in gt.get(uid_str, []))
        for item_id, s in item_scores.items():
            scores.append(float(s))
            labels.append(1.0 if str(item_id) in gt_items else 0.0)

    if not scores or not labels or sum(labels) == 0:
        raise RuntimeError("Not enough positive examples to fit calibration.")

    return np.asarray(scores, dtype=np.float32), np.asarray(labels, dtype=np.float32)


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


def compute_brier(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((probs - labels) ** 2))


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.clip(
        np.searchsorted(bin_edges[1:-1], probs, side="right"), 0, n_bins - 1
    )
    ece = 0.0
    n = len(probs)
    for b in range(n_bins):
        mask = bin_indices == b
        if not np.any(mask):
            continue
        acc = float(np.mean(labels[mask]))
        conf = float(np.mean(probs[mask]))
        weight = float(np.sum(mask)) / n
        ece += weight * abs(acc - conf)
    return ece


def fit_temperature(logits: np.ndarray, labels: np.ndarray, T_min: float = 0.01, T_max: float = 10.0) -> float:

    def brier_at_T(T: float) -> float:
        p = sigmoid(logits / T)
        return compute_brier(p, labels)

    res = minimize_scalar(brier_at_T, bounds=(T_min, T_max), method="bounded")
    return float(res.x)


def fit_platt(logits: np.ndarray, labels: np.ndarray) -> LogisticRegression:

    clf = LogisticRegression(solver="lbfgs")
    clf.fit(logits.reshape(-1, 1), labels)
    return clf


def fit_isotonic(probs: np.ndarray, labels: np.ndarray) -> IsotonicRegression:
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(probs, labels)
    return iso


def calibrate_run(run_id: str) -> Dict[str, Any]:
    logits, labels = load_reranker_scores(run_id)


    probs_raw = sigmoid(logits)
    brier_raw = compute_brier(probs_raw, labels)
    ece_raw = compute_ece(probs_raw, labels)


    T_opt = fit_temperature(logits, labels)
    probs_temp = sigmoid(logits / T_opt)
    brier_temp = compute_brier(probs_temp, labels)
    ece_temp = compute_ece(probs_temp, labels)


    platt = fit_platt(logits, labels)
    probs_platt = platt.predict_proba(logits.reshape(-1, 1))[:, 1]
    brier_platt = compute_brier(probs_platt, labels)
    ece_platt = compute_ece(probs_platt, labels)

    iso = fit_isotonic(probs_raw, labels)
    probs_iso = iso.predict(probs_raw)
    brier_iso = compute_brier(probs_iso, labels)
    ece_iso = compute_ece(probs_iso, labels)

    result = {
        "run_id": run_id,
        "n_samples": int(len(logits)),
        "positive_rate": float(labels.mean()),
        "brier_raw": brier_raw,
        "ece_raw": ece_raw,
        "temperature": T_opt,
        "brier_temp": brier_temp,
        "ece_temp": ece_temp,
        "brier_platt": brier_platt,
        "ece_platt": ece_platt,
        "brier_isotonic": brier_iso,
        "ece_isotonic": ece_iso,
    }


    out_json = {
        "run_id": run_id,
        "temperature": T_opt,
        "platt": {
            "coef": platt.coef_.tolist(),
            "intercept": platt.intercept_.tolist(),
        },
        "isotonic": {
            "x_thresholds": iso.X_thresholds_.tolist(),
            "y_thresholds": iso.y_thresholds_.tolist(),
        },
        "metrics": result,
    }
    out_path = OUTPUT_DIR / f"calibration_{run_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2, ensure_ascii=False)

    print(f"Saved calibration for {run_id} to {out_path}")
    print(
        f"Brier (probs): raw={brier_raw:.4f}, temp={brier_temp:.4f}, "
        f"Platt={brier_platt:.4f}, Isotonic={brier_iso:.4f}"
    )
    print(
        f"ECE: raw={ece_raw:.4f}, temp={ece_temp:.4f}, "
        f"Platt={ece_platt:.4f}, Isotonic={ece_iso:.4f}  (T={T_opt:.4f})"
    )

    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Fit calibration (Platt / Isotonic) for reranker scores of a given run_id "
            "and report Brier scores before/after."
        )
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="run_id (stem of results/<run_id>.json) to calibrate",
    )
    args = parser.parse_args()

    calibrate_run(args.run_id)


if __name__ == "__main__":
    main()

