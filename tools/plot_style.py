from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Optional

AXES_FONTSIZE = 9
TITLE_FONTSIZE = 11
LEGEND_FONTSIZE = 8

MODEL_DISPLAY_NAMES = {
    "random": "Random",
    "popularity": "Popularity",
    "embedding_cosine": "EmbCos",
    "candidates_only": "CandOnly",
    "ours_with_reranker": "Ours+Rerank",
}

MODEL_ORDER = ["random", "popularity", "embedding_cosine", "candidates_only", "ours_with_reranker"]


def apply_paper_style():
    plt.rcParams["font.size"] = AXES_FONTSIZE
    plt.rcParams["axes.titlesize"] = TITLE_FONTSIZE
    plt.rcParams["axes.labelsize"] = AXES_FONTSIZE
    plt.rcParams["xtick.labelsize"] = AXES_FONTSIZE
    plt.rcParams["ytick.labelsize"] = AXES_FONTSIZE
    plt.rcParams["legend.fontsize"] = LEGEND_FONTSIZE


def short_name(model_key: str) -> str:
    return MODEL_DISPLAY_NAMES.get(model_key, model_key.replace("_", " ").title())


def bootstrap_ci(
    data: List[float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
) -> Tuple[Optional[float], Optional[float]]:
    if not data or len(data) == 0:
        return None, None
    data = np.asarray(data)
    n = len(data)
    bootstrap_means = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        bootstrap_means.append(float(np.mean(data[idx])))
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
    return float(lower), float(upper)


def save_fig_paper(path_base: Path, fmt: str = "both") -> None:
    path_base.parent.mkdir(parents=True, exist_ok=True)
    if fmt in ("both", "pdf"):
        plt.savefig(path_base.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    if fmt in ("both", "svg"):
        plt.savefig(path_base.with_suffix(".svg"), dpi=300, bbox_inches="tight")
    if fmt == "png":
        plt.savefig(path_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    # Always save PNG for quick preview
    if fmt != "png":
        plt.savefig(path_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
