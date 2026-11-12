
import os
import pandas as pd
from pathlib import Path
from .config import DATA_DIR, PROCESSED_DIR
from .utils import logger, save_json

def prepare_serendipity2018(src_dir: Path, out_dir: Path):

    src_dir = Path(src_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Preparing Serendipity-2018...")


    items = []
    interactions = []
    users = []

    possible_movies = list(src_dir.glob("**/movies*.csv")) + list(src_dir.glob("**/movies*.tsv"))
    # Serendipity-SAC2018 uses training.csv (userId, movieId, rating); also check ratings*.csv
    possible_ratings = (
        list(src_dir.glob("**/ratings*.csv")) + list(src_dir.glob("**/ratings*.tsv")) +
        list(src_dir.glob("**/training*.csv")) + list(src_dir.glob("**/training*.tsv"))
    )
    if possible_movies:
        movies = pd.read_csv(
            possible_movies[0],
            sep=",",
            engine="python",
            quotechar='"',
            escapechar="\\",
            on_bad_lines="skip",
            encoding="utf-8",
        )
        movies = movies.rename(columns={c: c.strip() for c in movies.columns})
        if 'movieId' in movies.columns:
            movies = movies[['movieId','title','genres']].rename(columns={'movieId':'item_id'})
        movies.to_csv(out_dir / "items_serendipity.csv", index=False)
        logger.info(f"Wrote items_serendipity.csv, shape={movies.shape}")
    else:
        if not (out_dir / "items_serendipity.csv").exists():
            logger.warning("movies file not found for Serendipity dataset; check your data folder")
        else:
            logger.debug("movies file not found; using existing items_serendipity.csv")

    if possible_ratings:
        ratings = pd.read_csv(possible_ratings[0], sep=None, engine='python')
        if 'userId' in ratings.columns:
            ratings = ratings.rename(columns={'userId': 'user_id', 'movieId': 'item_id', 'rating': 'rating'})
        ratings.to_csv(out_dir / "interactions_serendipity.csv", index=False)
        logger.info(f"Wrote interactions_serendipity.csv, shape={ratings.shape}")
    else:
        if not (out_dir / "interactions_serendipity.csv").exists():
            logger.warning("ratings/training file not found for Serendipity dataset; check your data folder")
        else:
            logger.debug("ratings/training file not found; using existing interactions_serendipity.csv")

def prepare_taobao(src_dir: Path, out_dir: Path):

    src_dir = Path(src_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not src_dir.exists():
        logger.warning("Taobao dataset folder not found (%s); skipping.", src_dir)
        return
    logger.info("Preparing Taobao-Serendipity...")

    csvs = list(src_dir.glob("**/*.csv")) + list(src_dir.glob("**/*.tsv"))
    if not csvs:
        if not (out_dir / "interactions_taobao.csv").exists():
            logger.warning("No CSV/TSV in Taobao folder (%s); skipping.", src_dir)
        return

    written = False
    for f in csvs:
        try:
            try:
                df = pd.read_csv(f, sep=None, engine="python", dtype=str)
            except Exception:
                df = pd.read_csv(f, sep=",", engine="c", low_memory=False)  # fallback: обычный csv

            def _pick_col(cols_lower, candidates):
                for cand in candidates:
                    for i, c in enumerate(cols_lower):
                        if c == cand or c.endswith(cand) or cand in c:
                            return i
                return None

            cols = list(df.columns)
            cols_lower = [str(c).lower() for c in cols]

            # --- pick user/item columns (supports UserId/ItemId etc.) ---
            u_idx = _pick_col(cols_lower, ["user_id", "userid", "user", "uid", "userId".lower()])
            i_idx = _pick_col(cols_lower, ["item_id", "itemid", "item", "goods_id", "product_id", "movieid", "itemId".lower()])
            t_idx = _pick_col(cols_lower, ["timestamp", "time", "ts", "datetime"])

            if u_idx is None or i_idx is None:
                # not an interactions file; try next
                logger.warning("Skipping %s: can't infer user/item columns from %s", f.name, cols)
                continue

            u_col = cols[u_idx]
            i_col = cols[i_idx]

            # --- keep only needed cols ---
            keep = [u_col, i_col] + ([cols[t_idx]] if t_idx is not None else [])
            out = df[keep].copy()

            out = out.rename(columns={u_col: "user_id", i_col: "item_id"})
            if t_idx is not None:
                out = out.rename(columns={cols[t_idx]: "timestamp"})
            else:
                out["timestamp"] = ""

            # --- optional: keep only positive interactions (Clicked OR Purchased) ---

            def _normalize_binary(series: pd.Series) -> pd.Series:
                # Handles: 1/0, "1", "1.0", " 1", True/False, "true"/"false", etc.
                s = series.astype(str).str.strip()
                num = pd.to_numeric(s, errors="coerce")
                if num.notna().any():
                    return (num.fillna(0) > 0).astype("int8")
                s2 = s.str.lower()
                return s2.isin({"1", "true", "t", "yes", "y"}).astype("int8")

            def _find_col(df_cols, target_lower: str):
                # strict match after strip/lower (works with spaces like "Purchase intention")
                for c in df_cols:
                    if str(c).strip().lower() == target_lower:
                        return c
                return None

            clicked_col = _find_col(df.columns, "clicked")
            purchased_col = _find_col(df.columns, "purchased")

            mask = None

            if clicked_col is not None:
                clicked = _normalize_binary(df[clicked_col])
                logger.info("Taobao: Clicked normalized value_counts=%s", clicked.value_counts(dropna=False).to_dict())
                mask = (clicked == 1)

            if purchased_col is not None:
                purchased = _normalize_binary(df[purchased_col])
                logger.info("Taobao: Purchased normalized value_counts=%s", purchased.value_counts(dropna=False).to_dict())
                mask = (purchased == 1) if mask is None else (mask | (purchased == 1))

            if mask is not None:
                positives = out.loc[mask].copy()

                # больше НЕ оставляем "все строки": иначе нули превращаются в "позитив"
                if positives.empty:
                    raise ValueError(
                        "Taobao: 0 positives after (Clicked OR Purchased) filtering. "
                        "Check labels in CSV."
                    )

                if len(positives) < 5000:
                    logger.warning(
                        "Taobao: only %d positive interactions after (Clicked OR Purchased) filtering. "
                        "Proceeding with positives only (metrics may be noisy).",
                        len(positives),
                    )

                out = positives
            else:
                logger.warning("Taobao: no Clicked/Purchased columns found; keeping all rows (check parser).")

            # --- end positives handling ---
            out["user_id"] = out["user_id"].astype(str)
            out["item_id"] = out["item_id"].astype(str)

            # --- write interactions ---
            interactions_path = out_dir / "interactions_taobao.csv"
            out[["user_id", "item_id", "timestamp"]].to_csv(interactions_path, index=False)

            # --- write items (minimal metadata) ---
            items_path = out_dir / "items_taobao.csv"
            items_df = (
                out[["item_id"]]
                .drop_duplicates()
                .assign(title=lambda x: "Item " + x["item_id"].astype(str))
            )
            items_df.to_csv(items_path, index=False)

            logger.info(
                "Taobao: wrote %s (rows=%d) and %s (items=%d) from %s",
                interactions_path.name, len(out),
                items_path.name, len(items_df),
                f.name,
            )

            written = True
            break

        except Exception as e:
            logger.warning("Failed to parse %s: %s", f.name, e)
            continue

    if not written and not (out_dir / "interactions_taobao.csv").exists():
        logger.warning("No interactions-like CSV/TSV in Taobao folder (%s); skipping.", src_dir)

def prepare_movielens25m(min_rating: float = 4.0):
    logger.info("Preparing MovieLens-25M...")
    ml_dir = DATA_DIR / "movieLens" / "ml-25m"

    ratings_path = ml_dir / "ratings.csv"
    movies_path = ml_dir / "movies.csv"

    if not ratings_path.exists() or not movies_path.exists():
        logger.warning(f"MovieLens not found at {ml_dir}. Expected ratings.csv and movies.csv")
        return

    # Read only required columns to reduce RAM usage
    ratings = pd.read_csv(
        ratings_path,
        usecols=["userId", "movieId", "rating", "timestamp"],
        dtype={"userId": "int32", "movieId": "int32", "rating": "float32", "timestamp": "int64"},
    )

    # Keep only positive interactions (implicit)
    ratings = ratings[ratings["rating"] >= float(min_rating)].copy()

    ratings.rename(columns={"userId": "user_id", "movieId": "item_id"}, inplace=True)
    ratings["user_id"] = ratings["user_id"].astype(str)
    ratings["item_id"] = ratings["item_id"].astype(str)

    interactions = ratings[["user_id", "item_id", "timestamp", "rating"]]
    out_inter = PROCESSED_DIR / "interactions_movielens.csv"
    interactions.to_csv(out_inter, index=False)
    logger.info(f"Wrote interactions_movielens.csv, shape={interactions.shape}")

    movies = pd.read_csv(movies_path, usecols=["movieId", "title", "genres"])
    movies.rename(columns={"movieId": "item_id"}, inplace=True)
    movies["item_id"] = movies["item_id"].astype(str)
    movies["title"] = movies["title"].fillna("").astype(str)
    movies["genres"] = movies["genres"].fillna("").astype(str)

    # Make a robust text field for embeddings/enrichment
    movies["text"] = movies["title"] + " Genres: " + movies["genres"]

    items = movies[["item_id", "title", "genres", "text"]]
    out_items = PROCESSED_DIR / "items_movielens.csv"
    items.to_csv(out_items, index=False)
    logger.info(f"Wrote items_movielens.csv, shape={items.shape}")


def run_all(datasets=None):
    """
    datasets:
      - None -> preprocess all supported datasets
      - str  -> preprocess only that dataset
      - list[str] -> preprocess listed datasets
    """
    if datasets is None:
        datasets = ["serendipity", "taobao", "movielens"]
    elif isinstance(datasets, str):
        datasets = [datasets]

    ds = {d.lower() for d in datasets}

    if "serendipity" in ds:
        seren_path = DATA_DIR / "serendipity-sac2018"
        prepare_serendipity2018(seren_path, PROCESSED_DIR)
    if "taobao" in ds:
        taobao_path = DATA_DIR / "Taobao-Serendipity-Dataset-master"
        prepare_taobao(taobao_path, PROCESSED_DIR)
    if "movielens" in ds:
        prepare_movielens25m()
    logger.info("Preprocessing finished. Check data/processed/ for outputs.")

if __name__ == "__main__":
    run_all()
