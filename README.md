````markdown
# cold-start-algorithm

End-to-end, reproducible evaluation pipeline for recommendation algorithms in **cold-start** and **few-shot personalization** settings.  
The project focuses on accuracy (HR@K, nDCG@K, MRR, MAP) *and* beyond-accuracy properties such as serendipity/novelty, catalog coverage, and exposure bias.

## TL;DR (Quickstart)

1) Download datasets and place them into `data/` (see **Datasets**).  
2) Install dependencies (`requirements.txt`).  
3) Run the full pipeline:

```bash
python -m tools.full_pipeline --clean --fast --rebuild-gt
````

---

## Requirements

* Python **3.12.7** (used during development)
* `pip`
* OS: Linux / macOS / Windows (WSL recommended if you run into dependency issues)

---

## Installation

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/nikita-zmanovskiy/cold-start-algorithm
cd cold-start-algorithm
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## Datasets

You need to download 3 datasets and place them into specific folders **inside this repository**.

### 1) MovieLens 25M

* Download: [https://files.grouplens.org/datasets/movielens/ml-25m.zip](https://files.grouplens.org/datasets/movielens/ml-25m.zip)
* Unpack to: `data/movieLens/ml-25m/`

Expected files (at minimum):

* `data/movieLens/ml-25m/ratings.csv`
* `data/movieLens/ml-25m/movies.csv`

Example (Linux/macOS):

```bash
mkdir -p data/movieLens
unzip /path/to/ml-25m.zip -d data/movieLens
# This should create: data/movieLens/ml-25m/...
```

### 2) Serendipity-2018 (SAC 2018)

* Download page: [https://grouplens.org/datasets/serendipity-2018/](https://grouplens.org/datasets/serendipity-2018/)
* Unpack to: `data/serendipity-sac2018/`

Example layout:

```text
data/serendipity-sac2018/
  (dataset files as provided by GroupLens)
```

> Note: The Serendipity dataset download may require following the instructions on the official dataset page.

### 3) Taobao-Serendipity

* Source: [https://github.com/greenblue96/Taobao-Serendipity-Dataset](https://github.com/greenblue96/Taobao-Serendipity-Dataset)
* Put into: `data/Taobao-Serendipity-Dataset-master/`

Option A (recommended): clone directly into the target folder

```bash
mkdir -p data
git clone https://github.com/greenblue96/Taobao-Serendipity-Dataset data/Taobao-Serendipity-Dataset-master
```

Option B: download ZIP from GitHub and unpack it so the final folder name is:

```text
data/Taobao-Serendipity-Dataset-master/
```

---

## Reproducing Results

### Run the end-to-end pipeline (recommended)

From the repository root:

```bash
python -m tools.full_pipeline --clean --fast --rebuild-gt
```

What the flags mean:

* `--clean`
  Starts from a clean state (removes/overwrites prior artifacts so results are reproducible).

* `--fast`
  Runs a **faster sanity-check** configuration (smaller evaluation setup) so you can verify everything works end-to-end.

* `--rebuild-gt`
  Rebuilds ground-truth / splits artifacts required for evaluation.

### Full (slow) reproduction

If you want a heavier run (more exhaustive than `--fast`), run without `--fast`:

```bash
python -m tools.full_pipeline --clean --rebuild-gt
```

---

## Outputs

After the pipeline finishes, the project will write artifacts such as:

* run logs / per-run metrics (e.g., `runs.jsonl`)
* aggregated results tables (CSV/JSON)
* figures/plots for analysis

(Exact output paths are defined by the pipeline scripts.)

---

## Troubleshooting

### “File not found” for datasets

Double-check the folder names and paths are exactly:

```text
data/movieLens/ml-25m/...
data/serendipity-sac2018/...
data/Taobao-Serendipity-Dataset-master/...
```

## Citation

If you use this codebase in academic work, please cite:

* Zenodo DOI: [https://doi.org/10.5281/zenodo.18772321](https://doi.org/10.5281/zenodo.18772321)

You can also use the provided `CITATION.cff`.

---

## License

See `LICENSE`

[1]: https://raw.githubusercontent.com/nikita-zmanovskiy/cold-start-algorithm/master/src/preprocess.py "raw.githubusercontent.com"
[2]: https://raw.githubusercontent.com/nikita-zmanovskiy/cold-start-algorithm/master/tools/full_pipeline.py "raw.githubusercontent.com"
[3]: https://raw.githubusercontent.com/nikita-zmanovskiy/cold-start-algorithm/master/CITATION.cff "raw.githubusercontent.com"
