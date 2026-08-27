# Unsupervised Multi-Domain Image Segmentation — Medical, Aerial & Scene Understanding

**Repo name:** `unsupervised-image-segmentation-toolkit`

A classical computer-vision + unsupervised machine-learning toolkit that segments four visually and structurally different scenes (natural photo, medical fundus image, aerial photo, RGB-D scan) — each with its own algorithm, all sharing a common CLI, evaluation framework, and package structure.

Built to demonstrate practical skills in **computer vision, unsupervised ML, and applied data science**: designing features by hand (color-space engineering, gradients, texture, depth), choosing the right clustering/segmentation algorithm per problem, evaluating rigorously against ground truth, and packaging the result as installable, tested, documented software rather than a one-off notebook.
## Tech stack

**Language:** Python 3.10+

- **OpenCV** — GrabCut, Sobel gradients, connected components, morphology, thresholding, contour detection
- **scikit-learn** — K-Means, DBSCAN, `StandardScaler`
- **scikit-image** — SLIC superpixels, morphology, connected-component labeling, color-space conversion
- **NumPy / SciPy** — feature engineering, grey dilation, statistics
- **Matplotlib** — result visualizations
- **Pandas** — metrics tables / CSV export
- **Streamlit** — interactive demo
- **pytest / flake8 / GitHub Actions** — testing and CI
## Try it

```bash
git clone https://github.com/<your-username>/unsupervised-image-segmentation-toolkit.git
cd unsupervised-image-segmentation-toolkit
pip install -e ".[dev,demo]"

# put your images in data/ (see data/README.md), then either:

# 1. Command line — run everything
python -m imgseg.pipeline --scene all --data-dir data --out-dir outputs --no-show

# 2. Interactive demo
streamlit run app.py
```

## What this project demonstrates

| Skill area | Where it shows up |
|---|---|
| **Unsupervised ML** | K-Means (4 different problems), DBSCAN with feature-space scaling, cluster selection via hand-crafted scoring functions |
| **Classical computer vision** | GrabCut, SLIC superpixels, Sobel gradients, morphological operations, connected-component analysis, directional structuring elements |
| **Feature engineering** | LAB/HSV color spaces, depth maps, spatial coordinates, local texture statistics, edge magnitude — combined per-scene into task-specific feature vectors |
| **Rigorous evaluation** | Every scene scored against ground truth with IoU, Dice/F1, Accuracy, Precision, Recall via one shared, unit-tested metrics module |
| **Software engineering** | Installable Python package (`pyproject.toml`), shared CLI interface across independent pipelines, `pytest` test suite, GitHub Actions CI (lint + test on every push), no code duplication between scenes |
| **Product thinking** | A Streamlit demo turns four research scripts into something a non-technical reviewer can click through in under a minute |

## Scenes

| # | Problem | Domain | Core algorithm |
|---|---|---|---|
| 1 | Segment a cat from sky/ground/trees | General scene understanding | GrabCut (seeded by color cues) → SLIC superpixel refinement → K-Means (k=3) on the background |
| 2 | Extract the optic disc from a retina photo | Medical imaging | Four binarization strategies (Otsu, fixed, percentile, K-Means) compared automatically, best one kept by IoU |
| 3 | Detect the road in an aerial photo | Remote sensing | LAB color + texture + edge cues → DBSCAN clustering → directional morphological closing → shape/purity component filtering |
| 4 | Isolate a standing person | RGB-D / robotics-style perception | Depth + spatial + gradient features → K-Means (k=7) → automatic cluster selection by size/centrality/depth-consistency score |

Each scene is a self-contained module (`imgseg/sceneN_*.py`) exposing the same shape: `build_parser()`, `run(args)`, `main()` — so it works identically from the shell, from the unified pipeline, or imported directly in a notebook.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌────────────────────┐
│   app.py    │────▶│              │────▶│ scene1_cat_seg.py  │
│ (Streamlit) │     │  imgseg/     │     │ scene2_optic_disc  │
└─────────────┘     │ pipeline.py  │────▶│ scene3_road_detect │
                     │ (CLI runner)│     │ scene4_person_extr │
┌─────────────┐     │              │     └─────────┬──────────┘
│ CLI (shell) │────▶│              │               │
└─────────────┘     └──────────────┘               ▼
                                          ┌───────────────────────┐
                                          │ imgseg/common/        │
                                          │  io_utils.py          │
                                          │  metrics.py           │
                                          │ (shared, unit-tested) │
                                          └───────────────────────┘
```

## Project structure

```
.
├── app.py                     # Streamlit interactive demo
├── pyproject.toml             # installable package + pytest config
├── requirements.txt
├── LICENSE                    # MIT
├── .github/workflows/ci.yml   # lint (flake8) + test (pytest) on every push
├── data/README.md             # expected input files per scene
├── imgseg/                    # the actual package
│   ├── pipeline.py            # unified multi-scene CLI orchestrator
│   ├── common/
│   │   ├── io_utils.py        # shared image I/O + mask helpers
│   │   └── metrics.py         # shared IoU / Dice / Accuracy / Precision / Recall
│   ├── scene1_cat_segmentation.py
│   ├── scene2_optic_disc.py
│   ├── scene3_road_detection.py
│   └── scene4_person_extraction.py
└── tests/
    ├── test_metrics.py
    ├── test_io_utils.py
    └── test_pipeline.py
```

## Usage

### Unified CLI

```bash
python -m imgseg.pipeline --list                          # see available scenes
python -m imgseg.pipeline --scene 1                        # run one scene
python -m imgseg.pipeline --scene 2 4                       # run a subset
python -m imgseg.pipeline --scene all --continue-on-error   # run everything, skip missing data gracefully
```

Env vars `DATA_DIR` / `OUT_DIR`, or the `--data-dir` / `--out-dir` flags, control where images are read from and results are written to. Add `--no-show` for headless/CI runs.

### Individual scene scripts

Each scene also runs standalone with the same flags:

```bash
python -m imgseg.scene2_optic_disc --data-dir data --out-dir outputs --no-show
```

### Interactive demo

```bash
streamlit run app.py
```

Pick a scene, point it at your data folder, click **Run pipeline**, and view the result figure, metrics, and intermediate masks inline.

## Data

Images are not included in the repo (see `.gitignore`). Drop them into `data/` following the layout in [`data/README.md`](data/README.md).

## Evaluation

All four scenes are scored against ground-truth masks using the same confusion-matrix-based metrics, implemented once in `imgseg/common/metrics.py` and covered by unit tests:

- **IoU** = TP / (TP + FP + FN)
- **Dice / F1** = 2·TP / (2·TP + FP + FN)
- **Accuracy** = (TP + TN) / (TP + TN + FP + FN)
- **Precision** = TP / (TP + FP)
- **Recall** = TP / (TP + FN)

## Development

```bash
pip install -e ".[dev]"
pytest -v                 # run the test suite
flake8 imgseg tests app.py --max-line-length=140 --extend-ignore=E203,W503
```

CI runs both on every push/PR via GitHub Actions (`.github/workflows/ci.yml`).




