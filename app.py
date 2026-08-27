"""Interactive demo for the image-segmentation pipeline.

Run with:
    streamlit run app.py

Lets you pick a scene, point it at a data folder (defaults to ./data),
run that scene's pipeline, and view the resulting figure plus metrics
without touching the command line.
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from imgseg import pipeline  # noqa: E402

st.set_page_config(page_title="Image Segmentation Pipeline", layout="wide")

st.title("🖼️ Multi-Scene Image Segmentation — Interactive Demo")
st.caption(
    "Classical computer-vision pipelines (GrabCut, SLIC, K-Means, DBSCAN) "
    "applied to four different segmentation problems, each evaluated against "
    "a ground-truth mask with IoU / Dice / Accuracy."
)

with st.sidebar:
    st.header("Configuration")
    scene_key = st.selectbox(
        "Scene",
        options=[s.key for s in pipeline.SCENES],
        format_func=lambda k: f"Scene {k} — {pipeline.SCENES_BY_KEY[k].title}",
    )
    data_dir = st.text_input("Data folder", value="data")
    run_button = st.button("▶ Run pipeline", type="primary", use_container_width=True)

spec = pipeline.SCENES_BY_KEY[scene_key]

st.subheader(f"Scene {spec.key} — {spec.title}")
st.write(f"Module: `imgseg/{spec.module.__name__.split('.')[-1]}.py`")

if run_button:
    with tempfile.TemporaryDirectory() as out_dir:
        with st.spinner("Running pipeline…"):
            log_buffer = io.StringIO()
            old_stdout = sys.stdout
            sys.stdout = log_buffer
            success = True
            try:
                pipeline.run_scene(spec, data_dir=data_dir, out_dir=out_dir, no_show=True)
            except Exception as exc:  # noqa: BLE001
                success = False
                st.error(f"Pipeline failed: {exc}")
            finally:
                sys.stdout = old_stdout

        st.text_area("Pipeline log", log_buffer.getvalue(), height=200)

        if success:
            out_path = Path(out_dir)
            images = sorted(out_path.rglob("*.png"))
            figures = [p for p in images if "figures" in p.parts or p.name.endswith("_result.png")]
            masks = [p for p in images if "masks" in p.parts]

            if figures:
                st.subheader("Result")
                for fig_path in figures:
                    st.image(str(fig_path), use_column_width=True)

            json_results = list(out_path.rglob("*.json"))
            if json_results:
                st.subheader("Metrics")
                for jp in json_results:
                    st.json(json.loads(jp.read_text()))

            csv_results = list(out_path.rglob("*.csv"))
            for cp in csv_results:
                st.subheader(f"Table — {cp.name}")
                st.dataframe(cp.read_text())

            if masks:
                with st.expander(f"Intermediate masks ({len(masks)})"):
                    cols = st.columns(min(4, len(masks)))
                    for i, mp in enumerate(masks):
                        cols[i % len(cols)].image(str(mp), caption=mp.name, use_column_width=True)
else:
    st.info(
        f"Set the data folder in the sidebar (must contain the images for "
        f"Scene {spec.key} — see `data/README.md`), then click **Run pipeline**."
    )

st.divider()
with st.expander("About this project"):
    st.markdown(
        """
This app is a thin interactive layer over the `imgseg` Python package — the
same code also runs headlessly from the command line via:

```bash
python -m imgseg.pipeline --scene all --data-dir data --out-dir outputs --no-show
```

Each scene is an independent segmentation pipeline (see the README for
details on the techniques used per scene: GrabCut, SLIC superpixels,
K-Means, DBSCAN, LAB/HSV color-space engineering, morphological
post-processing) sharing common I/O and evaluation utilities
(`imgseg/common/`).
        """
    )
