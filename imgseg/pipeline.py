"""Unified pipeline CLI for the image-segmentation project.

Runs one, several, or all four scene pipelines from a single entry point,
each with its own algorithm (GrabCut+SLIC+K-Means, thresholding ensemble,
DBSCAN, depth-aware K-Means) but a shared CLI shape, shared I/O and metrics
utilities, and shared output layout.

Examples:
    python -m imgseg.pipeline --scene all --data-dir data --out-dir outputs --no-show
    python -m imgseg.pipeline --scene 2 4 --data-dir data --no-show
    python -m imgseg.pipeline --list
"""

from __future__ import annotations

import argparse
import sys
import traceback
from dataclasses import dataclass

from . import scene1_cat_segmentation as scene1
from . import scene2_optic_disc as scene2
from . import scene3_road_detection as scene3
from . import scene4_person_extraction as scene4


@dataclass
class SceneSpec:
    key: str
    title: str
    module: object


SCENES: list[SceneSpec] = [
    SceneSpec("1", "Cat / Sky / Ground / Trees (GrabCut + SLIC + K-Means)", scene1),
    SceneSpec("2", "Optic disc extraction (thresholding ensemble)", scene2),
    SceneSpec("3", "Road detection (DBSCAN)", scene3),
    SceneSpec("4", "Person silhouette from RGB-D (K-Means)", scene4),
]
SCENES_BY_KEY = {s.key: s for s in SCENES}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--scene",
        nargs="+",
        default=["all"],
        help="Which scene(s) to run: one or more of 1 2 3 4, or 'all' (default).",
    )
    parser.add_argument("--data-dir", default="data", help="Folder containing all scene images/ground truths.")
    parser.add_argument("--out-dir", default="outputs", help="Folder to write all scene outputs to.")
    parser.add_argument("--no-show", action="store_true", help="Don't open matplotlib windows (recommended for batch/CI runs).")
    parser.add_argument("--list", action="store_true", help="List available scenes and exit.")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep running remaining scenes if one fails (useful when only some data files are present).",
    )
    return parser


def resolve_scenes(keys: list[str]) -> list[SceneSpec]:
    if "all" in keys:
        return SCENES
    resolved = []
    for k in keys:
        if k not in SCENES_BY_KEY:
            valid = ", ".join(SCENES_BY_KEY)
            raise SystemExit(f"Unknown scene '{k}'. Valid options: {valid}, all")
        resolved.append(SCENES_BY_KEY[k])
    return resolved


def run_scene(spec: SceneSpec, data_dir: str, out_dir: str, no_show: bool) -> bool:
    """Run a single scene module's pipeline. Returns True on success."""
    print(f"\n### Scene {spec.key} — {spec.title} ###")
    scene_parser: argparse.ArgumentParser = spec.module.build_parser()
    cli_args = ["--data-dir", data_dir, "--out-dir", out_dir]
    if no_show:
        cli_args.append("--no-show")
    args = scene_parser.parse_args(cli_args)
    try:
        spec.module.run(args)
        return True
    except FileNotFoundError as exc:
        print(f"  [skipped] {exc}")
        return False


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list:
        for s in SCENES:
            print(f"  {s.key}: {s.title}")
        return 0

    scenes_to_run = resolve_scenes(args.scene)
    results: dict[str, bool] = {}

    for spec in scenes_to_run:
        try:
            results[spec.key] = run_scene(spec, args.data_dir, args.out_dir, args.no_show)
        except Exception:
            results[spec.key] = False
            print(f"  [error] Scene {spec.key} raised an exception:")
            traceback.print_exc()
            if not args.continue_on_error:
                break

    print("\n=== Pipeline summary ===")
    for spec in scenes_to_run:
        status = "OK" if results.get(spec.key) else "FAILED / SKIPPED"
        print(f"  Scene {spec.key}: {status}")

    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
