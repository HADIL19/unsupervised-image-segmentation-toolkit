"""Scene 2 — Optic disc extraction from a fundus (retina) image.

Compares four binarization strategies (fixed threshold, Otsu, percentile,
K-Means) and automatically keeps the one with the highest IoU against the
ground-truth mask.

Usage:
    python src/scene2_optic_disc.py
    python src/scene2_optic_disc.py --data-dir data --out-dir outputs --no-show
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from sklearn.cluster import KMeans

from .common.io_utils import largest_component, read_rgb
from .common.metrics import evaluate

SEED = 42


# ==============================================================
# Image helpers
# ==============================================================

def vers_gris(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img.astype(np.uint8)
    return (0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]).astype(np.uint8)


def lisser(img: np.ndarray, k: int = 5, s: float = 1.0) -> np.ndarray:
    return cv2.GaussianBlur(img, (k, k), sigmaX=s)


def binariser_fixe(img: np.ndarray, t: int) -> np.ndarray:
    g = vers_gris(img)
    return np.where(g >= t, 255, 0).astype(np.uint8)


def binariser_otsu(img: np.ndarray) -> tuple[np.ndarray, float]:
    g = vers_gris(img)
    t, result = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return result.astype(np.uint8), float(t)


def binariser_percentile(img: np.ndarray, p: int = 94) -> tuple[np.ndarray, float]:
    g = vers_gris(img)
    t = float(np.percentile(g[g > 0], p))
    return np.where(g >= t, 255, 0).astype(np.uint8), t


def binariser_kmeans(img: np.ndarray, k: int = 2) -> np.ndarray:
    g = vers_gris(img)
    pixels = g.reshape(-1, 1).astype(np.float32)
    pmin, pmax = pixels.min(), pixels.max()
    pn = (pixels - pmin) / (pmax - pmin) if pmax > pmin else pixels
    etiq = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit_predict(pn).reshape(g.shape)
    moyennes = [g[etiq == c].mean() for c in range(k)]
    return np.where(etiq == int(np.argmax(moyennes)), 255, 0).astype(np.uint8)


def gt_vers_binaire(gt: np.ndarray) -> np.ndarray:
    g = vers_gris(gt)
    etiq = KMeans(n_clusters=2, random_state=SEED, n_init=10).fit_predict(
        g.reshape(-1, 1).astype(np.float32)).reshape(g.shape)
    moy = [g[etiq == c].mean() for c in range(2)]
    return (etiq == int(np.argmax(moy))).astype(np.uint8)


def superposer(img: np.ndarray, masque: np.ndarray, couleur=(50, 220, 120), alpha: float = 0.5) -> np.ndarray:
    fond = img[:, :, :3].astype(np.float32)
    result = fond.copy()
    result[masque > 0] = (1 - alpha) * result[masque > 0] + alpha * np.array(couleur, np.float32)
    return np.clip(result, 0, 255).astype(np.uint8)


# ==============================================================
# CLI
# ==============================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "data"), help="Folder containing Scene_2.png and GT2.png")
    parser.add_argument("--out-dir", default=os.environ.get("OUT_DIR", "outputs"), help="Folder to write masks/figures/metrics to")
    parser.add_argument("--img-name", default="Scene_2.png")
    parser.add_argument("--gt-name", default="GT2.png")
    parser.add_argument("--no-show", action="store_true", help="Don't open a matplotlib window (useful in CI / headless runs)")
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def run(args: argparse.Namespace) -> None:

    out_masks = Path(args.out_dir) / "masks"
    out_figures = Path(args.out_dir) / "figures"
    out_results = Path(args.out_dir) / "resultats"
    for d in (out_masks, out_figures, out_results):
        d.mkdir(parents=True, exist_ok=True)

    img_path = os.path.join(args.data_dir, args.img_name)
    gt_path = os.path.join(args.data_dir, args.gt_name)

    print("=" * 50)
    print("  Scene 2 - Optic disc extraction")
    print("=" * 50)

    img_orig = read_rgb(img_path)
    H, W = img_orig.shape[:2]
    img_lisse = lisser(img_orig)
    print(f"  Image: {W}x{H} px")

    # --- Candidate segmentations ---
    _, t_otsu = binariser_otsu(img_lisse)
    _, t_p94 = binariser_percentile(img_lisse, 94)

    masques_bruts = {
        "Otsu": binariser_otsu(img_lisse)[0],
        "Seuil 128": binariser_fixe(img_lisse, 128),
        "Percentile94": binariser_percentile(img_lisse, 94)[0],
        "KMeans k=2": binariser_kmeans(img_lisse, k=2),
    }
    masques = {nom: largest_component(m) for nom, m in masques_bruts.items()}

    # --- Ground truth ---
    gt_img = read_rgb(gt_path)
    if gt_img.shape[:2] != (H, W):
        gt_img = cv2.resize(gt_img, (W, H), interpolation=cv2.INTER_NEAREST)
    gt_bin = gt_vers_binaire(gt_img)

    # --- Evaluation ---
    resultats = []
    for nom, masque in masques.items():
        s = evaluate(gt_bin, masque > 0)
        row = s.as_dict()
        row["methode"] = nom
        row["f1"] = row.pop("dice")  # keep historical column name
        resultats.append(row)
        print(f"  {nom:14s} | IoU={s.iou:.3f}  F1={s.dice:.3f}  Acc={s.accuracy:.3f}")

    df = pd.DataFrame(resultats).sort_values("iou", ascending=False).reset_index(drop=True)
    meilleure = df.iloc[0]["methode"]
    masque_best = masques[meilleure]
    best_s = df.iloc[0]
    print(f"\n  Best method: {meilleure}  (IoU={best_s['iou']:.3f})")

    # --- Visualization ---
    fig = plt.figure(figsize=(20, 13), facecolor="#0f0f1a")
    fig.suptitle("SCENE 2 - Optic disc segmentation", fontsize=16, fontweight="bold", color="white", y=0.98)

    gs = GridSpec(2, 4, figure=fig, hspace=0.38, wspace=0.12, left=0.03, right=0.97, top=0.93, bottom=0.04)
    title_kw = dict(fontsize=10, fontweight="bold", color="white", pad=6)

    def show(ax, data, titre, cmap=None, border_color="#444466"):
        ax.imshow(data, cmap=cmap)
        ax.set_title(titre, **title_kw)
        ax.axis("off")
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color(border_color)
            sp.set_linewidth(2)

    show(fig.add_subplot(gs[0, 0]), img_orig, "Original image")
    show(fig.add_subplot(gs[0, 1]), vers_gris(img_orig), "Grayscale", cmap="gray")
    show(fig.add_subplot(gs[0, 2]), gt_bin, "Ground Truth", cmap="gray", border_color="#22aaff")
    show(fig.add_subplot(gs[0, 3]), superposer(img_orig, masque_best), f"Overlay - {meilleure}", border_color="#22ff88")

    diff = np.zeros((*masque_best.shape, 3), dtype=np.uint8)
    diff[np.logical_and(gt_bin.astype(bool), (masque_best > 0))] = [50, 220, 80]
    diff[np.logical_and(~gt_bin.astype(bool), (masque_best > 0))] = [220, 60, 60]
    diff[np.logical_and(gt_bin.astype(bool), ~(masque_best > 0))] = [60, 120, 220]

    show(fig.add_subplot(gs[1, 0]), masques[meilleure], f"Final mask - {meilleure}", cmap="gray", border_color="#22ff88")
    show(fig.add_subplot(gs[1, 1]), diff, "TP(green) / FP(red) / FN(blue)", border_color="#ffaa00")

    ax_m = fig.add_subplot(gs[1, 2:])
    ax_m.set_facecolor("#13132a")
    ax_m.axis("off")
    for sp in ax_m.spines.values():
        sp.set_visible(True)
        sp.set_color("#22ff88")
        sp.set_linewidth(2)

    ax_m.text(
        0.5, 0.95, f"METRICS - {meilleure}", ha="center", va="top",
        fontsize=12, fontweight="bold", color="#22ff88", transform=ax_m.transAxes,
    )

    labels = ["IoU", "Dice/F1", "Accuracy", "Precision", "Recall"]
    vals = [best_s["iou"], best_s["f1"], best_s["accuracy"], best_s["precision"], best_s["recall"]]
    colors = ["#f1c40f", "#2ecc71", "#3498db", "#e67e22", "#9b59b6"]

    for i, (lbl, val, col) in enumerate(zip(labels, vals, colors)):
        y = 0.76 - i * 0.14
        ax_m.text(0.02, y, lbl, ha="left", va="center", fontsize=13, color=col, fontweight="bold", transform=ax_m.transAxes)
        ax_m.text(0.98, y, f"{val:.4f}", ha="right", va="center", fontsize=13, color="white", fontweight="bold", transform=ax_m.transAxes)
        ax_m.add_patch(plt.Rectangle((0.15, y - 0.05), 0.78 * val, 0.07, transform=ax_m.transAxes, color=col, alpha=0.4, clip_on=False))
        ax_m.add_patch(plt.Rectangle(
            (0.15, y - 0.05), 0.78, 0.07, transform=ax_m.transAxes,
            fill=False, edgecolor=col, alpha=0.3, linewidth=1, clip_on=False,
        ))

    fig_path = out_figures / "scene2_resultat_final.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    if not args.no_show:
        plt.show()
    plt.close(fig)
    print(f"\n  -> {fig_path}")

    # --- Save masks + metrics ---
    for nom, masque in masques.items():
        nf = nom.lower().replace(" ", "_")
        cv2.imwrite(str(out_masks / f"scene2_{nf}.png"), masque)
    cv2.imwrite(str(out_masks / "scene2_best.png"), masque_best)
    cv2.imwrite(str(out_masks / "scene2_gt.png"), (gt_bin * 255).astype(np.uint8))

    cols = ["methode", "iou", "f1", "accuracy", "precision", "recall"]
    df[cols].to_csv(out_results / "scene2_metriques.csv", index=False)

    with open(out_results / "scene2_resultats.json", "w", encoding="utf-8") as f:
        json.dump({"meilleure": meilleure, "seuil_otsu": t_otsu, "seuil_p94": t_p94, "scores": resultats}, f, indent=2)

    print(f"  -> {out_masks}/   + {out_results}/")
    print("\n[Done]")


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
