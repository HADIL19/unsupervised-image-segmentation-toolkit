"""Scene 1 — Multi-class scene segmentation (Cat / Sky / Ground / Trees).

Pipeline: LAB/HSV color-cue masks for "certain background" -> GrabCut
(two-phase, seeded with those cues) -> SLIC superpixel refinement -> tail
recovery via spatial contiguity -> K-Means (k=3) on the remaining
background pixels -> semantic cluster naming -> pixel-accuracy / mean-IoU
evaluation against a 4-class ground truth.

Usage:
    python src/scene1_cat_segmentation.py
    python src/scene1_cat_segmentation.py --data-dir data --out-dir outputs --no-show
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.stats import mode
from skimage import color, morphology, segmentation
from skimage.measure import label as sk_label
from sklearn.cluster import KMeans

SEED = 42

# Manually annotated bounding box around the cat's body (not the tail).
# This is a one-off calibration for this specific image, not derived from
# the ground truth — it only seeds GrabCut, which then refines the shape.
CAT_RECT = (14, 103, 439, 313)  # (x, y, w, h)

PALETTE = {
    "Cat": [245, 210, 75],
    "Sky": [100, 185, 235],
    "Ground": [85, 170, 55],
    "Trees": [30, 95, 30],
}


# ==============================================================
# Color-cue "certain background" masks
# ==============================================================

def certain_background_mask(L: np.ndarray, a_ch: np.ndarray, b_ch: np.ndarray) -> np.ndarray:
    """Regions that are ~100% certainly NOT the cat, from LAB color cues:
    sky (light blue), dark trees (dark green) and front grass (light green).
    """
    sky = ((b_ch < -4) & (L > 55) & (a_ch > -8) & (a_ch < 8)).astype(np.uint8)
    sky = morphology.opening(sky.astype(bool), morphology.disk(3)).astype(np.uint8)

    trees = ((a_ch < -6) & (L < 55) & (b_ch > -5)).astype(np.uint8)
    grass = ((a_ch < -5) & (L > 40)).astype(np.uint8)

    merged = np.maximum(sky, np.maximum(trees, grass))
    merged = morphology.dilation(merged.astype(bool), morphology.disk(2))
    return merged.astype(np.uint8)


# ==============================================================
# GrabCut foreground extraction
# ==============================================================

def run_grabcut(bgr: np.ndarray, sure_bg_color: np.ndarray, rect) -> np.ndarray:
    """Two-phase GrabCut: seed with the manual rect, then refine with a
    mask built from strong FG/BG seeds derived from the color cues.
    Returns a binary (0/1) foreground mask.
    """
    H, W = bgr.shape[:2]
    mask_gc = np.zeros((H, W), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    cv2.grabCut(bgr, mask_gc, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

    rx, ry, rw, rh = rect
    margin = 0.20
    cx1, cx2 = int(rx + rw * margin), int(rx + rw * (1 - margin))
    cy1, cy2 = int(ry + rh * margin), int(ry + rh * (1 - margin))

    center_fg = np.zeros((H, W), np.uint8)
    center_fg[cy1:cy2, cx1:cx2] = 1
    center_fg[sure_bg_color > 0] = 0  # exclude confident background from the FG seed

    bg_seeds = np.zeros((H, W), np.uint8)
    bg_seeds[:ry, :] = 1
    bg_seeds[ry + rh:, :] = 1
    bg_seeds[:, :rx] = 1
    bg_seeds[:, rx + rw:] = 1
    bg_seeds = np.maximum(bg_seeds, sure_bg_color)

    ke = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    kb = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_gc[cv2.erode(center_fg, ke, iterations=1) > 0] = cv2.GC_FGD
    mask_gc[cv2.erode(bg_seeds, kb, iterations=1) > 0] = cv2.GC_BGD

    # Image border is always background.
    mask_gc[: int(H * 0.08), :] = cv2.GC_BGD
    mask_gc[int(H * 0.93):, :] = cv2.GC_BGD
    mask_gc[:, :6] = cv2.GC_BGD
    mask_gc[:, W - 6:] = cv2.GC_BGD

    cv2.grabCut(bgr, mask_gc, rect, bgd_model, fgd_model, 10, cv2.GC_INIT_WITH_MASK)

    return np.where((mask_gc == cv2.GC_FGD) | (mask_gc == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)


# ==============================================================
# SLIC superpixel refinement
# ==============================================================

def slic_refine(scene_arr: np.ndarray, gc_fg: np.ndarray, sure_bg_color: np.ndarray) -> np.ndarray:
    """Reclassify GrabCut's foreground at the superpixel level to smooth
    boundaries and drop isolated mis-classified pixels."""
    H, W = gc_fg.shape
    segments = segmentation.slic(scene_arr, n_segments=400, compactness=15, sigma=1, start_label=0)
    sp_mask = np.zeros((H, W), dtype=np.uint8)
    for sp_id in np.unique(segments):
        sp_pix = segments == sp_id
        fg_ratio = gc_fg[sp_pix].mean()
        bg_ratio = sure_bg_color[sp_pix].mean()
        if fg_ratio > 0.55 and bg_ratio < 0.30:
            sp_mask[sp_pix] = 1
    return sp_mask


# ==============================================================
# Tail recovery
# ==============================================================

def recover_tail(sp_mask: np.ndarray, L, a_ch, b_ch, sure_bg_color: np.ndarray) -> np.ndarray:
    """The tail's color is close to the background (low contrast), so it
    is recovered separately using spatial contiguity with the body."""
    H, W = sp_mask.shape
    body_dilated = morphology.dilation(sp_mask.astype(bool), morphology.disk(10))

    tail_color = (
        (L > 15) & (L < 68) & (np.abs(a_ch) < 15) & (np.abs(b_ch) < 15) & (sure_bg_color == 0)
    ).astype(np.uint8)

    tail_zone = np.zeros((H, W), np.uint8)
    tail_zone[: int(H * 0.58), : int(W * 0.38)] = 1
    tail_candidates = tail_color * tail_zone
    connected_tail = tail_candidates.astype(bool) & body_dilated

    combined_fg = sp_mask.astype(bool) | connected_tail
    combined_fg = morphology.closing(combined_fg, morphology.disk(5))
    combined_fg = morphology.remove_small_holes(combined_fg, area_threshold=3000)

    labeled_fg = sk_label(combined_fg)
    if labeled_fg.max() > 0:
        sizes = [(labeled_fg == i).sum() for i in range(1, labeled_fg.max() + 1)]
        combined_fg = labeled_fg == (np.argmax(sizes) + 1)

    combined_fg[sure_bg_color.astype(bool)] = False
    return combined_fg


def refine_foreground_mask(combined_fg: np.ndarray, sure_bg_color: np.ndarray) -> np.ndarray:
    """Bilateral smoothing of the (binary, cast to float) mask to soften
    the boundary before hard re-thresholding."""
    fp = combined_fg.astype(np.float32)
    fp_b = cv2.bilateralFilter(fp, d=9, sigmaColor=0.3, sigmaSpace=9)
    fg_refined = (fp_b > 0.42).astype(np.uint8)
    fg_refined[fp_b > 0.75] = 1
    fg_refined[fp_b < 0.18] = 0
    fg_refined[sure_bg_color > 0] = 0
    return fg_refined


# ==============================================================
# Background K-Means + semantic naming
# ==============================================================

def segment_background(fg_mask: np.ndarray, lab, hsv_sk) -> tuple[np.ndarray, KMeans]:
    H, W = fg_mask.shape
    lab_norm = lab / np.array([100, 128, 128])
    hue_n, sat_n, val_n = hsv_sk[:, :, 0], hsv_sk[:, :, 1], hsv_sk[:, :, 2]
    rows_n = np.repeat(np.arange(H), W).reshape(H, W) / H
    cols_n = np.tile(np.arange(W), H).reshape(H, W) / W

    bg_mask = fg_mask == 0
    bg_idx = np.where(bg_mask)

    feats = np.hstack([
        lab_norm[bg_idx] * 3.0,
        hue_n[bg_idx].reshape(-1, 1) * 1.0,
        sat_n[bg_idx].reshape(-1, 1) * 2.5,
        val_n[bg_idx].reshape(-1, 1) * 1.0,
        rows_n[bg_idx].reshape(-1, 1) * 2.5,
        cols_n[bg_idx].reshape(-1, 1) * 0.3,
    ])

    km = KMeans(n_clusters=3, n_init=30, random_state=SEED, max_iter=500)
    bg_labels = km.fit_predict(feats)

    lmap = np.zeros((H, W), dtype=np.int32)
    lmap[bg_mask] = bg_labels + 1
    lmap[fg_mask == 1] = 0
    return lmap, km


def name_clusters(lmap: np.ndarray, km: KMeans, H: int) -> dict:
    info = []
    for k in range(1, 4):
        r_idx, _ = np.where(lmap == k)
        center = km.cluster_centers_[k - 1]
        info.append({"id": k, "row": r_idx.mean() if len(r_idx) else H / 2, "L": center[0] * 100})

    info.sort(key=lambda x: x["row"])
    id2name = {0: "Cat", info[0]["id"]: "Sky"}
    remaining = sorted(info[1:], key=lambda x: x["L"])
    id2name[remaining[0]["id"]] = "Trees"
    id2name[remaining[1]["id"]] = "Ground"
    return id2name


def cleanup_background_labels(lmap: np.ndarray, fg_mask: np.ndarray) -> np.ndarray:
    """Drop small noisy regions per class, then fill them by nearest-label
    dilation, so every background pixel ends up with a coherent label."""
    lmap = lmap.copy()
    for class_id in (1, 2, 3):
        labeled_pp = sk_label(lmap == class_id)
        for region_id in range(1, labeled_pp.max() + 1):
            if (labeled_pp == region_id).sum() < 400:
                lmap[labeled_pp == region_id] = -1

    unassigned = lmap == -1
    if unassigned.sum() > 0:
        assigned = lmap.copy()
        assigned[unassigned] = 0
        for _ in range(6):
            dilated = ndi.grey_dilation(np.maximum(assigned, 0).astype(np.int32), size=(3, 3))
            assigned = np.where(assigned <= 0, dilated, assigned)
        lmap[unassigned] = assigned[unassigned]

    lmap[fg_mask == 1] = 0
    return lmap


# ==============================================================
# Evaluation against ground truth
# ==============================================================

def evaluate_multiclass(lmap: np.ndarray, gt_img: Image.Image, W: int, H: int) -> tuple[float, float]:
    """Ground truth is an RGB image without explicit class IDs, so it is
    itself clustered into 4 classes (K-Means) and matched to the predicted
    labels by majority vote, then scored with pixel accuracy and mean IoU.
    """
    gt_res = np.array(gt_img.resize((W, H), Image.LANCZOS))
    gt_feat = gt_res.reshape(-1, 3).astype(np.float32) / 255
    km_gt = KMeans(n_clusters=4, n_init=15, random_state=0)
    gt_lbl = km_gt.fit_predict(gt_feat)

    pred_f = lmap.reshape(-1)
    matched = np.zeros_like(pred_f)
    for k in range(4):
        mk = pred_f == k
        if mk.any():
            matched[mk] = mode(gt_lbl[mk], keepdims=True).mode[0]

    acc = (matched == gt_lbl).mean()
    ious = []
    for c in np.unique(gt_lbl):
        p, g = matched == c, gt_lbl == c
        union = (p | g).sum()
        if union > 0:
            ious.append((p & g).sum() / union)
    return float(acc), float(np.mean(ious))


# ==============================================================
# CLI
# ==============================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "data"), help="Folder containing Scene_1.png and GT1.png")
    parser.add_argument("--out-dir", default=os.environ.get("OUT_DIR", "outputs"), help="Folder to write the result figure to")
    parser.add_argument("--img-name", default="Scene_1.png")
    parser.add_argument("--gt-name", default="GT1.png")
    parser.add_argument("--no-show", action="store_true", help="Don't open a matplotlib window (useful in CI / headless runs)")
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "scene1_result.png"

    scene_path = os.path.join(args.data_dir, args.img_name)
    gt_path = os.path.join(args.data_dir, args.gt_name)

    print("=" * 50)
    print("  Scene 1 - Cat / Sky / Ground / Trees segmentation")
    print("=" * 50)

    scene_img = Image.open(scene_path).convert("RGB")
    gt_img = Image.open(gt_path).convert("RGB")
    scene_arr = np.array(scene_img)
    H, W, _ = scene_arr.shape
    bgr = cv2.cvtColor(scene_arr, cv2.COLOR_RGB2BGR)
    print(f"  Image size: {W}x{H}")

    # --- Color spaces ---
    lab = color.rgb2lab(scene_arr.astype(np.float32) / 255.0)
    L, a_ch, b_ch = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
    hsv_sk = color.rgb2hsv(scene_arr.astype(np.float32) / 255.0)

    # --- Foreground extraction ---
    sure_bg_color = certain_background_mask(L, a_ch, b_ch)
    gc_fg = run_grabcut(bgr, sure_bg_color, CAT_RECT)
    gc_fg[sure_bg_color > 0] = 0

    sp_mask = slic_refine(scene_arr, gc_fg, sure_bg_color)
    combined_fg = recover_tail(sp_mask, L, a_ch, b_ch, sure_bg_color)
    fg_mask = refine_foreground_mask(combined_fg, sure_bg_color)
    print(f"  Cat pixels: {fg_mask.sum()} ({fg_mask.sum() * 100 // (H * W)}%)")

    # --- Background segmentation ---
    lmap, km = segment_background(fg_mask, lab, hsv_sk)
    id2name = name_clusters(lmap, km, H)
    print(f"  Cluster names: {id2name}")
    lmap = cleanup_background_labels(lmap, fg_mask)

    coloured = np.zeros((H, W, 3), dtype=np.uint8)
    for k in range(4):
        coloured[lmap == k] = PALETTE[id2name[k]]

    # --- Evaluation ---
    acc, miou = evaluate_multiclass(lmap, gt_img, W, H)
    print(f"\n{'=' * 45}")
    print(f"  Pixel Accuracy : {acc * 100:.1f}%")
    print(f"  Mean IoU       : {miou * 100:.1f}%")
    print(f"{'=' * 45}\n")

    # --- Visualization ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor("#0d1117")

    gt_display = np.array(gt_img.resize((W, H), Image.LANCZOS))
    for ax, img, ttl in zip(
        axes,
        [scene_arr, coloured, gt_display],
        ["Original Scene", "GrabCut + Color-BG Removal + K-Means (k=4)", "Ground Truth"],
    ):
        ax.imshow(img)
        ax.set_title(ttl, color="white", fontsize=11, fontweight="bold", pad=9)
        ax.axis("off")

    patches = [mpatches.Patch(color=np.array(c) / 255, label=n) for n, c in PALETTE.items()]
    axes[1].legend(handles=patches, loc="lower right", framealpha=0.92, facecolor="#161b22", labelcolor="white", fontsize=10)

    fig.text(
        0.5, 0.01,
        f"Pixel Accuracy: {acc * 100:.1f}%   |   Mean IoU: {miou * 100:.1f}%"
        f"   |   GrabCut + Color-BG Mask + SLIC + K-Means",
        ha="center", color="#58a6ff", fontsize=10, fontweight="bold",
    )
    plt.suptitle("Image Segmentation - Scene 1: Cat / Sky / Ground / Trees", color="white", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout(pad=1.5)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    if not args.no_show:
        plt.show()
    plt.close(fig)
    print(f"  -> {output_path}")


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
