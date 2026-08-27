"""Scene 3 — Road detection in an aerial/satellite image.

Pipeline: LAB color cues + local texture + Sobel edges -> candidate pixels
-> DBSCAN clustering -> cluster linking (collinearity) -> directional
morphological closing -> color-attraction merge -> region growing ->
shape/purity-based component filtering.

Usage:
    python src/scene3_road_detection.py
    python src/scene3_road_detection.py --data-dir data --out-dir outputs --no-show
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from .common.metrics import evaluate


# =========================================================
# Helpers
# =========================================================

def angle_diff(a: float, b: float) -> float:
    d = abs(a - b) % 180
    return min(d, 180 - d)


def connect_clusters(road_mask: np.ndarray, cluster_pts: dict, max_dist: int = 60, angle_thresh: int = 35) -> np.ndarray:
    """Draw a connecting line between two DBSCAN clusters that are close
    together and roughly collinear (likely the same road, split by noise)."""
    if len(cluster_pts) < 2:
        return road_mask
    centroids = {lab: np.mean(pts, axis=0).astype(int) for lab, pts in cluster_pts.items() if len(pts) > 5}
    if len(centroids) < 2:
        return road_mask

    def cluster_angle(pts):
        if len(pts) < 5:
            return 0
        vx, vy, cx, cy = cv2.fitLine(pts.reshape(-1, 1, 2).astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01)
        return np.degrees(np.arctan2(vy, vx))

    labs = list(centroids.keys())
    for i in range(len(labs)):
        for j in range(i + 1, len(labs)):
            p1, p2 = centroids[labs[i]], centroids[labs[j]]
            dist = np.linalg.norm(p1 - p2)
            if dist > max_dist:
                continue
            theta = np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))
            ang1 = cluster_angle(cluster_pts[labs[i]])
            ang2 = cluster_angle(cluster_pts[labs[j]])
            if abs(angle_diff(theta, ang1)) < angle_thresh or abs(angle_diff(theta, ang2)) < angle_thresh:
                cv2.line(road_mask, tuple(p1[::-1]), tuple(p2[::-1]), 1, thickness=6)
    return road_mask


def directional_close(mask: np.ndarray, orientations=(0, 45, 90, 135), length: int = 25, thickness: int = 4) -> np.ndarray:
    """Morphological closing with a rotated line kernel at several
    orientations, to bridge gaps along the road's direction."""
    result = mask.copy()
    for ang in orientations:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (length, thickness))
        rot_matrix = cv2.getRotationMatrix2D((length // 2, thickness // 2), ang, 1.0)
        kernel_rot = cv2.warpAffine(kernel, rot_matrix, (length, thickness))
        kernel_rot = (kernel_rot > 0).astype(np.uint8)
        closed = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel_rot, iterations=1)
        result = cv2.bitwise_or(result, closed)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "data"))
    parser.add_argument("--out-dir", default=os.environ.get("OUT_DIR", "outputs"))
    parser.add_argument("--img-name", default="Scene_3.png")
    parser.add_argument("--gt-name", default="GT3.png")
    parser.add_argument("--no-show", action="store_true")
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img_path = os.path.join(args.data_dir, args.img_name)
    gt_path = os.path.join(args.data_dir, args.gt_name)

    img_bgr = cv2.imread(img_path)
    gt_bgr = cv2.imread(gt_path)
    if img_bgr is None or gt_bgr is None:
        raise FileNotFoundError(f"Could not read '{img_path}' or '{gt_path}' - check data/README.md")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W = img_rgb.shape[:2]

    # --- Ground truth ---
    gt_bgr = cv2.resize(gt_bgr, (W, H), interpolation=cv2.INTER_NEAREST)
    R, G, B_ = gt_bgr[:, :, 2], gt_bgr[:, :, 1], gt_bgr[:, :, 0]
    gt_mask = ((R > 30) & (R > G + 10) & (R > B_ + 10)).astype(np.uint8)
    if gt_mask.sum() < 100:
        gray_gt = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2GRAY)
        gt_mask = (gray_gt > 127).astype(np.uint8)

    # --- Road cues ---
    img_smooth = cv2.GaussianBlur(img_rgb, (5, 5), 0)
    img_lab = cv2.cvtColor(img_smooth, cv2.COLOR_RGB2LAB).astype(np.float32)
    L = img_lab[:, :, 0]
    a = img_lab[:, :, 1] - 128.0
    b_ = img_lab[:, :, 2] - 128.0
    chroma = np.sqrt(a ** 2 + b_ ** 2)

    road_color = ((L > 95) & (L < 195) & (chroma < 15)).astype(np.uint8)

    gray = cv2.cvtColor(img_smooth, cv2.COLOR_RGB2GRAY).astype(np.float32)
    local_mean = cv2.blur(gray, (9, 9))
    local_sq = cv2.blur(gray ** 2, (9, 9))
    local_std = np.sqrt(np.clip(local_sq - local_mean ** 2, 0, None))
    smooth_mask = (local_std < 14).astype(np.uint8)

    sobelx = cv2.Sobel(gray / 255.0, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray / 255.0, cv2.CV_32F, 0, 1, ksize=3)
    edge_mag_norm = np.clip(np.sqrt(sobelx ** 2 + sobely ** 2), 0, 1)
    non_edge = (edge_mag_norm < 0.12).astype(np.uint8)

    combined = (road_color & smooth_mask & non_edge).astype(np.uint8)

    # --- DBSCAN ---
    candidate_yx = np.argwhere(combined == 1)
    cluster_pts = {}

    if len(candidate_yx) > 100:
        y_sub = candidate_yx[:, 0] / H
        x_sub = candidate_yx[:, 1] / W
        L_sub = L[candidate_yx[:, 0], candidate_yx[:, 1]] / 255.0
        c_sub = chroma[candidate_yx[:, 0], candidate_yx[:, 1]] / 30.0

        feat_sub = np.stack([x_sub * 0.8, y_sub * 0.8, L_sub * 0.6, c_sub * 0.4], axis=-1)
        feat_scaled = StandardScaler().fit_transform(feat_sub)

        db = DBSCAN(eps=0.42, min_samples=20, n_jobs=-1)
        labels = db.fit_predict(feat_scaled)

        label_map = np.full((H, W), -1, dtype=np.int32)
        label_map[candidate_yx[:, 0], candidate_yx[:, 1]] = labels

        road_dbscan = np.zeros((H, W), dtype=np.uint8)
        for lab in np.unique(labels):
            if lab == -1:
                continue
            mask = (label_map == lab)
            size = np.sum(mask) / (H * W)
            c_mean = np.mean(chroma[mask])
            L_mean = np.mean(L[mask])
            if 0.002 < size < 0.2 and c_mean < 14 and L_mean > 90:
                road_dbscan[mask] = 1
                cluster_pts[lab] = np.argwhere(mask)
    else:
        road_dbscan = combined.copy()

    if len(cluster_pts) > 1:
        road_dbscan = connect_clusters(road_dbscan, cluster_pts, max_dist=60, angle_thresh=35)

    road_dbscan = directional_close(road_dbscan, length=25, thickness=4)

    # --- Merge with color attraction ---
    color_road = road_color & smooth_mask
    k_attract = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    dbscan_dilated = cv2.dilate(road_dbscan, k_attract)
    merged = ((color_road & dbscan_dilated) | road_dbscan).astype(np.uint8)

    # --- Region growing ---
    k_grow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    for _ in range(2):
        dilated = cv2.dilate(merged, k_grow)
        grow = (dilated == 1) & (color_road == 1) & (edge_mag_norm < 0.12)
        merged[grow] = 1

    # --- Component filter (shape + purity) ---
    nlabels, labels_cc, stats, _ = cv2.connectedComponentsWithStats(merged, 8)
    final_mask = np.zeros((H, W), dtype=np.uint8)
    valid_components = []

    for i in range(1, nlabels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 400:
            continue
        comp_mask = (labels_cc == i)
        pts = np.column_stack(np.where(comp_mask))
        if len(pts) < 20:
            continue

        try:
            if len(pts) >= 5:
                (_, _), (major, minor), _ = cv2.fitEllipse(pts.astype(np.int32))
                eccentricity = (
                    np.sqrt(1 - (minor / major) ** 2) if major > minor
                    else np.sqrt(1 - (major / minor) ** 2) if minor > 0 else 0
                )
            else:
                eccentricity = 0
        except cv2.error:
            eccentricity = 0

        color_overlap = np.sum(road_color[comp_mask]) / area
        is_elongated = eccentricity > 0.55
        is_pure = color_overlap > 0.6
        is_large = area > 2000
        is_very_pure = color_overlap > 0.8

        accept = (
            (is_elongated and (is_pure or is_large))
            or (is_very_pure and area > 500)
            or (is_large and is_pure)
            or (eccentricity > 0.7 and area > 600 and color_overlap > 0.5)
        )
        if accept:
            valid_components.append((i, area, eccentricity, color_overlap))

    accepted_labels = set()
    for i, area, ecc, purity in valid_components:
        comp_mask = (labels_cc == i)
        touches_accepted = False
        if accepted_labels:
            kernel_check = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            dilated_comp = cv2.dilate(comp_mask.astype(np.uint8), kernel_check)
            touches_accepted = any(
                np.any(dilated_comp & (labels_cc == acc_label)) for acc_label in accepted_labels
            )
        if touches_accepted or area > 1500 or purity > 0.75:
            final_mask[comp_mask] = 1
            accepted_labels.add(i)

    # --- Final cleanup ---
    nlabels_final, labels_final, stats_final, _ = cv2.connectedComponentsWithStats(final_mask, 8)
    clean_mask = np.zeros_like(final_mask)
    for i in range(1, nlabels_final):
        if stats_final[i, cv2.CC_STAT_AREA] > 300:
            clean_mask[labels_final == i] = 1

    kernel_fill = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    pred = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel_fill, iterations=1)

    # --- Metrics ---
    scores = evaluate(gt_mask, pred)
    print("\n====== ROAD DETECTION METRICS ======")
    print(scores)

    # --- Visualization ---
    dbscan_viz = np.zeros((H, W, 3), dtype=np.uint8)
    for lab, pts in cluster_pts.items():
        if len(pts) > 10:
            col = np.random.randint(100, 255, 3).tolist()
            dbscan_viz[pts[:, 0], pts[:, 1]] = col

    dbscan_overlay = img_rgb.copy()
    alpha_db = 0.6
    for c in range(3):
        dbscan_overlay[:, :, c] = np.where(
            np.any(dbscan_viz != 0, axis=2),
            (1 - alpha_db) * img_rgb[:, :, c] + alpha_db * dbscan_viz[:, :, c],
            img_rgb[:, :, c],
        ).astype(np.uint8)

    pred_overlay = img_rgb.copy()
    alpha_pred = 0.5
    pred_color = [0, 220, 255]
    for c in range(3):
        pred_overlay[:, :, c] = np.where(
            pred == 1,
            (1 - alpha_pred) * img_rgb[:, :, c] + alpha_pred * pred_color[c],
            img_rgb[:, :, c],
        ).astype(np.uint8)
    contours, _ = cv2.findContours(pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(pred_overlay, contours, -1, (0, 255, 180), 2)

    gt_overlay = img_rgb.copy()
    alpha_gt = 0.5
    gt_color = [50, 255, 50]
    for c in range(3):
        gt_overlay[:, :, c] = np.where(
            gt_mask == 1,
            (1 - alpha_gt) * img_rgb[:, :, c] + alpha_gt * gt_color[c],
            img_rgb[:, :, c],
        ).astype(np.uint8)
    contours_gt, _ = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(gt_overlay, contours_gt, -1, (0, 200, 0), 2)

    metrics_img = np.ones((H, W, 3), dtype=np.uint8) * 20
    metrics_lines = [
        ("METRICS", (255, 220, 50), 0.9, 2),
        (f"IoU: {scores.iou:.4f}", (100, 220, 255), 0.75, 1),
        (f"Dice: {scores.dice:.4f}", (100, 220, 255), 0.75, 1),
        (f"Accuracy: {scores.accuracy:.4f}", (100, 255, 150), 0.75, 1),
        (f"Precision: {scores.precision:.4f}", (100, 255, 150), 0.75, 1),
        (f"Recall: {scores.recall:.4f}", (100, 255, 150), 0.75, 1),
    ]
    y_start = max(30, H // 8)
    line_gap = max(28, H // 14)
    for i, (text, color, scale, thick) in enumerate(metrics_lines):
        y = y_start + i * line_gap
        if y < H - 10:
            cv2.putText(metrics_img, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor("#0e0e0e")
    panels = [
        (dbscan_overlay, "DBSCAN Clusters"),
        (pred_overlay, "Predicted Road"),
        (gt_overlay, "Ground Truth Road"),
        (metrics_img, "Metrics"),
    ]
    for idx, (img, title) in enumerate(panels):
        ax = axes.flat[idx]
        ax.set_facecolor("#0e0e0e")
        ax.imshow(img)
        ax.set_title(title, color="white", fontsize=14, fontweight="bold", pad=10)
        ax.axis("off")

    plt.suptitle("Road Detection Results", color="white", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout()

    fig_path = out_dir / "scene3_result.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    if not args.no_show:
        plt.show()
    plt.close(fig)

    cv2.imwrite(str(out_dir / "scene3_mask.png"), (pred * 255).astype(np.uint8))
    print(f"-> {fig_path}")


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
