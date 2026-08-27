"""Scene 4 — Standing-person silhouette extraction from an RGB + depth pair.

Pipeline: depth + spatial + gradient features -> K-Means (k=7) -> automatic
cluster selection (size / centrality / depth-consistency score) -> depth
refinement -> ground-plane removal -> morphological cleanup.

Usage:
    python src/scene4_person_extraction.py
    python src/scene4_person_extraction.py --data-dir data --out-dir outputs --no-show
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from .common.io_utils import read_rgb
from .common.metrics import evaluate

N_CLUSTERS = 7


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "data"))
    parser.add_argument("--out-dir", default=os.environ.get("OUT_DIR", "outputs"))
    parser.add_argument("--rgb-name", default="scene4.png")
    parser.add_argument("--depth-name", default="scene4_2.png")
    parser.add_argument("--gt-name", default="GT4.png")
    parser.add_argument("--no-show", action="store_true")
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def build_features(img_color: np.ndarray, blur_depth: np.ndarray) -> np.ndarray:
    h, w = blur_depth.shape
    x_coords = np.tile(np.arange(w), h).reshape(-1, 1) / w
    y_coords = np.repeat(np.arange(h), w).reshape(-1, 1) / h

    gray = cv2.cvtColor(img_color, cv2.COLOR_RGB2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(grad_x ** 2 + grad_y ** 2)
    gradient = gradient / (gradient.max() + 1e-6)

    x_depth = blur_depth.reshape(-1, 1)
    x_gradient = gradient.reshape(-1, 1)
    return np.hstack([x_depth * 3.0, x_coords, y_coords, x_gradient])


def select_person_cluster(segmented: np.ndarray, blur_depth: np.ndarray) -> np.ndarray:
    """Score every cluster on size / centrality / depth-consistency and
    return the connected component of the best-scoring one."""
    h, w = blur_depth.shape
    center = np.array([w / 2, h / 2])
    best_score, best_mask = -1.0, None

    for cid in range(N_CLUSTERS):
        mask = (segmented == cid).astype(np.uint8) * 255
        num_labels, labels_cc, stats, _ = cv2.connectedComponentsWithStats(mask)
        if num_labels <= 1:
            continue

        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        area = stats[largest, cv2.CC_STAT_AREA]
        cx = stats[largest, cv2.CC_STAT_LEFT] + stats[largest, cv2.CC_STAT_WIDTH] / 2
        cy = stats[largest, cv2.CC_STAT_TOP] + stats[largest, cv2.CC_STAT_HEIGHT] / 2
        dist = np.linalg.norm(np.array([cx, cy]) - center)
        component = (labels_cc == largest)
        mean_depth = blur_depth[component].mean()

        size_ratio = area / blur_depth.size
        size_score = np.exp(-((size_ratio - 0.15) ** 2) / 0.01)
        pos_score = max(0, 1 - dist / (w / 2))
        depth_score = 1 - abs(mean_depth - np.median(blur_depth)) / (blur_depth.max() - blur_depth.min())
        score = 0.4 * size_score + 0.4 * pos_score + 0.2 * depth_score

        if score > best_score:
            best_score = score
            best_mask = component.astype(np.uint8) * 255

    return best_mask


def refine_by_depth(mask: np.ndarray, blur_depth: np.ndarray) -> np.ndarray:
    cluster_pixels = blur_depth[mask == 255]
    mean_depth = np.mean(cluster_pixels)
    std_depth = np.std(cluster_pixels)
    threshold = std_depth * 1.0

    refined = np.zeros_like(mask)
    refined[(mask == 255) & (np.abs(blur_depth - mean_depth) < threshold)] = 255
    return keep_largest(refined)


def keep_largest(mask: np.ndarray) -> np.ndarray:
    num_labels, labels_cc, stats, _ = cv2.connectedComponentsWithStats(mask)
    if num_labels <= 1:
        return mask.copy()
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    out = np.zeros_like(mask)
    out[labels_cc == largest] = 255
    return out


def remove_ground_plane(mask: np.ndarray, w: int, h: int) -> np.ndarray:
    """Erase rows below 60% of the image height that are almost entirely
    foreground — those are ground / floor, not the person."""
    clean = mask.copy()
    for r in range(h - 1, int(h * 0.60), -1):
        white = np.sum(clean[r, :] > 0)
        if white > w * 0.75:
            clean[r, :] = 0
    return keep_largest(clean)


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img_color = read_rgb(os.path.join(args.data_dir, args.rgb_name))
    img_depth = cv2.imread(os.path.join(args.data_dir, args.depth_name), 0)
    if img_depth is None:
        raise FileNotFoundError(f"Could not read depth map at '{args.depth_name}' in {args.data_dir}")
    img_depth = cv2.resize(img_depth, (img_color.shape[1], img_color.shape[0]), interpolation=cv2.INTER_NEAREST)

    blur = cv2.GaussianBlur(img_depth, (5, 5), 0)
    h, w = blur.shape

    features = build_features(img_color, blur)

    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)
    segmented = labels.reshape(blur.shape)

    best_mask = select_person_cluster(segmented, blur)
    refined = refine_by_depth(best_mask, blur)
    clean = remove_ground_plane(refined, w, h)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    final_mask = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)

    silhouette = np.zeros_like(img_color)
    silhouette[final_mask == 255] = [255, 255, 255]

    # --- Visualization ---
    fig = plt.figure(figsize=(12, 6))
    plt.subplot(131)
    plt.imshow(segmented, cmap="jet")
    plt.title("Clusters")

    plt.subplot(132)
    plt.imshow(final_mask, cmap="gray")
    plt.title("Final Mask")

    plt.subplot(133)
    plt.imshow(silhouette)
    plt.title("Silhouette")

    plt.tight_layout()
    fig_path = out_dir / "scene4_result.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    if not args.no_show:
        plt.show()
    plt.close(fig)
    print(f"-> {fig_path}")

    cv2.imwrite(str(out_dir / "scene4_mask.png"), final_mask)
    cv2.imwrite(str(out_dir / "scene4_silhouette.png"), cv2.cvtColor(silhouette, cv2.COLOR_RGB2BGR))

    # --- Evaluation ---
    gt_path = os.path.join(args.data_dir, args.gt_name)
    gt_img = cv2.imread(gt_path, 0)
    if gt_img is not None:
        gt_img = cv2.resize(gt_img, (w, h))
        _, gt_binary = cv2.threshold(gt_img, 127, 255, cv2.THRESH_BINARY)
        scores = evaluate(gt_binary > 0, final_mask > 0)
        print("\n==== EVALUATION METRICS ====")
        print(scores)
    else:
        print(f"Ground truth not found at '{gt_path}' - skipping evaluation")


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
