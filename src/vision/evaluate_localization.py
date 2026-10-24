
from pathlib import Path
import csv
import json

import cv2
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = PROJECT_ROOT / "data/raw/metal_nut"
EVALUATION_DIR = PROJECT_ROOT / "reports/full_evaluation"

PREDICTIONS_CSV = EVALUATION_DIR / "predictions.csv"
OUTPUT_CSV = EVALUATION_DIR / "localization_results.csv"
OUTPUT_JSON = EVALUATION_DIR / "localization_metrics.json"

EPS = 1e-8


# ============================================================
# LOAD BINARY MASK
# ============================================================

def load_binary_mask(path, target_shape=None):

    mask = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE,
    )

    if mask is None:
        raise FileNotFoundError(
            f"Could not load mask: {path}"
        )

    if target_shape is not None:
        height, width = target_shape

        if mask.shape != (height, width):
            mask = cv2.resize(
                mask,
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )

    return mask > 127


# ============================================================
# CALCULATE PIXEL METRICS
# ============================================================

def calculate_metrics(ground_truth, prediction):

    gt = ground_truth.astype(bool)
    pred = prediction.astype(bool)

    tp = int(np.count_nonzero(gt & pred))
    fp = int(np.count_nonzero(~gt & pred))
    fn = int(np.count_nonzero(gt & ~pred))
    tn = int(np.count_nonzero(~gt & ~pred))

    precision = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)

    dice = (2 * tp) / (2 * tp + fp + fn + EPS)
    iou = tp / (tp + fp + fn + EPS)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "dice": dice,
        "iou": iou,
    }


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():

    if not PREDICTIONS_CSV.exists():
        raise FileNotFoundError(PREDICTIONS_CSV)

    with PREDICTIONS_CSV.open(
        newline="",
        encoding="utf-8",
    ) as file:
        inspections = list(csv.DictReader(file))

    results = []

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0

    categories = [
        "good",
        "bent",
        "color",
        "flip",
        "scratch",
    ]

    for index, inspection in enumerate(inspections, start=1):

        category = inspection["category"]

        image_path = PROJECT_ROOT / inspection["image"]

        predicted_mask_path = (
            Path(inspection["output_directory"]) / "mask.png"
        )

        predicted_mask = load_binary_mask(
            predicted_mask_path
        )

        # ----------------------------------------------------
        # GROUND TRUTH
        # ----------------------------------------------------

        if category == "good":
            ground_truth = np.zeros_like(
                predicted_mask,
                dtype=bool,
            )

        else:
            ground_truth_path = (
                DATASET_DIR
                / "ground_truth"
                / category
                / f"{image_path.stem}_mask.png"
            )

            ground_truth = load_binary_mask(
                ground_truth_path,
                target_shape=predicted_mask.shape,
            )

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        metrics = calculate_metrics(
            ground_truth,
            predicted_mask,
        )

        total_tp += metrics["tp"]
        total_fp += metrics["fp"]
        total_fn += metrics["fn"]
        total_tn += metrics["tn"]

        results.append({
            "image": inspection["image"],
            "category": category,
            "pixel_precision": metrics["precision"],
            "pixel_recall": metrics["recall"],
            "pixel_dice": metrics["dice"],
            "pixel_iou": metrics["iou"],
            "tp": metrics["tp"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "tn": metrics["tn"],
        })

        print(
            f"[{index}/{len(inspections)}] "
            f"{category}/{image_path.name} | "
            f"IoU={metrics['iou']:.4f} | "
            f"Dice={metrics['dice']:.4f}"
        )

    # --------------------------------------------------------
    # OVERALL MICRO-AVERAGED METRICS
    # --------------------------------------------------------

    overall_precision = total_tp / (
        total_tp + total_fp + EPS
    )

    overall_recall = total_tp / (
        total_tp + total_fn + EPS
    )

    overall_dice = (2 * total_tp) / (
        2 * total_tp + total_fp + total_fn + EPS
    )

    overall_iou = total_tp / (
        total_tp + total_fp + total_fn + EPS
    )

    summary = {
        "images_evaluated": len(results),
        "pixel_precision": overall_precision,
        "pixel_recall": overall_recall,
        "pixel_dice_f1": overall_dice,
        "pixel_iou": overall_iou,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "total_tn": total_tn,
        "category_results": {},
    }

    # --------------------------------------------------------
    # CATEGORY BREAKDOWN
    # --------------------------------------------------------

    for category in categories:

        subset = [
            r for r in results
            if r["category"] == category
        ]

        tp = sum(r["tp"] for r in subset)
        fp = sum(r["fp"] for r in subset)
        fn = sum(r["fn"] for r in subset)

        summary["category_results"][category] = {
            "images": len(subset),
            "pixel_dice": (
                2 * tp / (2 * tp + fp + fn + EPS)
            ),
            "pixel_iou": (
                tp / (tp + fp + fn + EPS)
            ),
            "pixel_recall": (
                tp / (tp + fn + EPS)
            ),
            "false_positive_pixels": fp,
            "missed_defect_pixels": fn,
        }

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    with OUTPUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].keys()),
        )

        writer.writeheader()
        writer.writerows(results)

    with OUTPUT_JSON.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    # --------------------------------------------------------
    # PRINT SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("PIXEL-LEVEL LOCALIZATION RESULTS")
    print("=" * 65)

    print(f"Images evaluated: {len(results)}")
    print(f"Pixel precision: {overall_precision:.4%}")
    print(f"Pixel recall:    {overall_recall:.4%}")
    print(f"Pixel Dice/F1:   {overall_dice:.4%}")
    print(f"Pixel IoU:       {overall_iou:.4%}")

    print("\nCATEGORY BREAKDOWN")

    for category, data in summary["category_results"].items():
        print(
            f"{category:8s} | "
            f"Dice: {data['pixel_dice']:.4%} | "
            f"IoU: {data['pixel_iou']:.4%} | "
            f"Recall: {data['pixel_recall']:.4%}"
        )

    print(f"\nCSV saved:  {OUTPUT_CSV}")
    print(f"JSON saved: {OUTPUT_JSON}")

    print("\nLOCALIZATION EVALUATION COMPLETE")


if __name__ == "__main__":
    main()
