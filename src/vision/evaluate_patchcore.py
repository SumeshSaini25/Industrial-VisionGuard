
from pathlib import Path
from time import perf_counter
from collections import Counter
import csv
import json

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from sys import path as sys_path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys_path.insert(0, str(PROJECT_ROOT))

from src.inference.predict import predict_image


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = PROJECT_ROOT / "data/raw/metal_nut/test"
OUTPUT_DIR = PROJECT_ROOT / "reports/full_evaluation"
INSPECTION_DIR = OUTPUT_DIR / "inspections"

CATEGORIES = [
    "good",
    "bent",
    "color",
    "flip",
    "scratch",
]


# ============================================================
# LOAD DATASET
# ============================================================

def load_test_images():
    samples = []

    for category in CATEGORIES:
        folder = DATASET_DIR / category

        if not folder.is_dir():
            raise FileNotFoundError(
                f"Missing dataset folder: {folder}"
            )

        images = sorted(folder.glob("*.png"))

        for image in images:
            samples.append({
                "path": image,
                "category": category,
                "true_label": 0 if category == "good" else 1,
            })

    if not samples:
        raise RuntimeError("No test images found.")

    return samples


# ============================================================
# EVALUATION
# ============================================================

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    samples = load_test_images()

    print("\n" + "=" * 65)
    print("INDUSTRIAL VISIONGUARD - FULL DATASET EVALUATION")
    print("=" * 65)
    print(f"Total images: {len(samples)}")
    print(f"Category distribution: {dict(Counter(s['category'] for s in samples))}")

    records = []

    start_total = perf_counter()

    for index, sample in enumerate(samples, start=1):

        print(
            f"\n[{index}/{len(samples)}] "
            f"{sample['category']}/{sample['path'].name}"
        )

        start = perf_counter()

        result = predict_image(
            image_path=sample["path"],
            output_dir=INSPECTION_DIR,
        )

        elapsed = perf_counter() - start

        predicted_label = (
            1 if result["status"] == "DEFECT" else 0
        )

        records.append({
            "image": str(sample["path"].relative_to(PROJECT_ROOT)),
            "category": sample["category"],
            "true_label": sample["true_label"],
            "predicted_label": predicted_label,
            "status": result["status"],
            "anomaly_score": result["anomaly_score"],
            "inference_seconds": round(elapsed, 4),
            "correct": predicted_label == sample["true_label"],
            "output_directory": result["output_directory"],
        })

    total_seconds = perf_counter() - start_total

    # --------------------------------------------------------
    # CALCULATE METRICS
    # --------------------------------------------------------

    y_true = np.array([r["true_label"] for r in records])
    y_pred = np.array([r["predicted_label"] for r in records])
    y_scores = np.array([r["anomaly_score"] for r in records])

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    metrics = {
        "total_images": len(records),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_true, y_pred, zero_division=0)
        ),
        "image_auroc": float(roc_auc_score(y_true, y_scores)),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "average_inference_seconds": float(
            np.mean([r["inference_seconds"] for r in records])
        ),
        "total_evaluation_seconds": round(total_seconds, 2),
    }

    # --------------------------------------------------------
    # CATEGORY BREAKDOWN
    # --------------------------------------------------------

    category_results = {}

    for category in CATEGORIES:
        subset = [
            record for record in records
            if record["category"] == category
        ]

        category_results[category] = {
            "total": len(subset),
            "correct": sum(r["correct"] for r in subset),
            "accuracy": (
                sum(r["correct"] for r in subset) / len(subset)
                if subset else 0.0
            ),
        }

    metrics["category_results"] = category_results

    # --------------------------------------------------------
    # SAVE CSV
    # --------------------------------------------------------

    csv_path = OUTPUT_DIR / "predictions.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(records[0].keys()),
        )

        writer.writeheader()
        writer.writerows(records)

    # --------------------------------------------------------
    # SAVE METRICS
    # --------------------------------------------------------

    metrics_path = OUTPUT_DIR / "metrics.json"

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("FINAL EVALUATION RESULTS")
    print("=" * 65)

    print(f"Images evaluated: {metrics['total_images']}")
    print(f"Accuracy:         {metrics['accuracy']:.4%}")
    print(f"Precision:        {metrics['precision']:.4%}")
    print(f"Recall:           {metrics['recall']:.4%}")
    print(f"F1 score:         {metrics['f1_score']:.4%}")
    print(f"Image AUROC:      {metrics['image_auroc']:.4%}")

    print("\nConfusion Matrix:")
    print(f"True negatives:  {tn}")
    print(f"False positives: {fp}")
    print(f"False negatives: {fn}")
    print(f"True positives:  {tp}")

    print("\nCategory Results:")

    for category, data in category_results.items():
        print(
            f"  {category:8s}: "
            f"{data['correct']}/{data['total']} correct "
            f"({data['accuracy']:.2%})"
        )

    print(f"\nAverage inspection time: {metrics['average_inference_seconds']:.4f}s")
    print(f"Total evaluation time:   {total_seconds:.2f}s")

    print(f"\nPredictions CSV: {csv_path}")
    print(f"Metrics JSON:    {metrics_path}")

    print("\nFULL EVALUATION COMPLETE")


if __name__ == "__main__":
    main()
