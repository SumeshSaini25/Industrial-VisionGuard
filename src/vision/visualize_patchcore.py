from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Patchcore


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_ROOT = PROJECT_ROOT / "data" / "raw"
RESULTS_ROOT = PROJECT_ROOT / "results" / "patchcore_baseline"

CHECKPOINT = (
    RESULTS_ROOT
    / "Patchcore"
    / "MVTecAD"
    / "metal_nut"
    / "v0"
    / "weights"
    / "lightning"
    / "model.ckpt"
)

OUTPUT_ROOT = PROJECT_ROOT / "reports" / "patchcore_visualizations"


def to_numpy(value):
    """Convert tensors / Anomalib masks to NumPy arrays."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()

    if hasattr(value, "numpy"):
        try:
            return value.numpy()
        except Exception:
            pass

    return np.asarray(value)


def squeeze_mask(value):
    """Convert a mask/map to a 2D NumPy array."""
    array = to_numpy(value)

    while array.ndim > 2:
        array = array[0]

    return array


def get_scalar(value, index):
    """Safely extract one scalar from a batched tensor/list."""
    item = value[index]

    if isinstance(item, torch.Tensor):
        return item.detach().cpu().item()

    return float(item)


def get_bool(value, index):
    """Safely extract one boolean from a batched tensor/list."""
    item = value[index]

    if isinstance(item, torch.Tensor):
        return bool(item.detach().cpu().item())

    return bool(item)


def save_visualization(
    image_path,
    anomaly_map,
    predicted_mask,
    ground_truth_mask,
    score,
    predicted_label,
    output_path,
):
    image = Image.open(image_path).convert("RGB")

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    # Anomaly heatmap
    axes[1].imshow(image)
    axes[1].imshow(anomaly_map, alpha=0.55, cmap="jet")
    axes[1].set_title(f"Anomaly Map\nScore: {score:.3f}")
    axes[1].axis("off")

    # Predicted mask
    axes[2].imshow(predicted_mask, cmap="gray")
    axes[2].set_title(
        f"Predicted Mask\n{'DEFECT' if predicted_label else 'GOOD'}"
    )
    axes[2].axis("off")

    # Ground truth
    axes[3].imshow(ground_truth_mask, cmap="gray")
    axes[3].set_title("Ground Truth")
    axes[3].axis("off")

    fig.suptitle(
        f"{Path(image_path).parent.name} / {Path(image_path).name}",
        fontsize=12,
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    print("=" * 70)
    print("Industrial VisionGuard - PatchCore Visualization")
    print("=" * 70)

    print(f"\nCheckpoint:\n{CHECKPOINT}")

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"PatchCore checkpoint not found:\n{CHECKPOINT}"
        )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("\nCreating dataset...")

    datamodule = MVTecAD(
        root=str(DATASET_ROOT),
        category="metal_nut",
        num_workers=0,
    )

    print("Creating PatchCore model...")

    model = Patchcore(
        backbone="wide_resnet50_2",
        layers=["layer2", "layer3"],
        coreset_sampling_ratio=0.1,
    )

    engine = Engine(
        default_root_dir=str(RESULTS_ROOT),
    )

    print("\nRunning inference...")

    predictions = engine.predict(
        model=model,
        datamodule=datamodule,
        ckpt_path=str(CHECKPOINT),
    )

    print(f"\nPrediction batches received: {len(predictions)}")

    # One representative image from each class
    selected = {
        "good": False,
        "bent": False,
        "color": False,
        "flip": False,
        "scratch": False,
    }

    saved_count = 0

    for batch_number, batch in enumerate(predictions, start=1):

        print(f"Processing batch {batch_number}...")

        image_paths = batch.image_path
        pred_scores = batch.pred_score
        pred_labels = batch.pred_label
        anomaly_maps = batch.anomaly_map
        predicted_masks = batch.pred_mask
        ground_truth_masks = batch.gt_mask

        for i, image_path in enumerate(image_paths):

            image_path = Path(image_path)

            defect_type = image_path.parent.name

            if defect_type not in selected:
                continue

            if selected[defect_type]:
                continue

            print(
                f"  Selecting {defect_type}: "
                f"{image_path.name}"
            )

            anomaly_map = squeeze_mask(anomaly_maps[i])
            predicted_mask = squeeze_mask(predicted_masks[i])
            ground_truth_mask = squeeze_mask(ground_truth_masks[i])

            score = get_scalar(pred_scores, i)
            predicted_label = get_bool(pred_labels, i)

            output_path = (
                OUTPUT_ROOT
                / f"{defect_type}_patchcore_result.png"
            )

            save_visualization(
                image_path=image_path,
                anomaly_map=anomaly_map,
                predicted_mask=predicted_mask,
                ground_truth_mask=ground_truth_mask,
                score=score,
                predicted_label=predicted_label,
                output_path=output_path,
            )

            selected[defect_type] = True
            saved_count += 1

    print("\n" + "=" * 70)
    print(f"Saved visualizations: {saved_count}")
    print(f"Output directory: {OUTPUT_ROOT}")
    print("=" * 70)

    print("\nSelected:")
    for defect_type, was_selected in selected.items():
        print(
            f"  {defect_type:8s}: "
            f"{'YES' if was_selected else 'NO'}"
        )

    print("\nVisualization complete.")


if __name__ == "__main__":
    main()