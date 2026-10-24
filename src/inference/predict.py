
from functools import lru_cache
from pathlib import Path
import argparse
import json
import uuid

import cv2
import numpy as np
import torch
from PIL import Image

from anomalib.engine import Engine
from anomalib.models import Patchcore


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT = (
    PROJECT_ROOT
    / "results/patchcore_baseline/Patchcore/MVTecAD"
    / "metal_nut/v0/weights/lightning/model.ckpt"
)

OUTPUT_ROOT = PROJECT_ROOT / "reports/inference"


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def as_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()

    return np.asarray(value)


def extract_map(value):
    array = as_numpy(value)
    return np.squeeze(array).astype(np.float32)


# ============================================================
# CACHED PATCHCORE PREDICTOR
# ============================================================

@lru_cache(maxsize=1)
def get_predictor():
    """
    Initialize PatchCore and Lightning Engine only once
    per Python process.

    The trained checkpoint will be restored during the
    first successful prediction.

    Subsequent predictions reuse the loaded model.
    """

    print("\nInitializing cached PatchCore predictor...")

    model = Patchcore(
        backbone="wide_resnet50_2",
        layers=["layer2", "layer3"],
        coreset_sampling_ratio=0.1,
    )

    engine = Engine(
        default_root_dir=str(
            PROJECT_ROOT / "results/inference"
        ),
        accelerator="auto",
        devices=1,
        enable_progress_bar=False,
        logger=False,
    )

    state = {
        "checkpoint_loaded": False
    }

    return model, engine, state


# ============================================================
# PATCHCORE INFERENCE
# ============================================================

def predict_image(image_path, output_dir=OUTPUT_ROOT):

    image_path = Path(image_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()

    # --------------------------------------------------------
    # VALIDATE INPUTS
    # --------------------------------------------------------

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    if not CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # CREATE UNIQUE INSPECTION DIRECTORY
    # --------------------------------------------------------

    inspection_id = uuid.uuid4().hex[:12]

    save_dir = (
        output_dir
        / f"{image_path.stem}_{inspection_id}"
    )

    save_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    # --------------------------------------------------------
    # SAVE PERMANENT ORIGINAL IMAGE
    # --------------------------------------------------------

    permanent_image_path = save_dir / "original.png"

    with Image.open(image_path) as input_image:
        rgb_image = input_image.convert("RGB")
        rgb_image.save(permanent_image_path)

        original = np.asarray(rgb_image)

    # --------------------------------------------------------
    # GET CACHED PATCHCORE MODEL
    # --------------------------------------------------------

    model, engine, state = get_predictor()

    if state["checkpoint_loaded"]:

        print("\nReusing cached PatchCore model...")

        checkpoint_path = None

    else:

        print("\nLoading PatchCore checkpoint...")
        checkpoint_path = str(CHECKPOINT)

    # --------------------------------------------------------
    # RUN INFERENCE
    # --------------------------------------------------------

    print("Running inference...")

    batches = engine.predict(
        model=model,
        ckpt_path=checkpoint_path,
        data_path=str(permanent_image_path),
        return_predictions=True,
    )

    if not batches:
        raise RuntimeError(
            "Model returned no predictions."
        )

    # Mark checkpoint as loaded only after
    # successful inference.
    state["checkpoint_loaded"] = True

    batch = batches[0]

    # --------------------------------------------------------
    # EXTRACT MODEL OUTPUTS
    # --------------------------------------------------------

    score = float(
        as_numpy(batch.pred_score).reshape(-1)[0]
    )

    is_defective = bool(
        as_numpy(batch.pred_label).reshape(-1)[0]
    )

    anomaly_map = extract_map(
        batch.anomaly_map[0]
    )

    predicted_mask = extract_map(
        batch.pred_mask[0]
    )

    # --------------------------------------------------------
    # RESIZE ANOMALY MAP AND MASK
    # --------------------------------------------------------

    height, width = original.shape[:2]

    anomaly_map = cv2.resize(
        anomaly_map,
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )

    predicted_mask = cv2.resize(
        predicted_mask.astype(np.uint8),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )

    predicted_mask = (
        (predicted_mask > 0).astype(np.uint8) * 255
    )

    # --------------------------------------------------------
    # GENERATE HEATMAP
    # --------------------------------------------------------

    # This normalization is for visualization only.
    # It does not affect the model's prediction.

    heatmap_values = cv2.normalize(
        anomaly_map,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    ).astype(np.uint8)

    heatmap = cv2.applyColorMap(
        heatmap_values,
        cv2.COLORMAP_JET,
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB,
    )

    # --------------------------------------------------------
    # GENERATE ANOMALY OVERLAY
    # --------------------------------------------------------

    overlay = cv2.addWeighted(
        original,
        0.65,
        heatmap,
        0.35,
        0,
    )

    # --------------------------------------------------------
    # GENERATE ANNOTATED IMAGE
    # --------------------------------------------------------

    annotated = original.copy()

    contours, _ = cv2.findContours(
        predicted_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    cv2.drawContours(
        annotated,
        contours,
        -1,
        (255, 0, 0),
        3,
    )

    # --------------------------------------------------------
    # DETERMINE MODEL PREDICTION
    # --------------------------------------------------------

    status = (
        "DEFECT"
        if is_defective
        else "GOOD"
    )

    # --------------------------------------------------------
    # PREPARE RESULT RECORD
    # --------------------------------------------------------

    result = {
        "inspection_id": inspection_id,
        "image": str(permanent_image_path),
        "status": status,
        "anomaly_score": round(score, 6),
        "predicted_defect_pixels": int(
            np.count_nonzero(predicted_mask)
        ),
        "model": "PatchCore",
        "checkpoint": str(CHECKPOINT),
        "output_directory": str(save_dir),
    }

    # --------------------------------------------------------
    # SAVE VISUALIZATIONS
    # --------------------------------------------------------

    Image.fromarray(
        heatmap
    ).save(
        save_dir / "heatmap.png"
    )

    Image.fromarray(
        overlay
    ).save(
        save_dir / "overlay.png"
    )

    Image.fromarray(
        predicted_mask
    ).save(
        save_dir / "mask.png"
    )

    Image.fromarray(
        annotated
    ).save(
        save_dir / "annotated.png"
    )

    # --------------------------------------------------------
    # SAVE JSON RESULT
    # --------------------------------------------------------

    with open(
        save_dir / "result.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            result,
            file,
            indent=4,
        )

    # --------------------------------------------------------
    # TERMINAL OUTPUT
    # --------------------------------------------------------

    print("\n" + "=" * 55)
    print("INDUSTRIAL VISIONGUARD - INSPECTION RESULT")
    print("=" * 55)

    print(f"Inspection ID: {inspection_id}")
    print(f"Image:         {permanent_image_path}")
    print(f"Status:        {status}")
    print(f"Score:         {score:.4f}")
    print(f"Output:        {save_dir}")

    print("=" * 55)

    return result


# ============================================================
# COMMAND-LINE INTERFACE
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Run local PatchCore industrial inspection"
        )
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to an industrial component image",
    )

    parser.add_argument(
        "--output",
        default=str(OUTPUT_ROOT),
        help="Directory for generated results",
    )

    args = parser.parse_args()

    predict_image(
        image_path=args.image,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
