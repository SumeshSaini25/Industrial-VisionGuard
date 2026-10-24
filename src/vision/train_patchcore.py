from pathlib import Path

import torch
from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Patchcore


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "data" / "raw"
RESULTS_ROOT = PROJECT_ROOT / "results" / "patchcore_baseline"


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    print("=" * 60)
    print("Industrial VisionGuard - PatchCore Baseline")
    print("=" * 60)

    print(f"Dataset: {DATASET_ROOT}")
    print(f"Results: {RESULTS_ROOT}")
    print(f"Device:  {device}")

    # Dataset
    datamodule = MVTecAD(
        root=str(DATASET_ROOT),
        category="metal_nut",
        num_workers=0,
    )

    # PatchCore
    model = Patchcore(
        backbone="wide_resnet50_2",
        layers=["layer2", "layer3"],
        coreset_sampling_ratio=0.1,
    )

    # Engine
    engine = Engine(
        default_root_dir=str(RESULTS_ROOT),
    )

    # Train
    print("\nStarting PatchCore training...")

    engine.fit(
        model=model,
        datamodule=datamodule,
    )

    print("\nTraining complete.")

    # Test
    print("\nStarting evaluation...")

    results = engine.test(
        model=model,
        datamodule=datamodule,
    )

    print("\nEvaluation complete.")
    print(results)


if __name__ == "__main__":
    main()