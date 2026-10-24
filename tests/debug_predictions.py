from pathlib import Path

import torch
from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Patchcore


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = PROJECT_ROOT / "data" / "raw"
RESULTS_ROOT = PROJECT_ROOT / "results" / "patchcore_baseline"

checkpoint = next(
    RESULTS_ROOT.rglob("*.ckpt")
)

datamodule = MVTecAD(
    root=str(DATASET_ROOT),
    category="metal_nut",
    num_workers=0,
)

model = Patchcore(
    backbone="wide_resnet50_2",
    layers=["layer2", "layer3"],
    coreset_sampling_ratio=0.1,
)

engine = Engine(
    default_root_dir=str(RESULTS_ROOT)
)

predictions = engine.predict(
    model=model,
    datamodule=datamodule,
    ckpt_path=str(checkpoint),
    return_predictions=True,
)

print("\n" + "=" * 70)
print("RAW PREDICTION STRUCTURE")
print("=" * 70)

print("Type:", type(predictions))
print("Length:", len(predictions))

for i, batch in enumerate(predictions):

    print(f"\n--- Batch {i} ---")
    print("Type:", type(batch))

    if isinstance(batch, dict):

        print("Keys:")
        for key in batch.keys():
            value = batch[key]

            print(
                f"  {key}: "
                f"type={type(value)}, "
                f"shape={getattr(value, 'shape', None)}, "
                f"length={len(value) if hasattr(value, '__len__') else 'N/A'}"
            )

    else:
        print("Value:", batch)

print("\n" + "=" * 70)
print("END DIAGNOSTIC")
print("=" * 70)