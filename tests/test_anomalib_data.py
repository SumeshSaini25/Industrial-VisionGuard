from anomalib.data import MVTecAD

DATASET_ROOT = "data/raw"

datamodule = MVTecAD(
    root=DATASET_ROOT,
    category="metal_nut",
)

print("MVTecAD datamodule created successfully.")
print("Dataset root:", DATASET_ROOT)
print("Category:", "metal_nut")

datamodule.setup()

print("\nDataset setup successful.")
print("Train samples:", len(datamodule.train_data))
print("Test samples:", len(datamodule.test_data))