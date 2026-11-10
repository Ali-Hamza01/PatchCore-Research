
from pathlib import Path

from anomalib.data import MVTecAD
from anomalib.models import Patchcore
from anomalib.engine import Engine

# =====================================
# 1. PATH CONFIGURATION
# =====================================

DATASET_PATH = Path(r"G:\AI-Research\Dataset\mv-tec ad")
RESULT_PATH = Path(r"G:\AI-Research\PatchCore-Research\results")

CATEGORY = "wood"

# =====================================
# 2. LOAD DATASET
# =====================================

print("\nLoading MVTec AD dataset...")

datamodule = MVTecAD(
    root=DATASET_PATH,
    category=CATEGORY,
    train_batch_size=1,
    eval_batch_size=1,
    num_workers=0,
)

# =====================================
# 3. INITIALIZE PATCHCORE
# =====================================

print("\nInitializing PatchCore...")

model = Patchcore(
    backbone="wide_resnet50_2",
    layers=["layer2", "layer3"],
    pre_trained=True,
    coreset_sampling_ratio=0.1,
    num_neighbors=9,
)

# =====================================
# 4. INITIALIZE ENGINE
# =====================================

engine = Engine(
    accelerator="cpu",
    devices=1,
    default_root_dir=RESULT_PATH,
)

# =====================================
# 5. TRAINING
# =====================================

print("\nStarting PatchCore training...")

engine.fit(
    model=model,
    datamodule=datamodule,
)

print("\nTraining completed!")

# =====================================
# 6. TESTING
# =====================================

print("\nStarting PatchCore testing...")

engine.test(
    model=model,
    datamodule=datamodule,
)

print("\nTesting completed!")

print("\nExperiment finished successfully!")
print("Results location:", RESULT_PATH)
