# Industrial VisionGuard

### AI-Based Visual Quality Inspection with Retrieval-Augmented Quality Guidance

**Industrial VisionGuard** is a local-first research prototype for visual anomaly detection and quality-control assistance. It uses **PatchCore** to flag anomalous metal-nut images and localize suspicious regions, then combines a **ChromaDB** knowledge base with a locally running **Qwen3 4B** model to generate *advisory* quality-control guidance. **Streamlit** provides inspection and evaluation dashboards.

> **Scope:** This is a college/research prototype evaluated on the public **MVTec AD Metal Nut** test set. It is not validated for factory deployment, does not determine engineering acceptance, and does not classify specific defect types.

## Features

- Binary **GOOD / DEFECT** anomaly detection with PatchCore
- Raw anomaly score, heatmap, overlay, predicted anomaly mask, and annotated image
- Persistent inspection outputs with unique inspection IDs
- Cached PatchCore predictor for faster repeated inference within a process
- Retrieval of educational quality-control guidance using Sentence Transformers and ChromaDB
- Local Qwen3-powered advisory reports through Ollama; no paid model API required
- Interactive inspection dashboard and downloadable text reports
- Evaluation dashboard with classification metrics, localization metrics, per-category breakdowns, and misclassification examples

## System architecture

```text
Input metal-nut image
        |
        v
PatchCore (trained on normal images)
        |
        +--> GOOD / DEFECT + raw anomaly score
        +--> heatmap / overlay / predicted mask
        |
        v
Fixed inspection findings
        |
        +-------------------------------+
        |                               |
        v                               v
QC question/context              Local knowledge base
        |                        (Markdown sections)
        |                               |
        |                     Sentence Transformer
        |                               |
        |                          ChromaDB
        |                               |
        +---------- retrieved context <-+
        |
        v
Ollama / Qwen3 4B
        |
        v
Advisory QC guidance + source references
        |
        v
Streamlit inspection dashboard / downloadable report

Saved evaluation artifacts --> Streamlit evaluation dashboard
```

**Important:** PatchCore identifies anomalies and their approximate locations. The dataset's *bent*, *color*, *flip*, and *scratch* categories are used for **benchmark analysis**, not as defect-type predictions made by the model.

## Technology stack

| Layer | Tools |
| --- | --- |
| Vision model | PyTorch, anomalib, PatchCore |
| Image processing | OpenCV, Pillow, NumPy |
| Knowledge retrieval | Sentence Transformers (`all-MiniLM-L6-v2`), ChromaDB |
| Local language model | Ollama, `qwen3:4b` |
| Interface | Streamlit |
| Evaluation | Python, CSV/JSON artifacts, dashboard visualizations |

## Dataset

The project uses the **Metal Nut** category of **MVTec AD**:

| Split/category | Images |
| --- | ---: |
| Train / good | 220 |
| Test / good | 22 |
| Test / bent | 25 |
| Test / color | 22 |
| Test / flip | 23 |
| Test / scratch | 23 |
| **Total test images** | **115** |

The local development dataset was obtained from a public Hugging Face mirror of the MVTec AD Metal Nut data. Verify the official dataset terms and the mirror's provenance before redistributing data. The dataset is subject to noncommercial licensing restrictions; **do not commit the dataset to the repository**.

Expected local dataset layout:

```text
data/raw/metal_nut/
├── train/good/
├── test/
│   ├── good/
│   ├── bent/
│   ├── color/
│   ├── flip/
│   └── scratch/
└── ground_truth/
```

## Repository layout

```text
Industrial-VisionGuard/
├── app/
│   ├── dashboard.py
│   └── evaluation_dashboard.py
├── rag/
│   ├── documents/metal_nut_quality_guide.md
│   ├── build_knowledge_base.py
│   ├── search_knowledge_base.py
│   └── rag_assistant.py
├── src/
│   ├── inference/
│   │   ├── predict.py
│   │   └── generate_report.py
│   └── vision/
│       ├── train_patchcore.py
│       ├── evaluate_patchcore.py
│       ├── evaluate_localization.py
│       └── visualize_patchcore.py
├── tests/
├── main.py
├── requirements.txt
└── README.md
```

Generated directories such as `data/`, `results/`, `reports/`, and `rag/chroma_db/` are local artifacts and may be absent from a fresh clone. The repository also contains an experimental `yolo26n.pt` weight file; **YOLO is not used in the final PatchCore inspection pipeline or the benchmark metrics reported below**.

## Local setup

### 1. Create a Python environment

The prototype was developed on an Apple Silicon Mac. The development environment used Python 3.14, though some third-party libraries emitted compatibility warnings. For a new installation, use a Python version supported by the pinned versions of PyTorch, anomalib, and the other dependencies.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate the virtual environment using `.venv\\Scripts\\activate` instead.

### 2. Prepare the dataset and checkpoint

Place the MVTec AD Metal Nut data in `data/raw/metal_nut/` as shown above. Training and inference expect local dataset and model files; they are **not automatically included** in a source-only clone.

The tested inference script expects the trained PatchCore checkpoint at:

```text
results/patchcore_baseline/Patchcore/MVTecAD/metal_nut/v0/weights/lightning/model.ckpt
```

To train a new model, inspect the training configuration and run:

```bash
python src/vision/train_patchcore.py
```

Check the resulting checkpoint path before using inference, as run/version directories can differ.

### 3. Build the local knowledge base

```bash
python rag/build_knowledge_base.py
```

The script indexes sections from `rag/documents/metal_nut_quality_guide.md` into a local ChromaDB collection. The guide is educational content, **not an approved industrial specification**.

Optional retrieval test:

```bash
python rag/search_knowledge_base.py --query "What should happen after an anomaly is detected?"
```

### 4. Install and start Ollama

Install Ollama using the instructions at <https://ollama.com/>. Make sure its local server is running, then pull the language model:

```bash
ollama pull qwen3:4b
```

The report assistant uses Ollama's local API, normally at `http://127.0.0.1:11434`. The AI guidance feature requires Ollama and the model to be available; PatchCore inference itself does not require them.

## Running the project

### Inspect a single image

From the repository root:

```bash
python src/inference/predict.py --image data/raw/metal_nut/test/scratch/000.png
```

Each inspection creates a uniquely named output directory under `reports/inference/`, containing:

```text
original.png
heatmap.png
overlay.png
mask.png
annotated.png
result.json
```

The score in `result.json` is a **raw anomaly score**, not a calibrated confidence percentage. The heatmap is normalized for visualization and must not be interpreted as a probability map.

### Generate advisory guidance from a saved result

```bash
python src/inference/generate_report.py --result path/to/result.json
```

This step uses local retrieval and Qwen3 to suggest manual verification and follow-up checks. The model-generated advice is separate from the fixed, Python-generated inspection findings.

### Launch the inspection dashboard

```bash
streamlit run app/dashboard.py
```

Upload a PNG or JPEG image, run inspection, review the visualizations, and optionally generate and download an advisory report.

### Launch the evaluation dashboard

```bash
streamlit run app/evaluation_dashboard.py
```

The evaluation dashboard reads previously generated benchmark artifacts; it does **not** rerun the full test set each time. Generate those artifacts first using the project's evaluation scripts and verify their expected paths if the dashboard reports missing files.

## Evaluation results

**Dataset:** MVTec AD Metal Nut, 115 test images. The reported results are from a saved test-set evaluation of the trained PatchCore model.

### Image-level anomaly detection

| Metric | Result |
| --- | ---: |
| Accuracy | **98.26%** (113/115) |
| Image AUROC | **99.80%** |
| Image F1 | **98.92%** |
| Average inspection time | **0.318 s/image** |

Confusion matrix (rows = actual, columns = predicted):

| Actual \\ Predicted | GOOD | DEFECT |
| --- | ---: | ---: |
| GOOD | 21 | 1 |
| DEFECT | 1 | 92 |

Image-level correctness grouped by *ground-truth dataset category*:

| Category | Correct / Total |
| --- | ---: |
| Good | 21 / 22 |
| Bent | 25 / 25 |
| Color | 21 / 22 |
| Flip | 23 / 23 |
| Scratch | 23 / 23 |

### Pixel-level anomaly localization

| Metric | Result |
| --- | ---: |
| Pixel precision | **80.52%** |
| Pixel recall | **87.11%** |
| Pixel Dice / F1 | **83.69%** |
| Pixel IoU | **71.95%** |

These aggregate pixel metrics were calculated from saved binary prediction masks and ground-truth annotations.

| Ground-truth defect category | Dice / F1 | IoU | Recall |
| --- | ---: | ---: | ---: |
| Bent | 62.29% | 45.23% | 92.58% |
| Color | 62.84% | 45.82% | 68.53% |
| Flip | 88.85% | 79.93% | 89.42% |
| Scratch | 64.52% | 47.62% | 73.21% |

GOOD images have no positive ground-truth defect pixels and are omitted from the per-defect localization comparison.

### Failure analysis

Two of 115 images were misclassified:

1. **False positive:** `test/good/016.png` was flagged **DEFECT**, score `0.519797`.
2. **False negative:** `test/color/004.png` was predicted **GOOD**, score `0.409667`.

The false-negative example demonstrates why a heatmap with visually noticeable regions does not guarantee that the model's thresholded prediction will flag the image or produce a nonempty mask. These cases are available in the evaluation dashboard for inspection.

## Responsible interpretation and limitations

- The model is trained and tested on a **single public benchmark category**. Performance on production camera feeds, other materials, and changing illumination is unknown.
- **GOOD / DEFECT** denotes the model's anomaly decision, not engineering PASS / FAIL.
- PatchCore **does not classify** bent, color, flip, or scratch defect types.
- Anomaly scores and normalized heatmap colors are **not calibrated probabilities**.
- Localization quality varies by dataset category and depends on the model's thresholded masks.
- The quality guide is educational; retrieved text and LLM responses are **advisory only** and cannot replace approved manufacturing standards or qualified human review.
- The benchmark timing depends on the development machine and cached execution conditions; it is not a production latency guarantee.
- The project has not been validated for industrial deployment or safety-critical acceptance decisions.

## Future work

- Evaluate on additional industrial categories and genuinely unseen capture conditions
- Test robustness to lighting, orientation, blur, and camera changes
- Calibrate anomaly thresholds using a dedicated validation set rather than the final test set
- Improve localization of fine scratches and subtle color anomalies
- Add controlled human-review workflows and validated plant-specific QC documents
- Package reproducible model artifacts and environment versions for deployment testing

## Reproducibility notes

For an independent reproduction, record the exact dependency versions, dataset source and license, training settings, random seeds, checkpoint checksum, and benchmark hardware. The `requirements.txt` and scripts in this repository are the starting point; the source-only checkout does not by itself contain the trained checkpoint or dataset.

## Academic use

Developed as a **B.Tech Artificial Intelligence & Data Science** college project. Results and implementation are presented as an educational research prototype, not as a certified industrial quality-inspection product.
