
from pathlib import Path
import json

import altair as alt
import pandas as pd
import streamlit as st
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPORT_DIR = PROJECT_ROOT / "reports" / "full_evaluation"

PREDICTIONS_CSV = REPORT_DIR / "predictions.csv"
METRICS_JSON = REPORT_DIR / "metrics.json"
LOCALIZATION_JSON = REPORT_DIR / "localization_metrics.json"
LOCALIZATION_CSV = REPORT_DIR / "localization_results.csv"

st.set_page_config(
    page_title="VisionGuard | Model Evaluation",
    page_icon="📊",
    layout="wide",
)


# ============================================================
# LOAD SAVED EVALUATION RESULTS
# ============================================================

@st.cache_data
def load_results():

    required_files = [
        PREDICTIONS_CSV,
        METRICS_JSON,
        LOCALIZATION_JSON,
        LOCALIZATION_CSV,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Required evaluation file missing: {path}"
            )

    predictions = pd.read_csv(PREDICTIONS_CSV)
    localization = pd.read_csv(LOCALIZATION_CSV)

    with METRICS_JSON.open(
        encoding="utf-8"
    ) as file:
        metrics = json.load(file)

    with LOCALIZATION_JSON.open(
        encoding="utf-8"
    ) as file:
        localization_metrics = json.load(file)

    return (
        predictions,
        localization,
        metrics,
        localization_metrics,
    )


try:
    (
        predictions,
        localization,
        metrics,
        localization_metrics,
    ) = load_results()

except Exception as error:
    st.error(str(error))
    st.stop()


# ============================================================
# HEADER
# ============================================================

st.title("Industrial VisionGuard")

st.subheader(
    "PatchCore Model Evaluation Dashboard"
)

st.caption(
    "MVTec AD Metal Nut | "
    "Saved Test-Set Evaluation | "
    "Research Prototype"
)

st.info(
    "This dashboard summarizes a public-dataset benchmark. "
    "It does not establish readiness for industrial deployment."
)

st.divider()


# ============================================================
# SECTION 1 — IMAGE-LEVEL ANOMALY DETECTION
# ============================================================

st.header("1. Image-Level Anomaly Detection")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Classification Accuracy",
    f"{metrics['accuracy'] * 100:.2f}%",
)

col2.metric(
    "Image AUROC",
    f"{metrics['image_auroc'] * 100:.2f}%",
)

col3.metric(
    "Image F1 Score",
    f"{metrics['f1_score'] * 100:.2f}%",
)

col4.metric(
    "Average Inspection",
    f"{metrics['average_inference_seconds']:.3f}s",
)

st.caption(
    f"Evaluated on {metrics['total_images']} test images. "
    "Anomaly scores are not calibrated probabilities."
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

st.subheader("Confusion Matrix")

cm = metrics["confusion_matrix"]

matrix_rows = [
    {
        "Actual": "GOOD",
        "Predicted": "GOOD",
        "Count": cm["true_negative"],
    },
    {
        "Actual": "GOOD",
        "Predicted": "DEFECT",
        "Count": cm["false_positive"],
    },
    {
        "Actual": "DEFECT",
        "Predicted": "GOOD",
        "Count": cm["false_negative"],
    },
    {
        "Actual": "DEFECT",
        "Predicted": "DEFECT",
        "Count": cm["true_positive"],
    },
]

cm_df = pd.DataFrame(matrix_rows)

label_order = ["GOOD", "DEFECT"]

base_cm = alt.Chart(cm_df).encode(
    x=alt.X(
        "Predicted:N",
        title="Predicted Label",
        sort=label_order,
        axis=alt.Axis(labelAngle=0),
    ),
    y=alt.Y(
        "Actual:N",
        title="Actual Label",
        sort=label_order,
    ),
)

cm_heatmap = base_cm.mark_rect().encode(
    color=alt.Color(
        "Count:Q",
        scale=alt.Scale(
            scheme="blues"
        ),
        legend=None,
    ),
    tooltip=[
        "Actual",
        "Predicted",
        "Count",
    ],
)

cm_text = base_cm.mark_text(
    fontSize=24,
    fontWeight="bold",
).encode(
    text=alt.Text("Count:Q"),
    color=alt.condition(
        alt.datum.Count > 46,
        alt.value("white"),
        alt.value("black"),
    ),
)

cm_chart = (
    (cm_heatmap + cm_text)
    .properties(
        width=420,
        height=300,
    )
)

st.altair_chart(
    cm_chart,
    use_container_width=False,
)

st.caption(
    "Rows represent actual labels. "
    "Columns represent model predictions."
)

cm1, cm2, cm3, cm4 = st.columns(4)

cm1.metric(
    "True Negatives",
    cm["true_negative"],
)

cm2.metric(
    "False Positives",
    cm["false_positive"],
)

cm3.metric(
    "False Negatives",
    cm["false_negative"],
)

cm4.metric(
    "True Positives",
    cm["true_positive"],
)


# ============================================================
# IMAGE CLASSIFICATION BY CATEGORY
# ============================================================

st.subheader(
    "Image Classification by Dataset Category"
)

category_rows = []

for category, values in metrics[
    "category_results"
].items():

    category_rows.append({
        "Category": category.title(),
        "Correct": values["correct"],
        "Total": values["total"],
        "Accuracy (%)": round(
            values["accuracy"] * 100,
            2,
        ),
    })

category_df = pd.DataFrame(category_rows)

st.dataframe(
    category_df,
    hide_index=True,
    use_container_width=True,
)

accuracy_chart = (
    alt.Chart(category_df)
    .mark_bar(
        color="#64B5F6",
        cornerRadiusTopLeft=4,
        cornerRadiusTopRight=4,
    )
    .encode(
        x=alt.X(
            "Category:N",
            title="Dataset Category",
            sort=None,
        ),
        y=alt.Y(
            "Accuracy (%):Q",
            title="Classification Accuracy (%)",
            scale=alt.Scale(
                domain=[0, 100]
            ),
        ),
        tooltip=[
            alt.Tooltip(
                "Category:N",
                title="Category",
            ),
            alt.Tooltip(
                "Correct:Q",
                title="Correct",
            ),
            alt.Tooltip(
                "Total:Q",
                title="Total",
            ),
            alt.Tooltip(
                "Accuracy (%):Q",
                title="Accuracy (%)",
                format=".2f",
            ),
        ],
    )
    .properties(height=320)
)

st.altair_chart(
    accuracy_chart,
    use_container_width=True,
)

st.caption(
    "These are binary anomaly-detection results grouped "
    "by ground-truth category. PatchCore does not "
    "classify defect types."
)

st.divider()


# ============================================================
# SECTION 2 — PIXEL-LEVEL LOCALIZATION
# ============================================================

st.header(
    "2. Pixel-Level Anomaly Localization"
)

p1, p2, p3, p4 = st.columns(4)

p1.metric(
    "Pixel Precision",
    f"{localization_metrics['pixel_precision'] * 100:.2f}%",
)

p2.metric(
    "Pixel Recall",
    f"{localization_metrics['pixel_recall'] * 100:.2f}%",
)

p3.metric(
    "Pixel Dice / F1",
    f"{localization_metrics['pixel_dice_f1'] * 100:.2f}%",
)

p4.metric(
    "Pixel IoU",
    f"{localization_metrics['pixel_iou'] * 100:.2f}%",
)

st.caption(
    "These are micro-averaged metrics calculated "
    "from saved binary masks and ground-truth annotations."
)


# ============================================================
# LOCALIZATION PERFORMANCE BY CATEGORY
# ============================================================

st.subheader(
    "Localization Performance by Defect Category"
)

localization_rows = []

for category, values in localization_metrics[
    "category_results"
].items():

    if category == "good":
        continue

    localization_rows.append({
        "Category": category.title(),
        "Dice / F1 (%)": round(
            values["pixel_dice"] * 100,
            2,
        ),
        "IoU (%)": round(
            values["pixel_iou"] * 100,
            2,
        ),
        "Recall (%)": round(
            values["pixel_recall"] * 100,
            2,
        ),
    })

localization_df = pd.DataFrame(
    localization_rows
)

st.dataframe(
    localization_df,
    hide_index=True,
    use_container_width=True,
)

# Convert from wide to long format.
# This lets Altair show grouped bars instead of stacked bars.

localization_long = localization_df.melt(
    id_vars="Category",
    value_vars=[
        "Dice / F1 (%)",
        "IoU (%)",
    ],
    var_name="Metric",
    value_name="Score",
)

localization_chart = (
    alt.Chart(localization_long)
    .mark_bar(
        cornerRadiusTopLeft=3,
        cornerRadiusTopRight=3,
    )
    .encode(
        x=alt.X(
            "Category:N",
            title="Defect Category",
            sort=None,
        ),
        xOffset=alt.XOffset(
            "Metric:N",
            sort=[
                "Dice / F1 (%)",
                "IoU (%)",
            ],
        ),
        y=alt.Y(
            "Score:Q",
            title="Localization Score (%)",
            scale=alt.Scale(
                domain=[0, 100]
            ),
        ),
        color=alt.Color(
            "Metric:N",
            title="Metric",
            scale=alt.Scale(
                domain=[
                    "Dice / F1 (%)",
                    "IoU (%)",
                ],
                range=[
                    "#64B5F6",
                    "#1565C0",
                ],
            ),
        ),
        tooltip=[
            "Category",
            "Metric",
            alt.Tooltip(
                "Score:Q",
                format=".2f",
            ),
        ],
    )
    .properties(height=350)
)

st.altair_chart(
    localization_chart,
    use_container_width=True,
)

st.caption(
    "Dice/F1 and IoU are displayed side by side, "
    "not stacked. GOOD images are excluded because "
    "they contain no positive ground-truth defect pixels."
)

st.divider()


# ============================================================
# SECTION 3 — MISCLASSIFICATION ANALYSIS
# ============================================================

st.header(
    "3. Misclassification Analysis"
)

errors = predictions[
    predictions["true_label"]
    != predictions["predicted_label"]
].copy()

st.write(
    f"**Misclassified images: "
    f"{len(errors)} / {len(predictions)}**"
)

if errors.empty:

    st.success(
        "No image-level misclassifications found."
    )

else:

    error_display = errors[
        [
            "image",
            "category",
            "status",
            "anomaly_score",
        ]
    ]

    st.dataframe(
        error_display,
        hide_index=True,
        use_container_width=True,
    )

    for _, record in errors.iterrows():

        category = record["category"]
        image_name = Path(record["image"]).name

        true_label = (
            "GOOD"
            if int(record["true_label"]) == 0
            else "DEFECT"
        )

        predicted_label = record["status"]

        st.subheader(
            f"{category.title()} — {image_name}"
        )

        st.write(
            f"**Actual:** {true_label} | "
            f"**Predicted:** {predicted_label} | "
            f"**Raw anomaly score:** "
            f"{record['anomaly_score']:.6f}"
        )

        if (
            true_label == "GOOD"
            and predicted_label == "DEFECT"
        ):

            st.warning(
                "False positive: A normal component "
                "was flagged as anomalous."
            )

        elif (
            true_label == "DEFECT"
            and predicted_label == "GOOD"
        ):

            st.error(
                "False negative: An anomalous component "
                "was not flagged."
            )

        folder = Path(
            record["output_directory"]
        )

        image_names = [
            ("Original", "original.png"),
            ("Heatmap", "heatmap.png"),
            ("Overlay", "overlay.png"),
            ("Predicted Mask", "mask.png"),
        ]

        columns = st.columns(4)

        for column, (label, filename) in zip(
            columns,
            image_names,
        ):

            image_path = folder / filename

            with column:

                if image_path.exists():

                    with Image.open(image_path) as img:
                        st.image(
                            img.copy(),
                            caption=label,
                            use_container_width=True,
                        )

                else:

                    st.warning(
                        f"Missing image: {filename}"
                    )

        st.caption(
            "Heatmap colors are normalized for visualization. "
            "They do not represent calibrated probabilities."
        )

        st.divider()


# ============================================================
# SECTION 4 — DETAILED RESULTS
# ============================================================

st.header("4. Detailed Results")

available_categories = sorted(
    predictions["category"].unique().tolist()
)

selected_category = st.selectbox(
    "Filter by category",
    ["All"] + available_categories,
)

if selected_category == "All":

    filtered = predictions.copy()

else:

    filtered = predictions[
        predictions["category"]
        == selected_category
    ].copy()


# Hide long internal file paths from the visible table.
# They remain available in the downloaded CSV.

display_columns = [
    "image",
    "category",
    "true_label",
    "predicted_label",
    "status",
    "anomaly_score",
    "inference_seconds",
    "correct",
]

st.dataframe(
    filtered[display_columns],
    hide_index=True,
    use_container_width=True,
    height=400,
)

st.caption(
    f"Showing {len(filtered)} of "
    f"{len(predictions)} evaluated images."
)


# ============================================================
# CSV DOWNLOADS
# ============================================================

download_col1, download_col2 = st.columns(2)

with download_col1:

    st.download_button(
        "Download Image Predictions CSV",
        data=PREDICTIONS_CSV.read_bytes(),
        file_name="visionguard_predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )

with download_col2:

    st.download_button(
        "Download Localization Results CSV",
        data=LOCALIZATION_CSV.read_bytes(),
        file_name="visionguard_localization.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.divider()


# ============================================================
# SECTION 5 — EVALUATION LIMITATIONS
# ============================================================

st.header("5. Evaluation Limitations")

st.markdown(
    """
    - Evaluated only on the MVTec AD Metal Nut test set.
    - Results do not establish performance on factory camera feeds.
    - PatchCore detects anomalies but does not classify defect types.
    - Image-level and pixel-level metrics measure different tasks.
    - Binary-mask localization metrics depend on the model's
      existing pixel threshold.
    - Heatmaps are visualization aids, not calibrated probabilities.
    - The model's raw anomaly score is not a confidence percentage.
    - The evaluation does not establish engineering acceptance
      or rejection criteria.
    - AI-generated quality-control guidance is advisory only.
    """
)

st.success(
    "Evaluation dashboard loaded successfully."
)
