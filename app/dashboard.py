
from pathlib import Path
import sys
import json
import hashlib
import tempfile
import uuid
from datetime import datetime
from io import BytesIO

import streamlit as st
from PIL import Image


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.predict import predict_image
from src.inference.generate_report import generate_inspection_report


REPORTS_DIR = PROJECT_ROOT / "reports" / "dashboard"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Industrial VisionGuard",
    page_icon="🔍",
    layout="wide",
)

st.title("Industrial VisionGuard")

st.caption(
    "AI-Based Visual Quality Inspection with "
    "Retrieval-Augmented Quality-Control Guidance"
)

st.divider()


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "inspection_result": None,
    "inspection_folder": None,
    "ai_report": None,
    "uploaded_file_hash": None,
}

for key, default_value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = default_value


def clear_inspection():
    st.session_state.inspection_result = None
    st.session_state.inspection_folder = None
    st.session_state.ai_report = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("System Information")

    st.markdown(
        """
        **Vision Model:** PatchCore

        **Inspection Dataset:** MVTec AD — Metal Nut

        **Knowledge Retrieval:** ChromaDB

        **Embedding Model:** all-MiniLM-L6-v2

        **Local Language Model:** Qwen3 4B

        **LLM Runtime:** Ollama
        """
    )

    st.divider()

    st.info(
        "PatchCore detects visual anomalies. "
        "It does not independently classify defect types "
        "or determine engineering acceptability."
    )

    st.caption(
        "Educational prototype. Not an approved "
        "industrial inspection system."
    )


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.header("1. Upload Component Image")

uploaded_file = st.file_uploader(
    "Upload a metal nut image",
    type=["png", "jpg", "jpeg"],
)

if uploaded_file is None:

    if st.session_state.uploaded_file_hash is not None:
        clear_inspection()
        st.session_state.uploaded_file_hash = None

    st.info("Upload an image to begin inspection.")

    st.stop()


uploaded_bytes = uploaded_file.getvalue()

current_hash = hashlib.sha256(
    uploaded_bytes
).hexdigest()

if current_hash != st.session_state.uploaded_file_hash:

    clear_inspection()

    st.session_state.uploaded_file_hash = current_hash


try:

    image = Image.open(
        BytesIO(uploaded_bytes)
    ).convert("RGB")

except Exception as error:

    st.error(f"Unable to open image: {error}")

    st.stop()


st.image(
    image,
    caption="Uploaded component",
    width="stretch",
)


# ============================================================
# RUN PATCHCORE INSPECTION
# ============================================================

st.header("2. AI Visual Inspection")

if st.button(
    "Run AI Inspection",
    type="primary",
    width="stretch",
):

    clear_inspection()

    try:

        with st.spinner(
            "Running PatchCore anomaly detection..."
        ):

            # A unique temporary folder prevents
            # output collisions between inspections.
            unique_id = uuid.uuid4().hex[:12]

            with tempfile.TemporaryDirectory(
                prefix=f"visionguard_{unique_id}_"
            ) as temporary_folder:

                temporary_path = Path(temporary_folder)

                image_path = (
                    temporary_path / "uploaded_component.png"
                )

                image.save(image_path)

                result = predict_image(
                    image_path=image_path,
                    output_dir=REPORTS_DIR,
                )

                # predict_image saves its output in a folder
                # named after the image's parent and stem.
                inspection_folder = Path(
                    result["output_directory"]
                    
                )

            if not inspection_folder.exists():

                raise FileNotFoundError(
                    "PatchCore completed, but the "
                    "inspection output folder was not found."
                )

            st.session_state.inspection_result = result

            st.session_state.inspection_folder = str(
                inspection_folder
            )

            st.session_state.ai_report = None

        st.success("Visual inspection completed.")

    except Exception as error:

        st.error(
            f"Inspection failed: {error}"
        )


# ============================================================
# DISPLAY INSPECTION RESULTS
# ============================================================

result = st.session_state.inspection_result

if result is None:
    st.stop()


inspection_folder = Path(
    st.session_state.inspection_folder
)

status = result["status"]

score = float(
    result["anomaly_score"]
)

st.subheader("PatchCore Results")

col1, col2 = st.columns(2)

with col1:

    st.metric(
        "Model Prediction",
        status,
    )

with col2:

    st.metric(
        "Raw Anomaly Score",
        f"{score:.4f}",
    )


if status == "DEFECT":

    st.warning(
        "PatchCore flagged a visual anomaly. "
        "This is not an automatic engineering rejection."
    )

elif status == "GOOD":

    st.success(
        "PatchCore classified the image as GOOD. "
        "This is not an engineering compliance certification."
    )

else:

    st.error(
        f"Unexpected model prediction: {status}"
    )


# ============================================================
# ANOMALY VISUALIZATIONS
# ============================================================

st.subheader("Anomaly Visualization")

visualization_files = [
    ("Original Image", "original.png"),
    ("Anomaly Heatmap", "heatmap.png"),
    ("Anomaly Overlay", "overlay.png"),
    ("Predicted Anomaly Mask", "mask.png"),
]

left_column, right_column = st.columns(2)

for index, (title, filename) in enumerate(
    visualization_files
):

    image_file = inspection_folder / filename

    column = (
        left_column
        if index % 2 == 0
        else right_column
    )

    with column:

        if image_file.exists():

            st.image(
                str(image_file),
                caption=title,
                width="stretch",
            )

        else:

            st.warning(
                f"Visualization unavailable: {filename}"
            )


with st.expander("View Raw PatchCore Output"):

    st.json(result)


st.divider()


# ============================================================
# GENERATE RAG REPORT
# ============================================================

st.header("3. AI-Assisted Quality-Control Report")

st.write(
    "Generate advisory quality-control guidance "
    "using retrieved documents and the local Qwen3 model."
)

if st.button(
    "Generate AI Quality Report",
    width="stretch",
):

    try:

        with st.spinner(
            "Retrieving quality-control guidance "
            "and generating the report..."
        ):

            ai_report = generate_inspection_report(
                result
            )

            st.session_state.ai_report = ai_report

        st.success(
            "AI quality-control report generated."
        )

    except Exception as error:

        st.session_state.ai_report = None

        st.error(
            f"Report generation failed: {error}"
        )


# ============================================================
# DISPLAY QUALITY REPORT
# ============================================================

report = st.session_state.ai_report

if report is None:
    st.stop()


# ------------------------------------------------------------
# DETERMINISTIC MODEL FINDING
# ------------------------------------------------------------

st.subheader("Verified Model Output")

finding = report["finding"]

finding_col1, finding_col2 = st.columns(2)

with finding_col1:

    st.metric(
        "PatchCore Prediction",
        finding["status"],
    )

with finding_col2:

    st.metric(
        "Raw Anomaly Score",
        f"{finding['anomaly_score']:.4f}",
    )

st.info(
    finding["interpretation"]
)

st.caption(
    "This interpretation is generated by fixed Python rules, "
    "not by the language model. The underlying prediction "
    "is still a model estimate, not an independently verified "
    "engineering finding."
)


# ------------------------------------------------------------
# QWEN ADVISORY GUIDANCE
# ------------------------------------------------------------

st.subheader("AI-Generated Advisory Guidance")

st.markdown(
    report["report"]
)

st.caption(
    "The advisory guidance is generated by Qwen3 using "
    "retrieved reference material. It may require "
    "independent verification."
)


# ------------------------------------------------------------
# RETRIEVED KNOWLEDGE SOURCES
# ------------------------------------------------------------

with st.expander(
    "View Retrieved Knowledge Sources"
):

    for index, source in enumerate(
        report["sources"],
        start=1,
    ):

        st.markdown(
            f"**Source {index}: "
            f"{source['source']}**"
        )

        st.caption(
            "Cosine distance: "
            f"{source['distance']:.4f}"
        )

        st.text(
            source["content"]
        )

        st.divider()


# ============================================================
# DOWNLOAD REPORT
# ============================================================

st.header("4. Download Inspection Report")


def create_text_report(report_data):

    inspection = report_data["inspection"]

    finding = report_data["finding"]

    sources = report_data["sources"]

    lines = [
        "INDUSTRIAL VISIONGUARD",
        "AI-ASSISTED QUALITY-CONTROL REPORT",
        "=" * 60,
        "",
        "INSPECTION SUMMARY",
        "-" * 60,
        f"Model prediction: {finding['status']}",
        (
            "Raw anomaly score: "
            f"{finding['anomaly_score']:.4f}"
        ),
        "",
        "The anomaly score is not a probability.",
        "",
        "DETERMINISTIC INSPECTION INTERPRETATION",
        "-" * 60,
        finding["interpretation"],
        "",
        "AI-GENERATED ADVISORY GUIDANCE",
        "-" * 60,
        report_data["report"],
        "",
        "RETRIEVED KNOWLEDGE SOURCES",
        "-" * 60,
    ]

    for index, source in enumerate(
        sources,
        start=1,
    ):

        lines.extend(
            [
                f"[{index}] {source['source']}",
                (
                    "Cosine distance: "
                    f"{source['distance']:.4f}"
                ),
                source["content"],
                "",
            ]
        )

    lines.extend(
        [
            "",
            "DISCLAIMER",
            "-" * 60,
            (
                "This is an educational AI-assisted "
                "inspection prototype."
            ),
            (
                "PatchCore predictions do not establish "
                "engineering compliance or final acceptance."
            ),
            (
                "AI-generated recommendations must be "
                "verified against approved specifications."
            ),
        ]
    )

    return "\n".join(lines)


text_report = create_text_report(
    report
)

st.download_button(
    label="Download Quality-Control Report (.txt)",
    data=text_report,
    file_name="visionguard_quality_report.txt",
    mime="text/plain",
    width="stretch",
)


st.divider()

st.caption(
    "Industrial VisionGuard | "
    "PatchCore + ChromaDB + Local Qwen3"
)
