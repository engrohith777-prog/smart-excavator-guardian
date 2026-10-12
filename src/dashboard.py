
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = ROOT / "data" / "processed" / "sensor_features.csv"
MODEL_FILE = ROOT / "models" / "cooler_condition_model.joblib"
REPORT_FILE = ROOT / "reports" / "figures" / "cooler_classification_report.csv"
MATRIX_FILE = ROOT / "reports" / "figures" / "cooler_confusion_matrix.csv"
IMPORTANCE_FILE = ROOT / "reports" / "figures" / "cooler_feature_importance.csv"

EXPECTED_LABELS = [3, 20, 100]

st.set_page_config(
    page_title="Smart Excavator Guardian",
    page_icon="🚜",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. PROFESSIONAL INDUSTRIAL DESIGN
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background-color: #0A1017;
        color: #EAF0F6;
    }

    [data-testid="stSidebar"] {
        background: #101C28;
        border-right: 1px solid #283849;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2 {
        color: #F6B942;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 1.6rem;
        padding-bottom: 2.5rem;
    }

    .hero {
        background: linear-gradient(120deg, #172B3B 0%, #101923 75%);
        border: 1px solid #35495A;
        border-radius: 16px;
        padding: 28px 30px;
        margin-bottom: 20px;
    }

    .eyebrow {
        color: #F6B942;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 3px;
        text-transform: uppercase;
    }

    .hero h1 {
        color: #F6B942;
        font-size: clamp(25px, 3vw, 38px);
        font-weight: 800;
        line-height: 1.2;
        margin: 12px 0 10px 0;
    }

    .hero p {
        color: #BDCAD6;
        font-size: 14px;
        margin-bottom: 0;
    }

    .status-strip {
        border: 1px solid #735D2B;
        background: #282516;
        color: #F3D98E;
        border-radius: 9px;
        padding: 12px 16px;
        margin: 12px 0 24px 0;
        font-size: 12px;
        line-height: 1.7;
    }

    .section-kicker {
        color: #F6B942;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 2px;
        margin-bottom: 5px;
    }

    div[data-testid="stMetric"] {
        background: #142330;
        border: 1px solid #304657;
        border-radius: 12px;
        padding: 17px 18px;
    }

    div[data-testid="stMetricLabel"] {
        color: #AFC0CF;
        font-size: 12px;
    }

    div[data-testid="stMetricValue"] {
        color: #F4F7FA;
        font-weight: 700;
    }

    div[data-testid="stMetricDelta"] {
        font-size: 12px;
    }

    .stButton > button,
    .stDownloadButton > button {
        border-radius: 8px;
        min-height: 42px;
        font-weight: 700;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #293B4B;
        border-radius: 9px;
        overflow: hidden;
    }

    hr {
        border-color: #283849;
        margin: 24px 0;
    }

    .small-note {
        color: #96A8B8;
        font-size: 12px;
        line-height: 1.6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. FILE LOADING AND MODEL RECOVERY
# ============================================================

@st.cache_data
def load_sensor_data(file_path):
    frame = pd.read_csv(file_path)

    if frame.empty:
        raise ValueError("The sensor feature file is empty.")

    if frame.columns.duplicated().any():
        raise ValueError("Duplicate sensor feature names were found.")

    if not all(pd.api.types.is_numeric_dtype(frame[c]) for c in frame.columns):
        raise ValueError("Sensor features must all be numeric.")

    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError("Sensor features contain missing or infinite values.")

    return frame


def find_predictor(obj, seen=None):
    """
    Recover a fitted estimator from a saved model object or nested dictionary.
    Returns the estimator or None.
    """
    if seen is None:
        seen = set()

    object_id = id(obj)

    if object_id in seen:
        return None

    seen.add(object_id)

    if callable(getattr(obj, "predict", None)):
        return obj

    if isinstance(obj, dict):
        # Check likely estimator keys first.
        preferred_keys = [
            "model",
            "classifier",
            "estimator",
            "best_model",
            "pipeline",
            "random_forest",
            "rf_model",
            "trained_model",
        ]

        for key in preferred_keys:
            if key in obj:
                found = find_predictor(obj[key], seen)
                if found is not None:
                    return found

        # Search remaining values for a fitted estimator.
        for value in obj.values():
            found = find_predictor(value, seen)
            if found is not None:
                return found

    elif isinstance(obj, (list, tuple)):
        for value in obj:
            found = find_predictor(value, seen)
            if found is not None:
                return found

    return None


@st.cache_resource
def load_trained_model(file_path):
    saved_object = joblib.load(file_path)
    estimator = find_predictor(saved_object)

    if estimator is None:
        if isinstance(saved_object, dict):
            details = ", ".join(map(str, saved_object.keys()))
        else:
            details = type(saved_object).__name__

        raise ValueError(
            "No object with a predict() method was found in the saved model. "
            f"Saved content: {details}"
        )

    return estimator


def align_features(frame, estimator):
    """
    Align input columns to the fitted model's expected feature order.
    Reject incompatible inputs instead of silently predicting with wrong columns.
    """
    expected_names = getattr(estimator, "feature_names_in_", None)

    if expected_names is not None:
        expected_names = list(expected_names)

        if set(expected_names) != set(frame.columns):
            missing = sorted(set(expected_names) - set(frame.columns))
            extra = sorted(set(frame.columns) - set(expected_names))

            raise ValueError(
                f"Model/data feature mismatch. Missing: {missing}. "
                f"Unexpected: {extra}."
            )

        frame = frame[expected_names]

    expected_count = getattr(estimator, "n_features_in_", None)

    if expected_count is not None and frame.shape[1] != expected_count:
        raise ValueError(
            f"The model expects {expected_count} features, "
            f"but the data contains {frame.shape[1]}."
        )

    return frame


# ============================================================
# 4. SIDEBAR NAVIGATION
# ============================================================

st.sidebar.markdown("# 🚜 EXCAVATOR")
st.sidebar.markdown("## GUARDIAN")
st.sidebar.caption("INTELLIGENT HYDRAULIC MONITORING")
st.sidebar.divider()

page = st.sidebar.radio(
    "WORKSPACE",
    [
        "System Overview",
        "AI Diagnostics",
        "Model Performance",
        "Sensor Explorer",
    ],
)

st.sidebar.divider()

st.sidebar.markdown("**PROJECT STATUS**")
st.sidebar.caption("Mode: Offline research prototype")
st.sidebar.caption("Source: Hydraulic test-rig dataset")
st.sidebar.caption("Interface: English")

st.sidebar.divider()
st.sidebar.markdown("**SYSTEM SCOPE**")
st.sidebar.caption(
    "Recorded sensor analysis only. No live excavator connection."
)


# ============================================================
# 5. LOAD PROJECT RESOURCES
# ============================================================

for required_file in [DATA_FILE, MODEL_FILE]:
    if not required_file.exists():
        st.error(f"Required project file is missing: {required_file}")
        st.stop()

try:
    sensor_data = load_sensor_data(str(DATA_FILE))
    model = load_trained_model(str(MODEL_FILE))
    sensor_data = align_features(sensor_data, model)

except Exception as error:
    st.error(f"Could not initialize the application: {error}")
    st.info(
        "Check the model file, sensor feature file, and model training script. "
        "The dashboard has stopped safely rather than making an unreliable prediction."
    )
    st.stop()


# ============================================================
# 6. GLOBAL HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Intelligent Condition Monitoring Platform</div>
        <h1>SMART EXCAVATOR GUARDIAN</h1>
        <p>
            Hydraulic sensor intelligence · AI-assisted classification ·
            Engineering diagnostics
        </p>
    </div>
    <div class="status-strip">
        <strong>RESEARCH PROTOTYPE</strong> &nbsp;|&nbsp;
        Uses recorded hydraulic test-rig measurements. Predictions have not
        been validated for real-world excavator safety or maintenance decisions.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 7. SYSTEM OVERVIEW
# ============================================================

if page == "System Overview":

    st.markdown('<div class="section-kicker">OVERVIEW</div>', unsafe_allow_html=True)
    st.title("System Overview")
    st.caption("A structured overview of the recorded dataset and analysis tools.")

    a, b, c, d = st.columns(4)

    a.metric("RECORDED CYCLES", f"{len(sensor_data):,}")
    b.metric("SENSOR FEATURES", f"{sensor_data.shape[1]}")
    c.metric("MODEL", "LOADED")
    d.metric("DATA MODE", "OFFLINE")

    st.divider()

    left, right = st.columns([1.7, 1])

    with left:
        st.subheader("Sensor Signal Analysis")

        selected_feature = st.selectbox(
            "Select a feature",
            sensor_data.columns.tolist(),
            key="overview_sensor",
        )

        chart_df = pd.DataFrame({
            "Recorded cycle": np.arange(1, len(sensor_data) + 1),
            "Sensor value": sensor_data[selected_feature].to_numpy(),
        })

        fig = px.line(
            chart_df,
            x="Recorded cycle",
            y="Sensor value",
            title=selected_feature,
        )

        fig.update_traces(line=dict(color="#F6B942", width=1.8))
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#101923",
            height=370,
            margin=dict(l=15, r=15, t=55, b=15),
        )

        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Feature Statistics")

        stats = sensor_data[selected_feature].describe()
        stats_df = stats.rename("Value").to_frame()
        stats_df["Value"] = stats_df["Value"].map(lambda x: f"{x:.4f}")

        st.dataframe(stats_df, use_container_width=True)

        st.subheader("Available Tools")
        st.markdown(
            """
            - **AI Diagnostics:** classify a recorded sample.
            - **Model Performance:** review saved evaluation results.
            - **Sensor Explorer:** investigate selected cycles.
            """
        )

    st.divider()
    st.subheader("Recent Dataset Rows")
    st.dataframe(sensor_data.tail(10), use_container_width=True)


# ============================================================
# 8. AI DIAGNOSTICS
# ============================================================

elif page == "AI Diagnostics":

    st.markdown('<div class="section-kicker">CLASSIFICATION ENGINE</div>', unsafe_allow_html=True)
    st.title("AI Diagnostics")
    st.caption("Run the trained classifier on one recorded sensor cycle.")

    st.info(
        "This model predicts the cooler-condition label used during training. "
        "It does not independently diagnose all hydraulic faults or calculate "
        "a validated machine-health score."
    )

    cycle = st.number_input(
        "Recorded cycle number",
        min_value=1,
        max_value=len(sensor_data),
        value=1,
        step=1,
    )

    if st.button("RUN CONDITION ANALYSIS", type="primary", use_container_width=True):

        sample = sensor_data.iloc[[int(cycle) - 1]]

        try:
            prediction = model.predict(sample)[0]

            st.divider()
            st.subheader("Classification Result")

            r1, r2 = st.columns(2)

            r1.metric("PREDICTED CONDITION LABEL", str(prediction))
            r2.metric("ANALYZED CYCLE", f"{int(cycle):,}")

            if callable(getattr(model, "predict_proba", None)):
                probabilities = model.predict_proba(sample)[0]
                classes = model.classes_

                probability_df = pd.DataFrame({
                    "Condition label": [str(x) for x in classes],
                    "Estimated probability (%)": probabilities * 100,
                })

                fig = px.bar(
                    probability_df,
                    x="Condition label",
                    y="Estimated probability (%)",
                    text_auto=".1f",
                    title="Model Output Probabilities",
                )

                fig.update_traces(marker_color="#F6B942")
                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="#101923",
                    yaxis_range=[0, 100],
                    height=370,
                )

                st.plotly_chart(fig, use_container_width=True)

                st.caption(
                    "These are model probabilities, not guaranteed real-world "
                    "confidence or a measure of machine safety."
                )
            else:
                st.info(
                    "This estimator does not expose predict_proba(). "
                    "Only its predicted label is displayed."
                )

            with st.expander("Inspect input feature values"):
                st.dataframe(
                    sample.T.rename(columns={sample.index[0]: "Sensor value"}),
                    use_container_width=True,
                )

        except Exception as error:
            st.error(f"Prediction failed: {error}")

    st.divider()
    st.subheader("Interpretation and Limitations")
    st.markdown(
        """
        - A prediction is a classification result for a recorded test sample.
        - Confirm the label definitions using the original dataset documentation.
        - Do not use this prototype to make real equipment safety decisions.
        """
    )


# ============================================================
# 9. MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    st.markdown('<div class="section-kicker">VALIDATION & EVALUATION</div>', unsafe_allow_html=True)
    st.title("Model Performance")
    st.caption("Review the previously saved held-out test evaluation.")

    if REPORT_FILE.exists():

        try:
            report = pd.read_csv(REPORT_FILE, index_col=0)

            if "accuracy" in report.index and "f1-score" in report.columns:
                accuracy = float(report.loc["accuracy", "f1-score"])
                st.metric("SAVED TEST ACCURACY", f"{accuracy * 100:.2f}%")
            else:
                st.warning(
                    "The report was loaded, but its accuracy row could not be identified."
                )

            st.subheader("Classification Report")
            st.dataframe(report, use_container_width=True)

        except Exception as error:
            st.error(f"Could not read the classification report: {error}")

    else:
        st.warning(
            "Classification report not found. Run "
            "`python src/evaluate_model_details.py` first."
        )

    if MATRIX_FILE.exists():

        try:
            matrix_df = pd.read_csv(MATRIX_FILE, index_col=0)
            matrix_values = matrix_df.to_numpy()

            st.divider()
            st.subheader("Confusion Matrix")
            st.caption("Rows represent actual labels; columns represent predicted labels.")

            fig = px.imshow(
                matrix_values,
                x=[str(x) for x in matrix_df.columns],
                y=[str(x) for x in matrix_df.index],
                text_auto=True,
                aspect="auto",
                color_continuous_scale=[
                    [0, "#172635"],
                    [0.5, "#367AA5"],
                    [1, "#F6B942"],
                ],
                labels={
                    "x": "Predicted label",
                    "y": "Actual label",
                    "color": "Sample count",
                },
                title="Held-Out Test Predictions",
            )

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                height=440,
            )

            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(matrix_df, use_container_width=True)

        except Exception as error:
            st.error(f"Could not display the confusion matrix: {error}")

    else:
        st.warning(
            "Confusion matrix not found. Run "
            "`python src/evaluate_model_details.py` first."
        )

    if IMPORTANCE_FILE.exists():
        st.divider()
        st.subheader("Saved Feature-Importance Analysis")

        try:
            importance = pd.read_csv(IMPORTANCE_FILE)
            st.dataframe(importance.head(15), use_container_width=True)
        except Exception as error:
            st.warning(f"Could not read feature importance: {error}")

    st.warning(
        "The saved 82.36% accuracy came from one grouped train/test split. "
        "It does not establish performance on unseen excavators or field conditions."
    )


# ============================================================
# 10. SENSOR EXPLORER
# ============================================================

elif page == "Sensor Explorer":

    st.markdown('<div class="section-kicker">SIGNAL INVESTIGATION</div>', unsafe_allow_html=True)
    st.title("Sensor Explorer")
    st.caption("Investigate individual features and export selected readings.")

    feature = st.selectbox(
        "Sensor feature",
        sensor_data.columns.tolist(),
        key="explorer_sensor",
    )

    c1, c2 = st.columns(2)

    with c1:
        first_cycle = st.number_input(
            "First cycle",
            min_value=1,
            max_value=len(sensor_data),
            value=1,
            step=1,
        )

    with c2:
        last_cycle = st.number_input(
            "Last cycle",
            min_value=int(first_cycle),
            max_value=len(sensor_data),
            value=min(len(sensor_data), int(first_cycle) + 199),
            step=1,
        )

    start = int(first_cycle) - 1
    stop = int(last_cycle)

    selected = sensor_data.iloc[start:stop][feature]

    cycle_df = pd.DataFrame({
        "Recorded cycle": np.arange(int(first_cycle), int(last_cycle) + 1),
        "Sensor value": selected.to_numpy(),
    })

    fig = px.line(
        cycle_df,
        x="Recorded cycle",
        y="Sensor value",
        title=f"{feature} · Selected Cycle Range",
    )

    fig.update_traces(line=dict(color="#F6B942", width=2))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#101923",
        height=400,
    )

    st.plotly_chart(fig, use_container_width=True)

    a, b, c, d = st.columns(4)
    a.metric("MEAN", f"{selected.mean():.4f}")
    b.metric("STANDARD DEVIATION", f"{selected.std():.4f}")
    c.metric("MINIMUM", f"{selected.min():.4f}")
    d.metric("MAXIMUM", f"{selected.max():.4f}")

    st.subheader("Selected Readings")
    st.dataframe(cycle_df, use_container_width=True)

    st.download_button(
        label="DOWNLOAD SELECTED DATA AS CSV",
        data=cycle_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{feature}_cycles_{int(first_cycle)}_{int(last_cycle)}.csv",
        mime="text/csv",
        use_container_width=True,
    )


# ============================================================
# 11. FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div class="small-note">
        <strong>SMART EXCAVATOR GUARDIAN</strong><br>
        AI-assisted hydraulic condition monitoring research prototype.<br>
        Data source: hydraulic test rig · Interface language: English ·
        Live machine connection: Not available
    </div>
    """,
    unsafe_allow_html=True,
)
