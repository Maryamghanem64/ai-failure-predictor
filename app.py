
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="PredictAI",
    page_icon="🎓",
    layout="wide"
)

# ==========================
# MODEL LOADING
# ==========================

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "predictai_model.joblib"
)

saved_data = joblib.load(MODEL_PATH)

model = saved_data["model"]
features = saved_data["features"]

# ==========================
# APPLICATION HEADER
# ==========================

st.title("🎓 PredictAI")










st.sidebar.title("PredictAI")
st.sidebar.caption("Student Early Warning System")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "Class Dashboard",
        "Student Analysis",
        "Model Performance"
    ]
)

st.sidebar.divider()
st.sidebar.success("Model ready")
st.sidebar.caption("Logistic Regression")
st.sidebar.caption(f"{len(features)} model features")
st.sidebar.divider()
st.sidebar.caption("Medad AI | SHIRAA 2026")

if page == "Student Analysis":
    st.title("Individual Student Analysis")
    st.caption("Estimate academic risk and explore support options.")
    st.header("Student Information")

    with st.form("student_form"):

        G1 = st.slider(
            "First-period grade (G1)", 0, 20, 10
        )

        studytime = st.selectbox(
            "Weekly study time",
            [1, 2, 3, 4],
            format_func=lambda x: {
                1: "Less than 2 hours",
                2: "2–5 hours",
                3: "5–10 hours",
                4: "More than 10 hours"
            }[x]
        )

        failures = st.selectbox(
            "Previous failures", [0, 1, 2, 3]
        )

        schoolsup = st.radio(
            "School educational support", ["Yes", "No"]
        )

        famsup = st.radio(
            "Family educational support", ["Yes", "No"]
        )

        paid = st.radio(
            "Extra paid classes", ["Yes", "No"]
        )

        activities = st.radio(
            "Extracurricular activities", ["Yes", "No"]
        )

        higher = st.radio(
            "Wants higher education", ["Yes", "No"]
        )

        submitted = st.form_submit_button(
            "Analyze Student Risk"
        )

    # ==========================
    # STUDENT RISK PREDICTION
    # ==========================

    if submitted:

        student_data = {
            "G1": G1,
            "studytime": studytime,
            "failures": failures,
            "schoolsup": int(schoolsup == "Yes"),
            "famsup": int(famsup == "Yes"),
            "paid": int(paid == "Yes"),
            "activities": int(activities == "Yes"),
            "higher": int(higher == "Yes")
        }

        student_df = pd.DataFrame([student_data])
        student_df = student_df[features]

        risk = model.predict_proba(student_df)[0, 1]

        st.divider()
        st.header("Student Risk Analysis")

        st.metric(
            "Estimated Failure Risk",
            f"{risk:.1%}"
        )

        st.progress(float(risk))

        if G1 < 10:
            st.warning(
                "Baseline: Priority academic review recommended."
            )
        else:
            st.info(
                "Baseline: No alert based on G1 alone."
            )

        st.caption(
            "Experimental, uncalibrated ML estimate. "
            "Not a confirmed outcome or a substitute "
            "for counselor assessment."
        )

        # ==========================
        # EXPLAINABLE AI
        # ==========================

        st.divider()
        st.header("AI Prediction Explanation")

        scaler = model.named_steps["standardscaler"]

        classifier = model.named_steps[
            "logisticregression"
        ]

        scaled_data = scaler.transform(student_df)

        contributions = (
            scaled_data[0] * classifier.coef_[0]
        )

        explanation = pd.DataFrame({
            "Feature": features,
            "Contribution": contributions
        })

        explanation = explanation.sort_values(
            "Contribution",
            ascending=False
        )

        st.bar_chart(
            explanation.set_index("Feature")
        )

        st.dataframe(
            explanation,
            hide_index=True
        )

        st.caption(
            "Contributions are in log-odds relative "
            "to the training-feature means. "
            "Positive values increase the estimated risk; "
            "negative values decrease it. "
            "They do not establish causes of failure."
        )

        # ==========================
        # EDUCATIONAL RECOMMENDATIONS
        # ==========================

        st.divider()
        st.header("Educational Recommendations")

        recommendations = []

        if G1 < 10:
            recommendations.append(
                "Review academic difficulties and identify "
                "learning gaps with the student."
            )

        if studytime == 1:
            recommendations.append(
                "Develop a realistic weekly study plan "
                "with the student."
            )

        if failures > 0:
            recommendations.append(
                "Schedule individual academic counseling "
                "and review previous learning challenges."
            )

        if schoolsup == "No":
            recommendations.append(
                "Explore available school-based "
                "academic support."
            )

        if famsup == "No":
            recommendations.append(
                "Discuss the student's support needs "
                "and available resources."
            )

        if not recommendations:
            recommendations.append(
                "Continue regular academic monitoring "
                "and maintain existing support."
            )

        for recommendation in recommendations:
            st.markdown(f"- {recommendation}")

        st.caption(
            "Rule-based suggestions for counselor review. "
            "Not automatically prescribed interventions."
        )

elif page == "Class Dashboard":
    st.divider()
    st.title("Counselor Dashboard")
    st.caption("Identify students who may benefit from academic support.")

    st.caption(
        "Use synthetic or de-identified student data "
        "for this prototype."
    )

    uploaded_file = st.file_uploader(
        "Upload student CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            students = pd.read_csv(
                uploaded_file,
                dtype={"student_id": "string"}
            )

            required_columns = ["student_id"] + features

            # ==========================
            # DATA VALIDATION
            # ==========================

            if not all(
                col in students.columns
                for col in required_columns
            ):
                raise ValueError(
                    "Missing required columns."
                )

            if students.empty:
                raise ValueError(
                    "The file is empty."
                )

            if students[required_columns].isna().any().any():
                raise ValueError(
                    "Missing student information."
                )

            students["student_id"] = (
                students["student_id"].str.strip()
            )

            if (
                students["student_id"].eq("").any()
                or students["student_id"].duplicated().any()
            ):
                raise ValueError(
                    "Student IDs must be non-empty and unique."
                )

            X_bulk = students[features].copy()

            valid_ranges = {
                "G1": range(21),
                "studytime": [1, 2, 3, 4],
                "failures": [0, 1, 2, 3]
            }

            for col, values in valid_ranges.items():

                X_bulk[col] = pd.to_numeric(
                    X_bulk[col],
                    errors="coerce"
                )

                if not X_bulk[col].isin(values).all():
                    raise ValueError(
                        f"Invalid values in {col}."
                    )

            # ==========================
            # DATA ENCODING
            # ==========================

            categorical_columns = [
                "schoolsup",
                "famsup",
                "paid",
                "activities",
                "higher"
            ]

            for col in categorical_columns:

                X_bulk[col] = (
                    X_bulk[col]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .map({"yes": 1, "no": 0})
                )

            if X_bulk.isna().any().any():
                raise ValueError(
                    "Invalid Yes/No values."
                )

            # ==========================
            # BATCH PREDICTION
            # ==========================

            probabilities = model.predict_proba(
                X_bulk
            )[:, 1]

            results = students[["student_id"]].copy()

            results["G1"] = X_bulk["G1"]

            results["Risk Score (%)"] = (
                probabilities * 100
            )

            results["Priority Review"] = (
                X_bulk["G1"] < 10
            )

            st.success("Class analysis completed!")

            # ==========================
            # COUNSELOR DASHBOARD
            # ==========================

            total = len(results)

            priority = int(
                results["Priority Review"].sum()
            )

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Total Students",
                total
            )

            col2.metric(
                "Priority Review (G1)",
                priority
            )

            col3.metric(
                "No Baseline Alert",
                total - priority
            )

            # ==========================
            # STUDENT RANKING
            # ==========================

            st.subheader("Student Ranking")

            ranking_method = st.radio(
                "Ranking Method",
                ["G1 Baseline", "AI Risk Score"],
                horizontal=True
            )

            if ranking_method == "AI Risk Score":

                ranked = results.sort_values(
                    "Risk Score (%)",
                    ascending=False,
                    kind="stable"
                )

            else:

                ranked = results.sort_values(
                    "G1",
                    ascending=True,
                    kind="stable"
                )

            st.dataframe(
                ranked.round(2),
                hide_index=True
            )

            # ==========================
            # FOLLOW-UP PLAN
            # ==========================

            st.subheader("Counselor Follow-up Plan")

            capacity = st.number_input(
                "Number of students you can follow up with",
                min_value=1,
                max_value=total,
                value=min(3, total),
                step=1
            )

            follow_up = ranked.head(int(capacity))

            st.dataframe(
                follow_up.round(2),
                hide_index=True
            )

            # ==========================
            # RISK VISUALIZATION
            # ==========================


            export_data = follow_up.copy()

            export_data["student_id"] = export_data["student_id"].map(
                lambda value: "'" + str(value)
                if str(value).lstrip().startswith(("=", "+", "-", "@"))
                else str(value)
            )

            csv_data = export_data.round(2).to_csv(index=False)

            st.download_button(
                label="Download Follow-up Report",
                data=csv_data.encode("utf-8-sig"),
                file_name="predictai_followup_report.csv",
                mime="text/csv",
                key="download_followup"
            )

            st.subheader("Student Risk Scores")

            st.bar_chart(
                ranked.set_index("student_id")[
                    "Risk Score (%)"
                ]
            )

            st.caption(
                "G1 is the default ranking method. "
                "AI scores are experimental and uncalibrated. "
                "All follow-up decisions require counselor review."
            )

        except Exception as e:

            st.error(
                f"Unable to analyze file: {e}"
            )

elif page == "Model Performance":
    st.title("Model Performance")
    st.caption("Explore our machine learning experiments and baseline comparison.")

    metrics = pd.DataFrame({
        "Model": [
            "G1 Baseline",
            "Logistic Regression - 3 Features",
            "Logistic Regression - 8 Features",
            "Tuned Random Forest",
            "Original Random Forest"
        ],
        "CV F1 (%)": [
            77.28, 76.51, 75.56, 74.07, 72.24
        ]
    })

    st.subheader("Cross-Validation Results")

    st.dataframe(metrics, hide_index=True)
    st.bar_chart(metrics.set_index("Model"))

    st.subheader("Risk Ranking Comparison")

    ranking_metrics = pd.DataFrame({
        "Method": [
            "G1 Baseline",
            "Original AI",
            "Reduced AI"
        ],
        "Detected out of 50 selections": [44, 43, 42]
    })

    st.dataframe(ranking_metrics, hide_index=True)

    st.info(
        "The traditional baseline performed better "
        "in our exploratory experiments. "
        "AI risk scores remain experimental."
    )

    st.caption(
        "Dataset: UCI Student Performance - Mathematics. "
        "395 records, 316 training and 79 testing. "
        "CV results are exploratory, not independent validation."
    )

# ==========================
# FOOTER
# ==========================

st.divider()

st.caption(
    "Developed by Medad AI | SHIRAA 2026"
)

