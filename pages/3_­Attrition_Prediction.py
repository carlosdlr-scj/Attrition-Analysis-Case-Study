import streamlit as st
import pandas as pd
import plotly.express as px
import shap
from model_utils import train_models

st.title("🤖 Employee Attrition Prediction")
st.caption("Business questions 3–4: What employee profiles appear to be at higher risk, and can a predictive model support proactive retention efforts?")

df, results, fitted, best = train_models()

show = results.copy()
for c in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]:
    show[c] = show[c].map(lambda x: f"{x:.3f}")

st.markdown("### Model comparison")
st.dataframe(show, use_container_width=True, hide_index=True)
with st.expander("CLICK FOR INSIGHTS"):
    st.write(
        "All the different metrics have something to add to model's understanding, but for this case I "
        " decided to use recall as the top metric, due to the importance of intercepting the biggest number"
        " of members who are about to leave the company. Financially speaking, it's important to prevent employee"
        " turnover from real at-risk members. Regarding this prior statement, is also important to check precision,"
        " because we don't want to send multiple incentives to people who are not truly in risk, "
        " losing money on non-existent risks. The table showed a tight battle between two models, random forest"
        " and gradient boosting, so the decission point could be to prioritize precision score, where the gradient boosting"
        " model is a clear winner. But due to the fact that the better model is computationally expensive, we decided to "
        " proceed with random forest as the chosen model." 
    )
st.success(
    f"Selected model: **{best}** — highest recall ({results.iloc[0].Recall:.1%}). "
    "Each model was first tuned with RandomizedSearchCV (15 iterations) and then refined with GridSearchCV. "
    "The interactive tool uses the final GridSearchCV model."
)

with st.expander("How the hyperparameter search was performed"):
    st.write(
        "RandomizedSearchCV explored a broad parameter space with 15 random combinations per model, "
        "using 3-fold cross-validation and recall as the optimization metric. GridSearchCV then searched "
        "a smaller refinement grid around each model's best randomized configuration. The final model was "
        "selected by highest holdout recall, keeping the same 80/20 stratified train/test split and random_state=42."
    )

st.markdown("### Interactive employee profile")
a, b, c = st.columns(3)
with a:
    satisfaction = st.number_input("Satisfaction level", 0.0, 1.0, 0.60, 0.01)
    evaluation = st.number_input("Last evaluation", 0.0, 1.0, 0.70, 0.01)
    projects = st.number_input("Number of projects", 1, 20, 4, 1)
with b:
    hours = st.number_input("Average monthly hours", 1, 500, 180, 1)
    tenure = st.number_input("Years at company", 1, 20, 3, 1)
    accident = st.selectbox("Work accident", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
with c:
    promotion = st.selectbox("Promotion in last 5 years", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    department = st.selectbox("Department", sorted(df.Department.unique()))
    salary = st.selectbox("Salary level", sorted(df.salary.unique()))

# Keep the current employee profile available so SHAP can explain this exact prediction.
emp = pd.DataFrame([{
    "satisfaction_level": satisfaction,
    "last_evaluation": evaluation,
    "number_project": projects,
    "average_montly_hours": hours,
    "time_spend_company": tenure,
    "Work_accident": accident,
    "promotion_last_5years": promotion,
    "Department": department,
    "salary": salary,
}])

if st.button("Estimate attrition risk", type="primary"):
    p = float(fitted[best].predict_proba(emp)[0, 1])
    pred = int(p >= 0.5)
    x, y = st.columns(2)
    x.metric("Estimated probability of leaving", f"{p:.1%}")
    y.metric("Prediction", "Higher likelihood of leaving" if pred else "Lower likelihood of leaving")
    (st.warning if pred else st.info)(
        "This profile is classified as higher likelihood of attrition at the 0.50 threshold."
        if pred else
        "This profile is classified as lower likelihood of attrition at the 0.50 threshold."
    )

st.caption("Model probabilities are estimates, not certainties. Use them as an early-warning signal and not as the sole basis for employment decisions.")
with st.expander("What should we expect from the chosen model?"):
    st.write(
        "After testing the data, the random forest model presented a recall score of 0.93, so around 9 out 10 "
        " employees who are about to leave the company, we would be able to identify and approach them to try" 
        " to give them an agreement before quitting."
    )
# ---------------------------------------------------------------------------
# SHAP explainability
# ---------------------------------------------------------------------------
st.markdown("### SHAP explainability")
st.caption(
    "SHAP explains the winning Random Forest by showing how individual features "
    "move predictions toward leaving or staying. These are model explanations, "
    "not causal effects."
)

try:
    winning_pipeline = fitted[best]
    preprocessor = winning_pipeline.named_steps["prep"]
    estimator = winning_pipeline.named_steps["model"]

    # The selected model is Random Forest based on the current validation results.
    if best != "Random Forest":
        st.info(
            f"The current winning model is {best}. The SHAP section below is configured "
            "for tree-based models and will be skipped if the winner is not a compatible tree model."
        )
        raise ValueError("Current winner is not the tree-based Random Forest expected by this SHAP visualization.")

    X = df[
        [
            "satisfaction_level",
            "last_evaluation",
            "number_project",
            "average_montly_hours",
            "time_spend_company",
            "Work_accident",
            "promotion_last_5years",
            "Department",
            "salary",
        ]
    ]

    # Use a representative sample for the global explanation to keep the page responsive.
    X_sample = X.sample(n=min(1000, len(X)), random_state=42)
    X_sample_t = preprocessor.transform(X_sample)
    if hasattr(X_sample_t, "toarray"):
        X_sample_t = X_sample_t.toarray()

    feature_names = preprocessor.get_feature_names_out()
    explainer = shap.TreeExplainer(estimator)
    shap_sample = explainer.shap_values(X_sample_t)

    # SHAP's binary-classifier return shape differs between versions.
    if isinstance(shap_sample, list):
        shap_left = shap_sample[1]
    elif getattr(shap_sample, "ndim", 0) == 3:
        shap_left = shap_sample[:, :, 1]
    else:
        shap_left = shap_sample

    shap_left = pd.DataFrame(shap_left, columns=feature_names)

    st.markdown("#### Overall explanation of the winning model")
    st.caption(
        "Mean absolute SHAP value across a representative sample. Larger values mean "
        "the feature has a stronger influence on the model's attrition predictions."
    )

    global_importance = (
        shap_left.abs()
        .mean()
        .sort_values(ascending=False)
        .head(15)
        .sort_values()
        .reset_index()
    )
    global_importance.columns = ["Feature", "Mean absolute SHAP"]

    fig_global = px.bar(
        global_importance,
        x="Mean absolute SHAP",
        y="Feature",
        orientation="h",
        title="Top features influencing attrition predictions",
        labels={
            "Mean absolute SHAP": "Mean absolute SHAP value",
            "Feature": "Model feature",
        },
    )
    st.plotly_chart(fig_global, use_container_width=True)

    st.markdown("#### Explanation for the current employee profile")
    st.caption(
        "Positive SHAP values push this profile toward the model's **Left** class; "
        "negative values push it toward **Stayed**."
    )

    emp_t = preprocessor.transform(emp)
    if hasattr(emp_t, "toarray"):
        emp_t = emp_t.toarray()

    shap_emp = explainer.shap_values(emp_t)
    if isinstance(shap_emp, list):
        shap_left_emp = shap_emp[1][0]
    elif getattr(shap_emp, "ndim", 0) == 3:
        shap_left_emp = shap_emp[0, :, 1]
    else:
        shap_left_emp = shap_emp[0]

    local = pd.DataFrame({
        "Feature": feature_names,
        "SHAP value": shap_left_emp,
    })
    local["Direction"] = local["SHAP value"].apply(
        lambda x: "Pushes toward Left" if x > 0 else "Pushes toward Stayed"
    )
    local = local.reindex(
        local["SHAP value"].abs().sort_values(ascending=False).head(15).index
    ).sort_values("SHAP value")

    fig_local = px.bar(
        local,
        x="SHAP value",
        y="Feature",
        orientation="h",
        color="Direction",
        color_discrete_map={
            "Pushes toward Left": "#D55E00",
            "Pushes toward Stayed": "#0072B2",
        },
        title="Why this employee received this prediction",
        labels={
            "SHAP value": "SHAP value",
            "Feature": "Model feature",
            "Direction": "Prediction direction",
        },
    )
    fig_local.add_vline(x=0, line_width=1)
    st.plotly_chart(fig_local, use_container_width=True)

except Exception as shap_error:
    st.warning(f"SHAP explanation could not be rendered: {shap_error}")

st.markdown("### Tenure outlier check for Logistic Regression")
q1 = df.time_spend_company.quantile(.25)
q3 = df.time_spend_company.quantile(.75)
iqr = q3 - q1
lo = q1 - 1.5 * iqr
hi = q3 + 1.5 * iqr
n = int(((df.time_spend_company < lo) | (df.time_spend_company > hi)).sum())
st.write(f"IQR bounds: **{lo:.2f} to {hi:.2f} years**. Statistical outliers detected: **{n:,} rows**.")
st.info("Tenure outliers were checked and retained for Logistic Regression because they can represent legitimate employee profiles and useful attrition signal.")
