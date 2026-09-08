import streamlit as st
st.set_page_config(page_title="Salifort Motors | Employee Retention",page_icon="📊",layout="wide")
st.title("Salifort Motors | Employee Retention")
st.markdown("By CARLOS DE LA RE.")
st.subheader("From employee data to retention decisions")
st.markdown("This dashboard analyzes employee attrition and builds a classification model to identify employee profiles with a higher likelihood of leaving.")
st.info("**Business questions**\n1. **What is the extent of employee attrition, and which employee groups experience the highest turnover?**\n2. **Which employee characteristics and workplace conditions are most strongly associated with employees leaving?**\n3. **What employee profiles appear to be at higher risk of attrition?**\n4. **Can a predictive model help Salifort Motors identify employees who are likely to leave and support proactive retention efforts?**")
st.markdown("### Project background")
st.write("Salifort Motors (License CCO: Public Domain) has experienced turmoil times, many employees leaving the company with no clear understanding of the specific drivers that are pushing them out of the squad, and no tool available to fight back against this turnover problem."
" Each member is valuable for this company, representing an important investment for it; would be great to see if they can act upon advertisement when a person is about to leave."
" It’s time consuming and expensive to search, interview and make the whole process to get a new employee in action, so having new ideas for employee retention will be beneficial for Salifort Motors.")
st.markdown("### Executive summary")
st.write(
    """
    This project analyzes employee attrition at **Salifort Motors** to identify
    patterns associated with employees leaving the company and to explore how
    data-driven methods can support employee retention. The analysis examines
    employee satisfaction, performance evaluations, workload, number of projects,
    tenure, promotions, work accidents, salary, and department.
    """
)

st.write(
    """
    The analysis identified several patterns associated with employee attrition.
    **Workload appears to be an important factor**, with higher numbers of projects
    and excessive working hours being particularly concerning. At the same time,
    employees working on relatively few projects also showed elevated attrition,
    suggesting that both excessive workload and insufficient engagement may
    negatively affect the employee experience.
    """
)

st.write(
    """
    Attrition also appears to increase during the **4th to 6th years of employment**,
    while the very low proportion of employees receiving promotions in the previous
    five years highlights a potential opportunity to strengthen career development
    and advancement pathways.
    """
)

st.write(
    """
    To complement the descriptive analysis, three machine learning classification
    models were evaluated to predict employee attrition. The selected model achieved
    a **recall of 0.932**, meaning that it correctly identified approximately
    **93% of the employees who actually left** in the test dataset.
    """
)

st.write(
    """
    This demonstrates the potential value of predictive analytics as an early-warning
    tool that could help the company identify higher-risk employee profiles and
    consider retention interventions before attrition occurs.
    """
)

st.write(
    """
    Finally, **SHAP explainability** was incorporated to provide insight into the
    factors contributing to individual model predictions. Rather than treating the
    model as a black box, SHAP helps identify which employee characteristics
    contribute toward a prediction of staying or leaving, providing additional
    context for potential retention strategies.
    """
)

st.write(
    """
    Based on these findings, Salifort Motors should consider **strengthening career
    development opportunities, reviewing workload and project allocation, examining
    working arrangements for employees with unusually low or high workloads, and
    using predictive analytics as a proactive retention-support tool**.
    """
)

st.write(
    """
    These recommendations should be considered as areas for further investigation,
    since the analysis identifies associations in the observed data and does not
    establish that any individual factor directly causes employee attrition.
    """
)

st.markdown("### Project scope")
st.write("The dataset contains employee satisfaction, evaluation, workload, tenure, accident, promotion, department, salary level, and attrition information. It does not contain dates, salary amounts, revenue, or turnover costs, so the analysis focuses on retention patterns, associations, and prediction rather than financial ROI or time-series trends.")
st.markdown("### Data-quality approach")
st.write("Exact duplicates, missing values, domain validity, and tenure outliers were checked. Statistical outliers were investigated and retained when they could represent legitimate employee profiles or useful attrition signal.")
