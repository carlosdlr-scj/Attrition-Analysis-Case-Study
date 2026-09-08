import streamlit as st

st.set_page_config(
    page_title="Recommendations",
    page_icon="💡",
    layout="wide"
)

st.title("💡 Recommendations")

st.markdown(
    """
    Based on the descriptive analysis, drivers of attrition, and employee
    attrition prediction model, the following recommendations are proposed
    to support Salifort Motors in improving employee retention.
    """
)

# ============================================================
# RETENTION OVERVIEW
# ============================================================

st.header("Retention Overview")

with st.container(border=True):

    st.subheader("1. Define and develop career plans")

    st.markdown(
        """
        While not all dimensions showed significant patterns, it is also a
        reality that very few people received a promotion in the last five
        years, with only **1.69% of employees being promoted**.

        This could be highly associated with low satisfaction levels and low
        evaluation scores. Developing clearer career paths and fostering a
        more positive workplace culture could help improve employee retention.
        """
    )


with st.container(border=True):

    st.subheader(
        "2. Provide professional mental health guidance for employees "
        "going through their 4th to 6th year"
    )

    st.markdown(
        """
        These three years appear to represent an important threshold for
        determining whether an employee will remain with the company and
        achieve stability or eventually leave.

        The chart showed an increasing pattern in employees leaving the
        company during this period, but after these difficult years, attrition
        appears to decrease significantly.

        Promotions could be another potential solution, when deserved and
        supported by performance.
        """
    )


# ============================================================
# DRIVERS OF ATTRITION
# ============================================================

st.header("Drivers of Attrition")

with st.container(border=True):

    st.subheader(
        "3. Limit the number of projects an employee has to handle "
        "simultaneously"
    )

    st.markdown(
        """
        Not a single employee who was working on 7 projects remained with
        the company.

        Similarly, among employees handling 6 projects, approximately
        **45% left the company**, representing almost half of them.
        """
    )


with st.container(border=True):

    st.subheader(
        "4. Set the total working hours for a full-time employee at "
        "approximately 150 hours per month"
    )

    st.markdown(
        """
        A rewarding system could also help employees manage excessive
        workloads and reduce burnout.

        Overworking clearly appeared as a potential signal of an employee's
        intention to leave, and this finding is closely linked to the high
        number of projects observed in many attrition cases.
        """
    )


with st.container(border=True):

    st.subheader(
        "5. Review job offers and working arrangements for part-time "
        "employees and newly hired employees"
    )

    st.markdown(
        """
        An important cluster of employees who left the company appeared to
        be working fewer hours than expected, which was associated with
        lower satisfaction levels.

        An impressive **54.2% of employees working on 2 projects left the
        company**, suggesting that both excessive and insufficient workloads
        may negatively affect employees' perception of their work experience.
        """
    )


# ============================================================
# EMPLOYEE ATTRITION PREDICTION
# ============================================================

st.header("Employee Attrition Prediction")

with st.container(border=True):

    st.subheader("6. Evaluate employees using the selected model")

    st.markdown(
        """
        A **0.932 recall score** shows that the model is capable of correctly
        identifying approximately **9 out of 10 employees who actually
        leave**.

        This provides the company with an opportunity to intervene before
        the employee leaves, potentially allowing for a new negotiation or
        retention strategy rather than facing the final settlement costs and
        the exhaustive process of hiring and training a replacement.
        """
    )


with st.container(border=True):

    st.subheader("7. Consider changing workplace practices based on SHAP values")

    st.markdown(
        """
        SHAP values indicate whether a feature had a significant influence
        on the model's prediction of whether an employee stays or leaves.

        These values should **not be interpreted as percentages**. Instead,
        they represent the contribution of each feature to the model's
        predicted outcome.
        """
    )


# ============================================================
# FINAL NOTE
# ============================================================

st.divider()

st.info(
    """
    **Important:** These recommendations are based on patterns and
    associations observed in the available dataset. They should be treated
    as areas for further investigation rather than proof that any individual
    factor directly causes employee attrition.
    """
)