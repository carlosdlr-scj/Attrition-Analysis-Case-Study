import streamlit as st
import pandas as pd
import plotly.express as px
from model_utils import load_data

df = load_data().copy()
df["employee_status"] = df["left"].map({0: "Stayed", 1: "Left"})

st.title("📊 Retention Overview")
st.caption("Business question 1: What is the extent of employee attrition, and which employee groups experience the highest turnover?")

with st.sidebar:
    st.header("Filters")
    sf = st.selectbox("Salary level", ["All"] + sorted(df.salary.unique()))
    dep = st.selectbox("Department", ["All"] + sorted(df.Department.unique()))
    tf = st.slider(
        "Tenure (years)",
        int(df.time_spend_company.min()),
        int(df.time_spend_company.max()),
        (int(df.time_spend_company.min()), int(df.time_spend_company.max())),
    )

f = df[df.time_spend_company.between(*tf)].copy()
f = f if sf == "All" else f[f.salary == sf]
f = f if dep == "All" else f[f.Department == dep]

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Employees in segment", f"{len(f):,}")
c2.metric("Employees who left", f"{int(f.left.sum()):,}")
c3.metric("Attrition rate", f"{f.left.mean() * 100:.1f}%")
c4.metric("Avg. satisfaction", f"{f.satisfaction_level.mean():.2f}")
c5.metric("Avg. last evaluation", f"{f.last_evaluation.mean():.2f}")

status = {"Stayed": "#0072B2", "Left": "#D55E00"}


def composition_bar(data, category, title, label, category_order=None):
    counts = (
        data.groupby([category, "employee_status"], observed=False)
        .size()
        .reset_index(name="Count")
    )
    counts["Percent"] = (
        counts["Count"]
        / counts.groupby(category, observed=False)["Count"].transform("sum")
        * 100
    )
    counts["employee_status"] = pd.Categorical(
        counts["employee_status"], categories=["Stayed", "Left"], ordered=True
    )
    counts = counts.sort_values([category, "employee_status"])

    fig = px.bar(
        counts,
        x=category,
        y="Percent",
        color="employee_status",
        barmode="stack",
        color_discrete_map=status,
        custom_data=["Count", "Percent"],
        category_orders={category: category_order} if category_order else None,
        labels={
            category: label,
            "Percent": "Share of employees (%)",
            "employee_status": "Employee status",
        },
        title=title,
    )
    fig.update_traces(
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Employee status: %{fullData.name}<br>"
            "Employees: %{customdata[0]:,}<br>"
            "Share: %{customdata[1]:.1f}%<extra></extra>"
        )
    )
    fig.update_yaxes(range=[0, 100], ticksuffix="%")
    fig.update_layout(legend_title_text="Employee status")
    return fig

st.markdown("### Attrition composition across employee groups")
st.caption("Each bar totals 100%. Hover over a segment to see the percentage and the underlying row count.")

plots = [
    ("Department", "Employee composition by department", "Department", sorted(f["Department"].dropna().unique())),
    ("salary", "Employee composition by salary level", "Salary level", [x for x in ["low", "medium", "high"] if x in set(f["salary"].dropna())]),
    ("time_spend_company", "Employee composition by tenure", "Years at company", sorted(f["time_spend_company"].dropna().unique())),
    ("promotion_last_5years", "Employee composition by promotion in the last 5 years", "Promotion in last 5 years", [0, 1]),
    ("Work_accident", "Employee composition by work accident history", "Work accident", [0, 1]),
]

def plot_for_column(col, title, label, order):
    if col in ["promotion_last_5years", "Work_accident"]:
        display_order = ["No", "Yes"]
        chart_df = f.copy()
        chart_df["display_category"] = chart_df[col].map({0: "No", 1: "Yes"})
        return composition_bar(chart_df, "display_category", title, label, display_order)
    return composition_bar(f, col, title, label, order)

# Row 1: Department + Salary
c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(plot_for_column(*plots[0]), use_container_width=True)
with c2:
    st.plotly_chart(plot_for_column(*plots[1]), use_container_width=True)
with st.expander("CLICK FOR INSIGHTS"):
    st.write(
        "Fairly consistent percentages could be seen in all departments, with a slightly better management department that had the minor percent of 11.9%" 
        " and the lowest member count with 436. Keep an eye on this dimensions, similar patterns across different values could mean we have manipulated"
        " or sinthetic data. Naturally, the attrition rate is expected to decrease as salaries rise from low to high, which is what we have on the chart"
        " , more money, more desire to stay."
        )

# Row 2: Work accident + Promotion
c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(plot_for_column(*plots[4]), use_container_width=True)
with c2:
    st.plotly_chart(plot_for_column(*plots[3]), use_container_width=True)
with st.expander("CLICK FOR INSIGHTS"):
    st.write(
        "Much more members didn't experiment an accident, but between those 1,850 who experienced one, only 5% left the company." 
        " A similar case happened with promoted members, very few members received a promotion in the last 5 years, but just a small 3.9%"
        " of those promoted members left; with a higher number of not promoted members who abandoned the company, would be important to check the"
        " promotion system, prioritazing the protection of great talents that could abandon the squad."
    )

# Row 3: Tenure only
st.plotly_chart(plot_for_column(*plots[2]), use_container_width=True)
with st.expander("CLICK FOR INSIGHTS"):
    st.write(
        "With a surprising 100% of stayed members at years 7, 8 and 10, seems to be a certain sense of security after "
        "enduring the first 6 years, could be a great opportunity to check the incentives reached at such periods at Salifort."
        " Important to consider, the total sum of those perfectly stayed members represents only 282 (2.35%) of all the members,"
        " stable position but hard to reach."
    )