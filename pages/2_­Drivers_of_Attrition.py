import streamlit as st
import pandas as pd
import plotly.express as px
from model_utils import load_data
df=load_data().copy(); df["employee_status"]=df["left"].map({0:"Stayed",1:"Left"})
st.title("🔎 Drivers of Attrition")
st.caption("Business question 2: Which employee characteristics and workplace conditions are most strongly associated with employees leaving?")
with st.sidebar:
 st.header("Filters"); sf=st.selectbox("Salary level",["All"]+sorted(df.salary.unique())); dep=st.selectbox("Department",["All"]+sorted(df.Department.unique())); tf=st.slider("Tenure (years)",int(df.time_spend_company.min()),int(df.time_spend_company.max()),(int(df.time_spend_company.min()),int(df.time_spend_company.max())))
f=df[df.time_spend_company.between(*tf)].copy(); f=f if sf=="All" else f[f.salary==sf]; f=f if dep=="All" else f[f.Department==dep]
status={"Stayed":"#0072B2","Left":"#D55E00"}
def scatter(x,y,title,xlab,ylab):
 fig=px.scatter(f,x=x,y=y,color="employee_status",color_discrete_map=status,opacity=.55,hover_data=["Department","salary","time_spend_company","number_project"],labels={x:xlab,y:ylab,"employee_status":"Employee status","Department":"Department","salary":"Salary","time_spend_company":"Tenure (years)","number_project":"Number of projects"},title=title); fig.update_layout(legend_title_text="Employee status"); return fig
a,b=st.columns(2)
with a: st.plotly_chart(scatter("average_montly_hours","satisfaction_level","Monthly hours vs satisfaction level","Average monthly hours","Satisfaction level"),use_container_width=True)
   
with b: st.plotly_chart(scatter("average_montly_hours","last_evaluation","Monthly hours vs last evaluation","Average monthly hours","Last evaluation"),use_container_width=True)
with st.expander("CLICK FOR INSIGHTS"):
    st.write(
        "Clearly on both charts we have a bunch of left employees around the same down-left or up-right area," 
        " with a few scatter points representing attrition on other areas. That could act as a disadvantage"
        " for the future model. Keep in mind that satisfaction level comes from the user, and the last evaluation"
        " is a number measured by the company. Psychology attributes could be hard to rate and interpret."
        " If the average monthly hours for a US worker is around 150 hours. How is it possible to have members working"
        " more than 200 hours, and even worst more than 290. Both scatter plots show an important reason to quit."  
    )
st.markdown("### Attrition composition by employee segment")
st.caption("Each bar totals 100%. Hover over a segment to see the percentage and the underlying row count.")
def stacked(data,cat,title,label):
 c=data.groupby([cat,"employee_status"],observed=False).size().reset_index(name="Count"); c["Percent"]=c["Count"]/c.groupby(cat,observed=False)["Count"].transform("sum")*100; c.employee_status=pd.Categorical(c.employee_status,categories=["Stayed","Left"],ordered=True); c=c.sort_values([cat,"employee_status"])
 fig=px.bar(c,x=cat,y="Percent",color="employee_status",barmode="stack",color_discrete_map=status,custom_data=["Count","Percent"],labels={cat:label,"Percent":"Share of employees (%)","employee_status":"Employee status"},title=title); fig.update_traces(hovertemplate="<b>%{x}</b><br>Employee status: %{fullData.name}<br>Employees: %{customdata[0]:,}<br>Share: %{customdata[1]:.1f}%<extra></extra>"); fig.update_yaxes(range=[0,100],ticksuffix="%"); fig.update_layout(legend_title_text="Employee status"); return fig
f["satisfaction_band"]=pd.cut(f.satisfaction_level,[0,.25,.5,.75,1.01],labels=["0–0.25","0.25–0.50","0.50–0.75","0.75–1.00"],include_lowest=True); f["hours_band"]=pd.cut(f.average_montly_hours,[-float("inf"),160,200,240,float("inf")],labels=["≤160","161–200","201–240","241+"]); f["evaluation_band"]=pd.cut(f.last_evaluation,[0,.6,.8,1.01],labels=["≤0.60","0.61–0.80",">0.80"],include_lowest=True)
a,b=st.columns(2)
with a: st.plotly_chart(stacked(f,"satisfaction_band","Employee composition by satisfaction level","Satisfaction band"),use_container_width=True)
with b: st.plotly_chart(stacked(f,"number_project","Employee composition by number of projects","Number of projects"),use_container_width=True)
with st.expander("CLICK FOR INSIGHTS"):
    st.write(
        "Each category from the satisfaction band has a significant members share, and it's natural to consider "
        "dissatisfaction as a driver to quit. The interesting categories are the two over 0.5, maybe the low 3.5%"
        " of left members is due to real feeling of satisfaction and stability, also the higher band of satisfaction"
        " could mean the neccesity from some members to reach new challenges after a high sensation of satisfaction."
        " The number of projects shows a straightforward message, overworking tends to force employees to quit, but an impressive "
        " 54.2% percent of people working just 2 projects decided to quit, a new psychological trait associated with underestimated" 
        " workload could be happening. 3 to 4 projects seems to be the comfort or stability zone."
    )

a,b=st.columns(2)
with a: st.plotly_chart(stacked(f,"hours_band","Employee composition by monthly-hours group","Monthly-hours group"),use_container_width=True)
with b: st.plotly_chart(stacked(f,"evaluation_band","Employee composition by last evaluation","Last evaluation"),use_container_width=True)

with st.expander("CLICK FOR INSIGHTS"):
    st.write(
        "Previously we discussed about the average monthly hours, and after watching this grouped chart "
        " ,it looks like there is an underwork and overwork problem regarding abandonment. Would be great to see"
         " the interaction between these two lower and higher categories with salaries to answer the question:"
         " Is my salary worth this overwork/underwork anguish? Due to the strange pattern found at the last evaluation chart"
         " a review of the evaluation process could be needed; the middle point has a percent of 3.9% of abandonment, while"
         " the lowest and highest position appear pretty similar and higher, around 22% and 23%, it's important to push higher" 
         " the stayed percent from top evaluated members."
    )

st.warning("These patterns are associations in observational data; they do not by themselves prove causation.")


