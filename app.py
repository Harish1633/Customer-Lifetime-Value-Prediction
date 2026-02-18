import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
import io

def generate_clv_pdf(data):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph("<b>Customer Lifetime Value Prediction Report</b>", styles["Title"]))
    story.append(Spacer(1, 12))
    for label, value in data.items():
        story.append(Paragraph(f"<b>{label}:</b> {value}", styles["Normal"]))
        story.append(Spacer(1, 8))
    doc.build(story)
    buffer.seek(0)
    return buffer
from datetime import datetime

if "activity_log" not in st.session_state:
    st.session_state.activity_log = []

def log_activity(action, status="Success"):
    st.session_state.activity_log.insert(
        0,  # newest on top
        {
            "action": action,
            "date": datetime.now().strftime("%b %d, %H:%M"),
            "status": status
        }
    )
if "logged_in" not in st.session_state:
    log_activity("Logged In")
    st.session_state.logged_in = True
st.set_page_config(
    page_title="CLV Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS

st.markdown("""<style>
    .kpi-grid {display: flex;gap: 20px;margin: 20px 0;}
    .kpi-card {flex: 1;background: #ffffff;border-radius: 14px;padding: 24px;box-shadow: 0 8px 20px rgba(0,0,0,0.06);text-align: center;}
    .kpi-title {font-size: 15px;color: #6b7280;font-weight: 600;margin-bottom: 10px;}
    .kpi-value {font-size: 30px;font-weight: 700;color: #111827;}
    .kpi-green { border-left: 5px solid #22c55e; }
    .kpi-blue  { border-left: 5px solid #3b82f6; }
    .kpi-purple{ border-left: 5px solid #a855f7; }
    </style>""",unsafe_allow_html=True)

st.markdown("""<style>
h1, h2, h3 {letter-spacing: 0.5px;}
.metric-card {background-color: #d6a2depadding: 20px;border-radius: 12px;text-align: center;box-shadow: 0 4px 10px rgba(0,0,0,0.05);
    height: 140px;display: flex;flex-direction: column;justify-content: center;}
.info-box {background-color: #eaf3ff;padding: 12px;border-radius: 8px;margin-top: 10px;color: #1f4fd8;font-size: 14px;}
.section {background-color: #a2e0a5;padding: 20px;border-radius: 12px;box-shadow: 0 4px 10px rgba(0,0,0,0.05);margin-top: 20px;}
</style>""", unsafe_allow_html=True)

model = joblib.load("clv_model.pkl")

st.set_page_config(page_title="Customer Lifetime Value Prediction",layout="wide")
st.markdown("<h1 style='text-align: center;'>Customer Lifetime Value (CLV) Prediction</h1>",unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size:16px;'>Machine Learning based CLV Analysis & Prediction</p>",unsafe_allow_html=True)
st.markdown("---")

# Sidebar navigation
st.sidebar.title("📊 Analysis Menu")
st.sidebar.markdown("Select a section to explore the project:")
page = st.sidebar.radio("Go to",["Dashboard", "CLV Prediction", "Retention Strategy", "Business Suggestions", "Project Overview"])
if page == "Dashboard":
    live_df = None
    st.markdown("## 📊 CLV Analytics Dashboard")
    st.markdown("---")
    st.markdown("### 📂 Data Source")
    uploaded_file = st.file_uploader("Upload customer CSV to generate live insights",type=["csv"])
    if uploaded_file is not None:
        live_df = pd.read_csv(uploaded_file, encoding="latin1")
        st.session_state["live_df"] = live_df
        st.success("Live data loaded successfully!")
        st.write("Preview of uploaded data:")
        st.dataframe(live_df.head())
        if uploaded_file is not None and "csv_uploaded" not in st.session_state:
            log_activity("Uploaded CSV")
            st.session_state.csv_uploaded = True
    else:
        st.markdown("""<div class="info-box">No file uploaded. Dashboard is currently using a representative sample dataset.</div>""",unsafe_allow_html=True)
    
    # KPIs (use example values if data not loaded)
    # ================= KPI CALCULATION =================

    if live_df is not None and not live_df.empty:

        st.markdown("### Customer value indicators derived from uploaded data")

    # Ensure CLV column exists
        if "CLV" in live_df.columns:

            total_customers = len(live_df)
            avg_clv = int(live_df["CLV"].mean())
            max_clv = int(live_df["CLV"].max())
        else:
            st.warning("CLV column not found in uploaded CSV.")
            total_customers = "N/A"
            avg_clv = "N/A"
            max_clv = "N/A"
    else:
        # Fallback demo values
        total_customers = 4000
        avg_clv = 3200
        max_clv = 25000

    st.markdown(f"""<div class="kpi-grid">
<div class="kpi-card kpi-green">
<div class="kpi-title">👥 Active Customers</div>
<div class="kpi-value">{total_customers:,}</div>
        </div>
<div class="kpi-card kpi-blue">
<div class="kpi-title">💰 Avg Revenue per Customer</div>
<div class="kpi-value">₹ {avg_clv:,}</div>
        </div>
<div class="kpi-card kpi-purple">
<div class="kpi-title">⭐ Top Customer Value</div>
<div class="kpi-value">₹ {max_clv:,}</div>
        </div>
    </div>""",unsafe_allow_html=True)
    
    
    st.markdown("""<div class="section"><h4>📈 Average CLV Trend by Tenure</h4></div>""",unsafe_allow_html=True)
    if live_df is None:
        st.markdown("""<div class="info-box">Upload a CSV file to view Average CLV Trend by Tenure.</div>""",unsafe_allow_html=True)
    else:
        required_cols = {"CLV", "TenureMonths"}
        if not required_cols.issubset(live_df.columns):
            st.warning("Required columns (CLV, TenureMonths) not found in uploaded file.")
        else:
            st.markdown("""<div class="info-box">This chart shows how average customer lifetime value changes with customer tenure.</div>""",unsafe_allow_html=True)
            import plotly.express as px
        # Prepare data for line chart
            trend_df = (live_df.groupby("TenureMonths", as_index=False)["CLV"].mean().sort_values("TenureMonths"))
        # Line chart
            fig = px.line(trend_df,x="TenureMonths",y="CLV",markers=True,title="Average CLV Trend by Tenure",labels={
                "TenureMonths": "Customer Tenure (Months)","CLV": "Average Customer Lifetime Value"})
            fig.update_layout(height=450,margin=dict(l=30, r=30, t=60, b=40))
            st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ===== CHARTS ROW =====
    left, right = st.columns(2)

    # -------- CLV TREND --------
    with left:
        st.markdown("""<div class="section"><h4>📊 CLV vs Purchase Frequency</h4></div>""",unsafe_allow_html=True)
        if live_df is None:
            st.markdown("""<div class="info-box">Upload a CSV file to view CLV vs Purchase Frequency.</div>""",unsafe_allow_html=True)
        else:
            required_cols = {"CLV", "Frequency"}
            if not required_cols.issubset(live_df.columns):
                st.warning("Required columns (CLV, Frequency) not found in uploaded file.")
            else:
                st.markdown("""<div class="info-box">This chart shows the relationship between purchase frequency and customer lifetime value.</div>""",unsafe_allow_html=True)
                import plotly.express as px
                fig = px.scatter(live_df,x="Frequency",y="CLV",
                color="CLV_Segment" if "CLV_Segment" in live_df.columns else None,title="CLV vs Purchase Frequency",
                labels={"Frequency": "Purchase Frequency","CLV": "Customer Lifetime Value"})
                fig.update_layout(height=450,margin=dict(l=30, r=30, t=60, b=40))
                st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # -------- CUSTOMER SEGMENTS --------
    with right:
        st.markdown("""<div class="section"><h4>👥 Customer Segments </h4>""", unsafe_allow_html=True)
        if live_df is None:
            st.markdown("""<div class="info-box">Upload a CSV file to view customer segmentation.</div>""", unsafe_allow_html=True)
        else:
            segment_col = None
            for col in live_df.columns:
                if "segment" in col.lower():
                    segment_col = col
                    break
            if segment_col is None:
                st.warning("No segmentation column found (e.g. CLV_Segment).")
            else:
                st.markdown("""<div class="info-box">This pie chart shows the distribution of customers across value segments.</div>""", unsafe_allow_html=True)
                segment_counts = live_df[segment_col].value_counts()
                fig = px.pie(names=segment_counts.index,values=segment_counts.values,title="Customer Segmentation",hole=0.5)
                st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("""<style>
    .activity-card {background: #ffffff;border-radius: 14px;padding: 14px 18px;margin-bottom: 12px;box-shadow: 0 6px 16px rgba(0,0,0,0.08);display: flex;justify-content: space-between;align-items: center;}
    
    .activity-left {display: flex;flex-direction: column;}
    .activity-title {font-weight: 600;font-size: 15px;}
    .activity-date {font-size: 12px;color: #6b7280;}
    .activity-status {padding: 5px 14px;border-radius: 999px;font-size: 12px;font-weight: 600;color: white;background-color: #22c55e;}
    </style>""",unsafe_allow_html=True)

    
    st.markdown("### 🕒 Recent Activity")
    if len(st.session_state.activity_log) == 0:
        st.caption("No recent activity")
    else:
        for item in st.session_state.activity_log[:5]:
            st.markdown(f"""<div class="activity-card">
            <div class="activity-left">
            <div class="activity-title">{item['action']}</div>
            <div class="activity-date">{item['date']}</div>
            </div>
            <div class="activity-status">{item['status']}</div>
            </div>""",unsafe_allow_html=True)  
    
if page == "CLV Prediction" :
    st.subheader("🔮 CLV Prediction")
    st.markdown("---")
    st.write("Enter customer behavior details to predict CLV.")
    col1, col2, col3 = st.columns(3)
    with col1:
        frequency = st.number_input("Purchase Frequency", min_value=1, step=1)
        st.markdown("<br>", unsafe_allow_html=True)
    with col2:
        avg_purchase_value = st.number_input("Average Purchase Value", min_value=0.0)
        st.markdown("<br>", unsafe_allow_html=True)
    with col3:
        lifespan = st.number_input("Customer Lifespan (days)", min_value=1)
        st.markdown("<br>", unsafe_allow_html=True)
    recency = st.number_input("Recency (days since last purchase)", min_value=0)
    st.markdown("<br>", unsafe_allow_html=True)
    revenue_per_day = st.number_input("Revenue Per Day", min_value=0.0)
    st.markdown("<br>", unsafe_allow_html=True)
    tenure_months = lifespan / 30
    if st.button("Predict CLV"):
        input_data = np.array([[frequency, avg_purchase_value, lifespan, recency, revenue_per_day, tenure_months]])
        predicted_clv = model.predict(input_data)[0]
        st.markdown(f"""<div style="padding:15px; border-radius:10px; background-color:#e8f5e9;font-size:18px; text-align:center;">
<b>Predicted Customer Lifetime Value</b><br>₹ {round(predicted_clv, 2)}</div>""",unsafe_allow_html=True)
        log_activity("Predicted CLV")
        st.session_state.report_data = {
        "Purchase Frequency": frequency,
        "Average Purchase Value": avg_purchase_value,
        "Customer Lifespan (days)": lifespan,
        "Recency (days)": recency,
        "Revenue Per Day": revenue_per_day,
        "Tenure (months)": round(tenure_months, 2),
        "Predicted CLV": f" {round(predicted_clv, 2)}"
    }
        if "report_data" in st.session_state:
            pdf_buffer = generate_clv_pdf(st.session_state.report_data)
            st.download_button(label="📄 Download Report (PDF)",data=pdf_buffer,file_name="CLV_Prediction_Report.pdf",mime="application/pdf")
        st.markdown("""
    <style>
    .activity-card {background: #ffffff;border-radius: 14px;padding: 14px 18px;margin-bottom: 12px;box-shadow: 0 6px 16px rgba(0,0,0,0.08);display: flex;
justify-content: space-between;align-items: center;}
    .activity-left {display: flex;flex-direction: column;}
    .activity-title {font-weight: 600;font-size: 15px;}
    .activity-date {font-size: 12px;color: #6b7280;}
    .activity-status {padding: 5px 14px;border-radius: 999px;font-size: 12px;font-weight: 600;color: white;background-color: #22c55e;}
    </style>""",unsafe_allow_html=True)
        st.markdown("### 🕒 Recent Activity")
        for item in st.session_state.activity_log[:1]:
            st.markdown(f"""<div class="activity-card"><div><b>{item['action']}</b><br><small>{item['date']}</small></div><span class="activity-status">{item['status']}</span></div>""",unsafe_allow_html=True)


    
if page == "Retention Strategy":
    live_df = st.session_state.get("live_df", None)
    st.header("🔔 Customer Retention Recommendations")
    st.caption("Actionable strategies based on customer behavior and CLV")
    live_df = st.session_state.get("live_df", None)
    if live_df is None or live_df.empty:
        st.warning("Please upload a CSV file from the Dashboard page.")
    else:
        required_cols = {"CLV", "Recency", "Frequency"}
        if not required_cols.issubset(live_df.columns):
            st.warning("Required columns (CLV, Recency, Frequency) not found in dataset.")
        else:
            def get_risk(row):
                if row["CLV"] > live_df["CLV"].quantile(0.75) and row["Recency"] > 60:
                    return "🔴 At Risk"
                elif row["CLV"] < live_df["CLV"].median() and row["Frequency"] > live_df["Frequency"].median():
                    return "🟡 Potential"
                elif row["CLV"] > live_df["CLV"].median() and row["Recency"] < 30:
                    return "🟢 Loyal"
                else:
                    return "⚪ Low Priority"
            def get_action(risk):
                return {
                    "🔴 At Risk": "Offer loyalty discounts or personal outreach",
                    "🟡 Potential": "Upsell premium products or bundles",
                    "🟢 Loyal": "Reward with exclusive benefits",
                    "⚪ Low Priority": "Standard promotional campaigns"
                }[risk]
            live_df = live_df.copy()  # avoid SettingWithCopyWarning
            live_df["Risk Level"] = live_df.apply(get_risk, axis=1)
            live_df["Recommended Action"] = live_df["Risk Level"].apply(get_action)
            st.dataframe(live_df[["CustomerID", "CLV", "Recency", "Frequency", "Risk Level", "Recommended Action"]].head(20),use_container_width=True)
            st.success("Retention recommendations generated successfully.")


if page == "Business Suggestions":

    st.subheader("💡 Business Suggestions")
    st.caption("Data-driven strategies based on customer value segmentation")

    st.markdown("""<style>
    .cards-container {display: flex;gap: 20px;}

    .strategy-card {background: #ffffff;border-radius: 16px;padding: 20px;box-shadow: 0 6px 18px rgba(0,0,0,0.08);flex: 1;display: flex;flex-direction: column;height: 100%;}

    .strategy-title {font-size: 20px;font-weight: 700;margin-bottom: 12px;}

    .strategy-desc {font-size: 15px;color: #4b5563;line-height: 1.6;flex-grow: 1;}
    
    .strategy-desc1 {font-size: 15px;color: #4b5563;line-height: 1.6;flex-grow: 1;}

    .high {border-left: 6px solid #22c55e;}

    .medium {border-left: 6px solid #facc15;}

    .low {border-left: 6px solid #ef4444;}
    </style>""",unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
            <div class="strategy-card high">
                <div class="strategy-title">⭐ High-Value Customers</div>
                <div class="strategy-desc">
                    • Provide loyalty rewards and exclusive benefits<br>
                    • Offer personalized discounts and early access<br>
                    • Maintain strong engagement through premium support
                </div>
            </div>""",unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <div class="strategy-card medium">
                <div class="strategy-title">📈 Medium-Value Customers</div>
                <div class="strategy-desc">
                    • Apply upselling and cross-selling strategies<br>
                    • Encourage repeat purchases with limited offers<br>
                    • Use targeted email or app notifications<br>
                    <br>
                </div>
            </div>""",unsafe_allow_html=True)

    with col3:
        st.markdown("""
            <div class="strategy-card low">
                <div class="strategy-title">⚠️ Low-Value Customers</div>
                <div class="strategy-desc1">
                    • Improve onboarding experience<br>
                    • Reduce churn risk with incentives<br>
                    • Re-engage using educational or promotional content<br>
                </div>
            </div>""",unsafe_allow_html=True)
    st.markdown("---")
    st.info("These recommendations help businesses optimize retention, revenue growth, and customer engagement based on lifetime value.")


if page == "Project Overview":
    # ===== PROBLEM STATEMENT BOX =====
    with st.container():
        st.markdown("### 🎯 Problem Statement")
        st.write(
            "Businesses often struggle to identify which customers generate the most long-term value. "
            "Without accurate insights, marketing efforts become inefficient, customer churn increases, "
            "and retention strategies lack clear direction."
        )
        st.info("This project addresses the problem by predicting Customer Lifetime Value (CLV) using machine learning.")
    st.markdown("---")
    # ===== OBJECTIVES & SOLUTION BOXES =====
    col1, col2 = st.columns(2)
    with col1:
        with st.container():
            st.markdown("### 📌 Project Objectives")
            st.write("""
                • Analyze customer purchase behavior  
                • Predict Customer Lifetime Value (CLV)  
                • Segment customers based on value  
                • Support data-driven business decisions  """)

    with col2:
        with st.container():
            st.markdown("### ⚙️ Solution Approach")
            st.write("""
                • Data preprocessing and feature engineering  
                • Machine learning-based CLV prediction  
                • Interactive dashboards and visualizations  
                • Retention & business recommendation engine  """)
    st.markdown("---")

    # ===== KEY FEATURES BOX =====
    st.markdown("### 🚀 Key Features")
    f1, f2, f3 = st.columns(3)
    with f1:
        st.metric("📊 Dashboard", "Interactive")
        st.caption("Visualizes customer data, CLV trends, and segmentation.")
    with f2:
        st.metric("🤖 ML Prediction", "CLV")
        st.caption("Predicts customer lifetime value using trained ML models.")
    with f3:
        st.metric("🔔 Recommendations", "Actionable")
        st.caption("Provides retention and business strategies based on CLV.")
    st.markdown("---")

    # ===== BUSINESS IMPACT BOX =====
    with st.container():
        st.markdown("### 📈 Business Impact")
        st.write(
            "By identifying high-value customers, businesses can focus on retention, "
            "reduce churn, and maximize revenue. This project demonstrates how "
            "machine learning can transform raw customer data into actionable insights."
        )
        st.success("This project bridges data analytics, machine learning, and real-world business decision-making.")

st.markdown("---")
st.caption("CLV Prediction using Machine Learning")
