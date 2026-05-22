"""Streamlit informal credit scoring demo dashboard for lenders."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Informal Credit Scoring", page_icon="💳", layout="wide")
st.title("💳 Informal Economy Credit Scoring — Nigerian Market Traders")
st.caption("Alternative data credit scoring | Mobile money + Airtime + Market association + GPS stability")

BAND_COLOURS = {
    "Excellent": "#006400", "Good": "#228B22", "Fair": "#FFA500",
    "Poor": "#FF4500", "Thin File": "#CC0000"
}


@st.cache_data
def generate_portfolio(n: int = 500) -> pd.DataFrame:
    np.random.seed(42)
    scores = np.random.choice(
        [np.random.randint(650, 850), np.random.randint(550, 680),
         np.random.randint(450, 560), np.random.randint(300, 460)],
        n, p=[0.40, 0.32, 0.18, 0.10]
    )
    scores = np.array([np.random.randint(300, 851) for _ in range(n)])

    def get_band(s):
        if s >= 750: return "Excellent"
        if s >= 680: return "Good"
        if s >= 580: return "Fair"
        if s >= 500: return "Poor"
        return "Thin File"

    return pd.DataFrame({
        "applicant_id": [f"NG-TRD-{i:05d}" for i in range(n)],
        "credit_score": scores,
        "score_band": [get_band(s) for s in scores],
        "default_probability": np.clip(1 - (scores - 300) / 550 * 0.90 + np.random.normal(0, 0.04, n), 0.02, 0.95),
        "gender": np.random.choice(["Female", "Male"], n, p=[0.62, 0.38]),
        "state": np.random.choice(["Lagos", "Kano", "Oyo", "Rivers", "Benue", "Kaduna"], n),
        "market_tenure_months": np.random.randint(1, 240, n),
        "monthly_txn_volume": np.random.lognormal(11.5, 0.8, n).astype(int),
        "loan_approved": scores >= 580,
        "max_loan_naira": [500000 if s >= 750 else 150000 if s >= 680 else
                           50000 if s >= 580 else 20000 if s >= 500 else 0 for s in scores],
    })


portfolio = generate_portfolio()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Applicants", len(portfolio))
with col2:
    approved = portfolio["loan_approved"].sum()
    st.metric("Approved for Loans", f"{approved} ({approved/len(portfolio)*100:.0f}%)")
with col3:
    avg_score = int(portfolio["credit_score"].mean())
    st.metric("Average Credit Score", avg_score)
with col4:
    total_credit = portfolio["max_loan_naira"].sum()
    st.metric("Total Credit Unlocked", f"₦{total_credit/1e6:.1f}M")

st.markdown("---")
tab1, tab2, tab3 = st.tabs(["📊 Portfolio Overview", "🔍 Score Applicant", "⚖️ Fairness Audit"])

with tab1:
    col_a, col_b = st.columns(2)
    with col_a:
        band_counts = portfolio["score_band"].value_counts()
        fig = px.pie(
            values=band_counts.values, names=band_counts.index,
            color=band_counts.index,
            color_discrete_map=BAND_COLOURS,
            title="Credit Score Band Distribution",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        fig2 = px.histogram(
            portfolio, x="credit_score", color="score_band",
            color_discrete_map=BAND_COLOURS,
            nbins=40, title="Score Distribution",
            labels={"credit_score": "Credit Score"},
        )
        for threshold, label in [(750, "Excellent"), (680, "Good"), (580, "Fair"), (500, "Poor")]:
            fig2.add_vline(x=threshold, line_dash="dash", opacity=0.5)
        st.plotly_chart(fig2, use_container_width=True)

    state_avg = portfolio.groupby("state")["credit_score"].mean().reset_index()
    fig3 = px.bar(
        state_avg.sort_values("credit_score", ascending=False),
        x="state", y="credit_score",
        color="credit_score", color_continuous_scale="RdYlGn",
        range_color=[400, 750],
        title="Average Credit Score by State",
    )
    st.plotly_chart(fig3, use_container_width=True)

with tab2:
    st.subheader("Score a New Applicant")
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        txn_volume = st.number_input("Monthly Mobile Money Volume (₦)", 5000, 2000000, 85000, 5000)
        txn_freq = st.slider("Monthly Transactions", 5, 150, 45)
        regularity = st.slider("Transaction Regularity (0-1)", 0.0, 1.0, 0.75, 0.01)
    with col_f2:
        counterparty = st.slider("Unique Counterparties (monthly)", 2, 80, 28)
        market_tenure = st.number_input("Market Tenure (months)", 1, 240, 36)
        dues_rate = st.slider("Association Dues Payment Rate", 0.0, 1.0, 0.88, 0.01)
    with col_f3:
        airtime_topup = st.number_input("Avg Airtime Top-up (₦)", 100, 10000, 1500, 100)
        sim_age = st.number_input("SIM Age (months)", 1, 120, 36)
        gps_stability = st.slider("GPS Location Stability (0-1)", 0.0, 1.0, 0.85, 0.01)

    if st.button("Calculate Credit Score"):
        import math
        score = int(np.clip(
            300
            + min(txn_volume / 2000000, 1) * 150
            + regularity * 120
            + min(counterparty / 80, 1) * 80
            + min(market_tenure / 240, 1) * 100
            + dues_rate * 80
            + min(airtime_topup / 10000, 1) * 50
            + min(sim_age / 120, 1) * 40
            + gps_stability * 80,
            300, 850
        ))

        def get_band(s):
            if s >= 750: return "Excellent"
            if s >= 680: return "Good"
            if s >= 580: return "Fair"
            if s >= 500: return "Poor"
            return "Thin File"

        band = get_band(score)
        colour = BAND_COLOURS[band]
        max_loan = 500000 if score >= 750 else 150000 if score >= 680 else 50000 if score >= 580 else 20000 if score >= 500 else 0

        st.markdown(f"""
        <div style='background:{colour};color:white;padding:20px;border-radius:8px'>
        <h2>Credit Score: {score}</h2>
        <h3>Band: {band}</h3>
        <b>Max Recommended Loan:</b> ₦{max_loan:,}<br>
        </div>
        """, unsafe_allow_html=True)

        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            domain={"x": [0, 1], "y": [0, 1]},
            gauge={
                "axis": {"range": [300, 850]},
                "bar": {"color": colour},
                "steps": [
                    {"range": [300, 500], "color": "#FFE0E0"},
                    {"range": [500, 580], "color": "#FFECC0"},
                    {"range": [580, 680], "color": "#FFFFC0"},
                    {"range": [680, 750], "color": "#E0FFE0"},
                    {"range": [750, 850], "color": "#C0FFC0"},
                ],
            },
        ))
        st.plotly_chart(gauge, use_container_width=True)

with tab3:
    st.subheader("Fairness Audit — Demographic Parity")
    gender_scores = portfolio.groupby("gender")["credit_score"].agg(["mean", "median", "count"])
    st.dataframe(gender_scores.style.background_gradient(subset=["mean"], cmap="RdYlGn"), use_container_width=True)
    gender_gap = abs(portfolio[portfolio["gender"] == "Female"]["credit_score"].mean() -
                     portfolio[portfolio["gender"] == "Male"]["credit_score"].mean())
    if gender_gap < 10:
        st.success(f"Gender parity gap: {gender_gap:.1f} points — within acceptable range (< 10 pts)")
    else:
        st.warning(f"Gender parity gap: {gender_gap:.1f} points — review feature importance for bias")

    fig4 = px.box(portfolio, x="gender", y="credit_score", color="gender",
                   title="Score Distribution by Gender", points="outliers")
    st.plotly_chart(fig4, use_container_width=True)

st.markdown("---")
st.caption("MOMAH MOSES .C. · Geospatial AI Engineer & Data Scientist · github.com/Momahmoses")
