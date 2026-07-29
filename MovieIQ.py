"""
MovieIQ - Stage 5: Streamlit Dashboard & Deployment
Interactive dashboard: filter by genre & min rating, view EDA + stats results,
and get a live success prediction from the trained Random Forest model.
Run with: streamlit run MovieIQ.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from data_prep import load_clean_data

st.set_page_config(page_title="MovieIQ", page_icon="🎬", layout="wide")

# ---------- Load data & model ----------
@st.cache_data
def get_data():
    df, dropped = load_clean_data("data/movies.csv")
    return df, dropped


@st.cache_resource
def get_model():
    with open("model/movieiq_model.pkl", "rb") as f:
        return pickle.load(f)


df, dropped_rows = get_data()
bundle = get_model()
model = bundle["model"]
genre_encoder = bundle["genre_encoder"]
FEATURES = bundle["features"]

# ---------- Sidebar filters ----------
st.sidebar.header("Filters")
genres = ["All"] + sorted(df["genre"].unique().tolist())
selected_genre = st.sidebar.selectbox("Genre", genres)
min_rating = st.sidebar.slider("Minimum vote average", 0.0, 10.0, 0.0, 0.1)

filtered = df.copy()
if selected_genre != "All":
    filtered = filtered[filtered["genre"] == selected_genre]
filtered = filtered[filtered["vote_average"] >= min_rating]

st.sidebar.markdown("---")
st.sidebar.metric("Movies in view", len(filtered))
st.sidebar.metric("Success rate in view", f"{filtered['success'].mean()*100:.1f}%" if len(filtered) else "n/a")

# ---------- Header ----------
st.title("🎬 MovieIQ — Predictive Analytics on Film Success")
st.caption("A movie is labeled **successful** when its revenue exceeds its budget.")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔍 EDA", "📈 Statistical Tests", "🎯 Predict a Movie"])

# ---------- Tab 1: Overview ----------
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total movies", len(df))
    col2.metric("Overall success rate", f"{df['success'].mean()*100:.1f}%")
    col3.metric("Avg budget", f"${df['budget'].mean():,.0f}")
    col4.metric("Avg revenue", f"${df['revenue'].mean():,.0f}")

    st.subheader("Filtered data")
    st.dataframe(
        filtered[["title", "genre", "budget", "revenue", "popularity",
                  "runtime", "vote_average", "success"]].reset_index(drop=True),
        use_container_width=True, height=300
    )

# ---------- Tab 2: EDA ----------
with tab2:
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Budget vs Revenue")
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.scatterplot(data=filtered, x="budget", y="revenue", hue="success",
                         palette={0: "#d62728", 1: "#2ca02c"}, alpha=0.6, ax=ax)
        ax.set_xlabel("Budget ($)"); ax.set_ylabel("Revenue ($)")
        st.pyplot(fig)
        if len(filtered) > 1:
            st.caption(f"Correlation: {filtered['budget'].corr(filtered['revenue']):.3f}")

    with c2:
        st.subheader("Success Rate by Genre")
        genre_success = df.groupby("genre")["success"].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.barplot(x=genre_success.values, y=genre_success.index,
                    hue=genre_success.index, palette="viridis", legend=False, ax=ax)
        ax.set_xlabel("Success rate")
        st.pyplot(fig)

    st.subheader("Feature Distributions by Success")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, ["popularity", "runtime", "vote_average"]):
        sns.boxplot(data=filtered, x="success", y=col, hue="success", ax=ax,
                    palette={0: "#d62728", 1: "#2ca02c"}, legend=False)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Failure", "Success"])
    st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(6, 5))
    num_cols = ["budget", "revenue", "popularity", "runtime", "vote_average", "success"]
    sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    st.pyplot(fig)

# ---------- Tab 3: Stats ----------
with tab3:
    st.subheader("T-Test — popularity: Success vs Failure")
    s = df[df.success == 1]["popularity"]
    f = df[df.success == 0]["popularity"]
    t_stat, p_val = stats.ttest_ind(s, f, equal_var=False)
    col1, col2, col3 = st.columns(3)
    col1.metric("Success mean popularity", f"{s.mean():.2f}")
    col2.metric("Failure mean popularity", f"{f.mean():.2f}")
    col3.metric("p-value", f"{p_val:.4f}")
    st.write("**H0:** mean popularity is equal for successful and unsuccessful movies.")
    st.success("Reject H0 — statistically significant difference (p < 0.05).") if p_val < 0.05 \
        else st.info("Fail to reject H0 — no significant difference (p ≥ 0.05).")

    st.subheader("Chi-Square Test — genre vs success")
    contingency = pd.crosstab(df["genre"], df["success"])
    chi2, p_val_chi, dof, _ = stats.chi2_contingency(contingency)
    col1, col2 = st.columns(2)
    col1.metric("Chi2 statistic", f"{chi2:.3f}")
    col2.metric("p-value", f"{p_val_chi:.4f}")
    st.write("**H0:** genre and success are independent.")
    st.success("Reject H0 — genre is associated with success (p < 0.05).") if p_val_chi < 0.05 \
        else st.info("Fail to reject H0 — no significant association between genre and success (p ≥ 0.05).")

# ---------- Tab 4: Predict ----------
with tab4:
    st.subheader("Predict Success for a New Movie")
    c1, c2 = st.columns(2)
    with c1:
        in_budget = st.number_input("Budget ($)", min_value=1000, value=50_000_000, step=1_000_000)
        in_popularity = st.slider("Popularity", 0.0, 100.0, 50.0)
        in_runtime = st.slider("Runtime (minutes)", 60, 220, 120)
    with c2:
        in_vote = st.slider("Expected vote average", 0.0, 10.0, 6.5, 0.1)
        in_genre = st.selectbox("Genre", sorted(df["genre"].unique().tolist()))

    if st.button("Predict", type="primary"):
        genre_code = genre_encoder.transform([in_genre])[0]
        X_new = pd.DataFrame([[in_budget, in_popularity, in_runtime, in_vote, genre_code]],
                              columns=FEATURES)
        pred = model.predict(X_new)[0]
        proba = model.predict_proba(X_new)[0]

        if pred == 1:
            st.success(f"✅ Predicted: **Success** (confidence: {proba[1]*100:.1f}%)")
        else:
            st.error(f"❌ Predicted: **Not Successful** (confidence: {proba[0]*100:.1f}%)")

        st.caption(
            "Note: this model's accuracy (≈72%) is below the naive baseline of always "
            "predicting success (≈81%), because popularity, runtime, and rating carry very "
            "little real signal for success in this dataset. Treat predictions as directional, not exact."
        )

st.markdown("---")
st.caption(f"Rows dropped during cleaning (zero/invalid budget or revenue): {dropped_rows} · Built with Streamlit")
