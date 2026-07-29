"""
MovieIQ - Stage 2: Exploratory Data Analysis
Generates all EDA charts into assets/.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from data_prep import load_clean_data

sns.set_style("whitegrid")
PALETTE = "viridis"


def scatter_budget_revenue(df):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="budget", y="revenue", hue="success",
                     palette={0: "#d62728", 1: "#2ca02c"}, alpha=0.6)
    plt.title("Budget vs Revenue")
    plt.xlabel("Budget ($)")
    plt.ylabel("Revenue ($)")
    plt.legend(title="Success", labels=["Failure", "Success"])
    plt.tight_layout()
    plt.savefig("assets/budget_vs_revenue.png", dpi=150)
    plt.close()

    corr = df["budget"].corr(df["revenue"])
    return corr


def genre_trends(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    genre_counts = df["genre"].value_counts()
    sns.barplot(x=genre_counts.values, y=genre_counts.index, hue=genre_counts.index,
                ax=axes[0], palette=PALETTE, legend=False)
    axes[0].set_title("Movie Count by Genre")
    axes[0].set_xlabel("Count")

    genre_success = df.groupby("genre")["success"].mean().sort_values(ascending=False)
    sns.barplot(x=genre_success.values, y=genre_success.index, hue=genre_success.index,
                ax=axes[1], palette=PALETTE, legend=False)
    axes[1].set_title("Success Rate by Genre")
    axes[1].set_xlabel("Success Rate")

    plt.tight_layout()
    plt.savefig("assets/genre_trends.png", dpi=150)
    plt.close()
    return genre_success


def feature_vs_success(df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, col in zip(axes, ["popularity", "runtime", "vote_average"]):
        sns.boxplot(data=df, x="success", y=col, hue="success", ax=ax,
                    palette={0: "#d62728", 1: "#2ca02c"}, legend=False)
        ax.set_xticklabels(["Failure", "Success"])
        ax.set_title(f"{col} vs Success")
    plt.tight_layout()
    plt.savefig("assets/feature_vs_success.png", dpi=150)
    plt.close()


def correlation_heatmap(df):
    plt.figure(figsize=(7, 6))
    num_cols = ["budget", "revenue", "popularity", "runtime", "vote_average", "success"]
    corr = df[num_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("assets/correlation_heatmap.png", dpi=150)
    plt.close()
    return corr


if __name__ == "__main__":
    df, _ = load_clean_data()

    corr_br = scatter_budget_revenue(df)
    print(f"Budget-Revenue correlation: {corr_br:.3f}")

    genre_success = genre_trends(df)
    print("\nSuccess rate by genre:")
    print(genre_success)

    feature_vs_success(df)

    corr = correlation_heatmap(df)
    print("\nFull correlation matrix:")
    print(corr)

    print("\nCharts saved to assets/")
