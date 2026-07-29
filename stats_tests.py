"""
MovieIQ - Stage 3: Statistical Testing
T-test on popularity (success vs failure) and Chi-square test on genre vs success.
"""
from scipy import stats
import pandas as pd
from data_prep import load_clean_data

ALPHA = 0.05


def run_ttest(df, col="popularity"):
    success_vals = df[df["success"] == 1][col]
    fail_vals = df[df["success"] == 0][col]
    t_stat, p_val = stats.ttest_ind(success_vals, fail_vals, equal_var=False)
    return t_stat, p_val, success_vals.mean(), fail_vals.mean()


def run_chi_square(df, col="genre"):
    contingency = pd.crosstab(df[col], df["success"])
    chi2, p_val, dof, expected = stats.chi2_contingency(contingency)
    return chi2, p_val, dof, contingency


if __name__ == "__main__":
    df, _ = load_clean_data()

    print("=" * 60)
    print("T-TEST: popularity, Success vs Failure")
    print("=" * 60)
    print("H0: mean popularity is equal for successful and unsuccessful movies")
    t_stat, p_val, mean_s, mean_f = run_ttest(df, "popularity")
    print(f"Success mean popularity: {mean_s:.3f} | Failure mean popularity: {mean_f:.3f}")
    print(f"t-statistic: {t_stat:.4f}, p-value: {p_val:.4f}")
    print("=> Reject H0 (significant difference)" if p_val < ALPHA
          else "=> Fail to reject H0 (no significant difference)")

    print("\n" + "=" * 60)
    print("CHI-SQUARE TEST: genre vs success")
    print("=" * 60)
    print("H0: genre and success are independent (no association)")
    chi2, p_val, dof, contingency = run_chi_square(df, "genre")
    print(f"Chi2: {chi2:.4f}, dof: {dof}, p-value: {p_val:.4f}")
    print("=> Reject H0 (genre is associated with success)" if p_val < ALPHA
          else "=> Fail to reject H0 (no significant association)")
