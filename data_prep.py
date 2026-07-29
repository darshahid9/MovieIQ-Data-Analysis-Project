"""
MovieIQ - Stage 1: Data Preparation
Loads movies.csv, cleans it, engineers the target and genre columns.
Import load_clean_data() from this module wherever needed (EDA, stats, modeling, app).
"""
import pandas as pd
import ast


def parse_genre(genres_str):
    """TMDB-style genres column is a stringified list of dicts, e.g. [{'id': 18, 'name': 'Drama'}].
    Returns the primary genre name, or 'Unknown' if the list is empty."""
    try:
        parsed = ast.literal_eval(genres_str)
        if isinstance(parsed, list) and len(parsed) > 0:
            return parsed[0]['name']
        return "Unknown"
    except (ValueError, SyntaxError, TypeError):
        return "Unknown"


def load_clean_data(path="data/movies.csv"):
    df = pd.read_csv(path)

    # --- Stage 1.2: handle zero/invalid budget & revenue ---
    # A 0 budget or revenue is not real financial data (movies aren't made for free,
    # and $0 revenue almost always means "not reported" rather than "earned nothing").
    # Left in, they'd corrupt the success ratio and blow up ratio-based features.
    before = len(df)
    df = df[(df["budget"] > 0) & (df["revenue"] > 0)].copy()
    dropped = before - len(df)

    # --- Stage 1.3: target variable ---
    df["success"] = (df["revenue"] > df["budget"]).astype(int)

    # --- Stage 1.4: genre processing ---
    df["genre"] = df["genres"].apply(parse_genre)

    return df, dropped


if __name__ == "__main__":
    df, dropped = load_clean_data()
    print(f"Rows dropped for zero/invalid budget or revenue: {dropped}")
    print(f"Final shape: {df.shape}")
    print("\nSummary statistics:")
    print(df[["budget", "revenue", "popularity", "runtime", "vote_average"]].describe())
    print("\nSuccess rate:")
    print(df["success"].value_counts(normalize=True))
    print("\nGenre counts:")
    print(df["genre"].value_counts())
