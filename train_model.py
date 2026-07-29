"""
MovieIQ - Stage 4: Predictive Modeling (Random Forest)
Trains a RandomForestClassifier to predict `success`, evaluates it, and saves the
fitted pipeline (model + genre encoder) to model/movieiq_model.pkl for the Streamlit app.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              confusion_matrix, classification_report)
from data_prep import load_clean_data

FEATURES = ["budget", "popularity", "runtime", "vote_average", "genre_encoded"]
TARGET = "success"


def build_features(df):
    df = df.copy()
    le = LabelEncoder()
    df["genre_encoded"] = le.fit_transform(df["genre"])
    return df, le


if __name__ == "__main__":
    df, _ = load_clean_data()
    df, genre_encoder = build_features(df)

    X = df[FEATURES]
    y = df[TARGET]

    # 80/20 split, stratified because success is imbalanced (~81% success)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=5,
        class_weight="balanced", random_state=42
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall:    {rec:.3f}")
    print("\nConfusion Matrix ([[TN, FP], [FN, TP]]):")
    print(cm)
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["Failure", "Success"]))

    # Baseline: predicting majority class every time
    baseline_acc = max(y_test.mean(), 1 - y_test.mean())
    print(f"Majority-class baseline accuracy: {baseline_acc:.3f}")

    # Confusion matrix plot
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Failure", "Success"], yticklabels=["Failure", "Success"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("assets/confusion_matrix.png", dpi=150)
    plt.close()

    # Feature importance plot
    importances = pd.Series(clf.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("\nFeature importances:")
    print(importances)

    plt.figure(figsize=(7, 5))
    sns.barplot(x=importances.values, y=importances.index, hue=importances.index,
                palette="viridis", legend=False)
    plt.title("Feature Importance (Random Forest)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig("assets/feature_importance.png", dpi=150)
    plt.close()

    # Save model + encoder for the Streamlit app
    with open("model/movieiq_model.pkl", "wb") as f:
        pickle.dump({"model": clf, "genre_encoder": genre_encoder, "features": FEATURES}, f)

    print("\nModel saved to model/movieiq_model.pkl")
