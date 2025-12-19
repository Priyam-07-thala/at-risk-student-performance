import pandas as pd

def assign_risk(row):
    score = (
        0.4 * row["avg_marks"] +
        0.25 * row["attendance"] +
        0.2 * row["assignment_completion"] +
        0.15 * (row["behavior_score"] * 10)
    )

    if score >= 85:
        return "Very Low"
    elif score >= 70:
        return "Low"
    elif score >= 50:
        return "Medium"
    else:
        return "High"


def preprocess_df(df):
    df = df.copy()

    df["avg_marks"] = df[["math score","reading score","writing score"]].mean(axis=1)

    import numpy as np
    df["attendance"] = np.clip(df["avg_marks"] + np.random.normal(0, 5, len(df)), 50, 100)
    df["assignment_completion"] = np.clip(df["avg_marks"] + np.random.normal(0, 10, len(df)), 40, 100)

    df["behavior_score"] = df["test preparation course"].apply(
        lambda x: 8 if x == "completed" else 5
    )

    df["risk_level"] = df.apply(assign_risk, axis=1)
    return df
