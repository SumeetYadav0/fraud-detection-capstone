import pandas as pd

def transform_features(df):
    df["balance_diff_orig"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    df["balance_diff_dest"] = df["newbalanceDest"] - df["oldbalanceDest"]
    df["error_balance_orig"] = df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig"]
    df["error_balance_dest"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
    return df.drop(columns=["oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"])
