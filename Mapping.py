import os
import re

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


STATE_FEATURE_NAMES = [
    "Disbursed amount (standardized)",
    "Asset cost (standardized)",
    "Loan-to-value (standardized)",
    "New-to-credit flag",
    "CIBIL score (standardized)",
    "Active accounts (standardized)",
    "Overdue accounts (standardized)",
    "Inquiry velocity (standardized)",
    "Recent delinquency (standardized)",
    "Average account age (standardized)",
    "Salaried, established credit",
    "Self-employed, established credit",
    "Salaried, new-to-credit",
    "Self-employed, new-to-credit",
    "Repo rate",
    "Inflation assumption",
    "Quarter index",
]


class Dataset:
    def __init__(self, csv_path: str):
        self.path = csv_path
        self.scaler = StandardScaler()
        self.mapping = {
            "2018-08": 0.0650,
            "2018-09": 0.0650,
            "2018-10": 0.0650,
            "default": 0.0650,
        }

    @staticmethod
    def _parse_date(date_str):
        if pd.isna(date_str) or not isinstance(date_str, str):
            return 0.0
        years = re.findall(r"(\d+)yrs", date_str)
        months = re.findall(r"(\d+)mon", date_str)
        total_years = int(years[0]) if years else 0
        total_months = int(months[0]) if months else 0
        return float(total_years * 12 + total_months)

    def _transform_frame(self, df: pd.DataFrame, fit_scaler: bool):
        df = df.copy()
        df["EMPLOYMENT_TYPE"] = df["EMPLOYMENT_TYPE"].fillna("Self employed")
        df["AVERAGE_ACCT_AGE_MONTHS"] = df["AVERAGE_ACCT_AGE"].apply(self._parse_date)
        df["CREDIT_HISTORY_LENGTH_MONTHS"] = df["CREDIT_HISTORY_LENGTH"].apply(self._parse_date)
        df["LOG_DISBURSED"] = np.log1p(df["DISBURSED_AMOUNT"])
        df["LOG_ASSET_COST"] = np.log1p(df["ASSET_COST"])
        df["LTV_NORM"] = df["LTV"] / 100.0
        df["CIBIL_IS_NTC"] = (df["PERFORM_CNS_SCORE"] == 0).astype(float)
        df["CIBIL_SCORE_NORM"] = np.where(
            df["PERFORM_CNS_SCORE"] > 0,
            (df["PERFORM_CNS_SCORE"] - 300) / 600.0,
            0.0,
        ).clip(0.0, 1.0)
        df["TOT_ACTIVE_ACCTS"] = (
            df["PRI_ACTIVE_ACCTS"] + df["SEC_ACTIVE_ACCTS"]
        ).clip(0, 20)
        df["TOT_OVERDUE_ACCTS"] = (
            df["PRI_OVERDUE_ACCTS"] + df["SEC_OVERDUE_ACCTS"]
        ).clip(0, 10)
        df["INQUIRY_VELOCITY"] = df["NO_OF_INQUIRIES"].clip(0, 10) / 10.0
        df["DELINQUENCY_RECENT"] = (
            df["DELINQUENT_ACCTS_IN_LAST_SIX_MONTHS"].clip(0, 5) / 5.0
        )

        financial_columns = [
            "LOG_DISBURSED",
            "LOG_ASSET_COST",
            "LTV_NORM",
            "CIBIL_IS_NTC",
            "CIBIL_SCORE_NORM",
            "TOT_ACTIVE_ACCTS",
            "TOT_OVERDUE_ACCTS",
            "INQUIRY_VELOCITY",
            "DELINQUENCY_RECENT",
            "AVERAGE_ACCT_AGE_MONTHS",
        ]
        if fit_scaler:
            scaled_financial = self.scaler.fit_transform(df[financial_columns])
        else:
            if not hasattr(self.scaler, "mean_"):
                raise RuntimeError("Fit the training scaler or load a checkpoint before inference.")
            scaled_financial = self.scaler.transform(df[financial_columns])

        conditions = [
            (df["EMPLOYMENT_TYPE"] == "Salaried") & (df["CIBIL_IS_NTC"] == 0),
            (df["EMPLOYMENT_TYPE"] == "Self employed") & (df["CIBIL_IS_NTC"] == 0),
            (df["EMPLOYMENT_TYPE"] == "Salaried") & (df["CIBIL_IS_NTC"] == 1),
            (df["EMPLOYMENT_TYPE"] == "Self employed") & (df["CIBIL_IS_NTC"] == 1),
        ]
        context_labels = np.select(conditions, [0, 1, 2, 3], default=2)
        context_features = np.eye(4, dtype=np.float32)[context_labels]

        disbursal_dates = pd.to_datetime(
            df["DISBURSAL_DATE"], format="%d-%m-%Y", errors="coerce"
        )
        year_month = disbursal_dates.dt.strftime("%Y-%m")
        repo_rates = year_month.map(self.mapping).fillna(self.mapping["default"]).values
        inflation = np.full(len(df), 0.052)
        quarter_index = (disbursal_dates.dt.month / 12.0).fillna(0.75).values
        macro_features = np.column_stack([repo_rates, inflation, quarter_index])

        states = np.hstack([scaled_financial, context_features, macro_features]).astype(np.float32)
        return states, context_labels

    def preprocess(self):
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"File not found: {self.path}")
        df = pd.read_csv(self.path)
        if "LOAN_DEFAULT" not in df.columns:
            raise ValueError("Training data must contain the LOAN_DEFAULT target column.")
        states, context_labels = self._transform_frame(df, fit_scaler=True)
        default_labels = df["LOAN_DEFAULT"].to_numpy(dtype=np.int64)
        print(
            f">>[Dataset] Training preprocessing complete. States: {states.shape}, "
            f"Labels: {default_labels.shape}"
        )
        return states, default_labels, context_labels

    def preprocess_unseen(self, csv_path=None):
        path = csv_path or self.path
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        df = pd.read_csv(path)
        states, context_labels = self.preprocess_frame(df)
        return df, states, context_labels

    def preprocess_frame(self, df: pd.DataFrame):
        required_columns = {
            "EMPLOYMENT_TYPE",
            "AVERAGE_ACCT_AGE",
            "CREDIT_HISTORY_LENGTH",
            "DISBURSED_AMOUNT",
            "ASSET_COST",
            "LTV",
            "PERFORM_CNS_SCORE",
            "PRI_ACTIVE_ACCTS",
            "SEC_ACTIVE_ACCTS",
            "PRI_OVERDUE_ACCTS",
            "SEC_OVERDUE_ACCTS",
            "NO_OF_INQUIRIES",
            "DELINQUENT_ACCTS_IN_LAST_SIX_MONTHS",
            "DISBURSAL_DATE",
        }
        missing_columns = sorted(required_columns.difference(df.columns))
        if missing_columns:
            raise ValueError(f"Application data is missing required columns: {missing_columns}")
        return self._transform_frame(df, fit_scaler=False)