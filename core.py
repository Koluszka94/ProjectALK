from __future__ import annotations
import sqlite3
import pandas as pd
import numpy as np
import logging

NUM_COLS = ["HeartRate", "Systolic", "Diastolic"]
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


# ==========================
# ===== LOADING / CLEANING =====
# ==========================
def normalize_symptoms(series: pd.Series) -> pd.Series:
    """
    Normalizuje kolumnę Symptoms:
    - usuwa spacje i konwertuje na lower()
    - puste stringi oraz 'none' → NaN
    """
    s = series.astype("string").str.strip().str.lower()
    return s.replace({"": np.nan, "none": np.nan})


def load_data(path: str) -> pd.DataFrame:
    """
    Wczytuje dane z pliku CSV i normalizuje kolumnę Symptoms.
    """
    logging.info(f"Loading CSV: {path}")
    df = pd.read_csv(path, dtype={"PatientID": "string"})
    if "Symptoms" in df.columns:
        df["Symptoms"] = normalize_symptoms(df["Symptoms"])
    else:
        df["Symptoms"] = pd.Series([np.nan] * len(df), dtype="string")
    return df


def load_sqlite(db_path: str, table: str) -> pd.DataFrame:
    """
    Wczytuje dane z bazy SQLite (SELECT * FROM table) i normalizuje kolumnę Symptoms.
    """
    if not table.isidentifier():
        raise ValueError(f"Invalid table name: {table}")

    logging.info(f"Loading table '{table}' from SQLite: {db_path}")
    con = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(f"SELECT * FROM {table}", con)
    finally:
        con.close()

    if "PatientID" in df.columns:
        df["PatientID"] = df["PatientID"].astype("string")
    if "Symptoms" not in df.columns:
        df["Symptoms"] = pd.Series([np.nan] * len(df), dtype="string")
    else:
        df["Symptoms"] = normalize_symptoms(df["Symptoms"])
    return df


def add_bp_numbers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rozbija 'BloodPressure' na numeryczne Systolic/Diastolic
    i wstawia je po kolumnie 'BloodPressure'.
    """
    if "BloodPressure" not in df.columns:
        logging.warning("Missing 'BloodPressure' column. Skipping BP split.")
        return df

    bp = df["BloodPressure"].astype("string").str.split("/", n=1, expand=True)
    df["Systolic"] = pd.to_numeric(bp[0], errors="coerce")
    df["Diastolic"] = pd.to_numeric(bp[1], errors="coerce")

    cols = [c for c in df.columns if c not in ("Systolic", "Diastolic")]
    insert_at = cols.index("BloodPressure") + 1 if "BloodPressure" in cols else len(cols)
    cols[insert_at:insert_at] = ["Systolic", "Diastolic"]
    return df.loc[:, cols]


# ==========================
# ===== FILTERING =====
# ==========================
def filter_patients(
    df: pd.DataFrame,
    *,
    age_min=None, age_max=None,
    gender=None,
    systolic_min=None, systolic_max=None,
    diastolic_min=None, diastolic_max=None,
    hr_min=None, hr_max=None,
    only_missing_symptom=None  
) -> pd.DataFrame:
    """Filtruje pacjentów wg zadanych parametrów."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    m = pd.Series(True, index=df.index)

    def apply_range(col, min_val, max_val):
        nonlocal m
        if col in df.columns:
            if min_val is not None:
                m &= df[col] >= min_val
            if max_val is not None:
                m &= df[col] <= max_val

    apply_range("Age", age_min, age_max)
    apply_range("Systolic", systolic_min, systolic_max)
    apply_range("Diastolic", diastolic_min, diastolic_max)
    apply_range("HeartRate", hr_min, hr_max)

    if gender is not None and "Gender" in df.columns:
        m &= df["Gender"].astype("string").str.upper() == str(gender).upper()

    if only_missing_symptom is True:
        m &= df["Symptoms"].isna()
    elif only_missing_symptom is False:
        m &= df["Symptoms"].notna()

    return df.loc[m].copy()

# ==========================
# ====== STATS ======
# ==========================
def basic_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Podstawowe statystyki opisowe:
    count, mean, median, std, min, q1, q3, max
    """
    cols = [c for c in NUM_COLS if c in df.columns]
    d = {
        "count": df[cols].count(),
        "mean": df[cols].mean(numeric_only=True),
        "median": df[cols].median(numeric_only=True),
        "std": df[cols].std(numeric_only=True),
        "min": df[cols].min(numeric_only=True),
        "q1": df[cols].quantile(0.25, numeric_only=True),
        "q3": df[cols].quantile(0.75, numeric_only=True),
        "max": df[cols].max(numeric_only=True),
    }
    out = pd.DataFrame(d).T
    return out[cols].T


def add_age_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Dodaje kolumnę AgeGroup z przedziałami wiekowymi."""
    if "Age" not in df.columns:
        return df
    bins = [0, 30, 50, 120]
    labels = ["<30", "30-50", "50+"]
    df["AgeGroup"] = pd.cut(df["Age"], bins=bins, labels=labels, right=False)
    return df


def group_summary(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """
    Uniwersalna funkcja grupująca:
    statystyki count/mean/median/min/max dla NUM_COLS w podziale po by.
    """
    cols = [c for c in NUM_COLS if c in df.columns]
    g = df.groupby(by, dropna=False)[cols].agg(["count", "mean", "median", "min", "max"])
    g.columns = ["_".join(col).strip() for col in g.columns.values]
    return g


def group_by_gender(df: pd.DataFrame) -> pd.DataFrame:
    return group_summary(df, "Gender")


def group_by_age(df: pd.DataFrame) -> pd.DataFrame:
    if "AgeGroup" not in df.columns:
        df = add_age_groups(df)
    return group_summary(df, "AgeGroup")


def compare_symptoms(df: pd.DataFrame) -> pd.DataFrame:
    if "Symptoms" not in df.columns:
        logging.warning("Column 'Symptoms' not found.")
        return pd.DataFrame()
    df = df.assign(SymptomStatus=df["Symptoms"].notna().map({True: "With symptoms", False: "Without symptoms"}))
    return group_summary(df, "SymptomStatus")