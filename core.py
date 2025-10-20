from __future__ import annotations
import sqlite3
import pandas as pd
import numpy as np
import logging


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