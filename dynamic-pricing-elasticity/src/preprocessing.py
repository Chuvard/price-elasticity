"""Cleaning, validation, and feature engineering for the pricing panel."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def load_data() -> pd.DataFrame:
    df = pd.read_csv(config.DATA_PATH, parse_dates=["date"])
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: drop impossible rows, enforce positivity for logs."""
    df = df.copy()
    df = df[df["units_sold"] > 0]
    df = df[df["net_price"] > 0]
    df = df[df["competitor_price_index"] > 0]
    df = df.dropna(subset=["units_sold", "net_price"])
    return df.reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create log transforms and model-ready fields."""
    df = df.copy()
    df["log_units"] = np.log(df["units_sold"])
    df["log_net_price"] = np.log(df["net_price"])
    df["log_competitor_index"] = np.log(df["competitor_price_index"])
    df["log_category_index"] = np.log(df["category_market_index"])
    df["log_inflation"] = np.log(df["inflation_index"])
    df["log_season"] = np.log(df["season_index"])
    df["month"] = df["month"].astype(int)
    return df


def preprocess() -> pd.DataFrame:
    df = load_data()
    df = clean(df)
    df = engineer_features(df)
    print(f"[preprocessing] {len(df):,} clean rows, {df.shape[1]} columns")
    return df
