"""Script standalone pour executer le pipeline complet."""

import sys
from pathlib import Path

# Ajouter src au path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from src.utils.io import load_csv, save_parquet
from src.utils.logging import get_logger, setup_logging
from src.utils.paths import RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR, ensure_dir

setup_logging(level="INFO", json_format=False)
logger = get_logger(__name__)


def load_and_prepare_data() -> pd.DataFrame:
    """Charge et prepare les donnees."""
    logger.info("chargement_donnees")
    
    # Charger train.csv
    train = load_csv(RAW_DATA_DIR / "train.csv")
    logger.info("train_charge", rows=len(train))
    
    # Charger stores
    stores = load_csv(RAW_DATA_DIR / "stores.csv")
    logger.info("stores_charge", rows=len(stores))
    
    # Charger oil
    oil = load_csv(RAW_DATA_DIR / "oil.csv")
    logger.info("oil_charge", rows=len(oil))
    
    # Charger holidays
    holidays = load_csv(RAW_DATA_DIR / "holidays_events.csv")
    logger.info("holidays_charge", rows=len(holidays))
    
    # Normaliser les types
    train["date"] = pd.to_datetime(train["date"])
    train["store_nbr"] = train["store_nbr"].astype("int32")
    train["family"] = train["family"].astype("category")
    train["sales"] = train["sales"].astype("float32")
    train["onpromotion"] = train["onpromotion"].astype("int32")
    
    stores["store_nbr"] = stores["store_nbr"].astype("int32")
    
    oil["date"] = pd.to_datetime(oil["date"])
    oil["dcoilwtico"] = pd.to_numeric(oil["dcoilwtico"], errors="coerce")
    oil["dcoilwtico"] = oil["dcoilwtico"].interpolate().ffill().bfill()
    
    holidays["date"] = pd.to_datetime(holidays["date"])
    
    # Merge
    df = train.merge(stores, on="store_nbr", how="left")
    df = df.merge(oil, on="date", how="left")
    
    # Holidays nationaux
    national = holidays[holidays["locale"] == "National"][["date"]].copy()
    national["is_holiday"] = True
    df = df.merge(national, on="date", how="left")
    df["is_holiday"] = df["is_holiday"].fillna(False)
    
    logger.info("donnees_fusionnees", rows=len(df), cols=len(df.columns))
    
    return df


def generate_price(df: pd.DataFrame) -> pd.DataFrame:
    """Genere le prix synthetique."""
    logger.info("generation_prix")
    
    price_base = {
        "AUTOMOTIVE": 25.0, "BABY CARE": 15.0, "BEAUTY": 20.0,
        "BEVERAGES": 3.5, "BOOKS": 12.0, "BREAD/BAKERY": 2.5,
        "CELEBRATION": 8.0, "CLEANING": 5.0, "DAIRY": 4.0,
        "DELI": 6.0, "EGGS": 3.0, "FROZEN FOODS": 7.0,
        "GROCERY I": 4.5, "GROCERY II": 3.5, "HARDWARE": 18.0,
        "HOME AND KITCHEN": 22.0, "HOME APPLIANCES": 45.0,
        "HOME CARE": 8.0, "LADIESWEAR": 28.0, "LAWN AND GARDEN": 15.0,
        "LINGERIE": 18.0, "LIQUOR,WINE,BEER": 12.0, "MAGAZINES": 5.0,
        "MEATS": 9.0, "PERSONAL CARE": 7.0, "PET SUPPLIES": 10.0,
        "PLAYERS AND ELECTRONICS": 55.0, "POULTRY": 8.0,
        "PREPARED FOODS": 6.0, "PRODUCE": 3.0,
        "SCHOOL AND OFFICE SUPPLIES": 8.0, "SEAFOOD": 12.0,
    }
    
    rng = np.random.default_rng(42)
    
    # Prix de base par famille
    df["base_price"] = df["family"].map(price_base).fillna(5.0)
    
    # Coefficient par magasin
    unique_stores = df["store_nbr"].unique()
    store_coefs = {s: rng.uniform(0.85, 1.15) for s in unique_stores}
    df["store_coef"] = df["store_nbr"].map(store_coefs)
    
    # Prix final avec promotion
    df["price"] = df["base_price"] * df["store_coef"]
    df.loc[df["onpromotion"] > 0, "price"] *= 0.85  # 15% discount
    df["price"] = df["price"].round(2)
    
    # Cleanup
    df = df.drop(columns=["base_price", "store_coef"])
    
    logger.info(
        "prix_genere",
        mean=float(df["price"].mean()),
        min=float(df["price"].min()),
        max=float(df["price"].max()),
    )
    
    return df


def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split temporel."""
    logger.info("split_temporel")
    
    df = df.sort_values("date")
    dates = df["date"].unique()
    n = len(dates)
    
    train_end = dates[int(n * 0.8)]
    val_end = dates[int(n * 0.9)]
    
    train = df[df["date"] <= train_end].copy()
    val = df[(df["date"] > train_end) & (df["date"] <= val_end)].copy()
    test = df[df["date"] > val_end].copy()
    
    logger.info("split_complete", train=len(train), val=len(val), test=len(test))
    
    return train, val, test


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute toutes les features."""
    df = df.copy()
    
    # Temporelles
    df["day_of_week"] = df["date"].dt.dayofweek.astype("int8")
    df["month"] = df["date"].dt.month.astype("int8")
    df["year"] = df["date"].dt.year.astype("int16")
    df["is_weekend"] = (df["date"].dt.dayofweek >= 5).astype("int8")
    df["day_of_month"] = df["date"].dt.day.astype("int8")
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype("int16")
    
    # Prix
    df["log_price"] = np.log1p(df["price"])
    
    # Promotion
    df["is_promo"] = (df["onpromotion"] > 0).astype("int8")
    
    # Encodage categorique
    for col in ["family", "city", "state", "type"]:
        if col in df.columns:
            df[f"{col}_encoded"] = df[col].astype("category").cat.codes.astype("int16")
    
    logger.info("features_ajoutees", cols=len(df.columns))
    
    return df


def main() -> None:
    """Execute le pipeline complet."""
    # Etape 1: Charger et preparer
    df = load_and_prepare_data()
    
    # Etape 2: Generer le prix
    df = generate_price(df)
    
    # Sauvegarder interim
    ensure_dir(INTERIM_DATA_DIR)
    save_parquet(df, INTERIM_DATA_DIR / "sales_with_price.parquet")
    logger.info("interim_sauvegarde", path=str(INTERIM_DATA_DIR / "sales_with_price.parquet"))
    
    # Etape 3: Split temporel
    train, val, test = temporal_split(df)
    
    # Etape 4: Features
    train = add_features(train)
    val = add_features(val)
    test = add_features(test)
    
    # Sauvegarder processed
    ensure_dir(PROCESSED_DATA_DIR)
    save_parquet(train, PROCESSED_DATA_DIR / "train_features.parquet")
    save_parquet(val, PROCESSED_DATA_DIR / "val_features.parquet")
    save_parquet(test, PROCESSED_DATA_DIR / "test_features.parquet")
    
    logger.info(
        "pipeline_termine",
        train_rows=len(train),
        val_rows=len(val),
        test_rows=len(test),
    )


if __name__ == "__main__":
    main()
