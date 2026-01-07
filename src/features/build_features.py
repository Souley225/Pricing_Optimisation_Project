"""Construction des features pour la modelisation."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.preprocessing import LabelEncoder

from src.utils.io import load_parquet, save_parquet
from src.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def add_temporal_features(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
    """Ajoute les features temporelles.

    Args:
        df: DataFrame source.
        date_column: Nom de la colonne date.

    Returns:
        DataFrame avec les features temporelles ajoutees.
    """
    df = df.copy()
    date_col = pd.to_datetime(df[date_column])

    df["day_of_week"] = date_col.dt.dayofweek.astype("int8")
    df["day_of_month"] = date_col.dt.day.astype("int8")
    df["week_of_year"] = date_col.dt.isocalendar().week.astype("int8")
    df["month"] = date_col.dt.month.astype("int8")
    df["quarter"] = date_col.dt.quarter.astype("int8")
    df["year"] = date_col.dt.year.astype("int16")
    df["is_weekend"] = (date_col.dt.dayofweek >= 5).astype("int8")
    df["is_month_start"] = date_col.dt.is_month_start.astype("int8")
    df["is_month_end"] = date_col.dt.is_month_end.astype("int8")

    logger.info("features_temporelles_ajoutees", n_features=9)

    return df


def add_price_features(
    df: pd.DataFrame,
    price_column: str,
    lag_periods: list[int],
    rolling_windows: list[int],
    log_transform: bool = True,
) -> pd.DataFrame:
    """Ajoute les features de prix.

    Args:
        df: DataFrame source.
        price_column: Nom de la colonne prix.
        lag_periods: Periodes de lag a calculer.
        rolling_windows: Fenetres pour moyennes mobiles.
        log_transform: Si True, ajoute le log du prix.

    Returns:
        DataFrame avec les features prix ajoutees.
    """
    df = df.copy()

    # Log du prix
    if log_transform:
        df["log_price"] = np.log1p(df[price_column])

    # Trier par store, family et date pour les calculs de lag
    df = df.sort_values(["store_nbr", "family", "date"])

    # Grouper par store et family
    group_cols = ["store_nbr", "family"]

    # Lags de prix
    for lag in lag_periods:
        col_name = f"price_lag_{lag}"
        df[col_name] = df.groupby(group_cols)[price_column].shift(lag)

    # Moyennes mobiles
    def rolling_mean(x: pd.Series, w: int) -> pd.Series:
        return x.rolling(window=w, min_periods=1).mean()

    for window in rolling_windows:
        col_name = f"price_rolling_mean_{window}"
        df[col_name] = df.groupby(group_cols)[price_column].transform(
            lambda x, w=window: rolling_mean(x, w)
        )

    # Variation de prix par rapport au lag 1
    if "price_lag_1" in df.columns:
        df["price_change"] = df[price_column] - df["price_lag_1"]
        df["price_change_pct"] = df["price_change"] / df["price_lag_1"].replace(0, np.nan)

    # Remplir les NaN introduits par les lags
    lag_cols = [c for c in df.columns if "lag" in c or "rolling" in c or "change" in c]
    for col in lag_cols:
        df[col] = df[col].fillna(0)

    logger.info(
        "features_prix_ajoutees",
        n_lags=len(lag_periods),
        n_rolling=len(rolling_windows),
        log_transform=log_transform,
    )

    return df


def add_promotion_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les features de promotion.

    Args:
        df: DataFrame source.

    Returns:
        DataFrame avec les features promotion ajoutees.
    """
    df = df.copy()

    # Flag binaire promotion
    df["is_promo"] = (df["onpromotion"] > 0).astype("int8")

    # Trier par store, family et date
    df = df.sort_values(["store_nbr", "family", "date"])
    group_cols = ["store_nbr", "family"]

    # Jours depuis derniere promo
    def days_since_promo(series: pd.Series) -> pd.Series:
        """Calcule le nombre de jours depuis la derniere promotion."""
        result = pd.Series(index=series.index, dtype="float32")
        last_promo = None
        for i, (idx, val) in enumerate(series.items()):
            if val > 0:
                result[idx] = 0
                last_promo = i
            elif last_promo is not None:
                result[idx] = i - last_promo
            else:
                result[idx] = -1  # Jamais eu de promo
        return result

    df["days_since_promo"] = df.groupby(group_cols)["onpromotion"].transform(days_since_promo)

    # Intensite de la promotion (nombre de produits en promo pour ce store/date)
    df["promo_intensity"] = df.groupby(["store_nbr", "date"])["is_promo"].transform("sum")

    logger.info("features_promotion_ajoutees", n_features=3)

    return df


def encode_categorical_features(
    df: pd.DataFrame,
    categorical_columns: list[str],
    method: str = "label",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Encode les variables categoriques.

    Args:
        df: DataFrame source.
        categorical_columns: Liste des colonnes a encoder.
        method: Methode d'encodage ('label', 'onehot').

    Returns:
        Tuple (DataFrame encode, dictionnaire des encodeurs).
    """
    df = df.copy()
    encoders: dict[str, Any] = {}

    for col in categorical_columns:
        if col not in df.columns:
            continue

        if method == "label":
            le = LabelEncoder()
            # Convertir en string pour gerer les valeurs manquantes
            df[col] = df[col].astype(str).fillna("unknown")
            df[f"{col}_encoded"] = le.fit_transform(df[col])
            encoders[col] = le

        elif method == "onehot":
            dummies = pd.get_dummies(df[col], prefix=col, dummy_na=True)
            df = pd.concat([df, dummies], axis=1)

    logger.info(
        "encodage_categorique",
        method=method,
        n_columns=len(categorical_columns),
    )

    return df, encoders


def build_feature_matrix(
    df: pd.DataFrame,
    target_column: str,
    exclude_columns: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    """Construit la matrice de features et le vecteur cible.

    Args:
        df: DataFrame avec toutes les features.
        target_column: Nom de la colonne cible.
        exclude_columns: Colonnes a exclure des features.

    Returns:
        Tuple (X features, y target).
    """
    # Colonnes a exclure
    exclude = set(exclude_columns + [target_column])

    # Garder uniquement les colonnes numeriques
    feature_cols = [
        col
        for col in df.columns
        if col not in exclude
        and df[col].dtype in ["int8", "int16", "int32", "int64", "float32", "float64"]
    ]

    X = df[feature_cols].copy()
    y = df[target_column].copy()

    # Remplir les NaN restants
    X = X.fillna(0)

    logger.info(
        "matrice_features_construite",
        n_features=len(feature_cols),
        n_samples=len(X),
        feature_names=feature_cols[:10],  # Log premiers 10 noms
    )

    return X, y


def run_build_features(cfg: DictConfig) -> None:
    """Execute le pipeline de feature engineering.

    Args:
        cfg: Configuration Hydra.
    """
    setup_logging(level="INFO", json_format=False)

    processed_dir = Path(cfg.data.processed_dir)

    for split_name in ["train", "val", "test"]:
        input_path = processed_dir / f"{split_name}.parquet"

        if not input_path.exists():
            logger.warning("fichier_manquant", path=str(input_path))
            continue

        df = load_parquet(input_path)
        logger.info("split_charge", name=split_name, rows=len(df))

        # Features temporelles
        df = add_temporal_features(df, "date")

        # Features de prix
        df = add_price_features(
            df,
            price_column="price",
            lag_periods=cfg.features.price.lag_periods,
            rolling_windows=cfg.features.price.rolling_windows,
            log_transform=cfg.features.price.log_transform,
        )

        # Features de promotion
        df = add_promotion_features(df)

        # Encodage categorique
        categorical_cols = ["family", "city", "state", "type"]
        df, encoders = encode_categorical_features(
            df,
            categorical_cols,
            method=cfg.features.encoding.method,
        )

        # Sauvegarder
        output_path = processed_dir / f"{split_name}_features.parquet"
        save_parquet(df, output_path)
        logger.info("features_sauvegardees", path=str(output_path), rows=len(df))


if __name__ == "__main__":
    import hydra

    @hydra.main(version_base=None, config_path="../../configs/hydra", config_name="config")
    def main(cfg: DictConfig) -> None:
        run_build_features(cfg)

    main()
