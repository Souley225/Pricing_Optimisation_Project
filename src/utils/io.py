"""Fonctions de lecture et ecriture de fichiers."""

from pathlib import Path
from typing import Any

import joblib
import pandas as pd


def load_csv(path: Path | str, **kwargs: Any) -> pd.DataFrame:
    """Charge un fichier CSV en DataFrame.

    Args:
        path: Chemin du fichier CSV.
        **kwargs: Arguments supplementaires pour pandas.read_csv.

    Returns:
        DataFrame contenant les donnees.
    """
    return pd.read_csv(path, **kwargs)


def save_csv(df: pd.DataFrame, path: Path | str, **kwargs: Any) -> None:
    """Sauvegarde un DataFrame en CSV.

    Args:
        df: DataFrame a sauvegarder.
        path: Chemin de destination.
        **kwargs: Arguments supplementaires pour DataFrame.to_csv.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, **kwargs)


def load_parquet(path: Path | str, **kwargs: Any) -> pd.DataFrame:
    """Charge un fichier Parquet en DataFrame.

    Args:
        path: Chemin du fichier Parquet.
        **kwargs: Arguments supplementaires pour pandas.read_parquet.

    Returns:
        DataFrame contenant les donnees.
    """
    return pd.read_parquet(path, **kwargs)


def save_parquet(df: pd.DataFrame, path: Path | str, **kwargs: Any) -> None:
    """Sauvegarde un DataFrame en Parquet.

    Args:
        df: DataFrame a sauvegarder.
        path: Chemin de destination.
        **kwargs: Arguments supplementaires pour DataFrame.to_parquet.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, **kwargs)


def load_model(path: Path | str) -> Any:
    """Charge un modele serialise avec joblib.

    Args:
        path: Chemin du fichier modele.

    Returns:
        Modele deserialise.
    """
    return joblib.load(path)


def save_model(model: Any, path: Path | str) -> None:
    """Sauvegarde un modele avec joblib.

    Args:
        model: Modele a sauvegarder.
        path: Chemin de destination.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
