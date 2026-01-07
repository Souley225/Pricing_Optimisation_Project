"""Telechargement et preparation du dataset Kaggle Store Sales."""

import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.utils.io import load_csv, save_parquet
from src.utils.logging import get_logger, setup_logging
from src.utils.paths import ensure_dir

logger = get_logger(__name__)


def download_kaggle_dataset(competition: str, output_dir: Path) -> None:
    """Telecharge un dataset depuis une competition Kaggle.

    Args:
        competition: Slug de la competition Kaggle.
        output_dir: Repertoire de destination.
    """
    ensure_dir(output_dir)
    logger.info("telechargement_dataset", competition=competition, output_dir=str(output_dir))

    try:
        subprocess.run(
            ["kaggle", "competitions", "download", "-c", competition, "-p", str(output_dir)],
            check=True,
            capture_output=True,
            text=True,
        )

        # Extraire le zip
        zip_path = output_dir / f"{competition}.zip"
        if zip_path.exists():
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(output_dir)
            zip_path.unlink()
            logger.info("extraction_terminee", path=str(output_dir))

    except subprocess.CalledProcessError as e:
        logger.error("erreur_telechargement", error=e.stderr)
        raise


def load_raw_data(raw_dir: Path) -> dict[str, pd.DataFrame]:
    """Charge tous les fichiers CSV bruts.

    Args:
        raw_dir: Repertoire contenant les fichiers CSV.

    Returns:
        Dictionnaire nom -> DataFrame.
    """
    datasets = {}

    # Fichiers principaux
    files = {
        "train": "train.csv",
        "stores": "stores.csv",
        "oil": "oil.csv",
        "holidays": "holidays_events.csv",
        "transactions": "transactions.csv",
    }

    for name, filename in files.items():
        path = raw_dir / filename
        if path.exists():
            datasets[name] = load_csv(path)
            logger.info("fichier_charge", name=name, rows=len(datasets[name]))
        else:
            logger.warning("fichier_manquant", name=name, path=str(path))

    return datasets


def normalize_dtypes(datasets: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Normalise les types de donnees.

    Args:
        datasets: Dictionnaire des DataFrames bruts.

    Returns:
        Dictionnaire des DataFrames avec types normalises.
    """
    result = {}

    # Train
    if "train" in datasets:
        df = datasets["train"].copy()
        df["date"] = pd.to_datetime(df["date"])
        df["store_nbr"] = df["store_nbr"].astype("int32")
        df["family"] = df["family"].astype("category")
        df["sales"] = df["sales"].astype("float32")
        df["onpromotion"] = df["onpromotion"].astype("int32")
        result["train"] = df

    # Stores
    if "stores" in datasets:
        df = datasets["stores"].copy()
        df["store_nbr"] = df["store_nbr"].astype("int32")
        df["city"] = df["city"].astype("category")
        df["state"] = df["state"].astype("category")
        df["type"] = df["type"].astype("category")
        df["cluster"] = df["cluster"].astype("int32")
        result["stores"] = df

    # Oil
    if "oil" in datasets:
        df = datasets["oil"].copy()
        df["date"] = pd.to_datetime(df["date"])
        df["dcoilwtico"] = pd.to_numeric(df["dcoilwtico"], errors="coerce")
        result["oil"] = df

    # Holidays
    if "holidays" in datasets:
        df = datasets["holidays"].copy()
        df["date"] = pd.to_datetime(df["date"])
        df["type"] = df["type"].astype("category")
        df["locale"] = df["locale"].astype("category")
        df["transferred"] = df["transferred"].astype("bool")
        result["holidays"] = df

    # Transactions
    if "transactions" in datasets:
        df = datasets["transactions"].copy()
        df["date"] = pd.to_datetime(df["date"])
        df["store_nbr"] = df["store_nbr"].astype("int32")
        df["transactions"] = df["transactions"].astype("int32")
        result["transactions"] = df

    return result


def clean_missing_values(datasets: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Nettoie les valeurs manquantes.

    Args:
        datasets: Dictionnaire des DataFrames.

    Returns:
        Dictionnaire des DataFrames nettoyes.
    """
    result = {}

    for name, df in datasets.items():
        df_clean = df.copy()
        missing_before = df_clean.isnull().sum().sum()

        if name == "oil":
            # Interpolation lineaire pour le prix du petrole
            df_clean["dcoilwtico"] = df_clean["dcoilwtico"].interpolate(method="linear")
            df_clean["dcoilwtico"] = df_clean["dcoilwtico"].ffill().bfill()

        elif name == "train":
            # Les ventes nulles restent nulles (pas de valeur manquante)
            pass

        missing_after = df_clean.isnull().sum().sum()
        if missing_before > 0:
            logger.info(
                "nettoyage_valeurs_manquantes",
                dataset=name,
                before=int(missing_before),
                after=int(missing_after),
            )

        result[name] = df_clean

    return result


def generate_synthetic_price(
    df: pd.DataFrame,
    price_base_by_family: dict[str, float],
    promotion_discount: float,
    store_coef_range: tuple[float, float],
    seed: int = 42,
) -> pd.DataFrame:
    """Genere un prix synthetique pour chaque observation.

    Le prix est construit a partir de:
    - Un prix de base par famille de produit
    - Un coefficient par magasin (variation geographique)
    - Une reduction si le produit est en promotion

    Args:
        df: DataFrame avec les ventes.
        price_base_by_family: Prix de base par famille.
        promotion_discount: Taux de reduction en promotion.
        store_coef_range: Intervalle du coefficient magasin (min, max).
        seed: Graine aleatoire pour reproductibilite.

    Returns:
        DataFrame avec la colonne 'price' ajoutee.
    """
    rng = np.random.default_rng(seed)

    df = df.copy()

    # Prix de base par famille
    df["base_price"] = df["family"].map(price_base_by_family)

    # Coefficient par magasin (deterministe base sur store_nbr et seed)
    unique_stores = df["store_nbr"].unique()
    store_coefficients = {
        store: rng.uniform(store_coef_range[0], store_coef_range[1]) for store in unique_stores
    }
    df["store_coef"] = df["store_nbr"].map(store_coefficients)

    # Prix avant promotion
    df["price_before_promo"] = df["base_price"] * df["store_coef"]

    # Application de la reduction si en promotion
    df["price"] = np.where(
        df["onpromotion"] > 0,
        df["price_before_promo"] * (1 - promotion_discount),
        df["price_before_promo"],
    )

    # Arrondi a 2 decimales
    df["price"] = df["price"].round(2)

    # Nettoyage des colonnes intermediaires
    df = df.drop(columns=["base_price", "store_coef", "price_before_promo"])

    logger.info(
        "prix_synthetique_genere",
        mean_price=float(df["price"].mean()),
        min_price=float(df["price"].min()),
        max_price=float(df["price"].max()),
    )

    return df


def merge_datasets(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Fusionne tous les datasets en un seul.

    Args:
        datasets: Dictionnaire des DataFrames.

    Returns:
        DataFrame fusionne.
    """
    df = datasets["train"].copy()

    # Merge avec stores
    if "stores" in datasets:
        df = df.merge(datasets["stores"], on="store_nbr", how="left")

    # Merge avec oil
    if "oil" in datasets:
        df = df.merge(datasets["oil"], on="date", how="left")

    # Merge avec holidays (type National uniquement)
    if "holidays" in datasets:
        holidays = datasets["holidays"]
        national_holidays = holidays[holidays["locale"] == "National"][["date"]].copy()
        national_holidays["is_holiday"] = True
        df = df.merge(national_holidays, on="date", how="left")
        df["is_holiday"] = df["is_holiday"].fillna(False)

    # Merge avec transactions
    if "transactions" in datasets:
        df = df.merge(
            datasets["transactions"],
            on=["date", "store_nbr"],
            how="left",
        )

    logger.info("datasets_fusionnes", rows=len(df), columns=len(df.columns))

    return df


def run_make_dataset(cfg: DictConfig) -> None:
    """Execute le pipeline de preparation des donnees.

    Args:
        cfg: Configuration Hydra.
    """
    setup_logging(level="INFO", json_format=False)

    raw_dir = Path(cfg.data.raw_dir)
    interim_dir = Path(cfg.data.interim_dir)

    # Telecharger si necessaire
    if not (raw_dir / "train.csv").exists():
        download_kaggle_dataset(cfg.data.kaggle.competition, raw_dir)

    # Charger les donnees brutes
    datasets = load_raw_data(raw_dir)

    # Normaliser les types
    datasets = normalize_dtypes(datasets)

    # Nettoyer les valeurs manquantes
    datasets = clean_missing_values(datasets)

    # Fusionner les datasets
    df = merge_datasets(datasets)

    # Generer le prix synthetique
    df = generate_synthetic_price(
        df,
        price_base_by_family=dict(cfg.features.price_base_by_family),
        promotion_discount=cfg.features.promotion.discount_rate,
        store_coef_range=(
            cfg.features.store.coefficient_min,
            cfg.features.store.coefficient_max,
        ),
        seed=cfg.features.store.seed,
    )

    # Sauvegarder
    ensure_dir(interim_dir)
    output_path = interim_dir / "sales_with_price.parquet"
    save_parquet(df, output_path)

    logger.info("dataset_sauvegarde", path=str(output_path), rows=len(df))


if __name__ == "__main__":
    import hydra

    @hydra.main(version_base=None, config_path="../../configs/hydra", config_name="config")
    def main(cfg: DictConfig) -> None:
        run_make_dataset(cfg)

    main()
