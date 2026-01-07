"""Split temporel du dataset."""

from pathlib import Path

import pandas as pd
from omegaconf import DictConfig

from src.utils.io import load_parquet, save_parquet
from src.utils.logging import get_logger, setup_logging
from src.utils.paths import ensure_dir

logger = get_logger(__name__)


def temporal_split(
    df: pd.DataFrame,
    date_column: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Realise un split temporel strict.

    Les donnees sont divisees chronologiquement pour eviter tout data leakage.

    Args:
        df: DataFrame a splitter.
        date_column: Nom de la colonne date.
        train_ratio: Proportion pour l'entrainement.
        val_ratio: Proportion pour la validation.
        test_ratio: Proportion pour le test.

    Returns:
        Tuple (train, val, test) DataFrames.

    Raises:
        ValueError: Si les ratios ne somment pas a 1.
    """
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        raise ValueError(f"Les ratios doivent sommer a 1.0, recu: {total_ratio}")

    # Trier par date
    df_sorted = df.sort_values(date_column).reset_index(drop=True)

    # Obtenir les dates uniques triees
    unique_dates = df_sorted[date_column].unique()
    n_dates = len(unique_dates)

    # Calculer les indices de coupure
    train_end_idx = int(n_dates * train_ratio)
    val_end_idx = int(n_dates * (train_ratio + val_ratio))

    train_end_date = unique_dates[train_end_idx - 1]
    val_end_date = unique_dates[val_end_idx - 1]

    # Splitter
    train_mask = df_sorted[date_column] <= train_end_date
    val_mask = (df_sorted[date_column] > train_end_date) & (df_sorted[date_column] <= val_end_date)
    test_mask = df_sorted[date_column] > val_end_date

    train_df = df_sorted[train_mask].copy()
    val_df = df_sorted[val_mask].copy()
    test_df = df_sorted[test_mask].copy()

    logger.info(
        "split_temporel",
        train_size=len(train_df),
        val_size=len(val_df),
        test_size=len(test_df),
        train_end_date=str(train_end_date),
        val_end_date=str(val_end_date),
    )

    return train_df, val_df, test_df


def run_split_dataset(cfg: DictConfig) -> None:
    """Execute le split temporel du dataset.

    Args:
        cfg: Configuration Hydra.
    """
    setup_logging(level="INFO", json_format=False)

    interim_dir = Path(cfg.data.interim_dir)
    processed_dir = Path(cfg.data.processed_dir)

    # Charger le dataset
    input_path = interim_dir / "sales_with_price.parquet"
    df = load_parquet(input_path)
    logger.info("dataset_charge", path=str(input_path), rows=len(df))

    # Split temporel
    train_df, val_df, test_df = temporal_split(
        df,
        date_column="date",
        train_ratio=cfg.data.split.train_ratio,
        val_ratio=cfg.data.split.val_ratio,
        test_ratio=cfg.data.split.test_ratio,
    )

    # Sauvegarder
    ensure_dir(processed_dir)

    save_parquet(train_df, processed_dir / "train.parquet")
    save_parquet(val_df, processed_dir / "val.parquet")
    save_parquet(test_df, processed_dir / "test.parquet")

    logger.info(
        "splits_sauvegardes",
        train_path=str(processed_dir / "train.parquet"),
        val_path=str(processed_dir / "val.parquet"),
        test_path=str(processed_dir / "test.parquet"),
    )


if __name__ == "__main__":
    import hydra

    @hydra.main(version_base=None, config_path="../../configs/hydra", config_name="config")
    def main(cfg: DictConfig) -> None:
        run_split_dataset(cfg)

    main()
