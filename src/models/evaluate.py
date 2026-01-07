"""Evaluation des modeles et calcul de l'elasticite prix."""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.features.build_features import build_feature_matrix
from src.utils.io import load_model, load_parquet
from src.utils.logging import get_logger, setup_logging
from src.utils.metrics import (
    compute_all_metrics,
    compute_elasticity_by_segment,
    price_elasticity,
)
from src.utils.mlflow_utils import log_artifact_file, log_metrics_dict, setup_mlflow, start_run
from src.utils.paths import FIGURES_DIR, MODELS_DIR, ensure_dir

logger = get_logger(__name__)


def evaluate_model(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    dataset_name: str,
) -> dict[str, float]:
    """Evalue un modele sur un ensemble de donnees.

    Args:
        model: Modele entraine.
        X: Features.
        y: Cible reelle.
        dataset_name: Nom du dataset pour le logging.

    Returns:
        Dictionnaire des metriques.
    """
    y_pred = model.predict(X)
    metrics = compute_all_metrics(y, y_pred)

    logger.info(f"evaluation_{dataset_name}", **metrics)

    return metrics


def compute_global_elasticity(df: pd.DataFrame) -> float:
    """Calcule l'elasticite prix globale.

    Args:
        df: DataFrame avec colonnes 'price' et 'sales'.

    Returns:
        Elasticite prix globale.
    """
    elasticity = price_elasticity(
        df["price"].values,
        df["sales"].values,
    )

    logger.info("elasticite_globale", elasticity=elasticity)

    return elasticity


def compute_segmented_elasticity(
    df: pd.DataFrame,
    segment_columns: list[str],
) -> dict[str, dict[str, float]]:
    """Calcule l'elasticite par segment.

    Args:
        df: DataFrame source.
        segment_columns: Colonnes de segmentation.

    Returns:
        Dictionnaire segment_col -> {segment_value: elasticite}.
    """
    results: dict[str, dict[str, float]] = {}

    for col in segment_columns:
        if col in df.columns:
            elasticities = compute_elasticity_by_segment(
                df,
                price_col="price",
                quantity_col="sales",
                segment_col=col,
            )
            results[col] = elasticities
            logger.info(f"elasticite_par_{col}", n_segments=len(elasticities))

    return results


def plot_demand_curve(
    df: pd.DataFrame,
    family: str,
    output_path: Path,
) -> None:
    """Trace la courbe de demande pour une famille de produits.

    Args:
        df: DataFrame source.
        family: Famille de produits.
        output_path: Chemin de sauvegarde.
    """
    family_df = df[df["family"] == family].copy()

    if len(family_df) == 0:
        logger.warning("famille_non_trouvee", family=family)
        return

    # Agreger par prix
    agg_df = family_df.groupby("price").agg({"sales": "mean"}).reset_index()
    agg_df = agg_df.sort_values("price")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(agg_df["price"], agg_df["sales"], alpha=0.6)
    ax.plot(agg_df["price"], agg_df["sales"], "r-", alpha=0.3)

    ax.set_xlabel("Prix")
    ax.set_ylabel("Ventes moyennes")
    ax.set_title(f"Courbe de demande - {family}")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info("courbe_demande_sauvegardee", family=family, path=str(output_path))


def plot_feature_importance(
    model: Any,
    feature_names: list[str],
    output_path: Path,
    top_n: int = 20,
) -> None:
    """Trace l'importance des features.

    Args:
        model: Modele LightGBM entraine.
        feature_names: Noms des features.
        output_path: Chemin de sauvegarde.
        top_n: Nombre de features a afficher.
    """
    importance = model.feature_importances_

    # Trier par importance
    sorted_idx = np.argsort(importance)[-top_n:]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(
        [feature_names[i] for i in sorted_idx],
        importance[sorted_idx],
    )

    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Features par Importance")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info("feature_importance_sauvegardee", path=str(output_path))


def plot_predictions_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
) -> None:
    """Trace predictions vs valeurs reelles.

    Args:
        y_true: Valeurs reelles.
        y_pred: Predictions.
        output_path: Chemin de sauvegarde.
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Echantillonner si trop de points
    n_samples = min(10000, len(y_true))
    idx = np.random.choice(len(y_true), n_samples, replace=False)

    ax.scatter(y_true[idx], y_pred[idx], alpha=0.3, s=5)

    # Ligne diagonale parfaite
    max_val = max(y_true[idx].max(), y_pred[idx].max())
    ax.plot([0, max_val], [0, max_val], "r--", lw=2, label="Prediction parfaite")

    ax.set_xlabel("Valeurs reelles")
    ax.set_ylabel("Predictions")
    ax.set_title("Predictions vs Valeurs Reelles")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info("predictions_vs_actual_sauvegarde", path=str(output_path))


def run_evaluation(cfg: DictConfig) -> None:
    """Execute le pipeline d'evaluation.

    Args:
        cfg: Configuration Hydra.
    """
    setup_logging(level="INFO", json_format=False)
    setup_mlflow(
        tracking_uri=cfg.train.mlflow.tracking_uri,
        experiment_name=cfg.train.mlflow.experiment_name,
    )

    processed_dir = Path(cfg.data.processed_dir)

    # Charger le modele et le scaler
    model = load_model(MODELS_DIR / "lightgbm_model.joblib")
    scaler = load_model(MODELS_DIR / "scaler.joblib")
    feature_names = load_model(MODELS_DIR / "feature_names.joblib")

    # Charger les donnees de test
    test_df = load_parquet(processed_dir / "test_features.parquet")
    logger.info("donnees_test_chargees", rows=len(test_df))

    # Construire la matrice de features
    X_test, y_test = build_feature_matrix(
        test_df,
        target_column=cfg.train.training.target_column,
        exclude_columns=list(cfg.train.training.exclude_columns),
    )
    X_test_scaled = scaler.transform(X_test)

    with start_run(run_name="evaluation"):
        # Evaluation du modele
        y_pred = model.predict(X_test_scaled)
        test_metrics = compute_all_metrics(y_test.values, y_pred)

        # Prefixer les metriques pour le test
        test_metrics_prefixed = {f"test_{k}": v for k, v in test_metrics.items()}
        log_metrics_dict(test_metrics_prefixed)

        # Elasticite globale
        global_elasticity = compute_global_elasticity(test_df)
        mlflow.log_metric("global_elasticity", global_elasticity)

        # Elasticite par segment
        segment_elasticities = compute_segmented_elasticity(
            test_df,
            segment_columns=["family", "city", "type"],
        )

        # Sauvegarder les elasticites
        mlflow.log_dict(
            {"global": global_elasticity, **segment_elasticities},
            "elasticities.json",
        )

        # Visualisations
        ensure_dir(FIGURES_DIR)

        # Feature importance
        fi_path = FIGURES_DIR / "feature_importance.png"
        plot_feature_importance(model, feature_names, fi_path)
        log_artifact_file(fi_path, "figures")

        # Predictions vs actual
        pva_path = FIGURES_DIR / "predictions_vs_actual.png"
        plot_predictions_vs_actual(y_test.values, y_pred, pva_path)
        log_artifact_file(pva_path, "figures")

        # Courbes de demande pour quelques familles
        sample_families = ["GROCERY I", "BEVERAGES", "PRODUCE"]
        for family in sample_families:
            if family in test_df["family"].values:
                curve_path = FIGURES_DIR / f"demand_curve_{family.replace(' ', '_')}.png"
                plot_demand_curve(test_df, family, curve_path)
                log_artifact_file(curve_path, "figures")

        logger.info("evaluation_terminee", **test_metrics)


if __name__ == "__main__":
    import hydra

    @hydra.main(version_base=None, config_path="../../configs/hydra", config_name="config")
    def main(cfg: DictConfig) -> None:
        run_evaluation(cfg)

    main()
