"""Metriques d'evaluation et metriques metier."""

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def rmse(y_true: NDArray[np.floating], y_pred: NDArray[np.floating]) -> float:
    """Calcule le Root Mean Squared Error.

    Args:
        y_true: Valeurs reelles.
        y_pred: Valeurs predites.

    Returns:
        RMSE.
    """
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: NDArray[np.floating], y_pred: NDArray[np.floating]) -> float:
    """Calcule le Mean Absolute Error.

    Args:
        y_true: Valeurs reelles.
        y_pred: Valeurs predites.

    Returns:
        MAE.
    """
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: NDArray[np.floating], y_pred: NDArray[np.floating]) -> float:
    """Calcule le Mean Absolute Percentage Error.

    Args:
        y_true: Valeurs reelles.
        y_pred: Valeurs predites.

    Returns:
        MAPE en pourcentage.
    """
    # Evite la division par zero
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def r2_score(y_true: NDArray[np.floating], y_pred: NDArray[np.floating]) -> float:
    """Calcule le coefficient de determination R2.

    Args:
        y_true: Valeurs reelles.
        y_pred: Valeurs predites.

    Returns:
        R2 score.
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - (ss_res / ss_tot)) if ss_tot != 0 else 0.0


def compute_all_metrics(
    y_true: NDArray[np.floating],
    y_pred: NDArray[np.floating],
) -> dict[str, float]:
    """Calcule toutes les metriques de regression.

    Args:
        y_true: Valeurs reelles.
        y_pred: Valeurs predites.

    Returns:
        Dictionnaire des metriques.
    """
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def price_elasticity(
    prices: NDArray[np.floating],
    quantities: NDArray[np.floating],
) -> float:
    """Calcule l'elasticite prix de la demande.

    Elasticite = (dQ/Q) / (dP/P) = (dQ/dP) * (P/Q)

    Utilise une regression log-log pour estimer l'elasticite.

    Args:
        prices: Prix observes.
        quantities: Quantites observees.

    Returns:
        Elasticite prix (valeur negative = demande elastique).
    """
    # Evite les valeurs nulles ou negatives pour le log
    valid_mask = (prices > 0) & (quantities > 0)
    log_p = np.log(prices[valid_mask])
    log_q = np.log(quantities[valid_mask])

    # Regression lineaire simple: log(Q) = a + b*log(P)
    # b est l'elasticite
    if len(log_p) < 2:
        return 0.0

    cov_matrix = np.cov(log_p, log_q)
    if cov_matrix[0, 0] == 0:
        return 0.0

    elasticity = cov_matrix[0, 1] / cov_matrix[0, 0]
    return float(elasticity)


def compute_elasticity_by_segment(
    df: pd.DataFrame,
    price_col: str,
    quantity_col: str,
    segment_col: str,
) -> dict[str, float]:
    """Calcule l'elasticite prix par segment.

    Args:
        df: DataFrame avec les donnees.
        price_col: Nom de la colonne prix.
        quantity_col: Nom de la colonne quantite.
        segment_col: Nom de la colonne segment.

    Returns:
        Dictionnaire segment -> elasticite.
    """
    elasticities = {}
    for segment in df[segment_col].unique():
        segment_data = df[df[segment_col] == segment]
        elasticities[str(segment)] = price_elasticity(
            segment_data[price_col].values,
            segment_data[quantity_col].values,
        )
    return elasticities


def expected_revenue(price: float, quantity: float) -> float:
    """Calcule le revenu attendu.

    Args:
        price: Prix unitaire.
        quantity: Quantite vendue.

    Returns:
        Revenu = prix * quantite.
    """
    return price * quantity


def expected_margin(price: float, quantity: float, cost_ratio: float = 0.6) -> float:
    """Calcule la marge attendue.

    Args:
        price: Prix unitaire.
        quantity: Quantite vendue.
        cost_ratio: Ratio cout/prix (par defaut 60%).

    Returns:
        Marge = (prix - cout) * quantite.
    """
    cost = price * cost_ratio
    return (price - cost) * quantity
