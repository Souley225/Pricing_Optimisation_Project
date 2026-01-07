"""Tests unitaires pour les modules models."""

import numpy as np

from src.utils.metrics import (
    compute_all_metrics,
    expected_margin,
    expected_revenue,
    mae,
    mape,
    price_elasticity,
    r2_score,
    rmse,
)


class TestRegressionMetrics:
    """Tests pour les metriques de regression."""

    def test_rmse(self) -> None:
        """Verifie le calcul du RMSE."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 4.1, 4.9])

        result = rmse(y_true, y_pred)

        # RMSE = sqrt(mean((0.1^2 + 0.1^2 + 0.1^2 + 0.1^2 + 0.1^2))) = 0.1
        np.testing.assert_almost_equal(result, 0.1, decimal=5)

    def test_mae(self) -> None:
        """Verifie le calcul du MAE."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.0, 2.5])

        result = mae(y_true, y_pred)

        # MAE = mean(abs([0.5, 0, 0.5])) = 1/3
        np.testing.assert_almost_equal(result, 1 / 3, decimal=5)

    def test_mape(self) -> None:
        """Verifie le calcul du MAPE."""
        y_true = np.array([100.0, 200.0, 150.0])
        y_pred = np.array([110.0, 190.0, 150.0])

        result = mape(y_true, y_pred)

        # MAPE = mean([10%, 5%, 0%]) * 100 = 5%
        np.testing.assert_almost_equal(result, 5.0, decimal=5)

    def test_r2_score(self) -> None:
        """Verifie le calcul du R2."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = y_true  # Prediction parfaite

        result = r2_score(y_true, y_pred)

        assert result == 1.0

    def test_compute_all_metrics(self) -> None:
        """Verifie compute_all_metrics."""
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([110.0, 190.0, 310.0])

        result = compute_all_metrics(y_true, y_pred)

        assert "rmse" in result
        assert "mae" in result
        assert "mape" in result
        assert "r2" in result


class TestBusinessMetrics:
    """Tests pour les metriques metier."""

    def test_expected_revenue(self) -> None:
        """Verifie le calcul du revenu."""
        result = expected_revenue(price=10.0, quantity=100.0)
        assert result == 1000.0

    def test_expected_margin(self) -> None:
        """Verifie le calcul de la marge."""
        result = expected_margin(price=10.0, quantity=100.0, cost_ratio=0.6)

        # Marge = (10 - 6) * 100 = 400
        assert result == 400.0

    def test_price_elasticity(self) -> None:
        """Verifie le calcul de l'elasticite."""
        # Demande elastique: quand le prix augmente, la quantite diminue
        prices = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        quantities = np.array([100.0, 90.0, 81.0, 73.0, 66.0])

        result = price_elasticity(prices, quantities)

        # L'elasticite devrait etre negative
        assert result < 0

    def test_price_elasticity_inelastic(self) -> None:
        """Verifie l'elasticite pour demande inelastique."""
        prices = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        quantities = np.array([100.0, 99.0, 98.0, 97.0, 96.0])

        result = price_elasticity(prices, quantities)

        # L'elasticite devrait etre negative mais proche de 0
        assert -1 < result < 0
