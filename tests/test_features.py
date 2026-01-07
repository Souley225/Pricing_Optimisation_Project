"""Tests unitaires pour les modules features."""

import numpy as np
import pandas as pd

from src.features.build_features import (
    add_price_features,
    add_promotion_features,
    add_temporal_features,
    build_feature_matrix,
    encode_categorical_features,
)


class TestAddTemporalFeatures:
    """Tests pour add_temporal_features."""

    def test_basic_features(self) -> None:
        """Verifie les features temporelles de base."""
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2023-01-15", "2023-06-20", "2023-12-25"]),
                "value": [1, 2, 3],
            }
        )

        result = add_temporal_features(df, "date")

        # Verifier les colonnes ajoutees
        expected_cols = [
            "day_of_week",
            "day_of_month",
            "week_of_year",
            "month",
            "quarter",
            "year",
            "is_weekend",
            "is_month_start",
            "is_month_end",
        ]
        for col in expected_cols:
            assert col in result.columns

        # Verifier quelques valeurs
        assert result["month"].iloc[0] == 1
        assert result["month"].iloc[1] == 6
        assert result["month"].iloc[2] == 12

    def test_weekend_detection(self) -> None:
        """Verifie la detection des weekends."""
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    [
                        "2023-01-02",  # Lundi
                        "2023-01-07",  # Samedi
                        "2023-01-08",  # Dimanche
                    ]
                ),
            }
        )

        result = add_temporal_features(df, "date")

        assert result["is_weekend"].iloc[0] == 0
        assert result["is_weekend"].iloc[1] == 1
        assert result["is_weekend"].iloc[2] == 1


class TestAddPriceFeatures:
    """Tests pour add_price_features."""

    def test_log_price(self) -> None:
        """Verifie le calcul du log du prix."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=5),
                "store_nbr": [1] * 5,
                "family": ["A"] * 5,
                "price": [10.0, 20.0, 15.0, 25.0, 30.0],
            }
        )

        result = add_price_features(
            df, "price", lag_periods=[1], rolling_windows=[3], log_transform=True
        )

        assert "log_price" in result.columns
        np.testing.assert_almost_equal(
            result["log_price"].iloc[0],
            np.log1p(10.0),
            decimal=5,
        )

    def test_lag_features(self) -> None:
        """Verifie les features de lag."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=5),
                "store_nbr": [1] * 5,
                "family": ["A"] * 5,
                "price": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )

        result = add_price_features(
            df, "price", lag_periods=[1, 2], rolling_windows=[], log_transform=False
        )

        assert "price_lag_1" in result.columns
        assert "price_lag_2" in result.columns

        # Premier lag doit etre 0 (NA rempli)
        assert result["price_lag_1"].iloc[0] == 0
        assert result["price_lag_1"].iloc[1] == 1.0


class TestAddPromotionFeatures:
    """Tests pour add_promotion_features."""

    def test_is_promo_flag(self) -> None:
        """Verifie le flag de promotion."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=3),
                "store_nbr": [1] * 3,
                "family": ["A"] * 3,
                "onpromotion": [0, 5, 0],
            }
        )

        result = add_promotion_features(df)

        assert "is_promo" in result.columns
        assert result["is_promo"].iloc[0] == 0
        assert result["is_promo"].iloc[1] == 1
        assert result["is_promo"].iloc[2] == 0


class TestEncodeCategoricalFeatures:
    """Tests pour encode_categorical_features."""

    def test_label_encoding(self) -> None:
        """Verifie l'encodage label."""
        df = pd.DataFrame(
            {
                "family": ["A", "B", "A", "C"],
                "city": ["Paris", "Lyon", "Paris", "Lyon"],
            }
        )

        result, encoders = encode_categorical_features(df, ["family", "city"], method="label")

        assert "family_encoded" in result.columns
        assert "city_encoded" in result.columns
        assert len(encoders) == 2

        # Verifier que les memes valeurs ont le meme encodage
        assert result["city_encoded"].iloc[0] == result["city_encoded"].iloc[2]
        assert result["city_encoded"].iloc[1] == result["city_encoded"].iloc[3]


class TestBuildFeatureMatrix:
    """Tests pour build_feature_matrix."""

    def test_exclude_columns(self) -> None:
        """Verifie l'exclusion des colonnes."""
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "date": pd.date_range("2023-01-01", periods=3),
                "sales": [100.0, 200.0, 150.0],
                "price": [10.0, 12.0, 11.0],
                "quantity": [10, 20, 15],
            }
        )

        X, y = build_feature_matrix(
            df,
            target_column="sales",
            exclude_columns=["id", "date"],
        )

        assert "id" not in X.columns
        assert "date" not in X.columns
        assert "sales" not in X.columns
        assert "price" in X.columns
        assert "quantity" in X.columns

        assert len(y) == 3
        assert y.iloc[0] == 100.0
