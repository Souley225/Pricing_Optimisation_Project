"""Tests unitaires pour les modules data."""

import numpy as np
import pandas as pd
import pytest

from src.data.make_dataset import (
    clean_missing_values,
    generate_synthetic_price,
    normalize_dtypes,
)
from src.data.split_dataset import temporal_split


class TestNormalizeDtypes:
    """Tests pour normalize_dtypes."""

    def test_train_dtypes(self) -> None:
        """Verifie les types du dataset train."""
        datasets = {
            "train": pd.DataFrame(
                {
                    "date": ["2023-01-01", "2023-01-02"],
                    "store_nbr": [1, 2],
                    "family": ["GROCERY I", "BEVERAGES"],
                    "sales": [100.0, 200.0],
                    "onpromotion": [0, 1],
                }
            )
        }

        result = normalize_dtypes(datasets)

        assert result["train"]["date"].dtype == "datetime64[ns]"
        assert result["train"]["store_nbr"].dtype == "int32"
        assert result["train"]["family"].dtype.name == "category"
        assert result["train"]["sales"].dtype == "float32"
        assert result["train"]["onpromotion"].dtype == "int32"

    def test_stores_dtypes(self) -> None:
        """Verifie les types du dataset stores."""
        datasets = {
            "stores": pd.DataFrame(
                {
                    "store_nbr": [1, 2],
                    "city": ["Quito", "Guayaquil"],
                    "state": ["Pichincha", "Guayas"],
                    "type": ["A", "B"],
                    "cluster": [1, 2],
                }
            )
        }

        result = normalize_dtypes(datasets)

        assert result["stores"]["store_nbr"].dtype == "int32"
        assert result["stores"]["city"].dtype.name == "category"
        assert result["stores"]["cluster"].dtype == "int32"


class TestCleanMissingValues:
    """Tests pour clean_missing_values."""

    def test_oil_interpolation(self) -> None:
        """Verifie l'interpolation du prix du petrole."""
        datasets = {
            "oil": pd.DataFrame(
                {
                    "date": pd.date_range("2023-01-01", periods=5),
                    "dcoilwtico": [50.0, np.nan, np.nan, 56.0, 58.0],
                }
            )
        }

        result = clean_missing_values(datasets)

        assert result["oil"]["dcoilwtico"].isna().sum() == 0
        # Verifier l'interpolation
        assert 50.0 < result["oil"]["dcoilwtico"].iloc[1] < 56.0


class TestGenerateSyntheticPrice:
    """Tests pour generate_synthetic_price."""

    def test_price_generation(self) -> None:
        """Verifie la generation du prix synthetique."""
        df = pd.DataFrame(
            {
                "family": ["GROCERY I", "BEVERAGES", "GROCERY I"],
                "store_nbr": [1, 1, 2],
                "onpromotion": [0, 1, 0],
            }
        )

        price_base = {"GROCERY I": 5.0, "BEVERAGES": 3.0}

        result = generate_synthetic_price(
            df,
            price_base_by_family=price_base,
            promotion_discount=0.15,
            store_coef_range=(0.9, 1.1),
            seed=42,
        )

        assert "price" in result.columns
        assert result["price"].min() > 0

        # Le prix en promotion devrait etre plus bas
        promo_prices = result[result["onpromotion"] > 0]["price"]
        non_promo_prices = result[(result["onpromotion"] == 0) & (result["family"] == "BEVERAGES")][
            "price"
        ]

        if len(promo_prices) > 0 and len(non_promo_prices) > 0:
            # Le prix en promo BEVERAGES devrait etre plus bas que le prix de base
            assert promo_prices.iloc[0] < 3.0

    def test_reproducibility(self) -> None:
        """Verifie la reproductibilite avec seed fixe."""
        df = pd.DataFrame(
            {
                "family": ["GROCERY I"] * 10,
                "store_nbr": list(range(10)),
                "onpromotion": [0] * 10,
            }
        )

        price_base = {"GROCERY I": 5.0}

        result1 = generate_synthetic_price(df, price_base, 0.15, (0.9, 1.1), seed=42)
        result2 = generate_synthetic_price(df, price_base, 0.15, (0.9, 1.1), seed=42)

        pd.testing.assert_frame_equal(result1, result2)


class TestTemporalSplit:
    """Tests pour temporal_split."""

    def test_split_ratios(self) -> None:
        """Verifie les ratios de split."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=100),
                "value": range(100),
            }
        )

        train, val, test = temporal_split(
            df, "date", train_ratio=0.8, val_ratio=0.1, test_ratio=0.1
        )

        total = len(train) + len(val) + len(test)
        assert total == 100

        # Verifier les proportions approximatives
        assert 0.75 <= len(train) / total <= 0.85
        assert 0.05 <= len(val) / total <= 0.15
        assert 0.05 <= len(test) / total <= 0.15

    def test_no_data_leakage(self) -> None:
        """Verifie l'absence de data leakage."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=100),
                "value": range(100),
            }
        )

        train, val, test = temporal_split(
            df, "date", train_ratio=0.8, val_ratio=0.1, test_ratio=0.1
        )

        # Toutes les dates de train doivent etre avant celles de val
        assert train["date"].max() < val["date"].min()

        # Toutes les dates de val doivent etre avant celles de test
        assert val["date"].max() < test["date"].min()

    def test_invalid_ratios(self) -> None:
        """Verifie la validation des ratios."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=10),
                "value": range(10),
            }
        )

        with pytest.raises(ValueError):
            temporal_split(df, "date", 0.5, 0.3, 0.3)  # Somme > 1
