"""Optimisation des prix pour maximiser revenu ou marge."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from scipy.optimize import minimize

from src.utils.io import load_model, load_parquet, save_parquet
from src.utils.logging import get_logger, setup_logging
from src.utils.metrics import expected_margin, expected_revenue
from src.utils.paths import MODELS_DIR, ensure_dir

logger = get_logger(__name__)


@dataclass
class OptimizationResult:
    """Resultat d'une optimisation de prix."""

    product_id: str
    current_price: float
    optimized_price: float
    current_volume: float
    expected_volume: float
    current_revenue: float
    expected_revenue: float
    current_margin: float
    expected_margin: float
    elasticity: float
    success: bool


class PriceOptimizer:
    """Optimiseur de prix base sur un modele de demande."""

    def __init__(
        self,
        model: Any,
        scaler: Any,
        feature_names: list[str],
        cost_ratio: float = 0.6,
    ) -> None:
        """Initialise l'optimiseur.

        Args:
            model: Modele de demande entraine.
            scaler: Scaler pour normaliser les features.
            feature_names: Noms des features.
            cost_ratio: Ratio cout/prix pour le calcul de marge.
        """
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names
        self.cost_ratio = cost_ratio

    def predict_demand(
        self,
        features: np.ndarray,
    ) -> float:
        """Predit la demande pour un ensemble de features.

        Args:
            features: Vecteur de features.

        Returns:
            Demande predite.
        """
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        return float(self.model.predict(features_scaled)[0])

    def _create_features_with_price(
        self,
        base_features: np.ndarray,
        price: float,
        price_idx: int,
    ) -> np.ndarray:
        """Cree un vecteur de features avec un nouveau prix.

        Args:
            base_features: Features de base.
            price: Nouveau prix.
            price_idx: Index de la feature prix.

        Returns:
            Features modifiees.
        """
        features = base_features.copy()
        features[price_idx] = price

        # Mettre a jour log_price si present
        log_price_idx = self._get_feature_idx("log_price")
        if log_price_idx is not None:
            features[log_price_idx] = np.log1p(price)

        return features

    def _get_feature_idx(self, name: str) -> int | None:
        """Recupere l'index d'une feature par son nom.

        Args:
            name: Nom de la feature.

        Returns:
            Index ou None si non trouve.
        """
        try:
            return self.feature_names.index(name)
        except ValueError:
            return None

    def optimize_price(
        self,
        base_features: np.ndarray,
        current_price: float,
        objective: str = "revenue",
        min_price: float | None = None,
        max_price: float | None = None,
        max_change: float = 0.2,
    ) -> tuple[float, float]:
        """Optimise le prix pour maximiser l'objectif.

        Args:
            base_features: Features de base (sans le prix optimal).
            current_price: Prix actuel.
            objective: Type d'objectif ('revenue' ou 'margin').
            min_price: Prix minimum absolu.
            max_price: Prix maximum absolu.
            max_change: Variation maximale relative.

        Returns:
            Tuple (prix optimal, valeur objectif).
        """
        price_idx = self._get_feature_idx("price")
        if price_idx is None:
            logger.error("feature_prix_non_trouvee")
            return current_price, 0.0

        # Bornes
        lower_bound = max(
            current_price * (1 - max_change),
            min_price if min_price else 0.01,
        )
        upper_bound = min(
            current_price * (1 + max_change),
            max_price if max_price else current_price * 2,
        )

        def objective_fn(price: np.ndarray) -> float:
            p = price[0]
            features = self._create_features_with_price(base_features, p, price_idx)
            demand = self.predict_demand(features)

            if objective == "revenue":
                return -expected_revenue(p, demand)  # Negatif car on minimise
            else:  # margin
                return -expected_margin(p, demand, self.cost_ratio)

        result = minimize(
            objective_fn,
            x0=[current_price],
            method="SLSQP",
            bounds=[(lower_bound, upper_bound)],
            options={"maxiter": 100, "ftol": 1e-6},
        )

        optimal_price = float(result.x[0])
        optimal_value = float(-result.fun)  # Remettre en positif

        return optimal_price, optimal_value

    def optimize_product(
        self,
        product_id: str,
        features: np.ndarray,
        current_price: float,
        current_sales: float,
        constraints: dict[str, Any] | None = None,
    ) -> OptimizationResult:
        """Optimise le prix pour un produit specifique.

        Args:
            product_id: Identifiant du produit.
            features: Features du produit.
            current_price: Prix actuel.
            current_sales: Ventes actuelles.
            constraints: Contraintes optionnelles.

        Returns:
            Resultat de l'optimisation.
        """
        constraints = constraints or {}

        # Optimiser
        optimal_price, expected_rev = self.optimize_price(
            features,
            current_price,
            objective=constraints.get("objective", "revenue"),
            min_price=constraints.get("min_price"),
            max_price=constraints.get("max_price"),
            max_change=constraints.get("max_change", 0.2),
        )

        # Calculer les metriques
        price_idx = self._get_feature_idx("price")
        if price_idx is not None:
            opt_features = self._create_features_with_price(features, optimal_price, price_idx)
            expected_vol = self.predict_demand(opt_features)
        else:
            expected_vol = current_sales

        # Calculer elasticite locale
        epsilon = 0.01
        if price_idx is not None:
            features_up = self._create_features_with_price(
                features, current_price * (1 + epsilon), price_idx
            )
            features_down = self._create_features_with_price(
                features, current_price * (1 - epsilon), price_idx
            )
            demand_up = self.predict_demand(features_up)
            demand_down = self.predict_demand(features_down)

            dq = demand_up - demand_down
            dp = 2 * epsilon * current_price
            elasticity = (dq / current_sales) / (dp / current_price) if current_sales > 0 else 0
        else:
            elasticity = 0.0

        return OptimizationResult(
            product_id=product_id,
            current_price=current_price,
            optimized_price=round(optimal_price, 2),
            current_volume=current_sales,
            expected_volume=round(expected_vol, 2),
            current_revenue=round(expected_revenue(current_price, current_sales), 2),
            expected_revenue=round(expected_revenue(optimal_price, expected_vol), 2),
            current_margin=round(expected_margin(current_price, current_sales, self.cost_ratio), 2),
            expected_margin=round(expected_margin(optimal_price, expected_vol, self.cost_ratio), 2),
            elasticity=round(elasticity, 4),
            success=True,
        )


def simulate_price_scenarios(
    optimizer: PriceOptimizer,
    features: np.ndarray,
    current_price: float,
    variations: list[float],
) -> pd.DataFrame:
    """Simule differents scenarios de prix.

    Args:
        optimizer: Optimiseur de prix.
        features: Features de base.
        current_price: Prix actuel.
        variations: Liste des variations relatives a tester.

    Returns:
        DataFrame avec les resultats de simulation.
    """
    results = []
    price_idx = optimizer._get_feature_idx("price")

    if price_idx is None:
        return pd.DataFrame()

    for var in variations:
        new_price = current_price * (1 + var)
        new_features = optimizer._create_features_with_price(features, new_price, price_idx)
        demand = optimizer.predict_demand(new_features)

        results.append(
            {
                "price_variation": var,
                "price": round(new_price, 2),
                "expected_demand": round(demand, 2),
                "expected_revenue": round(expected_revenue(new_price, demand), 2),
                "expected_margin": round(
                    expected_margin(new_price, demand, optimizer.cost_ratio), 2
                ),
            }
        )

    return pd.DataFrame(results)


def run_optimization(cfg: DictConfig) -> None:
    """Execute le pipeline d'optimisation des prix.

    Args:
        cfg: Configuration Hydra.
    """
    setup_logging(level="INFO", json_format=False)

    processed_dir = Path(cfg.data.processed_dir)
    output_dir = Path(cfg.optimize.output.dir)

    # Charger le modele
    model = load_model(MODELS_DIR / "lightgbm_model.joblib")
    scaler = load_model(MODELS_DIR / "scaler.joblib")
    feature_names = load_model(MODELS_DIR / "feature_names.joblib")

    # Creer l'optimiseur
    optimizer = PriceOptimizer(
        model=model,
        scaler=scaler,
        feature_names=feature_names,
        cost_ratio=cfg.optimize.objective.cost_ratio,
    )

    # Charger les donnees
    test_df = load_parquet(processed_dir / "test_features.parquet")

    # Agreger par produit (store + family)
    product_df = (
        test_df.groupby(["store_nbr", "family"])
        .agg(
            {
                "price": "mean",
                "sales": "mean",
            }
        )
        .reset_index()
    )
    product_df["product_id"] = product_df["store_nbr"].astype(str) + "_" + product_df["family"]

    logger.info("produits_a_optimiser", n_products=len(product_df))

    # Optimiser un echantillon
    sample_size = min(100, len(product_df))
    sample_df = product_df.sample(sample_size, random_state=cfg.seed)

    results = []
    for _, row in sample_df.iterrows():
        # Recuperer les features moyennes pour ce produit
        product_data = test_df[
            (test_df["store_nbr"] == row["store_nbr"]) & (test_df["family"] == row["family"])
        ]

        if len(product_data) == 0:
            continue

        # Utiliser les features moyennes
        numeric_cols = [c for c in feature_names if c in product_data.columns]
        if len(numeric_cols) == 0:
            continue

        features = product_data[numeric_cols].mean().values

        result = optimizer.optimize_product(
            product_id=row["product_id"],
            features=features,
            current_price=row["price"],
            current_sales=row["sales"],
            constraints={
                "objective": cfg.optimize.objective.type,
                "min_price": cfg.optimize.constraints.absolute_min,
                "max_price": cfg.optimize.constraints.absolute_max,
                "max_change": cfg.optimize.constraints.max_price_change,
            },
        )

        results.append(
            {
                "product_id": result.product_id,
                "current_price": result.current_price,
                "optimized_price": result.optimized_price,
                "price_change_pct": round(
                    (result.optimized_price - result.current_price) / result.current_price * 100, 2
                ),
                "current_volume": result.current_volume,
                "expected_volume": result.expected_volume,
                "current_revenue": result.current_revenue,
                "expected_revenue": result.expected_revenue,
                "revenue_change_pct": round(
                    (result.expected_revenue - result.current_revenue)
                    / result.current_revenue
                    * 100
                    if result.current_revenue > 0
                    else 0,
                    2,
                ),
                "elasticity": result.elasticity,
            }
        )

    # Sauvegarder les resultats
    results_df = pd.DataFrame(results)
    ensure_dir(output_dir)
    save_parquet(results_df, output_dir / "recommendations.parquet")

    # Statistiques
    logger.info(
        "optimisation_terminee",
        n_products=len(results_df),
        avg_price_change=float(results_df["price_change_pct"].mean()),
        avg_revenue_uplift=float(results_df["revenue_change_pct"].mean()),
        avg_elasticity=float(results_df["elasticity"].mean()),
    )


if __name__ == "__main__":
    import hydra

    @hydra.main(version_base=None, config_path="../../configs/hydra", config_name="config")
    def main(cfg: DictConfig) -> None:
        run_optimization(cfg)

    main()
