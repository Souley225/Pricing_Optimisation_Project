"""API FastAPI pour les recommandations de prix."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.models.optimize_prices import PriceOptimizer
from src.models.predict import DemandPredictor
from src.utils.logging import get_logger, setup_logging
from src.utils.metrics import expected_margin, expected_revenue

logger = get_logger(__name__)


# Schemas Pydantic v2
class PriceConstraints(BaseModel):
    """Contraintes pour l'optimisation du prix."""

    min_price: float | None = Field(None, ge=0, description="Prix minimum absolu")
    max_price: float | None = Field(None, ge=0, description="Prix maximum absolu")
    max_change: float = Field(0.2, ge=0, le=1, description="Variation max relative")


class RecommendPriceRequest(BaseModel):
    """Requete de recommandation de prix."""

    product_id: str = Field(..., description="Identifiant du produit")
    current_price: float = Field(..., gt=0, description="Prix actuel")
    current_volume: float = Field(100.0, ge=0, description="Volume actuel")
    constraints: PriceConstraints | None = Field(None, description="Contraintes optionnelles")
    features: dict[str, float] | None = Field(None, description="Features optionnelles")

    model_config = {
        "json_schema_extra": {
            "example": {
                "product_id": "GROCERY_I_1",
                "current_price": 4.50,
                "current_volume": 100.0,
                "constraints": {"min_price": 3.50, "max_price": 6.00, "max_change": 0.2},
            }
        }
    }


class RecommendPriceResponse(BaseModel):
    """Reponse de recommandation de prix."""

    recommended_price: float = Field(..., description="Prix recommande")
    expected_volume: float = Field(..., description="Volume attendu")
    expected_revenue: float = Field(..., description="Revenu attendu")
    expected_margin: float = Field(..., description="Marge attendue")
    price_change_pct: float = Field(..., description="Variation de prix en %")
    revenue_uplift_pct: float = Field(..., description="Hausse de revenu en %")
    model_version: str = Field(..., description="Version du modele")

    model_config = {
        "json_schema_extra": {
            "example": {
                "recommended_price": 4.15,
                "expected_volume": 125.5,
                "expected_revenue": 520.33,
                "expected_margin": 130.08,
                "price_change_pct": -7.78,
                "revenue_uplift_pct": 15.5,
                "model_version": "1.0.0",
            }
        }
    }


class HealthResponse(BaseModel):
    """Reponse du health check."""

    status: str
    model_loaded: bool
    timestamp: str
    version: str


class SimulateRequest(BaseModel):
    """Requete de simulation de prix."""

    product_id: str
    current_price: float = Field(..., gt=0)
    price_variations: list[float] = Field(
        default=[-0.2, -0.1, -0.05, 0, 0.05, 0.1, 0.2],
        description="Variations relatives a simuler",
    )
    features: dict[str, float] | None = None


class SimulationResult(BaseModel):
    """Resultat d'une simulation."""

    price_variation: float
    price: float
    expected_volume: float
    expected_revenue: float
    expected_margin: float


class SimulateResponse(BaseModel):
    """Reponse de simulation."""

    product_id: str
    current_price: float
    simulations: list[SimulationResult]


# Etat global de l'application
class AppState:
    """Etat de l'application."""

    predictor: DemandPredictor | None = None
    optimizer: PriceOptimizer | None = None
    model_version: str = "1.0.0"
    cost_ratio: float = 0.6


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Gestion du cycle de vie de l'application."""
    setup_logging(level="INFO", json_format=True)

    # Charger le modele au demarrage (MLflow avec fallback local)
    try:
        state.predictor = DemandPredictor(use_mlflow=True)
        state.predictor.load()

        # Utiliser la version du modele charge
        state.model_version = state.predictor.get_model_version()

        # Scaler et feature names depuis le predictor
        scaler = state.predictor.scaler
        feature_names = state.predictor.get_feature_names()

        if scaler is not None and feature_names:
            state.optimizer = PriceOptimizer(
                model=state.predictor.model,
                scaler=scaler,
                feature_names=feature_names,
                cost_ratio=state.cost_ratio,
            )

        logger.info(
            "api_demarree",
            model_version=state.model_version,
            model_loaded=True,
            optimizer_ready=state.optimizer is not None,
        )
    except Exception as e:
        logger.error("erreur_chargement_modele", error=str(e))
        # Continuer meme sans modele pour le health check

    yield

    # Cleanup
    logger.info("api_arretee")


app = FastAPI(
    title="Pricing Optimization API",
    description="API de recommandation de prix basee sur l'estimation de la demande",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """Verifie l'etat de sante de l'API."""
    return HealthResponse(
        status="healthy",
        model_loaded=state.predictor is not None and state.predictor._loaded,
        timestamp=datetime.utcnow().isoformat(),
        version=state.model_version,
    )


@app.post("/recommend_price", response_model=RecommendPriceResponse, tags=["Recommendations"])
async def recommend_price(request: RecommendPriceRequest) -> RecommendPriceResponse:
    """Recommande un prix optimal pour un produit."""
    if state.optimizer is None or state.predictor is None:
        raise HTTPException(status_code=503, detail="Modele non disponible")

    # Preparer les features
    if request.features:
        features_dict = request.features.copy()
    else:
        # Features par defaut
        features_dict = {
            "price": request.current_price,
            "log_price": np.log1p(request.current_price),
        }

    features_dict["price"] = request.current_price

    # Creer le vecteur de features
    feature_names = state.predictor.get_feature_names()
    features = np.zeros(len(feature_names))

    for name, value in features_dict.items():
        if name in feature_names:
            idx = feature_names.index(name)
            features[idx] = value

    # Contraintes
    constraints = {}
    if request.constraints:
        constraints = {
            "min_price": request.constraints.min_price,
            "max_price": request.constraints.max_price,
            "max_change": request.constraints.max_change,
        }

    # Optimiser
    result = state.optimizer.optimize_product(
        product_id=request.product_id,
        features=features,
        current_price=request.current_price,
        current_sales=request.current_volume,
        constraints=constraints,
    )

    # Calculer les variations
    price_change = (result.optimized_price - request.current_price) / request.current_price * 100
    current_rev = expected_revenue(request.current_price, request.current_volume)
    revenue_uplift = (
        (result.expected_revenue - current_rev) / current_rev * 100 if current_rev > 0 else 0
    )

    return RecommendPriceResponse(
        recommended_price=result.optimized_price,
        expected_volume=result.expected_volume,
        expected_revenue=result.expected_revenue,
        expected_margin=result.expected_margin,
        price_change_pct=round(price_change, 2),
        revenue_uplift_pct=round(revenue_uplift, 2),
        model_version=state.model_version,
    )


@app.post("/simulate", response_model=SimulateResponse, tags=["Simulations"])
async def simulate_prices(request: SimulateRequest) -> SimulateResponse:
    """Simule differents scenarios de prix."""
    if state.optimizer is None or state.predictor is None:
        raise HTTPException(status_code=503, detail="Modele non disponible")

    # Preparer les features
    if request.features:
        features_dict = request.features.copy()
    else:
        features_dict = {
            "price": request.current_price,
            "log_price": np.log1p(request.current_price),
        }

    feature_names = state.predictor.get_feature_names()
    features = np.zeros(len(feature_names))

    for name, value in features_dict.items():
        if name in feature_names:
            idx = feature_names.index(name)
            features[idx] = value

    # Simuler chaque variation
    simulations = []
    price_idx = state.optimizer._get_feature_idx("price")

    for var in request.price_variations:
        new_price = request.current_price * (1 + var)

        if price_idx is not None:
            sim_features = state.optimizer._create_features_with_price(
                features, new_price, price_idx
            )
            demand = state.optimizer.predict_demand(sim_features)
        else:
            demand = 100.0  # Valeur par defaut

        simulations.append(
            SimulationResult(
                price_variation=var,
                price=round(new_price, 2),
                expected_volume=round(demand, 2),
                expected_revenue=round(expected_revenue(new_price, demand), 2),
                expected_margin=round(expected_margin(new_price, demand, state.cost_ratio), 2),
            )
        )

    return SimulateResponse(
        product_id=request.product_id,
        current_price=request.current_price,
        simulations=simulations,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.serving.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
