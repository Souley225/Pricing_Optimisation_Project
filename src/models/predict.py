"""Module de prediction pour le service de recommandation."""

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.io import load_model
from src.utils.logging import get_logger
from src.utils.paths import MODELS_DIR

logger = get_logger(__name__)


def load_model_with_fallback(
    model_name: str = "demand-model",
    stage: str = "Production",
    local_path: Path | None = None,
) -> tuple[Any, str]:
    """Charge un modele depuis MLflow ou en fallback local.

    Tente d'abord de charger depuis le MLflow Model Registry.
    Si indisponible, charge depuis le chemin local.

    Args:
        model_name: Nom du modele dans le registry MLflow.
        stage: Stage du modele (Production, Staging, etc.).
        local_path: Chemin local de fallback.

    Returns:
        Tuple (modele, version/source).
    """
    # Essayer MLflow d'abord
    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
        mlflow.set_tracking_uri(tracking_uri)

        client = MlflowClient()
        versions = client.get_latest_versions(model_name, stages=[stage])

        if versions:
            version = versions[0].version
            model_uri = f"models:/{model_name}/{stage}"
            model = mlflow.pyfunc.load_model(model_uri)

            logger.info(
                "modele_charge_mlflow",
                model_name=model_name,
                version=version,
                stage=stage,
            )

            return model, f"mlflow:{version}"
        else:
            logger.warning(
                "aucune_version_mlflow",
                model_name=model_name,
                stage=stage,
            )

    except Exception as e:
        logger.warning(
            "mlflow_indisponible",
            error=str(e),
            fallback="local",
        )

    # Fallback local
    if local_path and local_path.exists():
        model = load_model(local_path)
        logger.info(
            "modele_charge_local",
            path=str(local_path),
        )
        return model, f"local:{local_path.name}"

    raise FileNotFoundError(
        f"Impossible de charger le modele: MLflow indisponible et chemin local {local_path} introuvable"
    )


class DemandPredictor:
    """Predicteur de demande pour le serving avec fallback local."""

    def __init__(
        self,
        model_path: Path | None = None,
        scaler_path: Path | None = None,
        features_path: Path | None = None,
        mlflow_model_name: str = "demand-model",
        mlflow_stage: str = "Production",
        use_mlflow: bool = True,
    ) -> None:
        """Initialise le predicteur.

        Args:
            model_path: Chemin du modele local. Par defaut MODELS_DIR/lightgbm_model.joblib.
            scaler_path: Chemin du scaler. Par defaut MODELS_DIR/scaler.joblib.
            features_path: Chemin des features. Par defaut MODELS_DIR/feature_names.joblib.
            mlflow_model_name: Nom du modele dans MLflow registry.
            mlflow_stage: Stage du modele MLflow.
            use_mlflow: Si True, tente de charger depuis MLflow d'abord.
        """
        self.model_path = model_path or MODELS_DIR / "lightgbm_model.joblib"
        self.scaler_path = scaler_path or MODELS_DIR / "scaler.joblib"
        self.features_path = features_path or MODELS_DIR / "feature_names.joblib"
        self.mlflow_model_name = mlflow_model_name
        self.mlflow_stage = mlflow_stage
        self.use_mlflow = use_mlflow

        self.model: Any = None
        self.scaler: Any = None
        self.feature_names: list[str] = []
        self.model_version: str = "unknown"
        self._loaded = False

    def load(self) -> None:
        """Charge le modele et le scaler avec fallback local."""
        if self._loaded:
            return

        # Charger le modele (MLflow ou local)
        if self.use_mlflow:
            try:
                self.model, self.model_version = load_model_with_fallback(
                    model_name=self.mlflow_model_name,
                    stage=self.mlflow_stage,
                    local_path=self.model_path,
                )
            except FileNotFoundError:
                # Fallback complet sur local
                if self.model_path.exists():
                    self.model = load_model(self.model_path)
                    self.model_version = f"local:{self.model_path.name}"
                else:
                    raise
        else:
            self.model = load_model(self.model_path)
            self.model_version = f"local:{self.model_path.name}"

        # Scaler et feature names toujours en local
        if self.scaler_path.exists():
            self.scaler = load_model(self.scaler_path)
        else:
            logger.warning("scaler_non_trouve", path=str(self.scaler_path))
            self.scaler = None

        if self.features_path.exists():
            self.feature_names = load_model(self.features_path)
        else:
            logger.warning("features_non_trouvees", path=str(self.features_path))
            self.feature_names = []

        self._loaded = True

        logger.info(
            "predicteur_charge",
            model_version=self.model_version,
            n_features=len(self.feature_names),
            has_scaler=self.scaler is not None,
        )

    def predict(self, features: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Predit la demande.

        Args:
            features: Features d'entree.

        Returns:
            Demande predite.
        """
        if not self._loaded:
            self.load()

        if isinstance(features, pd.DataFrame):
            features = features.values

        # Appliquer le scaler si disponible
        if self.scaler is not None:
            features_scaled = self.scaler.transform(features)
        else:
            features_scaled = features

        # Prediction (gere les modeles MLflow pyfunc et sklearn)
        if hasattr(self.model, "predict"):
            predictions = self.model.predict(features_scaled)
        else:
            # MLflow pyfunc
            predictions = self.model.predict(pd.DataFrame(features_scaled))
            if isinstance(predictions, pd.DataFrame):
                predictions = predictions.values.flatten()

        # S'assurer que les predictions sont positives
        predictions = np.maximum(predictions, 0)

        return predictions

    def predict_single(self, features_dict: dict[str, float]) -> float:
        """Predit la demande pour une seule observation.

        Args:
            features_dict: Dictionnaire feature_name -> value.

        Returns:
            Demande predite.
        """
        if not self._loaded:
            self.load()

        # Creer le vecteur de features
        features = np.zeros(len(self.feature_names))

        for name, value in features_dict.items():
            if name in self.feature_names:
                idx = self.feature_names.index(name)
                features[idx] = value

        return float(self.predict(features.reshape(1, -1))[0])

    def get_feature_names(self) -> list[str]:
        """Retourne les noms des features.

        Returns:
            Liste des noms de features.
        """
        if not self._loaded:
            self.load()
        return self.feature_names.copy()

    def get_model_version(self) -> str:
        """Retourne la version du modele charge.

        Returns:
            Version ou source du modele.
        """
        if not self._loaded:
            self.load()
        return self.model_version


# Singleton pour le serving
_predictor: DemandPredictor | None = None


def get_predictor(use_mlflow: bool = True) -> DemandPredictor:
    """Retourne le predicteur singleton.

    Args:
        use_mlflow: Si True, tente MLflow d'abord.

    Returns:
        Instance du predicteur.
    """
    global _predictor
    if _predictor is None:
        _predictor = DemandPredictor(use_mlflow=use_mlflow)
    return _predictor
