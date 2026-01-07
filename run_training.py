"""Script standalone pour entrainer les modeles."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler

from src.utils.io import load_parquet, save_model
from src.utils.logging import get_logger, setup_logging
from src.utils.metrics import compute_all_metrics
from src.utils.paths import MODELS_DIR, PROCESSED_DATA_DIR, ensure_dir

setup_logging(level="INFO", json_format=False)
logger = get_logger(__name__)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Retourne les colonnes de features numeriques."""
    exclude = {"id", "date", "sales", "family", "city", "state", "type", "cluster"}
    return [
        c for c in df.columns
        if c not in exclude and df[c].dtype in ["int8", "int16", "int32", "int64", "float32", "float64"]
    ]


def main() -> None:
    """Entraine les modeles."""
    logger.info("chargement_donnees")
    
    train = load_parquet(PROCESSED_DATA_DIR / "train_features.parquet")
    val = load_parquet(PROCESSED_DATA_DIR / "val_features.parquet")
    
    logger.info("donnees_chargees", train=len(train), val=len(val))
    
    # Features
    feature_cols = get_feature_columns(train)
    logger.info("features_selectionnees", n=len(feature_cols))
    
    X_train = train[feature_cols].fillna(0).values
    y_train = train["sales"].values
    X_val = val[feature_cols].fillna(0).values
    y_val = val["sales"].values
    
    # Scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Sauvegarder scaler et features
    ensure_dir(MODELS_DIR)
    save_model(scaler, MODELS_DIR / "scaler.joblib")
    save_model(feature_cols, MODELS_DIR / "feature_names.joblib")
    logger.info("scaler_sauvegarde")
    
    # 1. ElasticNet baseline
    logger.info("entrainement_elasticnet")
    elasticnet = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=1000, random_state=42)
    elasticnet.fit(X_train_scaled, y_train)
    
    y_pred_en = elasticnet.predict(X_val_scaled)
    en_metrics = compute_all_metrics(y_val, y_pred_en)
    logger.info("elasticnet_metrics", **en_metrics)
    
    save_model(elasticnet, MODELS_DIR / "elasticnet_model.joblib")
    
    # 2. LightGBM
    logger.info("entrainement_lightgbm")
    lgbm = lgb.LGBMRegressor(
        objective="regression",
        metric="rmse",
        n_estimators=500,
        num_leaves=31,
        learning_rate=0.05,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        verbose=-1,
        random_state=42,
    )
    
    lgbm.fit(
        X_train_scaled,
        y_train,
        eval_set=[(X_val_scaled, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100),
        ],
    )
    
    y_pred_lgbm = lgbm.predict(X_val_scaled)
    lgbm_metrics = compute_all_metrics(y_val, y_pred_lgbm)
    logger.info("lightgbm_metrics", **lgbm_metrics)
    
    save_model(lgbm, MODELS_DIR / "lightgbm_model.joblib")
    
    # Resultats
    logger.info(
        "entrainement_termine",
        elasticnet_rmse=en_metrics["rmse"],
        lightgbm_rmse=lgbm_metrics["rmse"],
        improvement=f"{(1 - lgbm_metrics['rmse'] / en_metrics['rmse']) * 100:.1f}%",
    )


if __name__ == "__main__":
    main()
