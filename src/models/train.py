"""Entrainement des modeles de demande."""

from pathlib import Path
from typing import Any

import lightgbm as lgb
import mlflow
import numpy as np
import optuna
from omegaconf import DictConfig, OmegaConf
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from src.features.build_features import build_feature_matrix
from src.utils.io import load_parquet, save_model
from src.utils.logging import get_logger, setup_logging
from src.utils.metrics import compute_all_metrics
from src.utils.mlflow_utils import (
    log_metrics_dict,
    log_model_lightgbm,
    log_model_sklearn,
    log_params_dict,
    setup_mlflow,
    start_run,
)
from src.utils.paths import MODELS_DIR, ensure_dir

logger = get_logger(__name__)


def train_elasticnet_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    alpha: float = 0.1,
    l1_ratio: float = 0.5,
    random_state: int = 42,
) -> ElasticNet:
    """Entraine un modele ElasticNet baseline.

    Args:
        X_train: Features d'entrainement.
        y_train: Cible d'entrainement.
        alpha: Parametre de regularisation.
        l1_ratio: Ratio L1/L2.
        random_state: Graine aleatoire.

    Returns:
        Modele ElasticNet entraine.
    """
    model = ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        max_iter=1000,
        random_state=random_state,
    )
    model.fit(X_train, y_train)

    logger.info(
        "elasticnet_entraine",
        alpha=alpha,
        l1_ratio=l1_ratio,
        n_coefs=int(np.sum(model.coef_ != 0)),
    )

    return model


def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: dict[str, Any],
    n_estimators: int = 1000,
    early_stopping_rounds: int = 50,
) -> lgb.LGBMRegressor:
    """Entraine un modele LightGBM.

    Args:
        X_train: Features d'entrainement.
        y_train: Cible d'entrainement.
        X_val: Features de validation.
        y_val: Cible de validation.
        params: Hyperparametres LightGBM.
        n_estimators: Nombre maximum d'arbres.
        early_stopping_rounds: Arret precoce.

    Returns:
        Modele LightGBM entraine.
    """
    model = lgb.LGBMRegressor(
        n_estimators=n_estimators,
        **params,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=early_stopping_rounds),
            lgb.log_evaluation(period=100),
        ],
    )

    logger.info(
        "lightgbm_entraine",
        best_iteration=model.best_iteration_,
        best_score=float(model.best_score_["valid_0"]["rmse"]) if model.best_score_ else None,
    )

    return model


def tune_lightgbm_optuna(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    search_space: dict[str, Any],
    n_trials: int = 50,
    timeout: int = 3600,
    seed: int = 42,
) -> dict[str, Any]:
    """Optimise les hyperparametres LightGBM avec Optuna.

    Args:
        X_train: Features d'entrainement.
        y_train: Cible d'entrainement.
        X_val: Features de validation.
        y_val: Cible de validation.
        search_space: Espace de recherche des hyperparametres.
        n_trials: Nombre d'essais.
        timeout: Timeout en secondes.
        seed: Graine aleatoire.

    Returns:
        Meilleurs hyperparametres trouves.
    """

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "verbose": -1,
            "seed": seed,
            "num_leaves": trial.suggest_int(
                "num_leaves",
                search_space["num_leaves"]["low"],
                search_space["num_leaves"]["high"],
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate",
                search_space["learning_rate"]["low"],
                search_space["learning_rate"]["high"],
                log=search_space["learning_rate"].get("log", False),
            ),
            "feature_fraction": trial.suggest_float(
                "feature_fraction",
                search_space["feature_fraction"]["low"],
                search_space["feature_fraction"]["high"],
            ),
            "bagging_fraction": trial.suggest_float(
                "bagging_fraction",
                search_space["bagging_fraction"]["low"],
                search_space["bagging_fraction"]["high"],
            ),
            "min_child_samples": trial.suggest_int(
                "min_child_samples",
                search_space["min_child_samples"]["low"],
                search_space["min_child_samples"]["high"],
            ),
        }

        model = lgb.LGBMRegressor(n_estimators=500, **params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=30),
            ],
        )

        y_pred = model.predict(X_val)
        rmse = float(np.sqrt(np.mean((y_val - y_pred) ** 2)))

        return rmse

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)

    logger.info(
        "optuna_tuning_termine",
        best_value=study.best_value,
        best_params=study.best_params,
        n_trials=len(study.trials),
    )

    # Ajouter les parametres fixes
    best_params = study.best_params.copy()
    best_params.update(
        {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "verbose": -1,
            "seed": seed,
        }
    )

    return best_params


def cross_validate_temporal(
    model_class: type,
    model_params: dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
) -> dict[str, float]:
    """Validation croisee temporelle.

    Args:
        model_class: Classe du modele.
        model_params: Parametres du modele.
        X: Features.
        y: Cible.
        n_splits: Nombre de splits.

    Returns:
        Metriques moyennes sur tous les folds.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    all_metrics: list[dict[str, float]] = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        model = model_class(**model_params)
        model.fit(X_train_fold, y_train_fold)

        y_pred = model.predict(X_val_fold)
        metrics = compute_all_metrics(y_val_fold, y_pred)
        all_metrics.append(metrics)

        logger.info(f"fold_{fold}_metrics", **metrics)

    # Moyenner les metriques
    avg_metrics = {
        metric: float(np.mean([m[metric] for m in all_metrics])) for metric in all_metrics[0]
    }

    return avg_metrics


def run_training(cfg: DictConfig) -> None:
    """Execute le pipeline d'entrainement.

    Args:
        cfg: Configuration Hydra.
    """
    setup_logging(level="INFO", json_format=False)
    setup_mlflow(
        tracking_uri=cfg.train.mlflow.tracking_uri,
        experiment_name=cfg.train.mlflow.experiment_name,
    )

    processed_dir = Path(cfg.data.processed_dir)

    # Charger les donnees
    train_df = load_parquet(processed_dir / "train_features.parquet")
    val_df = load_parquet(processed_dir / "val_features.parquet")

    logger.info("donnees_chargees", train_rows=len(train_df), val_rows=len(val_df))

    # Construire les matrices
    X_train, y_train = build_feature_matrix(
        train_df,
        target_column=cfg.train.training.target_column,
        exclude_columns=list(cfg.train.training.exclude_columns),
    )
    X_val, y_val = build_feature_matrix(
        val_df,
        target_column=cfg.train.training.target_column,
        exclude_columns=list(cfg.train.training.exclude_columns),
    )

    # Scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    feature_names = list(X_train.columns)

    # Sauvegarder le scaler
    ensure_dir(MODELS_DIR)
    save_model(scaler, MODELS_DIR / "scaler.joblib")
    save_model(feature_names, MODELS_DIR / "feature_names.joblib")

    with start_run(run_name="training_pipeline", tags=dict(cfg.train.mlflow.tags)):
        # 1. Baseline ElasticNet
        with start_run(run_name="elasticnet_baseline", nested=True):
            baseline_params = OmegaConf.to_container(cfg.model.baseline.params)
            log_params_dict(baseline_params, prefix="elasticnet")

            elasticnet = train_elasticnet_baseline(
                X_train_scaled,
                y_train.values,
                **baseline_params,
            )

            y_pred_baseline = elasticnet.predict(X_val_scaled)
            baseline_metrics = compute_all_metrics(y_val.values, y_pred_baseline)
            log_metrics_dict(baseline_metrics)

            log_model_sklearn(elasticnet, artifact_path="elasticnet_model")

            logger.info("elasticnet_evaluation", **baseline_metrics)

        # 2. LightGBM avec tuning optionnel
        with start_run(run_name="lightgbm", nested=True):
            if cfg.model.optuna.enabled:
                # Tuning Optuna
                best_params = tune_lightgbm_optuna(
                    X_train_scaled,
                    y_train.values,
                    X_val_scaled,
                    y_val.values,
                    search_space=OmegaConf.to_container(cfg.model.optuna.search_space),
                    n_trials=cfg.model.optuna.n_trials,
                    timeout=cfg.model.optuna.timeout,
                    seed=cfg.seed,
                )
            else:
                best_params = OmegaConf.to_container(cfg.model.lightgbm.params)

            log_params_dict(best_params, prefix="lightgbm")

            lgbm_model = train_lightgbm(
                X_train_scaled,
                y_train.values,
                X_val_scaled,
                y_val.values,
                params=best_params,
                n_estimators=cfg.model.lightgbm.training.n_estimators,
                early_stopping_rounds=cfg.model.lightgbm.training.early_stopping_rounds,
            )

            y_pred_lgbm = lgbm_model.predict(X_val_scaled)
            lgbm_metrics = compute_all_metrics(y_val.values, y_pred_lgbm)
            log_metrics_dict(lgbm_metrics)

            log_model_lightgbm(
                lgbm_model,
                artifact_path="lightgbm_model",
                registered_model_name="demand-model",
            )

            # Sauvegarder localement aussi
            save_model(lgbm_model, MODELS_DIR / "lightgbm_model.joblib")

            # Feature importance
            mlflow.log_dict(
                {
                    name: float(imp)
                    for name, imp in zip(feature_names, lgbm_model.feature_importances_)
                },
                "feature_importance.json",
            )

            logger.info("lightgbm_evaluation", **lgbm_metrics)

    logger.info(
        "entrainement_termine",
        baseline_rmse=baseline_metrics["rmse"],
        lightgbm_rmse=lgbm_metrics["rmse"],
    )


if __name__ == "__main__":
    import hydra

    @hydra.main(version_base=None, config_path="../../configs/hydra", config_name="config")
    def main(cfg: DictConfig) -> None:
        run_training(cfg)

    main()
