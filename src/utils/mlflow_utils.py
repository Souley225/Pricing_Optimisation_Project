"""Utilitaires pour MLflow."""

import os
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient


def setup_mlflow(
    tracking_uri: str | None = None,
    experiment_name: str = "default",
) -> None:
    """Configure MLflow avec l'URI de tracking et l'experience.

    Args:
        tracking_uri: URI du serveur MLflow. Par defaut, utilise la variable
                      d'environnement MLFLOW_TRACKING_URI ou ./mlruns.
        experiment_name: Nom de l'experience MLflow.
    """
    uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_name)


def get_or_create_experiment(name: str) -> str:
    """Recupere ou cree une experience MLflow.

    Args:
        name: Nom de l'experience.

    Returns:
        ID de l'experience.
    """
    client = MlflowClient()
    experiment = client.get_experiment_by_name(name)

    if experiment is None:
        experiment_id = client.create_experiment(name)
    else:
        experiment_id = experiment.experiment_id

    return experiment_id


@contextmanager
def start_run(
    run_name: str | None = None,
    tags: dict[str, str] | None = None,
    nested: bool = False,
) -> Generator[mlflow.ActiveRun, None, None]:
    """Context manager pour un run MLflow.

    Args:
        run_name: Nom du run.
        tags: Tags a ajouter au run.
        nested: Si True, permet les runs imbriques.

    Yields:
        Le run MLflow actif.
    """
    with mlflow.start_run(run_name=run_name, nested=nested) as run:
        if tags:
            mlflow.set_tags(tags)
        yield run


def log_params_dict(params: dict[str, Any], prefix: str = "") -> None:
    """Logue un dictionnaire de parametres.

    Args:
        params: Dictionnaire de parametres.
        prefix: Prefixe optionnel pour les cles.
    """
    flat_params = _flatten_dict(params, prefix)
    # MLflow limite les valeurs a 500 caracteres
    truncated = {k: str(v)[:500] for k, v in flat_params.items()}
    mlflow.log_params(truncated)


def log_metrics_dict(metrics: dict[str, float], step: int | None = None) -> None:
    """Logue un dictionnaire de metriques.

    Args:
        metrics: Dictionnaire de metriques.
        step: Etape optionnelle pour le tracking.
    """
    mlflow.log_metrics(metrics, step=step)


def log_artifact_file(path: Path | str, artifact_path: str | None = None) -> None:
    """Logue un fichier comme artefact.

    Args:
        path: Chemin du fichier local.
        artifact_path: Chemin de destination dans les artefacts.
    """
    mlflow.log_artifact(str(path), artifact_path)


def log_model_sklearn(
    model: Any,
    artifact_path: str = "model",
    registered_model_name: str | None = None,
) -> None:
    """Logue un modele sklearn.

    Args:
        model: Modele sklearn a loguer.
        artifact_path: Chemin dans les artefacts.
        registered_model_name: Nom pour l'enregistrement dans le registry.
    """
    mlflow.sklearn.log_model(
        model,
        artifact_path=artifact_path,
        registered_model_name=registered_model_name,
    )


def log_model_lightgbm(
    model: Any,
    artifact_path: str = "model",
    registered_model_name: str | None = None,
) -> None:
    """Logue un modele LightGBM.

    Args:
        model: Modele LightGBM a loguer.
        artifact_path: Chemin dans les artefacts.
        registered_model_name: Nom pour l'enregistrement dans le registry.
    """
    mlflow.lightgbm.log_model(
        model,
        artifact_path=artifact_path,
        registered_model_name=registered_model_name,
    )


def load_model_from_registry(
    model_name: str,
    version: str | None = None,
    stage: str | None = None,
) -> Any:
    """Charge un modele depuis le registry MLflow.

    Args:
        model_name: Nom du modele enregistre.
        version: Version specifique a charger.
        stage: Stage du modele (None, Staging, Production, Archived).

    Returns:
        Modele charge.
    """
    if version:
        model_uri = f"models:/{model_name}/{version}"
    elif stage:
        model_uri = f"models:/{model_name}/{stage}"
    else:
        model_uri = f"models:/{model_name}/latest"

    return mlflow.pyfunc.load_model(model_uri)


def _flatten_dict(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Aplatit un dictionnaire imbrique.

    Args:
        d: Dictionnaire a aplatir.
        prefix: Prefixe pour les cles.

    Returns:
        Dictionnaire aplati.
    """
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key).items())
        else:
            items.append((new_key, v))
    return dict(items)
