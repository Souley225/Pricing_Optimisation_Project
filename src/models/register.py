"""Enregistrement du modele dans le registry MLflow."""

from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

from src.utils.logging import get_logger, setup_logging
from src.utils.mlflow_utils import setup_mlflow

logger = get_logger(__name__)


def register_model(
    model_name: str,
    run_id: str | None = None,
    model_path: str = "lightgbm_model",
    stage: str = "Staging",
) -> str:
    """Enregistre un modele dans le MLflow Model Registry.

    Args:
        model_name: Nom du modele dans le registry.
        run_id: ID du run MLflow contenant le modele.
        model_path: Chemin de l'artefact modele dans le run.
        stage: Stage cible (None, Staging, Production, Archived).

    Returns:
        Version du modele enregistre.
    """
    client = MlflowClient()

    if run_id:
        model_uri = f"runs:/{run_id}/{model_path}"
    else:
        # Utiliser le dernier run
        experiment = mlflow.get_experiment_by_name("pricing-optimization")
        if experiment is None:
            raise ValueError("Experimentation 'pricing-optimization' non trouvee")

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=1,
        )

        if not runs:
            raise ValueError("Aucun run trouve dans l'experimentation")

        run_id = runs[0].info.run_id
        model_uri = f"runs:/{run_id}/{model_path}"

    # Enregistrer le modele
    result = mlflow.register_model(model_uri, model_name)
    version = result.version

    logger.info(
        "modele_enregistre",
        model_name=model_name,
        version=version,
        run_id=run_id,
    )

    # Transitionner vers le stage cible
    if stage:
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
        )
        logger.info(
            "modele_transition",
            model_name=model_name,
            version=version,
            stage=stage,
        )

    return version


def promote_model(
    model_name: str,
    version: str | None = None,
    from_stage: str = "Staging",
    to_stage: str = "Production",
) -> None:
    """Promeut un modele vers un nouveau stage.

    Args:
        model_name: Nom du modele.
        version: Version specifique. Si None, utilise la derniere version du stage source.
        from_stage: Stage source.
        to_stage: Stage cible.
    """
    client = MlflowClient()

    if version is None:
        # Trouver la derniere version dans le stage source
        versions = client.get_latest_versions(model_name, stages=[from_stage])
        if not versions:
            raise ValueError(f"Aucune version trouvee dans le stage {from_stage}")
        version = versions[0].version

    # Archiver la version actuelle en production si presente
    if to_stage == "Production":
        prod_versions = client.get_latest_versions(model_name, stages=["Production"])
        for pv in prod_versions:
            client.transition_model_version_stage(
                name=model_name,
                version=pv.version,
                stage="Archived",
            )
            logger.info(
                "version_archivee",
                model_name=model_name,
                version=pv.version,
            )

    # Promouvoir
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=to_stage,
    )

    logger.info(
        "modele_promu",
        model_name=model_name,
        version=version,
        from_stage=from_stage,
        to_stage=to_stage,
    )


def get_production_model(model_name: str) -> tuple[Any, str]:
    """Charge le modele en production.

    Args:
        model_name: Nom du modele.

    Returns:
        Tuple (modele, version).
    """
    client = MlflowClient()

    versions = client.get_latest_versions(model_name, stages=["Production"])
    if not versions:
        raise ValueError(f"Aucun modele en production pour {model_name}")

    version = versions[0].version
    model_uri = f"models:/{model_name}/Production"
    model = mlflow.pyfunc.load_model(model_uri)

    logger.info(
        "modele_production_charge",
        model_name=model_name,
        version=version,
    )

    return model, version


if __name__ == "__main__":
    setup_logging(level="INFO", json_format=False)
    setup_mlflow(experiment_name="pricing-optimization")

    # Enregistrer le modele
    version = register_model(
        model_name="demand-model",
        model_path="lightgbm_model",
        stage="Staging",
    )

    print(f"Modele enregistre: demand-model v{version}")
