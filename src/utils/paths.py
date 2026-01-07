"""Gestion centralisee des chemins du projet."""

from pathlib import Path


def get_project_root() -> Path:
    """Retourne le chemin racine du projet."""
    return Path(__file__).resolve().parents[2]


def get_data_dir() -> Path:
    """Retourne le repertoire des donnees."""
    return get_project_root() / "data"


def get_raw_data_dir() -> Path:
    """Retourne le repertoire des donnees brutes."""
    return get_data_dir() / "raw"


def get_interim_data_dir() -> Path:
    """Retourne le repertoire des donnees intermediaires."""
    return get_data_dir() / "interim"


def get_processed_data_dir() -> Path:
    """Retourne le repertoire des donnees traitees."""
    return get_data_dir() / "processed"


def get_external_data_dir() -> Path:
    """Retourne le repertoire des donnees externes."""
    return get_data_dir() / "external"


def get_models_dir() -> Path:
    """Retourne le repertoire des modeles."""
    return get_project_root() / "models"


def get_reports_dir() -> Path:
    """Retourne le repertoire des rapports."""
    return get_project_root() / "reports"


def get_figures_dir() -> Path:
    """Retourne le repertoire des figures."""
    return get_reports_dir() / "figures"


def get_configs_dir() -> Path:
    """Retourne le repertoire des configurations."""
    return get_project_root() / "configs"


def get_logs_dir() -> Path:
    """Retourne le repertoire des logs."""
    return get_project_root() / "logs"


def ensure_dir(path: Path) -> Path:
    """Cree un repertoire s'il n'existe pas et retourne le chemin."""
    path.mkdir(parents=True, exist_ok=True)
    return path


# Chemins pre-calcules pour import direct
PROJECT_ROOT = get_project_root()
DATA_DIR = get_data_dir()
RAW_DATA_DIR = get_raw_data_dir()
INTERIM_DATA_DIR = get_interim_data_dir()
PROCESSED_DATA_DIR = get_processed_data_dir()
EXTERNAL_DATA_DIR = get_external_data_dir()
MODELS_DIR = get_models_dir()
REPORTS_DIR = get_reports_dir()
FIGURES_DIR = get_figures_dir()
CONFIGS_DIR = get_configs_dir()
LOGS_DIR = get_logs_dir()
