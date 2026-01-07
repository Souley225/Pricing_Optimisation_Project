"""Configuration du logging structure JSON."""

import logging
import sys
from typing import Any

import structlog
from structlog.types import Processor


def setup_logging(
    level: str = "INFO",
    json_format: bool = True,
    log_file: str | None = None,
) -> None:
    """Configure le logging structure pour l'application.

    Args:
        level: Niveau de log (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        json_format: Si True, utilise le format JSON. Sinon, format console colore.
        log_file: Chemin optionnel vers un fichier de log.
    """
    # Processeurs communs
    common_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if json_format:
        # Format JSON pour production
        processors: list[Processor] = [
            *common_processors,
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Format console colore pour developpement
        processors = [
            *common_processors,
            structlog.dev.ConsoleRenderer(colors=True),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level.upper())),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configuration du logging standard pour les librairies tierces
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper()),
    )

    # Reduire le niveau de log des librairies bruyantes
    for logger_name in ["httpx", "httpcore", "urllib3", "mlflow"]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def get_logger(name: str | None = None) -> Any:
    """Obtient un logger structure.

    Args:
        name: Nom du logger. Utilise pour le contexte.

    Returns:
        Logger structure configure.
    """
    logger = structlog.get_logger()
    if name:
        logger = logger.bind(logger_name=name)
    return logger
