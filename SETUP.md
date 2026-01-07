# Guide de Configuration

Ce document decrit les etapes pour configurer l'environnement de developpement.

---

## Prerequisites

- Python 3.11+
- Poetry 1.7+
- Git
- Docker et Docker Compose (optionnel, pour le deploiement)
- Compte Kaggle avec API configuree

---

## 1. Python Environment

### Option A: pyenv (recommande)

```bash
# Installer pyenv (Windows: utiliser pyenv-win)
# https://github.com/pyenv-win/pyenv-win

# Installer Python 3.11
pyenv install 3.11.8
pyenv local 3.11.8
```

### Option B: Python direct

Telecharger Python 3.11+ depuis https://www.python.org/downloads/

---

## 2. Poetry

```bash
# Installer Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Configurer Poetry pour creer le venv dans le projet
poetry config virtualenvs.in-project true

# Installer les dependances
poetry install
```

---

## 3. Kaggle API

```bash
# Creer le fichier de configuration
# Windows: C:\Users\<username>\.kaggle\kaggle.json
# Linux/Mac: ~/.kaggle/kaggle.json

# Contenu:
{
  "username": "votre_username",
  "key": "votre_api_key"
}

# Restreindre les permissions (Linux/Mac)
chmod 600 ~/.kaggle/kaggle.json
```

Obtenir votre API key: https://www.kaggle.com/settings > API > Create New Token

---

## 4. DVC

```bash
# Initialiser DVC
dvc init

# Configurer le remote (optionnel, pour partage)
dvc remote add -d storage s3://your-bucket/dvc
```

---

## 5. Pre-commit (optionnel)

```bash
# Installer les hooks
poetry run pre-commit install

# Executer manuellement
poetry run pre-commit run --all-files
```

---

## 6. Variables d'Environnement

```bash
# Copier le template
cp .env.example .env

# Editer selon vos besoins
# Notamment MLFLOW_TRACKING_URI si vous utilisez un serveur central
```

---

## 7. Verification

```bash
# Verifier l'installation
poetry run python -c "import pandas; import mlflow; import lightgbm; print('OK')"

# Lancer les tests
poetry run pytest tests/ -v

# Verifier la qualite du code
poetry run ruff check .
poetry run mypy src/
```

---

## Troubleshooting

### LightGBM ne s'installe pas

Sur Windows, vous pourriez avoir besoin de Visual C++ Build Tools:
https://visualstudio.microsoft.com/visual-cpp-build-tools/

### Erreur Kaggle API

Verifiez que le fichier kaggle.json est au bon emplacement et contient des credentials valides.

### MLflow ne demarre pas

Verifiez que le port 5000 n'est pas deja utilise:
```bash
# Windows
netstat -ano | findstr :5000

# Linux/Mac
lsof -i :5000
```
