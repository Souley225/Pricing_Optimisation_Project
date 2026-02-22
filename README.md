# Optimisation des Prix — Tarification basée sur la demande

> **Application d'aide à la décision qui recommande le prix optimal pour chaque produit, afin de maximiser le chiffre d'affaires tout en tenant compte du comportement d'achat des clients.**

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Licence-MIT-green?style=for-the-badge" alt="License"/>
  <img src="https://img.shields.io/badge/MLflow-Tracking-0194E2?style=for-the-badge&logo=mlflow&logoColor=white" alt="MLflow"/>
  <img src="https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker"/>
  <img src="https://img.shields.io/badge/DVC-Pipeline-945DD6?style=for-the-badge&logo=dvc&logoColor=white" alt="DVC"/>
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI"/>
</p>

---

### Contexte

<p align="center">
  <img src="https://img.shields.io/badge/Secteur-Retail-667eea?style=flat-square" alt="Secteur"/>
  <img src="https://img.shields.io/badge/Enjeu-Maximisation_Revenus-667eea?style=flat-square" alt="Enjeu"/>
</p>

Dans le secteur de la grande distribution, fixer le bon prix est un equilibre delicat entre volume de ventes et marge. Ce projet propose une **Methode predictive** permettant d'optimiser les prix de vente en estimant l'elasticite de la demande, afin que les equipes pricing puissent maximiser leurs revenus.

### Solution

| Fonctionnalite                    | Description                                                                          |
| --------------------------------- | ------------------------------------------------------------------------------------ |
| **Recommandation de prix**  | Chaque produit recoit un prix optimal calcule a partir de l'estimation de la demande |
| **Simulation de scenarios** | Visualisation de l'impact de differents prix sur les ventes et revenus               |
| **Analyse de sensibilite**  | Mesure de l'elasticite-prix pour comprendre le comportement client                   |
| **Interface web**           | Application accessible pour consulter les recommandations sans competence technique  |

### Resultats

- Amelioration du modele : **-23% RMSE** par rapport au baseline
- Application **deployee et fonctionnelle** — testable via le lien ci-dessous
- Pipeline **reproductible** — les resultats peuvent etre regeneres a tout moment

---

## Demo en Ligne

<p align="center">
  <a href="https://pricing-optimization-ui.onrender.com/" target="_blank">
    <img src="https://img.shields.io/badge/Tester_la_Demo-En_Ligne-667eea?style=for-the-badge&logo=streamlit&logoColor=white" alt="Demo"/>
  </a>
</p>

| Element               | Lien                                                                                                  |
| --------------------- | ----------------------------------------------------------------------------------------------------- |
| Application Streamlit | [https://pricing-optimisation-project.onrender.com/](https://pricing-optimisation-project.onrender.com/) |
| API FastAPI (Swagger) | [https://pricing-optimization-api.onrender.com/docs](https://pricing-optimization-api.onrender.com/docs) |

> **Note** : L'application est hebergee sur Render (plan gratuit). Le premier chargement peut prendre quelques secondes si le service est en veille.

**Pour tester la demo :**

1. Cliquez sur le lien ci-dessus
2. Ouvrez le menu lateral pour configurer un produit (prix actuel, volume de ventes)
3. Cliquez sur "Calculer le prix optimal" pour obtenir la recommandation
4. Explorez les onglets "Simulation" et "Sensibilite" pour approfondir l'analyse

---

## Apercu

Pipeline complet de Machine Learning pour l'**optimisation des prix** base sur l'estimation de la demande. Ce projet implemente les meilleures pratiques **MLOps** : versioning des donnees, suivi des experiences, optimisation des hyperparametres et deploiement conteneurise.

**Objectif** : Recommander des prix optimaux qui maximisent le revenu en tenant compte de l'elasticite-prix de la demande.

---

## Stack Technologique

| Composant              | Outil                    | Description                                               |
| ---------------------- | ------------------------ | --------------------------------------------------------- |
| Configuration          | **Hydra**          | Gestion centralisee et modulaire des configurations YAML  |
| Versioning des donnees | **DVC**            | Reproductibilite et tracabilite des pipelines de donnees  |
| Suivi des experiences  | **MLflow**         | Logging des metriques, artefacts et registre de modeles   |
| Optimisation           | **Optuna**         | Recherche automatique des hyperparametres optimaux        |
| Optimisation de prix   | **SciPy**          | Algorithmes d'optimisation sous contraintes               |
| API de prediction      | **FastAPI**        | Service REST haute performance avec documentation Swagger |
| Interface utilisateur  | **Streamlit**      | Application web interactive pour les recommandations      |
| Visualisation          | **Plotly**         | Graphiques interactifs pour l'analyse des scenarios       |
| Conteneurisation       | **Docker Compose** | Orchestration multi-services pour le deploiement          |
| CI/CD                  | **GitHub Actions** | Automatisation des tests et du deploiement                |

---

## Informations du Projet

| Element                     | Valeur                                  |
| --------------------------- | --------------------------------------- |
| Dataset                     | `mmitchell/online-retail-ii` (Kaggle) |
| Variable cible              | Volume de ventes (regression)           |
| Langage                     | Python 3.11+                            |
| Gestionnaire de dependances | Poetry                                  |
| Licence                     | MIT                                     |

---

## Architecture

### Pipeline MLOps

```
                                    +-------------------+
                                    |   GitHub Actions  |
                                    |   (CI/CD)         |
                                    +-------------------+
                                            |
                     +----------------------+----------------------+
                     |                      |                      |
                     v                      v                      v
              +-----------+         +--------------+       +--------------+
              |   Lint    |         |    Tests     |       |    Build     |
              |   (Ruff)  |         |   (Pytest)   |       |   (Docker)   |
              +-----------+         +--------------+       +--------------+

                                   PIPELINE DE DONNEES
+----------------+     +------------------+     +-------------------+
|   Raw Data     | --> |  Feature Eng.    | --> |   Model Training  |
|   (Kaggle)     |     |  (DVC Pipeline)  |     |   (Optuna tuning) |
+----------------+     +------------------+     +-------------------+
                                                         |
                                                         v
                                               +-------------------+
                                               |   MLflow          |
                                               |   - Tracking      |
                                               |   - Model Registry|
                                               +-------------------+
                                                         |
                              Chargement du meilleur modele (Production)
                                                         |
                                                         v
+----------------+     +------------------+     +-------------------+
|   Streamlit    | <-- |   FastAPI        | <-- |  Price Optimizer  |
|   Dashboard    |     |   /recommend     |     |   (SciPy)         |
+----------------+     +------------------+     +-------------------+
```

### Flux MLflow

| Etape                 | Description                                                                          |
| --------------------- | ------------------------------------------------------------------------------------ |
| **Tracking**    | Chaque run d'entrainement est enregistre avec ses hyperparametres et metriques       |
| **Comparaison** | Interface MLflow UI pour comparer les experiences et selectionner le meilleur modele |
| **Registry**    | Le meilleur modele est promu au stage "Production" dans le registre                  |
| **Serving**     | L'API charge automatiquement le modele marque "Production" depuis le registre        |

> **Note** : Le Model Registry MLflow permet de versionner les modeles et de gerer leur cycle de vie (Staging → Production → Archived). Meme si le registry distant n'est pas encore configure, le systeme est pret pour cette integration.

---

## Structure du Projet

```
mlops-pricing-optimization/
├── README.md                    # Documentation principale
├── SETUP.md                     # Guide d'installation detaille
├── GUIDE_UTILISATEUR.md         # Guide d'utilisation de l'interface
├── pyproject.toml               # Configuration Poetry et outils
├── dvc.yaml                     # Definition du pipeline DVC
├── compose.yaml                 # Configuration Docker Compose
├── Makefile                     # Commandes utilitaires
│
├── configs/                     # Configurations Hydra
│   ├── data.yaml
│   ├── features.yaml
│   ├── model.yaml
│   └── optimization.yaml
│
├── src/                         # Code source principal
│   ├── data/                    # Telechargement et preparation des donnees
│   ├── features/                # Feature engineering
│   ├── models/                  # Entrainement, evaluation, optimisation
│   ├── serving/                 # API FastAPI
│   ├── ui/                      # Application Streamlit
│   └── utils/                   # Utilitaires partages
│
├── docker/                      # Dockerfiles
│   ├── Dockerfile.api
│   └── Dockerfile.ui
│
├── data/                        # Donnees (gerees par DVC)
│   ├── raw/                     # Donnees brutes
│   ├── interim/                 # Donnees intermediaires
│   └── processed/               # Features et artefacts
│
├── models/                      # Modeles entraines
├── mlruns/                      # Experiences MLflow
├── tests/                       # Tests unitaires
└── .github/workflows/           # Pipelines CI/CD
```

---

## Installation

### Prerequis

Avant de commencer, assurez-vous d'avoir installe :

- **Python 3.11** ou superieur (via pyenv recommande)
- **Poetry** pour la gestion des dependances
- **Git** et **DVC** pour le versioning
- **Docker** et **Docker Compose** pour le deploiement local
- Un **compte Kaggle** avec le fichier d'authentification `~/.kaggle/kaggle.json`

### Etapes d'installation

**1. Cloner le repository**

```bash
git clone https://github.com/Souley225/mlops-pricing-optimization.git
cd mlops-pricing-optimization
```

**2. Installer les dependances**

```bash
poetry install
```

**3. Configurer les variables d'environnement**

```bash
cp .env.example .env
```

**4. Configurer l'API Kaggle**

Creez le fichier d'authentification Kaggle et securisez-le :

```bash
# Linux/macOS
chmod 600 ~/.kaggle/kaggle.json

# Windows (PowerShell)
# Creez le fichier manuellement dans %USERPROFILE%\.kaggle\kaggle.json
```

Pour le guide complet, consultez [SETUP.md](SETUP.md).

---

## Execution du Pipeline

Le pipeline complet est orchestre par **DVC** et configure via **Hydra**.

### Execution complete

```bash
dvc repro
```

### Etapes du pipeline

| Etape        | Commande          | Description                                             |
| ------------ | ----------------- | ------------------------------------------------------- |
| `data`     | `make data`     | Telechargement et preparation des donnees Kaggle        |
| `features` | `make features` | Feature engineering et generation du prix synthetique   |
| `train`    | `make train`    | Entrainement avec optimisation Optuna et logging MLflow |
| `evaluate` | `make evaluate` | Evaluation des metriques et calcul de l'elasticite      |
| `optimize` | `make optimize` | Generation des recommandations de prix optimaux         |

### Features creees

| Feature                    | Description                                    |
| -------------------------- | ---------------------------------------------- |
| **price_synthetic**  | Prix synthetique genere a partir des quantiles |
| **day_of_week**      | Jour de la semaine de la transaction           |
| **month**            | Mois de la transaction                         |
| **is_weekend**       | Indicateur de week-end                         |
| **category_encoded** | Encodage des categories de produits            |

---

## Modeles Utilises

Le pipeline teste automatiquement plusieurs algorithmes et selectionne le meilleur :

| Modele             | Description                                                   |
| ------------------ | ------------------------------------------------------------- |
| **LightGBM** | Gradient boosting optimise pour la vitesse (modele principal) |
| ElasticNet         | Modele de reference lineaire avec regularisation              |

L'optimisation des hyperparametres est realisee par **Optuna** avec validation croisee.

### Metriques de Performance

| Metrique | Baseline (ElasticNet) | LightGBM |
| -------- | --------------------- | -------- |
| RMSE     | 100.67                | 77.59    |
| MAE      | 38.50                 | 29.85    |
| R²      | 0.65                  | 0.78     |

---

## Deploiement Local avec Docker

### Lancer les services

```bash
# Construire les images
docker compose build

# Demarrer l'infrastructure
docker compose up -d
```

### Services disponibles

| Service     | URL                        | Description                       |
| ----------- | -------------------------- | --------------------------------- |
| MLflow UI   | http://localhost:5000      | Suivi des experiences et registre |
| API FastAPI | http://localhost:8000/docs | Documentation interactive Swagger |
| Streamlit   | http://localhost:8501      | Interface utilisateur graphique   |

### Arreter les services

```bash
docker compose down
```

---

## Utilisation de l'API

### Endpoint de recommandation

`POST /recommend_price`

### Exemple de requete

```json
{
  "product_id": "GROCERY_I_1",
  "current_price": 4.50,
  "current_volume": 100,
  "constraints": {
    "min_price": 3.50,
    "max_price": 6.00,
    "max_change": 0.20
  }
}
```

### Reponse

```json
{
  "recommended_price": 4.15,
  "price_change_pct": -7.8,
  "expected_volume": 1250.5,
  "expected_revenue": 5189.58,
  "expected_margin": 1297.39,
  "revenue_uplift_pct": 15.2,
  "model_version": "1.2.0"
}
```

### Endpoint de simulation

`POST /simulate`

Permet de simuler l'impact de plusieurs variations de prix sur un produit.

---

## Interface Streamlit

L'application Streamlit permet de :

| Onglet                   | Fonctionnalite                                               |
| ------------------------ | ------------------------------------------------------------ |
| **Recommandation** | Obtenir le prix optimal en un clic                           |
| **Simulation**     | Visualiser l'impact de differents prix sur revenus et ventes |
| **Sensibilite**    | Analyser l'elasticite-prix du produit                        |

Consultez le [Guide Utilisateur](GUIDE_UTILISATEUR.md) pour une visite guidee complete.

---

## Tests et Qualite du Code

### Executer les tests

```bash
poetry run pytest tests/ -v
```

### Avec couverture

```bash
poetry run pytest tests/ --cov=src --cov-report=html
```

### Verifier la conformite du code

```bash
# Linting
make lint

# Formatage
make format

# Verification des types
make typecheck
```

---

## Integration Continue

Les workflows **GitHub Actions** automatisent :

| Workflow   | Description                                             |
| ---------- | ------------------------------------------------------- |
| Linting    | Analyse statique avec Ruff et MyPy                      |
| Tests      | Execution des tests unitaires avec Pytest               |
| Build      | Construction et publication des images Docker           |
| Keep-Alive | Ping regulier pour maintenir les services Render actifs |

---

## Deploiement Cloud

Le projet est configure pour un deploiement sur **Render** via Blueprint.

### Configuration MLflow distant (optionnel)

Pour connecter MLflow a un backend cloud, definissez les variables dans `.env` :

```env
MLFLOW_TRACKING_URI=https://votre-mlflow-server.com
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
```

---

## Commandes Utiles

| Commande                 | Description                           |
| ------------------------ | ------------------------------------- |
| `dvc repro`            | Executer le pipeline complet          |
| `make api`             | Lancer l'API FastAPI en local         |
| `make ui`              | Lancer l'interface Streamlit en local |
| `make lint`            | Verifier la qualite du code           |
| `make test`            | Executer les tests unitaires          |
| `docker compose up -d` | Demarrer tous les services Docker     |
| `docker compose down`  | Arreter les services Docker           |
| `poetry run mlflow ui` | Lancer l'interface MLflow locale      |

---

## Contribution

Les contributions sont bienvenues. Pour contribuer :

1. Forkez le repository
2. Creez une branche (`git checkout -b feature/NouvelleFeature`)
3. Committez vos modifications (`git commit -m 'Ajout de NouvelleFeature'`)
4. Poussez la branche (`git push origin feature/NouvelleFeature`)
5. Ouvrez une Pull Request

Assurez-vous que les tests passent et que le code respecte les standards de formatage avant de soumettre.

---

## Contact

**Auteur** : [Souley225](https://github.com/Souley225)

**Repository** : [mlops-pricing-optimization](https://github.com/Souley225/mlops-pricing-optimization)

---

## Licence

Ce projet est distribue sous licence **MIT**. Voir le fichier `LICENSE` pour plus de details.

---

<p align="center">
  <sub>Projet MLOps d'optimisation des prix basee sur l'estimation de la demande</sub>
</p>
