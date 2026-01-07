"""Tests pour l'API FastAPI."""

import pytest
from fastapi.testclient import TestClient

# Note: Ces tests necessitent un modele entraine
# Ils sont marques comme skip par defaut


@pytest.fixture
def client():
    """Client de test FastAPI."""
    from src.serving.api import app

    return TestClient(app)


class TestHealthEndpoint:
    """Tests pour l'endpoint /health."""

    def test_health_check(self, client: TestClient) -> None:
        """Verifie que l'endpoint health repond."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()

        assert "status" in data
        assert "model_loaded" in data
        assert "timestamp" in data
        assert "version" in data


@pytest.mark.skip(reason="Necessite un modele entraine")
class TestRecommendEndpoint:
    """Tests pour l'endpoint /recommend_price."""

    def test_recommend_price_basic(self, client: TestClient) -> None:
        """Verifie une recommandation basique."""
        response = client.post(
            "/recommend_price",
            json={
                "product_id": "TEST_1",
                "current_price": 10.0,
                "current_volume": 100.0,
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert "recommended_price" in data
        assert "expected_volume" in data
        assert "expected_revenue" in data
        assert "expected_margin" in data
        assert "model_version" in data

    def test_recommend_price_with_constraints(self, client: TestClient) -> None:
        """Verifie une recommandation avec contraintes."""
        response = client.post(
            "/recommend_price",
            json={
                "product_id": "TEST_1",
                "current_price": 10.0,
                "current_volume": 100.0,
                "constraints": {
                    "min_price": 8.0,
                    "max_price": 12.0,
                    "max_change": 0.15,
                },
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Le prix recommande doit respecter les contraintes
        assert 8.0 <= data["recommended_price"] <= 12.0

    def test_recommend_price_invalid_input(self, client: TestClient) -> None:
        """Verifie la validation des entrees."""
        response = client.post(
            "/recommend_price",
            json={
                "product_id": "TEST_1",
                "current_price": -5.0,  # Prix negatif invalide
            },
        )

        assert response.status_code == 422  # Validation error


@pytest.mark.skip(reason="Necessite un modele entraine")
class TestSimulateEndpoint:
    """Tests pour l'endpoint /simulate."""

    def test_simulate_basic(self, client: TestClient) -> None:
        """Verifie une simulation basique."""
        response = client.post(
            "/simulate",
            json={
                "product_id": "TEST_1",
                "current_price": 10.0,
                "price_variations": [-0.1, 0, 0.1],
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert "product_id" in data
        assert "current_price" in data
        assert "simulations" in data
        assert len(data["simulations"]) == 3
