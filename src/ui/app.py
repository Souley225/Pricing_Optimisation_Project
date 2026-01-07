"""Interface Streamlit pour l'optimisation des prix."""

import os
from typing import Any

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Configuration de la page
st.set_page_config(
    page_title="Pricing Optimization",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Configuration API
API_URL = os.getenv("API_URL", "http://localhost:8000")


def check_api_health() -> bool:
    """Verifie que l'API est disponible."""
    try:
        response = httpx.get(f"{API_URL}/health", timeout=5.0)
        return response.status_code == 200
    except Exception:
        return False


def get_price_recommendation(
    product_id: str,
    current_price: float,
    current_volume: float,
    constraints: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Appelle l'API pour obtenir une recommandation de prix."""
    try:
        payload = {
            "product_id": product_id,
            "current_price": current_price,
            "current_volume": current_volume,
        }
        if constraints:
            payload["constraints"] = constraints

        response = httpx.post(
            f"{API_URL}/recommend_price",
            json=payload,
            timeout=10.0,
        )

        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Erreur API: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Erreur de connexion: {e}")
        return None


def get_price_simulation(
    product_id: str,
    current_price: float,
    variations: list[float],
) -> dict[str, Any] | None:
    """Appelle l'API pour simuler des prix."""
    try:
        response = httpx.post(
            f"{API_URL}/simulate",
            json={
                "product_id": product_id,
                "current_price": current_price,
                "price_variations": variations,
            },
            timeout=10.0,
        )

        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception:
        return None


def render_header() -> None:
    """Affiche l'entete de l'application."""
    st.title("Optimisation des Prix")
    st.markdown("**Systeme de recommandation de prix base sur l'estimation de la demande**")

    # Status de l'API
    api_status = check_api_health()
    if api_status:
        st.sidebar.success("API connectee")
    else:
        st.sidebar.error("API non disponible")


def render_sidebar() -> dict[str, Any]:
    """Affiche la barre laterale et retourne les parametres."""
    st.sidebar.header("Parametres")

    product_id = st.sidebar.text_input(
        "ID Produit",
        value="GROCERY_I_1",
        help="Identifiant unique du produit",
    )

    current_price = st.sidebar.number_input(
        "Prix actuel",
        min_value=0.01,
        max_value=1000.0,
        value=4.50,
        step=0.10,
        format="%.2f",
    )

    current_volume = st.sidebar.number_input(
        "Volume actuel",
        min_value=0.0,
        max_value=100000.0,
        value=100.0,
        step=10.0,
    )

    st.sidebar.subheader("Contraintes")

    use_constraints = st.sidebar.checkbox("Appliquer des contraintes", value=True)

    constraints = None
    if use_constraints:
        min_price = st.sidebar.number_input(
            "Prix minimum",
            min_value=0.01,
            max_value=current_price,
            value=current_price * 0.7,
            step=0.10,
            format="%.2f",
        )

        max_price = st.sidebar.number_input(
            "Prix maximum",
            min_value=current_price,
            max_value=current_price * 2,
            value=current_price * 1.3,
            step=0.10,
            format="%.2f",
        )

        max_change = st.sidebar.slider(
            "Variation max (%)",
            min_value=5,
            max_value=50,
            value=20,
            step=5,
        )

        constraints = {
            "min_price": min_price,
            "max_price": max_price,
            "max_change": max_change / 100,
        }

    return {
        "product_id": product_id,
        "current_price": current_price,
        "current_volume": current_volume,
        "constraints": constraints,
    }


def render_recommendation(params: dict[str, Any]) -> None:
    """Affiche la recommandation de prix."""
    st.header("Recommandation de Prix")

    if st.button("Obtenir une recommandation", type="primary"):
        with st.spinner("Calcul en cours..."):
            result = get_price_recommendation(
                product_id=params["product_id"],
                current_price=params["current_price"],
                current_volume=params["current_volume"],
                constraints=params["constraints"],
            )

        if result:
            # Metriques principales
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Prix recommande",
                    f"{result['recommended_price']:.2f}",
                    f"{result['price_change_pct']:+.1f}%",
                )

            with col2:
                st.metric(
                    "Volume attendu",
                    f"{result['expected_volume']:.0f}",
                )

            with col3:
                st.metric(
                    "Revenu attendu",
                    f"{result['expected_revenue']:.2f}",
                    f"{result['revenue_uplift_pct']:+.1f}%",
                )

            with col4:
                st.metric(
                    "Marge attendue",
                    f"{result['expected_margin']:.2f}",
                )

            # Details
            st.info(f"Version du modele: {result['model_version']}")


def render_simulation(params: dict[str, Any]) -> None:
    """Affiche la simulation de prix."""
    st.header("Simulation de Scenarios")

    # Parametres de simulation
    col1, col2 = st.columns([1, 3])

    with col1:
        n_points = st.slider(
            "Nombre de points",
            min_value=5,
            max_value=21,
            value=11,
            step=2,
        )

        max_var = st.slider(
            "Variation max (%)",
            min_value=10,
            max_value=50,
            value=30,
            step=5,
        )

    variations = [
        round(-max_var / 100 + i * (2 * max_var / 100) / (n_points - 1), 3) for i in range(n_points)
    ]

    if st.button("Lancer la simulation"):
        with st.spinner("Simulation en cours..."):
            result = get_price_simulation(
                product_id=params["product_id"],
                current_price=params["current_price"],
                variations=variations,
            )

        if result and result.get("simulations"):
            df = pd.DataFrame(result["simulations"])

            # Graphique interactif
            fig = go.Figure()

            # Courbe de revenu
            fig.add_trace(
                go.Scatter(
                    x=df["price"],
                    y=df["expected_revenue"],
                    mode="lines+markers",
                    name="Revenu",
                    line={"color": "#2E86AB", "width": 3},
                    marker={"size": 8},
                )
            )

            # Courbe de volume
            fig.add_trace(
                go.Scatter(
                    x=df["price"],
                    y=df["expected_volume"],
                    mode="lines+markers",
                    name="Volume",
                    yaxis="y2",
                    line={"color": "#A23B72", "width": 3, "dash": "dot"},
                    marker={"size": 8},
                )
            )

            # Prix actuel
            fig.add_vline(
                x=params["current_price"],
                line_dash="dash",
                line_color="gray",
                annotation_text="Prix actuel",
            )

            fig.update_layout(
                title="Impact du Prix sur le Revenu et le Volume",
                xaxis_title="Prix",
                yaxis_title="Revenu",
                yaxis2={
                    "title": "Volume",
                    "overlaying": "y",
                    "side": "right",
                },
                legend={
                    "orientation": "h",
                    "yanchor": "bottom",
                    "y": 1.02,
                    "xanchor": "right",
                    "x": 1,
                },
                height=500,
            )

            st.plotly_chart(fig, use_container_width=True)

            # Tableau des resultats
            with st.expander("Voir les donnees detaillees"):
                df_display = df.copy()
                df_display["price_variation"] = df_display["price_variation"].apply(
                    lambda x: f"{x:+.1%}"
                )
                df_display.columns = ["Variation", "Prix", "Volume", "Revenu", "Marge"]
                st.dataframe(df_display, use_container_width=True)


def render_elasticity_analysis(params: dict[str, Any]) -> None:
    """Affiche l'analyse d'elasticite."""
    st.header("Analyse d'Elasticite")

    st.markdown("""
    L'elasticite prix de la demande mesure la sensibilite de la demande
    aux variations de prix. Une elasticite de -2 signifie qu'une hausse
    de prix de 1% entraine une baisse de demande de 2%.
    """)

    # Calcul de l'elasticite via simulation
    variations = [-0.05, 0, 0.05]
    result = get_price_simulation(
        product_id=params["product_id"],
        current_price=params["current_price"],
        variations=variations,
    )

    if result and len(result.get("simulations", [])) >= 3:
        sims = result["simulations"]

        # Elasticite = (dQ/Q) / (dP/P)
        q_low = sims[0]["expected_volume"]
        q_high = sims[2]["expected_volume"]
        p_low = sims[0]["price"]
        p_high = sims[2]["price"]

        dq = q_high - q_low
        dp = p_high - p_low
        q_avg = (q_high + q_low) / 2
        p_avg = (p_high + p_low) / 2

        elasticity = (dq / q_avg) / (dp / p_avg) if dp != 0 and q_avg != 0 else 0

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Elasticite estimee", f"{elasticity:.2f}")

        with col2:
            if abs(elasticity) > 1:
                st.info("Demande elastique - sensible au prix")
            else:
                st.info("Demande inelastique - peu sensible au prix")

        with col3:
            recommendation = "Baisser" if elasticity < -1 else "Maintenir ou augmenter"
            st.info(f"Strategie suggeree: {recommendation} le prix")


def main() -> None:
    """Point d'entree principal."""
    render_header()
    params = render_sidebar()

    # Tabs pour les differentes vues
    tab1, tab2, tab3 = st.tabs(
        [
            "Recommandation",
            "Simulation",
            "Elasticite",
        ]
    )

    with tab1:
        render_recommendation(params)

    with tab2:
        render_simulation(params)

    with tab3:
        render_elasticity_analysis(params)


if __name__ == "__main__":
    main()
