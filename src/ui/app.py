"""Interface Streamlit pour l'optimisation des prix."""

import os
import time
from typing import Any

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ----- Configuration de la page -----
st.set_page_config(
    page_title="Optimisation des Prix",
    page_icon="shopping_trolley",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----- Bloc de styles CSS -----
st.markdown("""
<style>
/* ===== NEW EDITORIAL LUXURY CSS ===== */
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Playfair+Display:ital,wght@0,400;0,600;0,700;0,800;1,400;1,600&display=swap');

:root {
    --bg-primary: #FAF8F5;
    --bg-card: #FFFFFF;
    --text-primary: #111111;
    --text-primary-input: #111111;
    --text-secondary: #4A4A4A;
    --text-muted: #737373;
    --text-inverse: #FAF8F5;
    
    --primary-dark: #163821;
    --accent-crimson: #8A1C1C;
    --border-color: #111111;
    --border-light: rgba(17, 17, 17, 0.15);
    
    --font-display: 'Playfair Display', serif;
    --font-sans: 'Outfit', sans-serif;
}

/* Base Resets */
*, *::before, *::after { box-sizing: border-box; }
html { scroll-behavior: smooth; }

.stApp {
    font-family: var(--font-sans) !important;
    background: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

/* Hide Streamlit default elements */
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }

/* Custom Header Header */
.slim-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: transparent;
    padding: 16px 0;
    border-bottom: 2px solid var(--border-color);
    margin-bottom: 32px;
}

.slim-header-left { display: flex; flex-direction: column; gap: 4px; }

.slim-header-title {
    font-family: var(--font-display);
    font-size: 2rem;
    font-weight: 800;
    color: var(--text-primary);
    font-style: italic;
    letter-spacing: -0.02em;
    line-height: 1;
}

.slim-header-sub {
    font-family: var(--font-sans);
    font-size: 0.75rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.15em;
    font-weight: 600;
}

.slim-header-right { display: flex; align-items: center; gap: 8px; }

.status-dot {
    width: 10px; height: 10px;
    border-radius: 0;
    display: inline-block;
    border: 1px solid var(--border-color);
}

.status-label {
    font-family: var(--font-sans);
    font-size: 0.75rem;
    color: var(--text-primary);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 700;
}

/* Typographic Overrides */
h1, h2, h3, h4, h5, h6 { 
    font-family: var(--font-display) !important; 
    color: var(--text-primary) !important; 
    font-weight: 700 !important; 
}

/* Custom Headings */
.section-label {
    font-family: var(--font-display) !important;
    font-size: 1.4rem !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
    font-style: italic !important;
    margin: 0 0 16px 0 !important;
    padding-bottom: 8px !important;
    border-bottom: 1px solid var(--border-light) !important;
}

.form-section-label {
    font-family: var(--font-sans) !important;
    font-size: 0.75rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    color: var(--text-primary) !important;
    margin: 0 0 16px 0 !important;
}

/* Result Cards */
.result-card {
    background: transparent;
    border: 1px solid var(--border-color);
    padding: 32px;
    margin-top: 24px;
    border-radius: 0;
    position: relative;
}

.result-card::before {
    content: '';
    position: absolute;
    top: -1px; left: -1px; width: 6px; height: calc(100% + 2px);
    background: var(--text-primary);
}

.result-label {
    font-family: var(--font-sans);
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: var(--text-secondary);
    margin: 0 0 12px 0;
}

.result-price-row { display: flex; align-items: baseline; gap: 16px; flex-wrap: wrap; }

.result-price {
    font-family: var(--font-display);
    font-size: 4rem;
    font-weight: 800;
    color: var(--text-primary);
    line-height: 1;
    letter-spacing: -0.04em;
}

.result-currency {
    font-family: var(--font-sans);
    font-size: 1.2rem;
    font-weight: 400;
    color: var(--text-secondary);
    vertical-align: super;
}

.result-badge {
    display: inline-flex; align-items: center;
    padding: 6px 12px;
    border: 1px solid var(--border-color);
    font-family: var(--font-sans);
    font-size: 0.85rem; font-weight: 600;
    border-radius: 0;
}

.result-badge-up { background: var(--text-primary); color: var(--text-inverse); }
.result-badge-down { background: transparent; color: var(--accent-crimson); border-color: var(--accent-crimson); }
.result-badge-flat { background: transparent; color: var(--text-secondary); }

/* Elasticity Cards */
.elasticity-card {
    border: 1px solid var(--border-color);
    padding: 24px;
    margin-top: 16px;
    border-radius: 0;
}
.elasticity-sensitive { border-left: 6px solid var(--accent-crimson); }
.elasticity-inelastic { border-left: 6px solid var(--primary-dark); }

.elasticity-title { font-family: var(--font-display); font-weight: 700; font-size: 1.4rem; font-style: italic; margin: 0 0 12px 0; color: var(--text-primary); }
.elasticity-body { font-family: var(--font-sans); line-height: 1.6; font-size: 0.95rem; margin: 0 0 16px 0; color: var(--text-secondary); }
.elasticity-value { font-family: var(--font-sans); font-size: 0.85rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-primary); }

/* Buttons */
.stButton > button {
    font-family: var(--font-sans) !important;
    min-height: 54px !important;
    font-size: 0.9rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    border-radius: 0 !important;
    border: 1px solid var(--border-color) !important;
    transition: all 0.2s ease !important;
    position: relative;
}

.stButton > button[kind="primary"] {
    background: var(--text-primary) !important;
    color: var(--text-inverse) !important;
}

.stButton > button[kind="primary"]:hover {
    box-shadow: 6px 6px 0px var(--accent-crimson) !important;
    transform: translate(-3px, -3px) !important;
}

.stButton > button[kind="secondary"] {
    background: transparent !important;
    color: var(--text-primary) !important;
}

.stButton > button[kind="secondary"]:hover {
    box-shadow: 6px 6px 0px var(--text-primary) !important;
    transform: translate(-3px, -3px) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 0; background: transparent; border-bottom: 2px solid var(--border-light); padding: 0; border-radius: 0;
}

.stTabs [data-baseweb="tab"] {
    font-family: var(--font-sans) !important;
    min-height: 48px !important;
    padding: 12px 24px !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    color: var(--text-secondary) !important;
    border-radius: 0 !important;
    border: none !important;
    background: transparent !important;
}

.stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: var(--text-primary) !important;
    border-bottom: 2px solid var(--text-primary) !important;
    background: transparent !important;
}

.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none !important; }

/* Metrics */
[data-testid="stMetric"] {
    background: transparent;
    border: none;
    border-bottom: 1px solid var(--border-color);
    padding: 16px 0 !important;
    border-radius: 0;
    box-shadow: none !important;
    margin-bottom: 12px;
}

[data-testid="stMetricLabel"] {
    font-family: var(--font-sans) !important;
    font-size: 0.75rem !important;
    color: var(--text-secondary) !important;
    font-weight: 600 !important;
    text-transform: uppercase; letter-spacing: 0.08em;
}

[data-testid="stMetricValue"] {
    font-family: var(--font-display) !important;
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    color: var(--text-primary) !important;
}

[data-testid="stMetricDelta"] > div { font-family: var(--font-sans) !important; font-weight: 600 !important; }

/* Form Inputs Solid Styling */
div[data-baseweb="input"],
div[data-baseweb="base-input"],
div[data-baseweb="input"] > input,
div[data-baseweb="base-input"] > input {
    background-color: #FFFFFF !important;
    color: #111111 !important;
    -webkit-text-fill-color: #111111 !important;
    caret-color: #8A1C1C !important;
    border-radius: 0 !important;
}

div[data-baseweb="input"] input,
div[data-baseweb="base-input"] input {
    background-color: #FFFFFF !important;
    color: #111111 !important;
    -webkit-text-fill-color: #111111 !important;
    font-family: var(--font-sans) !important;
    font-size: 1rem !important;
}

.stNumberInput > div > div > div, 
.stTextInput > div > div > div,
.stSelectbox > div > div > div {
    border: 1px solid #111111 !important;
    border-radius: 0 !important;
    background-color: #FFFFFF !important;
    box-shadow: 4px 4px 0px rgba(0,0,0,0.05) !important;
}

.stNumberInput > div > div > div:focus-within, 
.stTextInput > div > div > div:focus-within,
.stSelectbox > div > div > div:focus-within {
    border: 1px solid #8A1C1C !important;
    box-shadow: 4px 4px 0px #8A1C1C !important;
}

label, [data-testid="stCheckbox"] label span, [data-testid="stCheckbox"] label p {
    font-family: var(--font-sans) !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    color: var(--text-primary) !important;
}

/* Slider Overrides - Minimal */
.stSlider > div > div > div { background: var(--text-primary) !important; height: 2px !important; }
.stSlider [data-baseweb="slider"] [role="slider"] {
    background: var(--bg-primary) !important;
    border: 2px solid var(--text-primary) !important;
    box-shadow: none !important;
    border-radius: 0 !important;
    width: 16px !important; height: 16px !important;
}

/* Expander/Accordion */
.streamlit-expanderHeader {
    font-family: var(--font-display) !important;
    font-size: 1.1rem !important;
    font-style: italic !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 1px solid var(--border-light) !important;
    border-radius: 0 !important;
    padding: 16px 0 !important;
}
.streamlit-expanderContent {
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    padding: 16px 0 !important;
}

/* Alerts */
.stSuccess, .stError, .stInfo {
    background: transparent !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 0 !important;
    box-shadow: 4px 4px 0px rgba(0,0,0,0.05) !important;
}
.stSuccess { border-left: 4px solid var(--primary-dark) !important; }
.stSuccess p, .stSuccess span { color: var(--text-primary) !important; font-family: var(--font-sans) !important; }
.stError { border-left: 4px solid var(--accent-crimson) !important; }
.stError p, .stError span { color: var(--accent-crimson) !important; }
.stInfo { border-left: 4px solid var(--text-secondary) !important; }
.stInfo p, .stInfo span { color: var(--text-primary) !important; }

/* Misc Separators */
hr { border: none !important; border-top: 1px solid var(--border-light) !important; margin: 32px 0 !important; }

/* Sidebar styling for editorial look */
[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid var(--border-color) !important;
}
</style>
""", unsafe_allow_html=True)

# ----- URL de l'API -----
API_URL = os.getenv("API_URL", "http://localhost:8000")


def check_api_health() -> bool:
    """Verifie la disponibilite de l'API."""
    try:
        response = httpx.get(f"{API_URL}/health", timeout=5.0)
        return response.status_code == 200
    except Exception:
        return False


def wake_up_api(timeout: int = 90) -> bool:
    """Tente de reveiller l'API si elle est en veille."""
    if st.session_state.get("api_healthy", False):
        return True

    if check_api_health():
        st.session_state.api_healthy = True
        return True

    message = st.empty()
    barre = st.progress(0)
    message.warning("Connexion a l'API en cours...")

    debut = time.time()
    while time.time() - debut < timeout:
        elapsed = time.time() - debut
        barre.progress(min(elapsed / 60, 0.9))
        if check_api_health():
            barre.empty()
            message.empty()
            st.session_state.api_healthy = True
            return True
        time.sleep(2)

    barre.empty()
    message.empty()
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
            st.error(f"Erreur API : {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Connexion impossible : {e}")
        return None


def get_price_simulation(
    product_id: str,
    current_price: float,
    variations: list[float],
) -> dict[str, Any] | None:
    """Appelle l'API pour simuler differents niveaux de prix."""
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
        return None
    except Exception:
        return None


def render_header() -> None:
    """Affiche l'en-tete compact avec le statut de l'API."""
    api_ok = wake_up_api()
    couleur_statut = "#06D6A0" if api_ok else "#D62828"
    libelle_statut = "Systeme pret" if api_ok else "Systeme indisponible"

    st.markdown(f"""
    <div class="slim-header">
        <div class="slim-header-left">
            <span class="slim-header-title">Optimisation des Prix</span>
            <span class="slim-header-sub">Outil d'aide a la decision tarifaire</span>
        </div>
        <div class="slim-header-right">
            <span class="status-dot" style="background:{couleur_statut};"></span>
            <span class="status-label">{libelle_statut}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_params_panel() -> dict[str, Any]:
    """Affiche le formulaire de configuration du produit."""
    st.markdown('<p class="section-label">Caracteristiques du produit</p>', unsafe_allow_html=True)
    st.divider()

    col_gauche, col_droite = st.columns(2, gap="large")

    with col_gauche:
        st.markdown('<p class="form-section-label">Informations du produit</p>', unsafe_allow_html=True)

        product_id = st.text_input(
            "Reference du produit",
            value="GROCERY_I_1",
            help="Code unique du produit (ex : SKU, reference interne)",
        )

        current_price = st.number_input(
            "Prix actuel (EUR)",
            min_value=0.01,
            max_value=1000.0,
            value=4.50,
            step=0.10,
            format="%.2f",
            help="Prix de vente actuel du produit",
        )

        current_volume = st.number_input(
            "Ventes par semaine",
            min_value=0.0,
            max_value=100000.0,
            value=100.0,
            step=10.0,
            help="Nombre moyen d'unites vendues par semaine",
        )

    with col_droite:
        st.markdown('<p class="form-section-label">Contraintes de prix</p>', unsafe_allow_html=True)

        use_constraints = st.checkbox(
            "Activer les limites de prix",
            value=False,
            help="Definir un prix minimum et maximum acceptables",
        )

        constraints = None
        if use_constraints:
            min_price = st.number_input(
                "Prix minimum (EUR)",
                min_value=0.01,
                max_value=current_price,
                value=round(current_price * 0.7, 2),
                step=0.10,
                format="%.2f",
            )

            max_price = st.number_input(
                "Prix maximum (EUR)",
                min_value=current_price,
                max_value=current_price * 2,
                value=round(current_price * 1.3, 2),
                step=0.10,
                format="%.2f",
            )

            max_change = st.slider(
                "Variation maximale (%)",
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

    st.divider()

    resultat = {
        "product_id":    product_id,
        "current_price": current_price,
        "current_volume": current_volume,
        "constraints":   constraints,
    }
    st.session_state["params_cache"] = resultat
    return resultat


def render_recommendation(params: dict[str, Any]) -> None:
    """Affiche la recommandation de prix optimale."""
    # Initialisation du state de session
    if "recommendation_result" not in st.session_state:
        st.session_state.recommendation_result = None
    if "recommendation_params" not in st.session_state:
        st.session_state.recommendation_params = None

    cle_params = (
        f"{params['product_id']}_"
        f"{params['current_price']}_"
        f"{params['current_volume']}_"
        f"{params['constraints']}"
    )

    # Reinitialisation si les parametres ont change
    if st.session_state.recommendation_params != cle_params:
        st.session_state.recommendation_result = None

    st.markdown('<p class="section-label">Recommandation de prix</p>', unsafe_allow_html=True)

    # Bouton d'action principal — toujours visible sans scroll
    if st.button("Calculer le prix optimal", type="primary", use_container_width=True):
        with st.spinner("Calcul en cours..."):
            calcul = get_price_recommendation(
                product_id=params["product_id"],
                current_price=params["current_price"],
                current_volume=params["current_volume"],
                constraints=params["constraints"],
            )
        st.session_state.recommendation_result = calcul
        st.session_state.recommendation_params = cle_params

    # Affichage du resultat
    calcul = st.session_state.recommendation_result
    if calcul:
        variation = calcul["price_change_pct"]

        if variation > 0:
            classe_badge = "result-badge-up"
            signe = "+"
        elif variation < 0:
            classe_badge = "result-badge-down"
            signe = ""
        else:
            classe_badge = "result-badge-flat"
            signe = ""

        st.markdown(f"""
        <div class="result-card">
            <p class="result-label">Prix recommande</p>
            <div class="result-price-row">
                <span class="result-price">
                    {calcul['recommended_price']:.2f}
                    <span class="result-currency">EUR</span>
                </span>
                <span class="result-badge {classe_badge}">
                    {signe}{variation:+.1f}%
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Metriques secondaires
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Revenu attendu",
                f"{calcul['expected_revenue']:.2f} EUR",
                f"{calcul['revenue_uplift_pct']:+.1f}%",
            )

        with col2:
            st.metric(
                "Ventes prevues",
                f"{calcul['expected_volume']:.0f} unites",
            )

        with col3:
            st.metric(
                "Marge attendue",
                f"{calcul['expected_margin']:.2f} EUR",
            )

        # Detail techniques discret
        with st.expander("Detail technique"):
            st.caption(f"Version du modele : {calcul['model_version']}")
            st.caption(f"Produit analyse : {params['product_id']}")


def render_simulation(params: dict[str, Any]) -> None:
    """Affiche la simulation de scenarios de prix."""
    st.markdown('<p class="section-label">Simulation de scenarios</p>', unsafe_allow_html=True)
    st.divider()

    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        n_points = st.slider(
            "Nombre de prix testes",
            min_value=5,
            max_value=21,
            value=11,
            step=2,
            help="Plus la valeur est elevee, plus l'analyse est precise",
        )

    with col_b:
        max_var = st.slider(
            "Amplitude des variations (%)",
            min_value=10,
            max_value=50,
            value=30,
            step=5,
            help="Ecart maximal par rapport au prix actuel",
        )

    prix_min_sim = params["current_price"] * (1 - max_var / 100)
    prix_max_sim = params["current_price"] * (1 + max_var / 100)
    st.caption(
        f"Plage testee : {prix_min_sim:.2f} EUR — {prix_max_sim:.2f} EUR  ({n_points} points)"
    )

    variations = [
        round(-max_var / 100 + i * (2 * max_var / 100) / (n_points - 1), 3)
        for i in range(n_points)
    ]

    if st.button("Lancer la simulation", type="primary", use_container_width=True):
        with st.spinner("Simulation en cours..."):
            sim = get_price_simulation(
                product_id=params["product_id"],
                current_price=params["current_price"],
                variations=variations,
            )

        if sim and sim.get("simulations"):
            df = pd.DataFrame(sim["simulations"])
            idx_optimal = df["expected_revenue"].idxmax()
            prix_optimal = df.loc[idx_optimal, "price"]
            revenu_max = df.loc[idx_optimal, "expected_revenue"]

            st.success(
                f"Prix optimal : {prix_optimal:.2f} EUR — "
                f"Revenu maximal : {revenu_max:.2f} EUR"
            )

            # Graphique interactif
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=df["price"],
                y=df["expected_revenue"],
                mode="lines+markers",
                name="Revenu (EUR)",
                line={"color": "#111111", "width": 3, "shape": "spline"},
                marker={
                    "size": 10,
                    "color": "#111111",
                    "line": {"color": "#FFFFFF", "width": 2},
                },
                hovertemplate=(
                    "<b>Prix :</b> %{x:.2f} EUR<br>"
                    "<b>Revenu :</b> %{y:.2f} EUR"
                    "<extra></extra>"
                ),
            ))

            fig.add_trace(go.Scatter(
                x=df["price"],
                y=df["expected_volume"],
                mode="lines+markers",
                name="Ventes (unites)",
                yaxis="y2",
                line={"color": "#8A1C1C", "width": 3, "dash": "dot", "shape": "spline"},
                marker={
                    "size": 10,
                    "color": "#8A1C1C",
                    "symbol": "diamond",
                    "line": {"color": "#FFFFFF", "width": 2},
                },
                hovertemplate=(
                    "<b>Prix :</b> %{x:.2f} EUR<br>"
                    "<b>Ventes :</b> %{y:.0f} unites"
                    "<extra></extra>"
                ),
            ))

            # Ligne du prix actuel
            fig.add_vline(
                x=params["current_price"],
                line_dash="dash",
                line_color="rgba(17,17,17,0.3)",
                line_width=2,
                annotation_text="Actuel",
                annotation_font_color="#111111",
            )

            # Ligne du prix optimal
            fig.add_vline(
                x=prix_optimal,
                line_dash="solid",
                line_color="#8A1C1C",
                line_width=3,
                annotation_text="Optimal",
                annotation_font_color="#8A1C1C",
                annotation_font_size=12,
                annotation_font_weight=600,
            )

            fig.update_layout(
                xaxis_title="Prix (EUR)",
                yaxis_title="Revenu (EUR)",
                xaxis={
                    "gridcolor": "rgba(0,0,0,0.05)",
                    "tickfont": {"family": "Outfit, sans-serif", "size": 11, "color": "#718096"},
                    "title_font": {"family": "Outfit, sans-serif", "size": 13, "color": "#111111"},
                },
                yaxis={
                    "gridcolor": "rgba(0,0,0,0.05)",
                    "title_font": {"family": "Outfit, sans-serif", "size": 13, "color": "#111111"},
                    "tickfont": {"family": "Outfit, sans-serif", "size": 11, "color": "#718096"},
                },
                yaxis2={
                    "title": "Ventes",
                    "overlaying": "y",
                    "side": "right",
                    "title_font": {"family": "Outfit, sans-serif", "size": 13, "color": "#8A1C1C"},
                    "tickfont": {"family": "Outfit, sans-serif", "size": 11, "color": "#718096"},
                },
                legend={
                    "orientation": "h",
                    "yanchor": "bottom",
                    "y": 1.02,
                    "xanchor": "center",
                    "x": 0.5,
                    "font": {"family": "Outfit, sans-serif", "size": 12},
                    "bgcolor": "rgba(255,255,255,0.9)",
                    "bordercolor": "rgba(0,0,0,0.08)",
                    "borderwidth": 1,
                },
                height=380,
                margin={"l": 60, "r": 60, "t": 40, "b": 60},
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                hoverlabel={
                    "bgcolor": "#0A2463",
                    "font_size": 13,
                    "font_family": "Inter, sans-serif",
                    "font_color": "white",
                },
            )

            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Tableau des resultats"):
                df_aff = df.copy()
                df_aff["price_variation"] = df_aff["price_variation"].apply(
                    lambda x: f"{x:+.1%}"
                )
                df_aff.columns = ["Variation", "Prix", "Ventes", "Revenu", "Marge"]
                st.dataframe(df_aff, use_container_width=True)


def render_elasticity_analysis(params: dict[str, Any]) -> None:
    """Affiche l'analyse de sensibilite au prix."""
    st.markdown('<p class="section-label">Sensibilite au prix</p>', unsafe_allow_html=True)
    st.divider()

    with st.spinner("Calcul de la sensibilite..."):
        variations = [-0.05, 0, 0.05]
        sim = get_price_simulation(
            product_id=params["product_id"],
            current_price=params["current_price"],
            variations=variations,
        )

    if sim and len(sim.get("simulations", [])) >= 3:
        sims = sim["simulations"]

        q_bas  = sims[0]["expected_volume"]
        q_haut = sims[2]["expected_volume"]
        p_bas  = sims[0]["price"]
        p_haut = sims[2]["price"]

        dq    = q_haut - q_bas
        dp    = p_haut - p_bas
        q_moy = (q_haut + q_bas) / 2
        p_moy = (p_haut + p_bas) / 2

        elasticite = (dq / q_moy) / (dp / p_moy) if dp != 0 and q_moy != 0 else 0

        st.metric(
            "Indice de sensibilite",
            f"{elasticite:.2f}",
            help=(
                "Un indice negatif indique que les ventes diminuent "
                "quand le prix augmente — attendu pour la plupart des produits."
            ),
        )

        st.markdown("---")

        if abs(elasticite) > 1:
            st.markdown(f"""
            <div class="elasticity-card elasticity-sensitive">
                <p class="elasticity-title">Forte sensibilite au prix</p>
                <p class="elasticity-body">
                    Une variation de prix entraine une variation proportionnellement
                    plus importante des ventes. Une prudence particuliere est
                    recommandee avant toute hausse tarifaire.
                </p>
                <p class="elasticity-value">Elasticite mesuree : {elasticite:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="elasticity-card elasticity-inelastic">
                <p class="elasticity-title">Faible sensibilite au prix</p>
                <p class="elasticity-body">
                    Les volumes de vente reagissent peu aux variations de prix.
                    Une marge de manoeuvre tarifaire a la hausse est envisageable.
                </p>
                <p class="elasticity-value">Elasticite mesuree : {elasticite:.2f}</p>
            </div>
            """, unsafe_allow_html=True)

        with st.expander("Methode de calcul"):
            st.markdown("""
            **Elasticite-prix** = variation relative des ventes / variation relative du prix

            - Elasticite < -1 : les ventes diminuent plus vite que le prix augmente
            - Elasticite entre -1 et 0 : les ventes diminuent moins vite que le prix augmente
            - Elasticite proche de 0 : les ventes sont peu sensibles au prix
            """)
    else:
        st.info("Donnees insuffisantes pour calculer la sensibilite.")


def render_sidebar():
    """Affiche la barre latérale avec le contexte du projet et les liens externes."""
    st.sidebar.markdown("""
    <h3 style="font-family: var(--font-sans); color: var(--text-primary); font-weight: 700; font-size: 1.1rem; margin-top: -10px;">À propos du projet</h3>
    
    <p style="font-family: var(--font-sans); color: var(--text-secondary); font-size: 0.95rem; line-height: 1.5; margin-bottom: 20px;">
    Cette application est une solution d'<b>optimisation de prix</b> de nouvelle génération. Basée sur l'estimation de l'élasticité de la demande, elle fournit le <b>prix de vente optimal</b> pour maximiser le chiffre d'affaires et la rentabilité.
    </p>
    
    <h3 style="font-family: var(--font-sans); color: var(--text-primary); font-weight: 700; font-size: 1.0rem; margin-top: 10px; margin-bottom: 12px;">Stack Technique</h3>
    <div style="display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 20px;">
        <span style="background: transparent; color: var(--text-primary); padding: 4px 10px; border-radius: 6px; font-size: 0.75rem; font-weight: 700; border: 1px solid var(--border-color);">Python</span>
        <span style="background: transparent; color: var(--text-primary); padding: 4px 10px; border-radius: 6px; font-size: 0.75rem; font-weight: 700; border: 1px solid var(--border-color);">Streamlit</span>
        <span style="background: transparent; color: var(--text-primary); padding: 4px 10px; border-radius: 6px; font-size: 0.75rem; font-weight: 700; border: 1px solid var(--border-color);">FastAPI</span>
        <span style="background: transparent; color: var(--text-primary); padding: 4px 10px; border-radius: 6px; font-size: 0.75rem; font-weight: 700; border: 1px solid var(--border-color);">MLflow</span>
        <span style="background: transparent; color: var(--text-primary); padding: 4px 10px; border-radius: 6px; font-size: 0.75rem; font-weight: 700; border: 1px solid var(--border-color);">DVC</span>
        <span style="background: transparent; color: var(--text-primary); padding: 4px 10px; border-radius: 6px; font-size: 0.75rem; font-weight: 700; border: 1px solid var(--border-color);">Optuna</span>
        <span style="background: transparent; color: var(--text-primary); padding: 4px 10px; border-radius: 6px; font-size: 0.75rem; font-weight: 700; border: 1px solid var(--border-color);">Docker</span>
    </div>
    
    <hr style="background: var(--border-light); border: none; height: 1px; margin: 20px 0;">
    
    <div class="ext-links" style="display: flex; flex-direction: column; gap: 10px;">
        <a href="https://www.linkedin.com/in/souleymanes-sall" target="_blank" style="width: 100%; justify-content: center; background: transparent;">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
                <path d="M19 0h-14c-2.761 0-5 2.239-5 5v14c0 2.761 2.239 5 5 5h14c2.762 0 5-2.239 5-5v-14c0-2.761-2.238-5-5-5zm-11 19h-3v-11h3v11zm-1.5-12.268c-.966 0-1.75-.79-1.75-1.764s.784-1.764 1.75-1.764 1.75.79 1.75 1.764-.783 1.764-1.75 1.764zm13.5 12.268h-3v-5.604c0-3.368-4-3.113-4 0v5.604h-3v-11h3v1.765c1.396-2.586 7-2.777 7 2.476v6.759z"/>
            </svg>
            LinkedIn
        </a>
        <a href="https://github.com/Souley225/Pricing_Optimisation_Project" target="_blank" style="width: 100%; justify-content: center; background: transparent;">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
                <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
            </svg>
            Code source
        </a>
    </div>
    """, unsafe_allow_html=True)


def main() -> None:
    """Point d'entree de l'application."""
    render_sidebar()
    render_header()

    tab1, tab2, tab3 = st.tabs([
        "Recommandation",
        "Simulation",
        "Sensibilite",
    ])

    with tab1:
        params = render_params_panel()
        render_recommendation(params)

    with tab2:
        params = st.session_state.get("params_cache", {
            "product_id":    "GROCERY_I_1",
            "current_price": 4.50,
            "current_volume": 100.0,
            "constraints":   None,
        })
        render_simulation(params)

    with tab3:
        params = st.session_state.get("params_cache", {
            "product_id":    "GROCERY_I_1",
            "current_price": 4.50,
            "current_volume": 100.0,
            "constraints":   None,
        })
        render_elasticity_analysis(params)


if __name__ == "__main__":
    main()
