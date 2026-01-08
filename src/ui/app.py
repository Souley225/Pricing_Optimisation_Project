"""Interface Streamlit pour l'optimisation des prix."""

import os
import time
from typing import Any

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Configuration de la page - collapsed sidebar for mobile
st.set_page_config(
    page_title="Pricing Optimization",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================================
# MOBILE-FIRST CSS STYLES
# ============================================================================
st.markdown("""
<style>
/* ===== Base Mobile Styles ===== */
.stApp {
    max-width: 100%;
}

/* Larger touch targets for mobile */
.stButton > button {
    min-height: 50px !important;
    font-size: 1.1rem !important;
    border-radius: 12px !important;
    margin: 8px 0 !important;
    width: 100% !important;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #2E86AB 0%, #1a5276 100%) !important;
    border: none !important;
    box-shadow: 0 4px 12px rgba(46, 134, 171, 0.3) !important;
}

.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 16px rgba(46, 134, 171, 0.4) !important;
}

/* Mobile-friendly metrics */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    border-radius: 12px;
    padding: 16px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    margin-bottom: 12px !important;
}

[data-testid="stMetricLabel"] {
    font-size: 0.85rem !important;
    color: #6c757d !important;
}

[data-testid="stMetricValue"] {
    font-size: 1.4rem !important;
    font-weight: 700 !important;
    color: #2E86AB !important;
}

/* Info boxes styling - improved contrast */
.user-note {
    background: #f0f7ff;
    border-left: 4px solid #2E86AB;
    border-radius: 8px;
    padding: 14px 16px;
    margin: 16px 0;
    font-size: 0.92rem;
    line-height: 1.6;
    color: #1a1a1a;
}

.user-note strong {
    color: #1a3a52;
}

.tip-box {
    background: #fffef5;
    border-left: 4px solid #d4a000;
    border-radius: 8px;
    padding: 14px 16px;
    margin: 16px 0;
    font-size: 0.92rem;
    line-height: 1.6;
    color: #1a1a1a;
}

.tip-box strong {
    color: #5a4800;
}

/* Sidebar mobile optimization */
[data-testid="stSidebar"] {
    min-width: 280px !important;
}

[data-testid="stSidebar"] .stNumberInput input,
[data-testid="stSidebar"] .stTextInput input {
    font-size: 16px !important; /* Prevents zoom on iOS */
    min-height: 44px !important;
}

/* Tabs mobile styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 0px;
    flex-wrap: wrap;
}

.stTabs [data-baseweb="tab"] {
    min-height: 48px !important;
    padding: 12px 16px !important;
    font-size: 0.95rem !important;
    flex: 1;
    justify-content: center;
}

/* Expander styling */
.streamlit-expanderHeader {
    font-size: 1rem !important;
    font-weight: 600 !important;
}

/* Headers */
h1 {
    font-size: 1.6rem !important;
    margin-bottom: 8px !important;
}

h2 {
    font-size: 1.3rem !important;
    color: #2E86AB !important;
    margin-top: 20px !important;
}

/* Responsive columns - stack on mobile */
@media (max-width: 768px) {
    [data-testid="column"] {
        width: 100% !important;
        flex: 100% !important;
        min-width: 100% !important;
    }
    
    h1 {
        font-size: 1.4rem !important;
    }
    
    h2 {
        font-size: 1.2rem !important;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 1.2rem !important;
    }
}

/* Plotly chart mobile optimization */
.js-plotly-plot {
    width: 100% !important;
}
</style>
""", unsafe_allow_html=True)

# Configuration API
API_URL = os.getenv("API_URL", "http://localhost:8000")


def check_api_health() -> bool:
    """Verifie que l'API est disponible."""
    try:
        response = httpx.get(f"{API_URL}/health", timeout=5.0)
        return response.status_code == 200
    except Exception:
        return False


def wake_up_api(timeout: int = 90) -> bool:
    """Tente de reveiller l'API si elle est endormie (Render Cold Start)."""
    # Verification initiale rapide
    if check_api_health():
        return True

    # Si echec, on lance la procedure de reveil
    status_placeholder = st.sidebar.empty()
    status_placeholder.warning("Reveil de l'API en cours...")
    
    progress_bar = st.sidebar.progress(0)
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        # Mise a jour de la progression (estimation)
        elapsed = time.time() - start_time
        progress = min(elapsed / 60, 0.9)  # 60s reference pour la barre
        progress_bar.progress(progress)
        
        if check_api_health():
            progress_bar.empty()
            status_placeholder.empty()
            return True
            
        time.sleep(2)
    
    progress_bar.empty()
    status_placeholder.empty()
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
    st.markdown("**Trouvez le meilleur prix pour maximiser vos revenus**")
    
    # Note de bienvenue pour utilisateurs non-techniques
    st.markdown("""
    <div class="user-note">
        <strong>Comment utiliser cette application ?</strong><br>
        1. Ouvrez le menu lateral (icone en haut a gauche) pour configurer votre produit<br>
        2. Choisissez un onglet ci-dessous selon votre besoin<br>
        3. Cliquez sur les boutons pour lancer les calculs
    </div>
    """, unsafe_allow_html=True)

    # Status de l'API
    if wake_up_api():
        st.sidebar.success("Systeme pret")
    else:
        st.sidebar.error("Systeme non disponible")
        st.sidebar.info("Patientez 1-2 min, le systeme demarre...")


def render_sidebar() -> dict[str, Any]:
    """Affiche la barre laterale et retourne les parametres."""
    st.sidebar.header("Configuration")
    
    # Note explicative
    st.sidebar.caption("Renseignez les informations de votre produit ci-dessous")

    product_id = st.sidebar.text_input(
        "Reference du produit",
        value="GROCERY_I_1",
        help="Le code unique de votre produit (ex: SKU, reference interne)",
    )

    current_price = st.sidebar.number_input(
        "Prix de vente actuel (EUR)",
        min_value=0.01,
        max_value=1000.0,
        value=4.50,
        step=0.10,
        format="%.2f",
        help="Le prix auquel vous vendez actuellement ce produit",
    )

    current_volume = st.sidebar.number_input(
        "Ventes par semaine",
        min_value=0.0,
        max_value=100000.0,
        value=100.0,
        step=10.0,
        help="Nombre d'unites vendues en moyenne par semaine",
    )

    st.sidebar.divider()
    st.sidebar.subheader("Limites de prix")
    st.sidebar.caption("Definissez les bornes acceptables pour le nouveau prix")

    use_constraints = st.sidebar.checkbox(
        "Activer les limites", 
        value=True,
        help="Cochez pour definir un prix minimum et maximum"
    )

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
    
    # Note explicative pour utilisateurs non-techniques
    st.markdown("""
    <div class="tip-box">
        <strong>Que fait cet outil ?</strong><br>
        Notre algorithme analyse les donnees de vente pour vous suggerer 
        le prix optimal qui maximisera vos revenus.
    </div>
    """, unsafe_allow_html=True)

    if st.button("Calculer le prix optimal", type="primary", use_container_width=True):
        with st.spinner("Analyse en cours..."):
            result = get_price_recommendation(
                product_id=params["product_id"],
                current_price=params["current_price"],
                current_volume=params["current_volume"],
                constraints=params["constraints"],
            )

        if result:
            # Resultat principal mis en avant
            st.success(f"**Prix recommande: {result['recommended_price']:.2f} EUR** ({result['price_change_pct']:+.1f}% vs actuel)")
            
            st.markdown("---")
            st.subheader("Impact estime")
            
            # Layout mobile-friendly: 2 colonnes au lieu de 4
            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Revenu attendu",
                    f"{result['expected_revenue']:.2f} EUR",
                    f"{result['revenue_uplift_pct']:+.1f}%",
                    help="Le chiffre d'affaires prevu avec ce nouveau prix"
                )
                
                st.metric(
                    "Ventes prevues",
                    f"{result['expected_volume']:.0f} unites",
                    help="Nombre d'unites que vous devriez vendre"
                )

            with col2:
                st.metric(
                    "Marge attendue",
                    f"{result['expected_margin']:.2f} EUR",
                    help="Le benefice estime sur les ventes"
                )
                
                # Interpretation simple
                if result['revenue_uplift_pct'] > 0:
                    st.success("Ce prix devrait augmenter vos revenus")
                elif result['revenue_uplift_pct'] < 0:
                    st.warning("Ce prix pourrait reduire vos revenus")
                else:
                    st.info("Le prix actuel semble optimal")

            # Details techniques dans un expander
            with st.expander("Details techniques"):
                st.caption(f"Version du modele: {result['model_version']}")
                st.caption(f"Produit analyse: {params['product_id']}")


def render_simulation(params: dict[str, Any]) -> None:
    """Affiche la simulation de prix."""
    st.header("Simulation de Scenarios")
    
    # Note explicative
    st.markdown("""
    <div class="tip-box">
        <strong>A quoi sert cette simulation ?</strong><br>
        Visualisez comment differents prix pourraient impacter vos ventes 
        et revenus. Le graphique vous aide a trouver le "point ideal".
    </div>
    """, unsafe_allow_html=True)

    # Parametres de simulation - empiles pour mobile
    st.subheader("Parametres de simulation")
    
    n_points = st.slider(
        "Precision de l'analyse",
        min_value=5,
        max_value=21,
        value=11,
        step=2,
        help="Plus de points = analyse plus fine mais plus lente",
    )

    max_var = st.slider(
        "Amplitude des variations (%)",
        min_value=10,
        max_value=50,
        value=30,
        step=5,
        help="Jusqu'a combien le prix peut varier (ex: 30% = de -30% a +30%)",
    )

    variations = [
        round(-max_var / 100 + i * (2 * max_var / 100) / (n_points - 1), 3) for i in range(n_points)
    ]

    if st.button("Lancer la simulation", type="primary", use_container_width=True):
        with st.spinner("Simulation en cours..."):
            result = get_price_simulation(
                product_id=params["product_id"],
                current_price=params["current_price"],
                variations=variations,
            )

        if result and result.get("simulations"):
            df = pd.DataFrame(result["simulations"])
            
            # Trouver le prix optimal
            best_idx = df["expected_revenue"].idxmax()
            best_price = df.loc[best_idx, "price"]
            best_revenue = df.loc[best_idx, "expected_revenue"]
            
            st.success(f"**Prix optimal trouve: {best_price:.2f} EUR** (Revenu max: {best_revenue:.2f} EUR)")

            # Graphique interactif - hauteur reduite pour mobile
            fig = go.Figure()

            # Courbe de revenu
            fig.add_trace(
                go.Scatter(
                    x=df["price"],
                    y=df["expected_revenue"],
                    mode="lines+markers",
                    name="Revenu (EUR)",
                    line={"color": "#2E86AB", "width": 3},
                    marker={"size": 10},  # Plus gros pour le tactile
                )
            )

            # Courbe de volume
            fig.add_trace(
                go.Scatter(
                    x=df["price"],
                    y=df["expected_volume"],
                    mode="lines+markers",
                    name="Ventes (unites)",
                    yaxis="y2",
                    line={"color": "#A23B72", "width": 3, "dash": "dot"},
                    marker={"size": 10},
                )
            )

            # Prix actuel
            fig.add_vline(
                x=params["current_price"],
                line_dash="dash",
                line_color="gray",
                annotation_text="Actuel",
            )
            
            # Prix optimal
            fig.add_vline(
                x=best_price,
                line_dash="solid",
                line_color="green",
                annotation_text="Optimal",
            )

            fig.update_layout(
                title="",
                xaxis_title="Prix (EUR)",
                yaxis_title="Revenu (EUR)",
                yaxis2={
                    "title": "Ventes",
                    "overlaying": "y",
                    "side": "right",
                },
                legend={
                    "orientation": "h",
                    "yanchor": "bottom",
                    "y": 1.02,
                    "xanchor": "center",
                    "x": 0.5,
                },
                height=350,  # Reduit pour mobile
                margin={"l": 50, "r": 50, "t": 30, "b": 50},
            )

            st.plotly_chart(fig, use_container_width=True)
            
            # Aide a la lecture
            st.markdown("""
            <div class="user-note">
                <strong>Comment lire ce graphique ?</strong><br>
                • <span style="color: #2E86AB;">Courbe bleue</span> = Revenu (ce que vous gagnez)<br>
                • <span style="color: #A23B72;">Courbe rose</span> = Nombre de ventes<br>
                • Ligne verte = Prix qui maximise vos revenus
            </div>
            """, unsafe_allow_html=True)

            # Tableau des resultats
            with st.expander("Voir tous les chiffres"):
                df_display = df.copy()
                df_display["price_variation"] = df_display["price_variation"].apply(
                    lambda x: f"{x:+.1%}"
                )
                df_display.columns = ["Variation", "Prix", "Ventes", "Revenu", "Marge"]
                st.dataframe(df_display, use_container_width=True)


def render_elasticity_analysis(params: dict[str, Any]) -> None:
    """Affiche l'analyse d'elasticite."""
    st.header("Sensibilite au Prix")
    
    # Explication simple pour non-techniques
    st.markdown("""
    <div class="tip-box">
        <strong>Qu'est-ce que c'est ?</strong><br>
        Cet outil mesure comment vos clients reagissent aux changements de prix. 
        Est-ce qu'ils achetent moins si vous augmentez le prix ? De combien ?
    </div>
    """, unsafe_allow_html=True)

    # Calcul de l'elasticite via simulation
    with st.spinner("Analyse de la sensibilite..."):
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
        
        # Resultat principal
        st.markdown("---")
        
        # Layout mobile: 1 metrique principale + interpretation
        st.metric(
            "Indice de sensibilite", 
            f"{elasticity:.2f}",
            help="Un chiffre negatif signifie que les ventes baissent quand le prix monte"
        )
        
        st.markdown("---")
        st.subheader("Ce que cela signifie pour vous")
        
        # Interpretation claire et actionnable
        if abs(elasticity) > 1:
            st.warning("""
            **Vos clients sont sensibles au prix**
            
            Si vous augmentez vos prix, vous risquez de perdre beaucoup de clients.
            
            ➔ **Conseil :** Soyez prudent avec les hausses de prix. 
            Privilegiez les petites augmentations progressives.
            """)
        else:
            st.success("""
            **Vos clients sont peu sensibles au prix**
            
            Meme si vous augmentez vos prix, vos clients resteront fideles.
            
            ➔ **Conseil :** Vous avez une marge de manoeuvre pour 
            augmenter vos prix et ameliorer vos revenus.
            """)
        
        # Details techniques caches
        with st.expander("Comprendre le calcul"):
            st.markdown("""
            L'**elasticite-prix** mesure le pourcentage de variation des ventes 
            pour chaque pourcentage de variation du prix.
            
            - **Elasticite < -1** : Les ventes baissent plus vite que le prix monte
            - **Elasticite entre -1 et 0** : Les ventes baissent moins vite que le prix monte
            - **Elasticite = 0** : Les ventes ne changent pas avec le prix
            
            *Exemple : Une elasticite de -2 signifie qu'une hausse de prix de 10% 
            entraine une baisse des ventes de 20%.*
            """)
    else:
        st.info("Chargement des donnees d'analyse...")


def main() -> None:
    """Point d'entree principal."""
    render_header()
    params = render_sidebar()

    # Tabs pour les differentes vues
    tab1, tab2, tab3 = st.tabs(
        [
            "Recommandation",
            "Simulation",
            "Sensibilite",
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
