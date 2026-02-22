"""Interface Streamlit pour l'optimisation des prix."""

import os
import time
from typing import Any

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Configuration de la page
st.set_page_config(
    page_title="Optimisation des Prix",
    page_icon="shopping_trolley",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================

# ============================================================================
st.markdown("""
<style>
/* ===== Google Fonts Import ===== */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ===== CSS Keyframe Animations ===== */
@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes shimmer {
    0% { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}

@keyframes pulse-glow {
    0%, 100% { box-shadow: 0 0 20px rgba(0, 56, 123, 0.3); }
    50% { box-shadow: 0 0 30px rgba(0, 56, 123, 0.5); }
}

@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-5px); }
}

@keyframes gradient-shift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

@keyframes countUp {
    from { opacity: 0; transform: scale(0.8); }
    to { opacity: 1; transform: scale(1); }
}

/* ===== Variables de design ===== */
:root {
    /* Palette primaire */
    --primary-gradient: linear-gradient(135deg, #0A2463 0%, #1B3A8C 50%, #00387b 100%);
    --primary-blue: #0A2463;
    --primary-blue-light: #1B3A8C;
    --primary-blue-dark: #061539;
    
    /* Couleurs d'accent */
    --accent-red: #D62828;
    --accent-red-light: #E85454;
    --accent-red-glow: rgba(214, 40, 40, 0.4);
    --accent-orange: #FF9F1C;
    --accent-orange-light: #FFB347;
    --success-green: #06D6A0;
    --success-green-light: #34E8B8;
    
    /* Neutral Palette */
    --bg-primary: #FAFBFC;
    --bg-secondary: #F0F2F5;
    --bg-card: #FFFFFF;
    --bg-glass: rgba(255, 255, 255, 0.85);
    --bg-glass-dark: rgba(10, 36, 99, 0.95);
    
    /* Text Colors */
    --text-primary: #1A1D21;
    --text-secondary: #4A5568;
    --text-muted: #718096;
    --text-inverse: #FFFFFF;
    
    /* Shadows - Layered Depth */
    --shadow-sm: 0 1px 3px rgba(0, 0, 0, 0.08);
    --shadow-md: 0 4px 12px rgba(0, 0, 0, 0.1);
    --shadow-lg: 0 10px 40px rgba(0, 0, 0, 0.12);
    --shadow-xl: 0 20px 60px rgba(0, 0, 0, 0.15);
    --shadow-glow-blue: 0 8px 32px rgba(10, 36, 99, 0.25);
    --shadow-glow-red: 0 8px 32px rgba(214, 40, 40, 0.25);
    --shadow-glow-green: 0 8px 32px rgba(6, 214, 160, 0.25);
    
    /* Border Radius */
    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 16px;
    --radius-xl: 24px;
    --radius-full: 9999px;
    
    /* Transitions */
    --transition-fast: 0.15s cubic-bezier(0.4, 0, 0.2, 1);
    --transition-base: 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    --transition-slow: 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    
    /* Typography */
    --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

/* ===== Global Reset & Base ===== */
*, *::before, *::after {
    box-sizing: border-box;
}

html {
    scroll-behavior: smooth;
}

.stApp {
    font-family: var(--font-sans) !important;
    background: var(--bg-primary) !important;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

/* Hide Streamlit branding elements */
#MainMenu, footer, header[data-testid="stHeader"] {
    visibility: hidden;
    height: 0;
}

/* ===== En-tete principal ===== */
.premium-header {
    background: var(--primary-gradient);
    background-size: 200% 200%;
    animation: gradient-shift 8s ease infinite;
    color: var(--text-inverse);
    padding: 28px 32px;
    border-radius: var(--radius-lg);
    margin-bottom: 24px;
    text-align: center;
    position: relative;
    overflow: hidden;
    box-shadow: var(--shadow-glow-blue);
}

.premium-header::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, transparent 50%, rgba(255,255,255,0.05) 100%);
    pointer-events: none;
}

.premium-header::after {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 60%);
    opacity: 0.5;
    pointer-events: none;
}

.premium-header h1 {
    color: var(--text-inverse) !important;
    margin: 0 !important;
    font-size: 1.75rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em;
    position: relative;
    z-index: 1;
}

.premium-header p {
    color: rgba(255, 255, 255, 0.9);
    margin: 10px 0 0 0;
    font-size: 1rem;
    font-weight: 400;
    position: relative;
    z-index: 1;
}

/* Variante d'en-tete */
.carrefour-header {
    background: var(--primary-gradient);
    background-size: 200% 200%;
    animation: gradient-shift 8s ease infinite;
    color: var(--text-inverse);
    padding: 28px 32px;
    border-radius: var(--radius-lg);
    margin-bottom: 24px;
    text-align: center;
    position: relative;
    overflow: hidden;
    box-shadow: var(--shadow-glow-blue);
}

.carrefour-header h1 {
    color: var(--text-inverse) !important;
    margin: 0 !important;
    font-size: 1.75rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em;
}

.carrefour-header p {
    color: rgba(255, 255, 255, 0.9);
    margin: 10px 0 0 0;
    font-size: 1rem;
}

/* ===== Cartes d'information ===== */
.user-note {
    background: var(--bg-glass);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(10, 36, 99, 0.1);
    border-left: 4px solid var(--primary-blue);
    border-radius: var(--radius-md);
    padding: 18px 20px;
    margin: 20px 0;
    font-size: 0.95rem;
    line-height: 1.7;
    color: var(--text-primary);
    box-shadow: var(--shadow-sm);
    animation: fadeInUp 0.5s ease-out forwards;
}

.user-note strong {
    color: var(--primary-blue);
    font-weight: 600;
}

.tip-box {
    background: linear-gradient(135deg, #FFF8F0 0%, #FFF5EB 100%);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 159, 28, 0.2);
    border-left: 4px solid var(--accent-orange);
    border-radius: var(--radius-md);
    padding: 18px 20px;
    margin: 20px 0;
    font-size: 0.95rem;
    line-height: 1.7;
    color: var(--text-primary);
    box-shadow: var(--shadow-sm);
    animation: fadeInUp 0.5s ease-out forwards;
}

.tip-box strong {
    color: #CC7000;
    font-weight: 600;
}

/* ===== Success Card ===== */
.success-card {
    background: linear-gradient(135deg, #E6FFF5 0%, #D4F7E8 100%);
    border: 1px solid rgba(6, 214, 160, 0.3);
    border-left: 4px solid var(--success-green);
    border-radius: var(--radius-md);
    padding: 18px 20px;
    margin: 20px 0;
    font-size: 0.95rem;
    line-height: 1.7;
    color: var(--text-primary);
    box-shadow: var(--shadow-glow-green);
    animation: fadeInUp 0.5s ease-out forwards;
}

/* ===== Boutons ===== */
.stButton > button {
    font-family: var(--font-sans) !important;
    min-height: 52px !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.01em;
    border-radius: var(--radius-md) !important;
    margin: 10px 0 !important;
    width: 100% !important;
    transition: all var(--transition-base) !important;
    position: relative;
    overflow: hidden;
}

.stButton > button::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
    transition: left 0.5s ease;
}

.stButton > button:hover::before {
    left: 100%;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--accent-red) 0%, var(--accent-red-light) 100%) !important;
    border: none !important;
    color: var(--text-inverse) !important;
    box-shadow: var(--shadow-glow-red) !important;
}

.stButton > button[kind="primary"]:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 12px 40px rgba(214, 40, 40, 0.4) !important;
}

.stButton > button[kind="primary"]:active {
    transform: translateY(-1px) scale(0.98) !important;
}

.stButton > button[kind="secondary"] {
    background: linear-gradient(135deg, var(--primary-blue) 0%, var(--primary-blue-light) 100%) !important;
    border: none !important;
    color: var(--text-inverse) !important;
    box-shadow: var(--shadow-glow-blue) !important;
}

.stButton > button[kind="secondary"]:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 12px 40px rgba(10, 36, 99, 0.4) !important;
}

/* ===== Cartes de metriques ===== */
[data-testid="stMetric"] {
    background: var(--bg-card);
    border-radius: var(--radius-md);
    padding: 20px !important;
    box-shadow: var(--shadow-md);
    border: 1px solid rgba(0, 0, 0, 0.05);
    border-left: 4px solid var(--primary-blue);
    margin-bottom: 16px !important;
    transition: all var(--transition-base);
    animation: fadeInUp 0.5s ease-out forwards;
}

[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-lg);
}

[data-testid="stMetricLabel"] {
    font-family: var(--font-sans) !important;
    font-size: 0.85rem !important;
    color: var(--text-muted) !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

[data-testid="stMetricValue"] {
    font-family: var(--font-sans) !important;
    font-size: 1.5rem !important;
    font-weight: 700 !important;
    color: var(--primary-blue) !important;
    animation: countUp 0.6s ease-out forwards;
}

[data-testid="stMetricDelta"] > div {
    font-family: var(--font-sans) !important;
    font-weight: 600 !important;
}

[data-testid="stMetricDelta"] svg + div,
[data-testid="stMetricDelta"] > div:first-child {
    color: #0B5E3D !important;
}

/* Negative delta styling */
[data-testid="stMetricDelta"][data-testid*="negative"] > div {
    color: var(--accent-red) !important;
}

/* ===== Fix Material Icons rendering as text ===== */
[data-testid="stSidebarCollapseButton"] span,
[data-testid="collapsedControl"] span,
button[kind="headerNoPadding"] span {
    font-family: 'Material Symbols Rounded', 'Material Icons' !important;
    font-size: 24px !important;
    -webkit-font-smoothing: antialiased;
}

/* Hide icon text fallback */
[data-testid="stSidebar"] button span:not(.material-icons) {
    font-size: 0 !important;
    visibility: hidden;
}

/* ===== Barre laterale ===== */
[data-testid="stSidebar"] {
    min-width: 300px !important;
    background: var(--bg-glass-dark) !important;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-right: 1px solid rgba(255, 255, 255, 0.1);
}

[data-testid="stSidebar"]::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(180deg, rgba(10, 36, 99, 0.98) 0%, rgba(6, 21, 57, 0.99) 100%);
    z-index: -1;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
    color: var(--text-inverse) !important;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
    color: var(--text-inverse) !important;
    border-bottom-color: rgba(255, 255, 255, 0.2) !important;
}

[data-testid="stSidebar"] .stNumberInput input,
[data-testid="stSidebar"] .stTextInput input {
    font-family: var(--font-sans) !important;
    font-size: 16px !important;
    min-height: 48px !important;
    border-radius: var(--radius-sm) !important;
    background: rgba(255, 255, 255, 0.1) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    color: var(--text-inverse) !important;
    transition: all var(--transition-fast);
}

[data-testid="stSidebar"] .stNumberInput input:focus,
[data-testid="stSidebar"] .stTextInput input:focus {
    background: rgba(255, 255, 255, 0.15) !important;
    border-color: var(--accent-orange) !important;
    box-shadow: 0 0 0 3px rgba(255, 159, 28, 0.2) !important;
}

[data-testid="stSidebar"] label {
    font-family: var(--font-sans) !important;
    color: var(--text-inverse) !important;
    font-weight: 500 !important;
}

[data-testid="stSidebar"] .stCaption {
    color: rgba(255, 255, 255, 0.7) !important;
}

[data-testid="stSidebar"] hr {
    border-color: rgba(255, 255, 255, 0.15) !important;
    margin: 1.5rem 0 !important;
}

/* ===== Onglets ===== */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background: var(--bg-secondary);
    border-radius: var(--radius-lg);
    padding: 6px;
    box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.06);
}

.stTabs [data-baseweb="tab"] {
    font-family: var(--font-sans) !important;
    min-height: 52px !important;
    padding: 14px 24px !important;
    font-size: 0.95rem !important;
    font-weight: 500 !important;
    flex: 1;
    justify-content: center;
    border-radius: var(--radius-md) !important;
    color: var(--text-secondary) !important;
    transition: all var(--transition-base) !important;
    position: relative;
}

.stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]) {
    background: rgba(10, 36, 99, 0.05) !important;
    color: var(--primary-blue) !important;
}

.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: var(--bg-card) !important;
    color: var(--primary-blue) !important;
    font-weight: 600 !important;
    box-shadow: var(--shadow-md) !important;
}

.stTabs [data-baseweb="tab-highlight"] {
    display: none !important;
}

.stTabs [data-baseweb="tab-border"] {
    display: none !important;
}

/* ===== Accordeons ===== */
.streamlit-expanderHeader {
    font-family: var(--font-sans) !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    color: var(--primary-blue) !important;
    padding: 16px 20px !important;
    background: var(--bg-secondary) !important;
    border-radius: var(--radius-md) !important;
    transition: all var(--transition-fast);
}

.streamlit-expanderHeader:hover {
    background: var(--bg-card) !important;
    box-shadow: var(--shadow-sm);
}

details[open] .streamlit-expanderHeader {
    border-bottom-left-radius: 0 !important;
    border-bottom-right-radius: 0 !important;
}

.streamlit-expanderContent {
    background: var(--bg-card) !important;
    border: 1px solid rgba(0, 0, 0, 0.05);
    border-top: none;
    border-radius: 0 0 var(--radius-md) var(--radius-md) !important;
    padding: 20px !important;
}

/* ===== Typography ===== */
h1 {
    font-family: var(--font-sans) !important;
    font-size: 1.75rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em;
    margin-bottom: 12px !important;
    color: var(--primary-blue) !important;
}

h2 {
    font-family: var(--font-sans) !important;
    font-size: 1.35rem !important;
    font-weight: 600 !important;
    letter-spacing: -0.01em;
    color: var(--primary-blue) !important;
    margin-top: 24px !important;
    padding-bottom: 10px;
    border-bottom: 3px solid var(--accent-orange);
    display: inline-block;
}

h3 {
    font-family: var(--font-sans) !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    color: var(--primary-blue-dark) !important;
}

p, li, span {
    font-family: var(--font-sans) !important;
}

/* ===== Alert Messages ===== */
.stSuccess {
    background: linear-gradient(135deg, #E6FFF5 0%, #D4F7E8 100%) !important;
    border-left: 4px solid #0B5E3D !important;
    border-radius: var(--radius-md) !important;
    box-shadow: var(--shadow-sm);
    animation: fadeInUp 0.4s ease-out forwards;
}

.stSuccess > div {
    color: #0B5E3D !important;
}

.stSuccess p, .stSuccess span, .stSuccess strong {
    color: #0B5E3D !important;
}

.stWarning {
    background: linear-gradient(135deg, #FFF8F0 0%, #FFF5EB 100%) !important;
    border-left: 4px solid #B45309 !important;
    border-radius: var(--radius-md) !important;
    box-shadow: var(--shadow-sm);
    animation: fadeInUp 0.4s ease-out forwards;
}

.stWarning > div {
    color: #92400E !important;
}

.stWarning p, .stWarning span, .stWarning strong {
    color: #92400E !important;
}

.stInfo {
    background: linear-gradient(135deg, #EBF4FF 0%, #E0ECFF 100%) !important;
    border-left: 4px solid var(--primary-blue) !important;
    border-radius: var(--radius-md) !important;
    box-shadow: var(--shadow-sm);
    animation: fadeInUp 0.4s ease-out forwards;
}

.stInfo > div {
    color: #1E3A5F !important;
}

.stInfo p, .stInfo span, .stInfo strong {
    color: #1E3A5F !important;
}

.stError {
    background: linear-gradient(135deg, #FFF0F0 0%, #FFE5E5 100%) !important;
    border-left: 4px solid var(--accent-red) !important;
    border-radius: var(--radius-md) !important;
    box-shadow: var(--shadow-sm);
    animation: fadeInUp 0.4s ease-out forwards;
}

.stError > div {
    color: #991B1B !important;
}

.stError p, .stError span, .stError strong {
    color: #991B1B !important;
}

/* ===== Sliders ===== */
.stSlider > div > div > div {
    background: var(--accent-orange) !important;
}

.stSlider [data-baseweb="slider"] [role="slider"] {
    background: var(--accent-orange) !important;
    border: 3px solid white !important;
    box-shadow: var(--shadow-md) !important;
    width: 20px !important;
    height: 20px !important;
}

/* ===== Checkbox ===== */
.stCheckbox > label > div[data-testid="stCheckbox"] > div {
    border-color: var(--primary-blue) !important;
}

.stCheckbox > label > div[data-testid="stCheckbox"] > div[aria-checked="true"] {
    background: var(--primary-blue) !important;
    border-color: var(--primary-blue) !important;
}

/* ===== Plotly Charts Wrapper ===== */
.js-plotly-plot {
    width: 100% !important;
    border-radius: var(--radius-md);
    overflow: hidden;
    box-shadow: var(--shadow-sm);
}

.plotly .modebar {
    right: 10px !important;
    top: 10px !important;
}

/* ===== Dividers ===== */
hr {
    border: none !important;
    height: 1px !important;
    background: linear-gradient(90deg, transparent, var(--bg-secondary), transparent) !important;
    margin: 24px 0 !important;
}

/* ===== Spinner / Loading ===== */
.stSpinner > div {
    border-top-color: var(--primary-blue) !important;
}

/* ===== DataFrames ===== */
.stDataFrame {
    border-radius: var(--radius-md) !important;
    overflow: hidden;
    box-shadow: var(--shadow-sm);
}

[data-testid="stDataFrame"] > div {
    border-radius: var(--radius-md) !important;
}

/* ===== Mobile Responsive ===== */
@media (max-width: 768px) {
    .premium-header, .carrefour-header {
        padding: 20px 16px;
        border-radius: var(--radius-md);
    }
    
    .premium-header h1, .carrefour-header h1 {
        font-size: 1.35rem !important;
    }
    
    .premium-header p, .carrefour-header p {
        font-size: 0.9rem;
    }
    
    [data-testid="column"] {
        width: 100% !important;
        flex: 100% !important;
        min-width: 100% !important;
    }
    
    h1 {
        font-size: 1.5rem !important;
    }
    
    h2 {
        font-size: 1.2rem !important;
    }
    
    [data-testid="stMetric"] {
        padding: 16px !important;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 1.3rem !important;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        padding: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 12px 16px !important;
        font-size: 0.9rem !important;
        min-height: 48px !important;
    }
    
    .stButton > button {
        min-height: 48px !important;
        font-size: 0.95rem !important;
    }
    
    .user-note, .tip-box, .success-card {
        padding: 14px 16px;
        font-size: 0.9rem;
    }
}

@media (max-width: 480px) {
    .premium-header h1, .carrefour-header h1 {
        font-size: 1.2rem !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 10px 12px !important;
        font-size: 0.85rem !important;
    }
    
    h1 {
        font-size: 1.3rem !important;
    }
    
    h2 {
        font-size: 1.1rem !important;
        display: block;
    }
}

/* ===== Print Styles ===== */
@media print {
    .stButton, [data-testid="stSidebar"] {
        display: none !important;
    }
    
    .premium-header, .carrefour-header {
        background: var(--primary-blue) !important;
        -webkit-print-color-adjust: exact;
        print-color-adjust: exact;
    }
}

/* ===== Accessibility Focus States ===== */
button:focus-visible,
input:focus-visible,
select:focus-visible {
    outline: 3px solid var(--accent-orange) !important;
    outline-offset: 2px;
}

/* ===== Smooth Scroll for Anchor Links ===== */
[id] {
    scroll-margin-top: 100px;
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
    """Tente de reveiller l'API si elle est en veille."""
    # Verification prealable dans la session
    if st.session_state.get("api_healthy", False):
        return True
    
    # Verification initiale rapide
    if check_api_health():
        st.session_state.api_healthy = True
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
            st.session_state.api_healthy = True
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
    # En-tete
    st.markdown("""
    <div class="carrefour-header">
        <h1>Optimisation des Prix</h1>
        <p>Outil d'aide a la decision pour maximiser vos revenus</p>
    </div>
    """, unsafe_allow_html=True)
    
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

    # Section liens
    st.sidebar.divider()
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 10px 0;">
        <a href="https://www.linkedin.com/in/souleymanes-sall" target="_blank" 
           style="display: inline-flex; align-items: center; gap: 8px; padding: 10px 16px; 
                  background: rgba(255,255,255,0.1); border-radius: 8px; 
                  color: #FFFFFF; text-decoration: none; font-size: 0.9rem;
                  margin-bottom: 8px; transition: all 0.2s;">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
                <path d="M19 0h-14c-2.761 0-5 2.239-5 5v14c0 2.761 2.239 5 5 5h14c2.762 0 5-2.239 5-5v-14c0-2.761-2.238-5-5-5zm-11 19h-3v-11h3v11zm-1.5-12.268c-.966 0-1.75-.79-1.75-1.764s.784-1.764 1.75-1.764 1.75.79 1.75 1.764-.783 1.764-1.75 1.764zm13.5 12.268h-3v-5.604c0-3.368-4-3.113-4 0v5.604h-3v-11h3v1.765c1.396-2.586 7-2.777 7 2.476v6.759z"/>
            </svg>
            Contactez moi
        </a>
        <br>
        <a href="https://github.com/Souley225/Pricing_Optimisation_Project" target="_blank" 
           style="display: inline-flex; align-items: center; gap: 8px; padding: 10px 16px; 
                  background: rgba(255,255,255,0.1); border-radius: 8px; 
                  color: #FFFFFF; text-decoration: none; font-size: 0.9rem;
                  transition: all 0.2s;">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
                <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
            </svg>
            Voir le code source
        </a>
    </div>
    """, unsafe_allow_html=True)

    return {
        "product_id": product_id,
        "current_price": current_price,
        "current_volume": current_volume,
        "constraints": constraints,
    }


def render_recommendation(params: dict[str, Any]) -> None:
    """Affiche la recommandation de prix."""
    st.header("Recommandation de Prix")
    
    # Initialisation du state pour le resultat
    if "recommendation_result" not in st.session_state:
        st.session_state.recommendation_result = None
    if "recommendation_params" not in st.session_state:
        st.session_state.recommendation_params = None
    
    # Note explicative pour utilisateurs non-techniques
    st.markdown("""
    <div class="tip-box">
        <strong>Que fait cet outil ?</strong><br>
        Notre algorithme analyse les donnees de vente pour vous suggerer 
        le prix optimal qui maximisera vos revenus.
    </div>
    """, unsafe_allow_html=True)

    # Cle unique basee sur les parametres pour detecter les changements
    current_params_key = f"{params['product_id']}_{params['current_price']}_{params['current_volume']}_{params['constraints']}"

    if st.button("Calculer le prix optimal", type="primary", use_container_width=True):
        with st.spinner("Analyse en cours..."):
            result = get_price_recommendation(
                product_id=params["product_id"],
                current_price=params["current_price"],
                current_volume=params["current_volume"],
                constraints=params["constraints"],
            )
        # Stocker le resultat dans le session state
        st.session_state.recommendation_result = result
        st.session_state.recommendation_params = current_params_key

    # Afficher le resultat s'il existe (persiste entre les reruns)
    result = st.session_state.recommendation_result
    
    # Reinitialiser si les parametres ont change
    if st.session_state.recommendation_params != current_params_key:
        result = None
        st.session_state.recommendation_result = None

    if result:
        # Resultat principal mis en avant
        st.success(f"**Prix recommande: {result['recommended_price']:.2f} EUR** ({result['price_change_pct']:+.1f}% vs actuel)")
        
        st.markdown("---")
        st.subheader("Impact estime")
        
        # Disposition en 2 colonnes
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
    
    # Explication des parametres
    st.markdown("""
    <div class="user-note">
        <strong>Comment configurer la simulation ?</strong><br>
        <span style="color: #0A2463; font-weight: 600;">Precision</span> : 
        Nombre de prix testes entre le minimum et le maximum. 
        Plus le nombre est eleve, plus l'analyse est detaillee.<br>
        <span style="color: #0A2463; font-weight: 600;">Amplitude</span> : 
        Etendue des variations de prix a tester. 
        Par exemple, 30% signifie que l'on teste des prix de -30% a +30% par rapport au prix actuel.
    </div>
    """, unsafe_allow_html=True)
    
    n_points = st.slider(
        "Precision de l'analyse",
        min_value=5,
        max_value=21,
        value=11,
        step=2,
        help="Nombre de prix differents a tester (5 = rapide, 21 = tres detaille)",
    )

    max_var = st.slider(
        "Amplitude des variations (%)",
        min_value=10,
        max_value=50,
        value=30,
        step=5,
        help="Ecart maximum par rapport au prix actuel",
    )
    
    # Afficher un apercu des prix qui seront testes
    min_price = params["current_price"] * (1 - max_var / 100)
    max_price = params["current_price"] * (1 + max_var / 100)
    st.caption(f"Prix testes : de **{min_price:.2f} EUR** a **{max_price:.2f} EUR** ({n_points} points)")

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

            # Graphique interactif
            fig = go.Figure()

            # Courbe de revenu
            fig.add_trace(
                go.Scatter(
                    x=df["price"],
                    y=df["expected_revenue"],
                    mode="lines+markers",
                    name="Revenu (EUR)",
                    line={"color": "#0A2463", "width": 3, "shape": "spline"},
                    marker={
                        "size": 12,
                        "color": "#0A2463",
                        "line": {"color": "#FFFFFF", "width": 2}
                    },
                    hovertemplate="<b>Prix:</b> %{x:.2f} EUR<br><b>Revenu:</b> %{y:.2f} EUR<extra></extra>",
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
                    line={"color": "#D62828", "width": 3, "dash": "dot", "shape": "spline"},
                    marker={
                        "size": 12,
                        "color": "#D62828",
                        "symbol": "diamond",
                        "line": {"color": "#FFFFFF", "width": 2}
                    },
                    hovertemplate="<b>Prix:</b> %{x:.2f} EUR<br><b>Ventes:</b> %{y:.0f} unites<extra></extra>",
                )
            )

            # Prix actuel
            fig.add_vline(
                x=params["current_price"],
                line_dash="dash",
                line_color="#718096",
                line_width=2,
                annotation_text="Actuel",
                annotation_font_color="#718096",
            )
            
            # Prix optimal (Success Green)
            fig.add_vline(
                x=best_price,
                line_dash="solid",
                line_color="#06D6A0",
                line_width=3,
                annotation_text="Optimal",
                annotation_font_color="#06D6A0",
                annotation_font_size=12,
                annotation_font_weight=600,
            )

            fig.update_layout(
                title="",
                xaxis_title="Prix (EUR)",
                yaxis_title="Revenu (EUR)",
                xaxis={
                    "title_font": {"family": "Inter, sans-serif", "size": 13, "color": "#4A5568"},
                    "tickfont": {"family": "Inter, sans-serif", "size": 11, "color": "#718096"},
                    "gridcolor": "rgba(0,0,0,0.05)",
                    "zerolinecolor": "rgba(0,0,0,0.1)",
                },
                yaxis={
                    "title_font": {"family": "Inter, sans-serif", "size": 13, "color": "#0A2463"},
                    "tickfont": {"family": "Inter, sans-serif", "size": 11, "color": "#718096"},
                    "gridcolor": "rgba(0,0,0,0.05)",
                },
                yaxis2={
                    "title": "Ventes",
                    "title_font": {"family": "Inter, sans-serif", "size": 13, "color": "#D62828"},
                    "tickfont": {"family": "Inter, sans-serif", "size": 11, "color": "#718096"},
                    "overlaying": "y",
                    "side": "right",
                },
                legend={
                    "orientation": "h",
                    "yanchor": "bottom",
                    "y": 1.02,
                    "xanchor": "center",
                    "x": 0.5,
                    "font": {"family": "Inter, sans-serif", "size": 12},
                    "bgcolor": "rgba(255,255,255,0.9)",
                    "bordercolor": "rgba(0,0,0,0.1)",
                    "borderwidth": 1,
                },
                height=380,
                margin={"l": 60, "r": 60, "t": 40, "b": 60},
                plot_bgcolor="rgba(250,251,252,1)",
                paper_bgcolor="rgba(255,255,255,1)",
                hoverlabel={
                    "bgcolor": "#0A2463",
                    "font_size": 13,
                    "font_family": "Inter, sans-serif",
                    "font_color": "white",
                    "bordercolor": "#0A2463",
                },
            )

            st.plotly_chart(fig, use_container_width=True)
            
            # Legende du graphique
            st.markdown("""
            <div class="user-note">
                <strong>Comment lire ce graphique ?</strong><br>
                <span style="display: inline-block; width: 12px; height: 12px; background: #0A2463; border-radius: 50%; margin-right: 8px;"></span>
                <span style="color: #0A2463; font-weight: 600;">Courbe bleue</span> = Revenu (ce que vous gagnez)<br>
                <span style="display: inline-block; width: 12px; height: 12px; background: #D62828; border-radius: 50%; margin-right: 8px;"></span>
                <span style="color: #D62828; font-weight: 600;">Courbe rouge</span> = Nombre de ventes<br>
                <span style="display: inline-block; width: 12px; height: 12px; background: #06D6A0; border-radius: 50%; margin-right: 8px;"></span>
                <span style="color: #06D6A0; font-weight: 600;">Ligne verte</span> = Prix qui maximise vos revenus
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
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #FFF8F0 0%, #FFF5EB 100%);
                border: 1px solid rgba(180, 83, 9, 0.3);
                border-left: 4px solid #B45309;
                border-radius: 12px;
                padding: 20px;
                margin: 16px 0;
            ">
                <p style="color: #78350F; font-weight: 700; font-size: 1.1rem; margin: 0 0 12px 0;">
                    Vos clients sont sensibles au prix
                </p>
                <p style="color: #92400E; margin: 0 0 12px 0; line-height: 1.6;">
                    Si vous augmentez vos prix, vous risquez de perdre beaucoup de clients.
                </p>
                <p style="color: #92400E; margin: 0; line-height: 1.6;">
                    <strong style="color: #78350F;">Conseil :</strong> Soyez prudent avec les hausses de prix. 
                    Privilegiez les petites augmentations progressives.
                </p>
                <p style="color: #78350F; font-weight: 600; margin: 12px 0 0 0; font-size: 0.9rem;">
                    Elasticite mesuree : {elasticity:.2f}
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #ECFDF5 0%, #D1FAE5 100%);
                border: 1px solid rgba(6, 95, 70, 0.3);
                border-left: 4px solid #059669;
                border-radius: 12px;
                padding: 20px;
                margin: 16px 0;
            ">
                <p style="color: #064E3B; font-weight: 700; font-size: 1.1rem; margin: 0 0 12px 0;">
                    Vos clients sont peu sensibles au prix
                </p>
                <p style="color: #065F46; margin: 0 0 12px 0; line-height: 1.6;">
                    Meme si vous augmentez vos prix, vos clients resteront fideles.
                </p>
                <p style="color: #065F46; margin: 0; line-height: 1.6;">
                    <strong style="color: #064E3B;">Conseil :</strong> Vous avez une marge de manoeuvre pour 
                    augmenter vos prix et ameliorer vos revenus.
                </p>
                <p style="color: #064E3B; font-weight: 600; margin: 12px 0 0 0; font-size: 0.9rem;">
                    Elasticite mesuree : {elasticity:.2f}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
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
