import re

with open('src/ui/app.py', 'r', encoding='utf-8') as f:
    text = f.read()

new_css = '''<style>
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

/* Form Inputs fix */
/* The base input div in Streamlit needs to be customized to avoid generic text inputs */
div[data-baseweb="input"],
div[data-baseweb="base-input"],
div[data-baseweb="input"] > input,
div[data-baseweb="base-input"] > input {
    background-color: transparent !important;
    color: var(--text-primary-input) !important;
    border-radius: 0 !important;
}

/* For actual text rendering inside the input */
div[data-baseweb="input"] input,
div[data-baseweb="base-input"] input {
    color: var(--text-primary-input) !important;
    font-family: var(--font-sans) !important;
    font-size: 1rem !important;
    -webkit-text-fill-color: var(--text-primary-input) !important;
}

/* The wrapper of the input */
.stNumberInput > div > div > div, 
.stTextInput > div > div > div {
    border: none !important;
    border-bottom: 1px solid var(--border-color) !important;
    border-radius: 0 !important;
    background-color: transparent !important;
    transition: border-bottom 0.2s ease !important;
    box-shadow: none !important;
}

.stNumberInput > div > div > div:focus-within, 
.stTextInput > div > div > div:focus-within {
    border-bottom: 2px solid var(--accent-crimson) !important;
    box-shadow: none !important;
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
</style>'''

# Regex to safely replace all the CSS
# In old app.py, it was bounded by <style> and </style>
pattern = re.compile(r'<style>.*?</style>', re.DOTALL)
text = pattern.sub(new_css, text)

# Sidebar UI fixes
text = text.replace('color: var(--primary-blue)', 'color: var(--text-primary)')
text = text.replace('color: var(--text-secondary)', 'color: var(--text-secondary)')
text = text.replace('background: rgba(10, 36, 99, 0.08)', 'background: transparent')
text = text.replace('border: 1px solid rgba(10, 36, 99, 0.1)', 'border: 1px solid var(--border-color)')
text = text.replace('background: rgba(0, 0, 0, 0.08)', 'background: var(--border-light)')

# Plotly chart elements fixes
text = text.replace('"color": "#0A2463"', '"color": "#111111"') 
text = text.replace('"color": "#D62828"', '"color": "#8A1C1C"') 
text = text.replace('"family": "Inter, sans-serif"', '"family": "Outfit, sans-serif"')
text = text.replace('"color": "#4A5568"', '"color": "#111111"')

text = text.replace('plot_bgcolor="rgba(250,251,252,1)"', 'plot_bgcolor="transparent"')
text = text.replace('paper_bgcolor="rgba(255,255,255,1)"', 'paper_bgcolor="transparent"')

text = text.replace('line_color="#06D6A0"', 'line_color="#8A1C1C"')
text = text.replace('annotation_font_color="#06D6A0"', 'annotation_font_color="#8A1C1C"')

text = text.replace('line_color="#718096"', 'line_color="rgba(17,17,17,0.3)"')
text = text.replace('annotation_font_color="#718096"', 'annotation_font_color="#111111"')

with open('src/ui/app.py', 'w', encoding='utf-8') as f:
    f.write(text)

print("CSS replacement completed.")
