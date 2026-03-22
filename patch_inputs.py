import re

with open('src/ui/app.py', 'r', encoding='utf-8') as f:
    text = f.read()

old_css = """/* Form Inputs fix */
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
}"""

new_css = """/* Form Inputs Solid Styling */
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
}"""

# Try direct replacement
if old_css in text:
    text = text.replace(old_css, new_css)
else:
    # Fallback regex just in case
    pattern = re.compile(r'/\* Form Inputs fix \*/.*?box-shadow: none !important;\n}', re.DOTALL)
    text = pattern.sub(new_css, text)

# Just to be absolutely sure text color is forced black on any input element:
text = text.replace('color: var(--text-primary-input) !important;', 'color: #111111 !important; -webkit-text-fill-color: #111111 !important;')

with open('src/ui/app.py', 'w', encoding='utf-8') as f:
    f.write(text)

print("Inputs CSS patched!")
