import streamlit as st

st.set_page_config(layout="wide")

# Importing the content of portfolio_factsheet.py
from simulation import show_portfolio_factsheet
from market_regime import show_market_regime
from model_comparison import show_model_comparison

# Set up session state for tab selection
if 'selected_tab' not in st.session_state:
    st.session_state['selected_tab'] = 'Full Backtesting/Simulation'

# Create tabs with a key to track the selected tab
tab_labels = ["Full Backtesting/Simulation", "Different Market Performances", "Traditional ML vs Generative AI"]
selected_tab = st.radio("Navigate to", tab_labels, index=tab_labels.index(st.session_state['selected_tab']), key='selected_tab_radio')

# Render the content of the selected tab
if selected_tab == "Full Backtesting/Simulation":
    show_portfolio_factsheet()
elif selected_tab == "Different Market Performances":
    show_market_regime()
elif selected_tab == "Traditional ML vs Generative AI":
    show_model_comparison()

# Update the session state with the current tab
st.session_state['selected_tab'] = selected_tab
