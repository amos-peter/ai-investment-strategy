import streamlit as st

st.set_page_config(layout="wide")


# Importing the content of portfolio_factsheet.py
from simulation import show_portfolio_factsheet
from market_regime import show_market_regime
from model_comparison import show_model_comparison

# Create tabs
tab1, tab2, tab3 = st.tabs(["Full Backtesting/Simulation", "Different Market Performances", "Traditional ML vs Generative AI"])

with tab1:
    show_portfolio_factsheet()

with tab2:
    show_market_regime()

with tab3:
    show_model_comparison()
