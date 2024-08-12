import streamlit as st
import pandas as pd
import altair as alt
import pickle
import os
from datetime import datetime, timedelta  # Add this import

# Load precomputed data
model_data_path = 'model_data'
with open(os.path.join(model_data_path, 'precomputed_data.pkl'), 'rb') as f:
    precomputed_data = pickle.load(f)

# Define the function to show portfolio factsheet
def show_portfolio_factsheet():
    st.title("Portfolio Backtesting/Simulation")

    # Filter options
    year_windows = ['all-time', 'ytd', 1, 2, 3, 4, 5, 6, 7]
    selected_window = st.selectbox("Select Year Window", year_windows)

    # Load precomputed data based on the selected window
    cumulative_returns = precomputed_data[selected_window]['cumulative_returns']
    financial_metrics = precomputed_data[selected_window]['financial_metrics']

    # Top container layout
    st.header("Portfolio Overview")
    col1, col2 = st.columns([1, 3])  # Adjust the ratio to make the right column wider

    with col1:
        st.subheader("Details")
        st.write(f"**Start Date:** {cumulative_returns.index.min().date()}")
        st.write(f"**End Date:** {cumulative_returns.index.max().date()}")
        st.write("**Portfolios:**")
        portfolios = cumulative_returns.columns.tolist()
        for portfolio in portfolios:
            st.write(f"- {portfolio}")

    with col2:
        st.subheader("Performance Chart")
        chart_data = cumulative_returns.reset_index().melt('date', var_name='portfolio', value_name='value')
        base = alt.Chart(chart_data).encode(
            x='date:T',
            y=alt.Y('value:Q', scale=alt.Scale(domain=[chart_data['value'].min() * 0.9, chart_data['value'].max() * 1.1])),
            color='portfolio:N'
        )
        line = base.mark_line()
        st.altair_chart(line, use_container_width=True)

    # Bottom container layout for Performance Details
    st.header("Performance Details")

    st.subheader("Financial Metrics")
    st.write(financial_metrics)

# Run the app
if __name__ == "__main__":
    show_portfolio_factsheet()
