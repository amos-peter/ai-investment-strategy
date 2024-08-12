import streamlit as st
import pandas as pd
import os
from datetime import datetime, timedelta
import quantstats as qs
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import altair as alt

@st.cache_data
def load_portfolio_return(file_path):
    portfolio_return = pd.read_csv(file_path).set_index('date')
    portfolio_return.index = pd.to_datetime(portfolio_return.index)
    return portfolio_return

@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path).set_index('date')
    data.index = pd.to_datetime(data.index)
    return data

@st.cache_data
def calculate_sim_stat(df):
    final_stat_df = pd.DataFrame()
    for col in df.columns:
        row_df = len(df[col].dropna())
        stat_df = qs.reports.metrics(df[col], mode='full', display=False)
        stat_df.columns = [col]
        stat_df.index = stat_df.index.str.strip().str.lower()
        stat_df.index = stat_df.index.str.strip()
        if row_df < 252:
            stat_list = ['start period', 'end period', 'cumulative return', 'sharpe', 'sortino']
            percentage_columns = ['cumulative return']
        else:
            stat_list = ['start period', 'end period', 'cumulative return', 'all-time (ann.)', 'sharpe', 'sortino']
            percentage_columns = ['cumulative return', 'all-time (ann.)']
        stat_df = stat_df[stat_df.index.isin(stat_list)]
        
        for col in percentage_columns:
            stat_df.loc[col] = round(stat_df.loc[col].astype(float)*100,3)
        stat_df = stat_df.rename(index={'cumulative return': 'cumulative return (%)', 'all-time (ann.)': 'annualized return (%)'})
        final_stat_df = pd.concat([final_stat_df, stat_df], axis=1)
    return final_stat_df

# Define the function to show portfolio factsheet
def show_portfolio_factsheet():
    # Path to the model data
    model_data_path = 'model_data'
    file_path = os.path.join(model_data_path, 'combine_sim_rtn.csv')

    # Load the portfolio return data
    portfolio_return = load_portfolio_return(file_path)

    # Function to filter data based on year window
    def filter_by_year_window(df, window):
        if window == 'all-time':
            filtered_df = df
        elif window == 'ytd':
            filtered_df = df[df.index.year == datetime.now().year]
        else:
            years_ago = datetime.now() - timedelta(days=365 * int(window))
            filtered_df = df[df.index >= years_ago]
        return filtered_df

    # Streamlit layout
    st.title("Portfolio Backtesting/Simulation")

    # Filter options
    year_windows = ['all-time', 'ytd', 1, 2, 3, 4, 5, 6, 7]
    selected_window = st.selectbox("Select Year Window", year_windows)
    filtered_df = filter_by_year_window(portfolio_return, selected_window)

    # Compute cumulative returns and normalize to 100
    normalized_df = compute_cumulative_returns(filtered_df)

    # Calculate financial metrics
    overall_sim_stats = calculate_sim_stat(filtered_df)

    # Load and preprocess data for regression metrics
    data_df = load_data(os.path.join(model_data_path, 'data_regime.csv'))

    get_target = [col for col in data_df.columns if "_22d_fwd_target" in col]
    raw_df = load_data(os.path.join(model_data_path, 'raw_pricing.csv'))

    asset_ls = raw_df.columns.tolist()
    pricing_df = raw_df[asset_ls].copy()

    model_gb = preprocess_data('model_sim_gb', asset_ls)
    model_rf = preprocess_data('model_sim_rf', asset_ls)
    model_svr = preprocess_data('model_sim_svr', asset_ls)
    model_gan_nn = preprocess_data('model_sim_nn', asset_ls)
    model_gan_lstm = preprocess_data('model_sim_lstm', asset_ls)
    model_gan_tcn = preprocess_data('model_sim_tcn', asset_ls)

    # Filter the target data based on the selected date range
    overall_data_df = data_df[data_df.index.isin(model_gb.index)][get_target].copy()
    overall_data_df = overall_data_df[(overall_data_df.index >= filtered_df.index.min()) & (overall_data_df.index <= filtered_df.index.max())]

    # Evaluate models
    overall_ml_gb = evaluate_model(model_gb, overall_data_df, asset_ls, get_target)
    overall_ml_rf = evaluate_model(model_rf, overall_data_df, asset_ls, get_target)
    overall_ml_svr = evaluate_model(model_svr, overall_data_df, asset_ls, get_target)
    overall_gan_nn = evaluate_model(model_gan_nn, overall_data_df, asset_ls, get_target)
    overall_gan_lstm = evaluate_model(model_gan_lstm, overall_data_df, asset_ls, get_target)
    overall_gan_tcn = evaluate_model(model_gan_tcn, overall_data_df, asset_ls, get_target)

    # Top container layout
    st.header("Portfolio Overview")
    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("Details")
        st.write(f"**Start Date:** {normalized_df.index.min().date()}")
        st.write(f"**End Date:** {normalized_df.index.max().date()}")
        st.write("**Portfolios:**")
        portfolios = normalized_df.columns.tolist()
        for portfolio in portfolios:
            st.write(f"- {portfolio}")

    with col2:
        st.subheader("Performance Chart")
        chart_data = normalized_df.reset_index().melt('date', var_name='portfolio', value_name='value')
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
    st.write(overall_sim_stats)

    st.subheader("Regression Metrics")

    # Divide regression metrics into rows
    row1_col1, row1_col2, row1_col3 = st.columns(3)
    row2_col1, row2_col2, row2_col3 = st.columns(3)

    with row1_col1:
        st.write("GAN - Neural Network")
        st.write(overall_gan_nn)

    with row1_col2:
        st.write("GAN - LSTM")
        st.write(overall_gan_lstm)

    with row1_col3:
        st.write("GAN - TCN")
        st.write(overall_gan_tcn)

    with row2_col1:
        st.write("Gradient Boosting")
        st.write(overall_ml_gb)

    with row2_col2:
        st.write("Random Forest")
        st.write(overall_ml_rf)

    with row2_col3:
        st.write("Support Vector Regression")
        st.write(overall_ml_svr)

if __name__ == "__main__":
    show_portfolio_factsheet()
