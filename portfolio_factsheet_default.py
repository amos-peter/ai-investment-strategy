import streamlit as st
import pandas as pd
import os
from datetime import datetime, timedelta
import altair as alt
import quantstats as qs
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Path to the model data
model_data_path = 'model_data'
file_path = os.path.join(model_data_path, 'combine_sim_rtn.csv')

# Load the portfolio return data
portfolio_return = pd.read_csv(file_path).set_index('date')
portfolio_return.index = pd.to_datetime(portfolio_return.index)

# Function to filter data based on year window
def filter_by_year_window(df, window):
    if window == 'ytd':
        filtered_df = df[df.index.year == datetime.now().year]
    else:
        years_ago = datetime.now() - timedelta(days=365 * int(window))
        filtered_df = df[df.index >= years_ago]
    return filtered_df

# Function to compute cumulative returns and normalize to a base value of 100
def compute_cumulative_returns(df):
    cumulative_returns = (1 + df).cumprod()
    return (cumulative_returns / cumulative_returns.iloc[0]) * 100

# Function to calculate financial metrics using quantstats
def calculate_sim_stat(df):
    final_stat_df = pd.DataFrame()
    for col in df.columns:
        # Calculate all stat
        stat_df = qs.reports.metrics(df[col], mode='full', display=False)
        stat_df.columns = [col]
        stat_df.index = stat_df.index.str.strip().str.lower()
        stat_df.index = stat_df.index.str.strip()
        final_stat_df = pd.concat([final_stat_df, stat_df], axis=1)
    return final_stat_df

# Function to preprocess model data
def preprocess_data(get_model, get_asset_col):
    model_df = pd.read_csv(os.path.join(model_data_path, f'{get_model}.csv')).set_index('date')
    model_df.index = pd.to_datetime(model_df.index)
    model_score_df = model_df.copy()
    model_score_df.columns = get_asset_col
    return model_score_df

# Function to evaluate model
def evaluate_model(model_df, target_df, sim_targets, real_targets):
    sim_comparison_df = pd.concat([model_df, target_df], axis=1)
    metrics = {}
    for sim_target, real_target in zip(sim_targets, real_targets):
        y_true = sim_comparison_df[real_target].dropna()
        y_pred = sim_comparison_df[sim_target].dropna()
        
        common_idx = y_true.index.intersection(y_pred.index)
        y_true = y_true.loc[common_idx]
        y_pred = y_pred.loc[common_idx]
        
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        metrics[sim_target] = {'mse': mse, 'rmse': rmse, 'mae': mae, 'r_square': r2}
    
    return pd.DataFrame(metrics).T

# Set the page configuration to wide mode
st.set_page_config(layout="wide")

# Streamlit layout
st.title("Portfolio Factsheet")

# Filter options
st.header("Filter Options")
filter_option = st.selectbox("Filter by", ["Yearly Window", "Date Range"])

if filter_option == "Yearly Window":
    year_windows = ['ytd', 1, 2, 3, 4, 5, 6, 7]
    selected_window = st.selectbox("Select Year Window", year_windows)
    filtered_df = filter_by_year_window(portfolio_return, selected_window)
else:
    date_from = st.date_input('From', value=portfolio_return.index.min().date())
    date_to = st.date_input('To', value=portfolio_return.index.max().date())
    filtered_df = portfolio_return.loc[date_from:date_to]

# Compute cumulative returns and normalize to 100
normalized_df = compute_cumulative_returns(filtered_df)

# Calculate financial metrics
overall_sim_stats = calculate_sim_stat(filtered_df)

# Load and preprocess data for regression metrics
data_df = pd.read_csv(os.path.join(model_data_path, 'data_regime.csv')).set_index('date')
data_df.index = pd.to_datetime(data_df.index)

get_target = [col for col in data_df.columns if "_22d_fwd_target" in col]
raw_df = pd.read_csv(os.path.join(model_data_path, 'raw_pricing.csv')).set_index('date')
raw_df.index = pd.to_datetime(raw_df.index)

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
col1, col2 = st.columns([1, 3])  # Adjust the ratio to make the right column wider

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

# Divide regression metrics into columns
col1, col2 = st.columns(2)

with col1:
    st.write("GAN - Neural Network")
    st.write(overall_gan_nn)

    st.write("GAN - LSTM")
    st.write(overall_gan_lstm)

    st.write("GAN - TCN")
    st.write(overall_gan_tcn)

with col2:
    st.write("Gradient Boosting")
    st.write(overall_ml_gb)
    
    st.write("Random Forest")
    st.write(overall_ml_rf)
    
    st.write("Support Vector Regression")
    st.write(overall_ml_svr)
