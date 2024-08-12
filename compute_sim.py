import pandas as pd
import quantstats as qs
import os
import pickle
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Define paths
model_data_path = 'model_data'
file_path = os.path.join(model_data_path, 'combine_sim_rtn.csv')

# Load the portfolio return data
portfolio_return = pd.read_csv(file_path).set_index('date')
portfolio_return.index = pd.to_datetime(portfolio_return.index)

# Function to compute cumulative returns and normalize to a base value of 100
def compute_cumulative_returns(df):
    cumulative_returns = (1 + df).cumprod()
    return (cumulative_returns / cumulative_returns.iloc[0]) * 100

# Function to calculate financial metrics using quantstats
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
            stat_df.loc[col] = round(stat_df.loc[col].astype(float) * 100, 3)
        stat_df = stat_df.rename(index={'cumulative return': 'cumulative return (%)', 'all-time (ann.)': 'annualized return (%)'})
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

# Precompute and save the cumulative returns, financial metrics, and regression metrics
precomputed_data = {}
for window in ['all-time', 'ytd', 1, 2, 3, 4, 5, 6, 7]:
    if window == 'all-time':
        filtered_df = portfolio_return
    elif window == 'ytd':
        filtered_df = portfolio_return[portfolio_return.index.year == datetime.now().year]
    else:
        years_ago = datetime.now() - timedelta(days=365 * int(window))
        filtered_df = portfolio_return[portfolio_return.index >= years_ago]
    
    cumulative_returns = compute_cumulative_returns(filtered_df)
    financial_metrics = calculate_sim_stat(filtered_df)
    
    # Load and preprocess data for regression metrics
    data_df = pd.read_csv(os.path.join(model_data_path, 'data_regime.csv')).set_index('date')
    data_df.index = pd.to_datetime(data_df.index)

    get_target = [col for col in data_df.columns if "_22d_fwd_target" in col]
    raw_df = pd.read_csv(os.path.join(model_data_path, 'raw_pricing.csv')).set_index('date')
    raw_df.index = pd.to_datetime(raw_df.index)

    asset_ls = raw_df.columns.tolist()
    
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
    regression_metrics = {
        'ml_gb': evaluate_model(model_gb, overall_data_df, asset_ls, get_target),
        'ml_rf': evaluate_model(model_rf, overall_data_df, asset_ls, get_target),
        'ml_svr': evaluate_model(model_svr, overall_data_df, asset_ls, get_target),
        'gan_nn': evaluate_model(model_gan_nn, overall_data_df, asset_ls, get_target),
        'gan_lstm': evaluate_model(model_gan_lstm, overall_data_df, asset_ls, get_target),
        'gan_tcn': evaluate_model(model_gan_tcn, overall_data_df, asset_ls, get_target),
    }
    
    precomputed_data[window] = {
        'cumulative_returns': cumulative_returns,
        'financial_metrics': financial_metrics,
        'regression_metrics': regression_metrics
    }

# Save the precomputed data to a .pkl file
with open(os.path.join(model_data_path, 'precomputed_data.pkl'), 'wb') as f:
    pickle.dump(precomputed_data, f)
