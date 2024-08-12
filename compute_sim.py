import pandas as pd
import quantstats as qs
import os
import pickle
from datetime import datetime, timedelta  # Add this import

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

# Precompute and save the cumulative returns and financial metrics
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
    
    precomputed_data[window] = {
        'cumulative_returns': cumulative_returns,
        'financial_metrics': financial_metrics
    }

# Save the precomputed data to a .pkl file
with open(os.path.join(model_data_path, 'precomputed_data.pkl'), 'wb') as f:
    pickle.dump(precomputed_data, f)
