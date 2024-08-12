import quantstats as qs
import streamlit as st
import pandas as pd
import os
import numpy as np
import altair as alt
import IPython
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Function to show market regime performances
def show_market_regime():
    # Path to the model data
    model_data_path = 'model_data'
    file_path = os.path.join(model_data_path, 'combine_sim_rtn.csv')

    # Load the portfolio return data
    portfolio_return = pd.read_csv(file_path).set_index('date')
    portfolio_return.index = pd.to_datetime(portfolio_return.index)

    # Function to calculate financial metrics using quantstats
    def calculate_sim_stat(df):
        final_stat_df = pd.DataFrame()
        for col in df.columns:
            stat_df = qs.reports.metrics(df[col], mode='full', display=False)
            stat_df.columns = [col]
            stat_df.index = stat_df.index.str.strip().str.lower()
            stat_df.index = stat_df.index.str.strip()
            stat_list = ['cumulative return', 'sharpe', 'sortino']
            stat_df = stat_df[stat_df.index.isin(stat_list)]
            percentage_columns = ['cumulative return']
            for col in percentage_columns:
                stat_df.loc[col] = round(stat_df.loc[col].astype(float) * 100, 3)
            stat_df = stat_df.rename(index={'cumulative return': 'cumulative return (%)'})
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

    # Streamlit layout
    st.title("Different Market Condition Performances")

    # Load and preprocess data for regression metrics
    data_df = pd.read_csv(os.path.join(model_data_path, 'data_regime.csv'))
    data_df['date'] = pd.to_datetime(data_df['date'])
    data_df.set_index('date', inplace=True)

    # Define regime labels
    regime_labels = {0: 'Warming', 1: 'Perfection', 2: 'Cooling', 3: 'Overheating'}
    data_df['regime_label'] = data_df['regime'].map(regime_labels)

    # Create an Altair chart for regime changes
    regime_chart = alt.Chart(data_df.reset_index()).mark_line(point=True).encode(
        x=alt.X('date:T', title='Date'),
        y=alt.Y('regime_label:N', title='Regime', sort=['Warming', 'Perfection', 'Cooling', 'Overheating']),
        tooltip=['date:T', 'regime_label:N']
    ).properties(
        width=600,
        height=250,  # Increased height for the chart
        title="Regime Changes Over Time"
    )

    st.altair_chart(regime_chart, use_container_width=True)

    # Regime Definitions - one sentence
    st.write(
        "The regimes are defined as Warming (high inflation, high growth), Perfection (low inflation, high growth), Cooling (low inflation, low growth), and Overheating (high inflation, low growth)."
    )

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

    regime_titles = [
        "Warming Market Condition",
        "Perfection Market Condition",
        "Cooling Market Condition",
        "Overheating Market Condition"
    ]

    for regime in range(4):
        # Filter the target data based on the current regime and matching indices
        regime_indices = data_df[data_df['regime'] == regime].index

        model_gb_regime = model_gb[model_gb.index.isin(regime_indices)]
        model_rf_regime = model_rf[model_rf.index.isin(regime_indices)]
        model_svr_regime = model_svr[model_svr.index.isin(regime_indices)]
        model_gan_nn_regime = model_gan_nn[model_gan_nn.index.isin(regime_indices)]
        model_gan_lstm_regime = model_gan_lstm[model_gan_lstm.index.isin(regime_indices)]
        model_gan_tcn_regime = model_gan_tcn[model_gan_tcn.index.isin(regime_indices)]

        regime_data_df = data_df[data_df.index.isin(model_gb_regime.index)][get_target].copy()

        # Evaluate models
        ml_gb_metrics = evaluate_model(model_gb_regime, regime_data_df, asset_ls, get_target)
        ml_rf_metrics = evaluate_model(model_rf_regime, regime_data_df, asset_ls, get_target)
        ml_svr_metrics = evaluate_model(model_svr_regime, regime_data_df, asset_ls, get_target)
        gan_nn_metrics = evaluate_model(model_gan_nn_regime, regime_data_df, asset_ls, get_target)
        gan_lstm_metrics = evaluate_model(model_gan_lstm_regime, regime_data_df, asset_ls, get_target)
        gan_tcn_metrics = evaluate_model(model_gan_tcn_regime, regime_data_df, asset_ls, get_target)

        # Calculate financial metrics for the current regime
        financial_metrics_df = portfolio_return[portfolio_return.index.isin(regime_indices)].copy()
        sim_stats = calculate_sim_stat(financial_metrics_df)

        # Regime section
        st.header(regime_titles[regime])

        st.subheader("Financial Metrics")
        st.write(sim_stats)

        st.subheader("Regression Metrics")

        # Divide regression metrics into rows
        row1_col1, row1_col2, row1_col3 = st.columns(3)
        row2_col1, row2_col2, row2_col3 = st.columns(3)

        with row1_col1:
            st.write("GAN - Neural Network")
            st.write(gan_nn_metrics)

        with row1_col2:
            st.write("GAN - LSTM")
            st.write(gan_lstm_metrics)

        with row1_col3:
            st.write("GAN - TCN")
            st.write(gan_tcn_metrics)

        with row2_col1:
            st.write("Gradient Boosting")
            st.write(ml_gb_metrics)

        with row2_col2:
            st.write("Random Forest")
            st.write(ml_rf_metrics)

        with row2_col3:
            st.write("Support Vector Regression")
            st.write(ml_svr_metrics)

# Run the app
if __name__ == "__main__":
    show_market_regime()
