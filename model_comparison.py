import streamlit as st

def show_model_comparison():
    st.header("Traditional ML vs Generative AI")
    
    # Write the introduction
    st.write(
        "One-way analyses of variance (ANOVA) are used to determine the significant differences between the groups considering GANs Models,"
        "and traditional ML Models in terms of regression and financial performance metrics. A confidence level of 95% (α value = 0.05) was used."
        "The mean differences in the significant values were calculated as group 2 – group 1, where group 1 will be GANS models except for"
        "comparison with 60/40 portfolio as it will be in group 1 while GANS portfolio in group 2"
    )
    st.subheader("Overall Comparison")
    col1, col2 = st.columns(2)

    with col1:
        st.write("#### Regression Metrics")
        st.table([
            ["Metric", "GAN Model", "ML_GB", "ML_RF", "ML_SVR"],
            ["MSE", "GAN_LSTM", "0.039 (Low)", "0.015 (Low)", "0.022 (Low)"],
            ["MSE", "GAN_NN", "0.040 (Low)", "0.015 (Low)", "0.022 (Low)"],
            ["MSE", "GAN_TCN", "0.041 (Low)", "0.017 (Low)", "0.024 (Low)"],
            ["RMSE", "GAN_LSTM", "0.130 (Low)", "0.064 (Low)", "0.085 (Low)"],
            ["RMSE", "GAN_NN", "0.132 (Low)", "0.066 (Low)", "0.087 (Low)"],
            ["RMSE", "GAN_TCN", "0.142 (Low)", "0.076 (Low)", "0.097 (Low)"],
            ["MAE", "GAN_LSTM", "0.150 (Low)", "0.085 (Low)", "0.100 (Low)"],
            ["MAE", "GAN_NN", "0.134 (Low)", "0.069 (Low)", "0.084 (Low)"],
            ["MAE", "GAN_TCN", "0.134 (Low)", "0.069 (Low)", "0.084 (Low)"],
            ["R-squared", "GAN_LSTM", "-0.516 (High)", "-0.222 (Medium)", "-0.303 (Medium)"],
            ["R-squared", "GAN_NN", "-0.515 (High)", "-0.221 (Medium)", "-0.303 (Medium)"],
            ["R-squared", "GAN_TCN", "-0.534 (High)", "-0.240 (Medium)", "-0.322 (Medium)"]
        ])
    
    with col2:
        st.write("#### Financial Metrics")
        st.table([
            ["GAN Portfolio", "60/40 Portfolio", "ML_GB Portfolio", "ML_RF Portfolio", "ML_SVR Portfolio"],
            ["GAN_LSTM", "2.793 (High)", "-1.454 (High)", "-0.522 (High)", "-0.381 (Medium)"],
            ["GAN_NN", "2.860 (High)", "-1.521 (High)", "-0.590 (High)", "-0.453 (High)"],
            ["GAN_TCN", "2.912 (High)", "-1.573 (High)", "-0.642 (High)", "-0.505 (Medium)"]
        ])

    st.subheader("Market Regime Comparison")
    # Warming Regime
    st.write("### Warming Regime")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("#### Regression Metrics")
        st.table([
            ["Metric", "GAN Model", "ML_GB", "ML_RF", "ML_SVR"],
            ["MSE", "GAN_LSTM", "0.041 (Low)", "0.018 (Low)", "0.026 (Low)"],
            ["MSE", "GAN_NN", "0.038 (Low)", "0.015 (Low)", "0.023 (Low)"],
            ["MSE", "GAN_TCN", "0.039 (Low)", "0.016 (Low)", "0.025 (Low)"],
            ["RMSE", "GAN_LSTM", "0.149 (Low)", "0.086 (Low)", "0.110 (Low)"],
            ["RMSE", "GAN_NN", "0.129 (Low)", "0.066 (Low)", "0.091 (Low)"],
            ["RMSE", "GAN_TCN", "0.135 (Low)", "0.072 (Low)", "0.097 (Low)"],
            ["MAE", "GAN_LSTM", "0.157 (Low)", "0.093 (Low)", "0.110 (Low)"],
            ["MAE", "GAN_NN", "0.133 (Low)", "0.069 (Low)", "0.086 (Low)"],
            ["MAE", "GAN_TCN", "0.129 (Low)", "0.065 (Low)", "0.083 (Low)"],
            ["R-Squared", "GAN_LSTM", "-0.577 (High)", "-0.264 (Medium)", "-0.381 (Medium)"],
            ["R-Squared", "GAN_NN", "-0.534 (High)", "-0.222 (Medium)", "-0.339 (Medium)"],
            ["R-Squared", "GAN_TCN", "-0.551 (High)", "-0.238 (Medium)", "-0.355 (Medium)"]
        ])
    
    with col2:
        st.write("#### Financial Metrics")
        st.table([
            ["GAN Portfolio", "60/40 Portfolio", "ML_GB Portfolio", "ML_RF Portfolio", "ML_SVR Portfolio"],
            ["GAN_LSTM", "2.822 (High)", "-1.263 (High)", "-0.452 (High)", "-0.381 (Medium)"],
            ["GAN_NN", "2.879 (High)", "-1.320 (High)", "-0.508 (High)", "-0.438 (Medium)"],
            ["GAN_TCN", "3.094 (High)", "-1.535 (High)", "-0.723 (High)", "-0.653 (High)"]
        ])

    # Perfection Regime
    st.write("### Perfection Regime")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("#### Regression Metrics")
        st.table([
            ["Metric", "GAN Model", "ML_GB", "ML_RF", "ML_SVR"],
            ["MSE", "GAN_LSTM", "0.038 (Low)", "-", "0.022 (Low)"],
            ["MSE", "GAN_NN", "0.037 (Low)", "-", "0.021 (Low)"],
            ["MSE", "GAN_TCN", "0.042 (Low)", "-", "0.026 (Low)"],
            ["RMSE", "GAN_LSTM", "0.126 (Low)", "-", "0.088 (Low)"],
            ["RMSE", "GAN_NN", "0.113 (Low)", "-", "0.075 (Low)"],
            ["RMSE", "GAN_TCN", "0.147 (Low)", "0.080 (Low)", "0.109 (Low)"],
            ["MAE", "GAN_LSTM", "0.144 (Low)", "0.081 (Low)", "0.100 (Low)"],
            ["MAE", "GAN_NN", "0.120 (Low)", "-", "0.076 (Low)"],
            ["MAE", "GAN_TCN", "0.137 (Low)", "0.073 (Low)", "0.093 (Low)"],
            ["R-Squared", "GAN_LSTM", "-0.513 (High)", "-", "-0.345 (Medium)"],
            ["R-Squared", "GAN_NN", "-0.456 (High)", "-", "-0.288 (Medium)"],
            ["R-Squared", "GAN_TCN", "-0.551 (High)", "-", "-0.383 (Medium)"]
        ])
    
    with col2:
        st.write("#### Financial Metrics")
        st.table([
            ["GAN Portfolio", "60/40 Portfolio", "ML_GB Portfolio", "ML_RF Portfolio", "ML_SVR Portfolio"],
            ["GAN_LSTM", "2.938 (High)", "-1.308 (High)", "-", "-0.838 (High)"],
            ["GAN_NN", "2.862 (High)", "-1.233 (High)", "-", "-0.762 (High)"],
            ["GAN_TCN", "2.944 (High)", "-1.314 (High)", "-", "-0.844 (High)"]
        ])

    # Cooling Regime
    st.write("### Cooling Regime")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("#### Regression Metrics")
        st.table([
            ["Metric", "GAN Model", "ML_GB", "ML_RF", "ML_SVR"],
            ["MSE", "GAN_LSTM", "0.041 (Low)", "0.017 (Low)", "0.026 (Low)"],
            ["MSE", "GAN_NN", "0.041 (Low)", "0.017 (Low)", "0.026 (Low)"],
            ["MSE", "GAN_TCN", "0.043 (Low)", "0.018 (Low)", "0.027 (Low)"],
            ["RMSE", "GAN_LSTM", "0.144 (Low)", "0.077 (Low)", "0.107 (Low)"],
            ["RMSE", "GAN_NN", "0.141 (Low)", "0.074 (Low)", "0.103 (Low)"],
            ["RMSE", "GAN_TCN", "0.153 (Low)", "0.086 (Low)", "0.116 (Low)"],
            ["MAE", "GAN_LSTM", "0.154 (Low)", "0.089 (Low)", "0.117 (Low)"],
            ["MAE", "GAN_NN", "0.135 (Low)", "0.070 (Low)", "0.098 (Low)"],
            ["MAE", "GAN_TCN", "0.141 (Low)", "0.076 (Low)", "0.103 (Low)"],
            ["R-Squared", "GAN_LSTM", "-0.547 (High)", "-0.253 (Medium)", "-0.388 (Medium)"],
            ["R-Squared", "GAN_NN", "-0.540 (High)", "-0.246 (Medium)", "-0.381 (Medium)"],
            ["R-Squared", "GAN_TCN", "-0.558 (High)", "-0.265 (Medium)", "-0.399 (Medium)"]
        ])
    
    with col2:
        st.write("#### Financial Metrics")
        st.table([
            ["GAN Portfolio", "60/40 Portfolio", "ML_GB Portfolio", "ML_RF Portfolio", "ML_SVR Portfolio"],
            ["GAN_LSTM", "2.843 (High)", "-1.446 (High)", "-", "-"],
            ["GAN_NN", "2.852 (High)", "-1.456 (High)", "-", "-"],
            ["GAN_TCN", "2.782 (High)", "-1.386 (High)", "-", "-"]
        ])

    # Overheating Regime
    st.write("### Stagflation Regime")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("#### Regression Metrics")
        st.table([
            ["Metric", "GAN Model", "ML_GB", "ML_RF", "ML_SVR"],
            ["MSE", "GAN_LSTM", "0.037 (Low)", "0.017 (Low)", "0.026 (Low)"],
            ["MSE", "GAN_NN", "0.043 (Low)", "0.017 (Low)", "0.026 (Low)"],
            ["MSE", "GAN_TCN", "0.041 (Low)", "0.018 (Low)", "0.027 (Low)"],
            ["RMSE", "GAN_LSTM", "0.112 (Low)", "0.075 (Low)", "0.088 (Low)"],
            ["RMSE", "GAN_NN", "0.142 (Low)", "0.076 (Low)", "0.072 (Low)"],
            ["RMSE", "GAN_TCN", "0.133 (Low)", "0.067 (Low)", "0.063 (Low)"],
            ["MAE", "GAN_LSTM", "0.141 (Low)", "0.075 (Low)", "0.066 (Low)"],
            ["MAE", "GAN_NN", "0.148 (Low)", "0.081 (Low)", "0.072 (Low)"],
            ["MAE", "GAN_TCN", "0.130 (Low)", "0.063 (Low)", "0.072 (Low)"],
            ["R-Squared", "GAN_LSTM", "-0.462 (High)", "-0.281 (Medium)", "-0.388 (Medium)"],
            ["R-Squared", "GAN_NN", "-0.570 (High)", "-0.281 (Medium)", "-0.381 (Medium)"],
            ["R-Squared", "GAN_TCN", "-0.540 (High)", "-0.250 (Medium)", "-0.399 (Medium)"]
        ])
    
    with col2:
        st.write("#### Financial Metrics")
        st.table([
            ["GAN Portfolio", "60/40 Portfolio", "ML_GB Portfolio", "ML_RF Portfolio", "ML_SVR Portfolio"],
            ["GAN_LSTM", "2.665 (High)", "-", "-", "-"],
            ["GAN_NN", "3.108 (High)", "-1.152 (High)", "-", "-"],
            ["GAN_TCN", "2.748 (High)", "-0.791 (High)", "-", "-"]
        ])

# Call the function
show_model_comparison()
