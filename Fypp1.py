#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
import base64
from sklearn.linear_model import LinearRegression
import plotly.graph_objs as go
import warnings

warnings.filterwarnings("ignore")


def convert_to_numeric(value):
    if isinstance(value, str):
        if 'M' in value:
            return float(value.replace('M', '').replace(',', '')) * 1e6
        elif 'B' in value:
            return float(value.replace('B', '').replace(',', '')) * 1e9
        else:
            return float(value.replace(',', ''))
    return value


def download_csv(dataframe):
    csv = dataframe.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="forecast.csv">Download Forecast Data</a>'
    return href


def add_bounds(df):
    df['upper_bound'] = df['yhat'] * 1.05
    df['lower_bound'] = df['yhat'] * 0.95
    return df


def naive_forecast(data, forecast_days, target_column):
    last_value = data[target_column].iloc[-1]
    future_dates = pd.date_range(start=data['Date'].iloc[-1] + pd.Timedelta(days=1), periods=forecast_days)
    forecast_df = pd.DataFrame({'ds': future_dates, 'yhat': [last_value] * forecast_days})
    return add_bounds(forecast_df)


def moving_average_forecast(data, forecast_days, window=5, target_column='Close'):
    ma = data[target_column].rolling(window=window).mean().iloc[-1]
    future_dates = pd.date_range(start=data['Date'].iloc[-1] + pd.Timedelta(days=1), periods=forecast_days)
    forecast_df = pd.DataFrame({'ds': future_dates, 'yhat': [ma] * forecast_days})
    return add_bounds(forecast_df)


def linear_regression_forecast(data, forecast_days, target_column):
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    data = data.dropna(subset=['Date', target_column])

    if data[target_column].isnull().sum() > 0:
        st.warning(f"Missing values found in {target_column}, applying selected fill method.")
        fill_method = st.selectbox("Select fill method for missing values in Linear Regression", 
                                   ['Mean', 'Median', 'Forward', 'Backward'])
        if fill_method == 'Mean':
            data[target_column] = data[target_column].fillna(data[target_column].mean())
        elif fill_method == 'Median':
            data[target_column] = data[target_column].fillna(data[target_column].median())
        elif fill_method == 'Forward':
            data[target_column] = data[target_column].ffill()
        elif fill_method == 'Backward':
            data[target_column] = data[target_column].bfill()

    data['ds_ordinal'] = data['Date'].map(pd.Timestamp.toordinal)

    model = LinearRegression()
    model.fit(data[['ds_ordinal']], data[target_column])

    last_date = data['Date'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)
    future_ordinals = future_dates.map(pd.Timestamp.toordinal)
    y_pred = model.predict(future_ordinals.to_numpy().reshape(-1, 1))

    forecast_df = pd.DataFrame({'ds': future_dates, 'yhat': y_pred})
    return add_bounds(forecast_df)


def plot_selection(forecast_df, original_df, target_column):
    plot_type = st.selectbox(
        "📊 Choose a plot type:",
        [
            "Original vs Forecast",
            "Interactive Plot",
            "Trend Plot (General)",
            "Forecast Distribution Plot (General)"
        ],
        key=f"plot_type_{target_column}"
    )

    if plot_type == "Trend Plot (General)":
        fig, ax = plt.subplots()
        ax.plot(original_df['Date'], original_df[target_column], label=f"{target_column} Price")
        plt.title(f"Trend of {target_column} Over Time")
        plt.xlabel("Date")
        plt.ylabel(f"{target_column} Price")
        plt.legend()
        st.pyplot(fig)

    elif plot_type == "Forecast Distribution Plot (General)":
        fig, ax = plt.subplots()
        ax.hist(forecast_df['yhat'], bins=30, color='orange', edgecolor='black')
        plt.title("Distribution of Forecasted Values")
        plt.xlabel("Forecasted Values")
        plt.ylabel("Frequency")
        st.pyplot(fig)

    elif plot_type == "Original vs Forecast":
        fig, ax = plt.subplots()
        ax.plot(original_df['Date'], original_df[target_column], label='Original')
        ax.plot(forecast_df['ds'], forecast_df['yhat'], label='Forecast', color='orange')
        ax.fill_between(forecast_df['ds'], forecast_df['lower_bound'], forecast_df['upper_bound'], color='orange', alpha=0.2, label='Forecast Range')
        plt.title(f"{target_column}: Original vs Forecast")
        plt.xlabel("Date")
        plt.ylabel(target_column)
        plt.legend()
        st.pyplot(fig)

    elif plot_type == "Interactive Plot":
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=original_df['Date'], y=original_df[target_column], mode='lines', name='Original'))
        fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat'], mode='lines', name='Forecast'))
        fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['upper_bound'], mode='lines', name='Upper Bound', line=dict(dash='dot')))
        fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['lower_bound'], mode='lines', name='Lower Bound', line=dict(dash='dot')))
        fig.update_layout(title=f"{target_column}: Interactive Forecast Plot", xaxis_title="Date", yaxis_title=target_column)
        st.plotly_chart(fig)

    st.markdown(f"📝 Forecast generated for **{target_column}** column.")


def main():
    st.title("📈 Stock Shares Insights & Forecasting Dashboard")
    uploaded_file = st.file_uploader("📤 Upload your CSV file", type="csv")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("📄 Raw Dataset:")
        st.dataframe(data)

        st.subheader("📌 Select Columns")
        all_columns = data.columns.tolist()
        date_col = st.selectbox("Select Date column", all_columns)
        target_columns = st.multiselect("Select Target column(s)", all_columns)

        if date_col and target_columns:
            data[date_col] = pd.to_datetime(data[date_col], infer_datetime_format=True)
            data = data[[date_col] + target_columns].rename(columns={date_col: "Date"})

            for col in target_columns:
                data[col] = data[col].apply(convert_to_numeric)

            st.subheader("🧼 Preprocessing")
            if data.isnull().sum().sum() > 0:
                st.warning("Missing values detected.")
                missing_opt = st.selectbox("Fill NaNs with", ['Drop', 'Mean', 'Median', 'Forward', 'Backward'])
                if missing_opt == 'Drop':
                    data.dropna(inplace=True)
                elif missing_opt == 'Mean':
                    for col in target_columns:
                        data[col] = data[col].fillna(data[col].mean())
                elif missing_opt == 'Median':
                    for col in target_columns:
                        data[col] = data[col].fillna(data[col].median())
                elif missing_opt == 'Forward':
                    for col in target_columns:
                        data[col] = data[col].ffill()
                elif missing_opt == 'Backward':
                    for col in target_columns:
                        data[col] = data[col].bfill()

            st.subheader("📈 Forecasting Model")
            model_choice = st.selectbox("Choose Model", ['Prophet', 'Naive', 'Moving Average', 'Linear Regression'])
            period = st.number_input("Forecast Period (days)", min_value=1, value=30)

            forecast_data = None

            if model_choice == 'Prophet':
                if len(data) > 10:
                    for target_column in target_columns:
                        prophet_df = data.rename(columns={"Date": "ds", target_column: "y"})
                        prophet_df = prophet_df.dropna()
                        model = Prophet()
                        model.fit(prophet_df)
                        future = model.make_future_dataframe(periods=period)
                        forecast = model.predict(future)
                        forecast_data = forecast[['ds', 'yhat']].tail(period)
                        forecast_data = add_bounds(forecast_data)
                        st.dataframe(forecast_data)
                        plot_selection(forecast_data, data, target_column)
                else:
                    st.warning("Not enough data for Prophet. Please upload a longer time series.")

            elif model_choice == 'Naive':
                for target_column in target_columns:
                    forecast_data = naive_forecast(data, period, target_column)
                    st.dataframe(forecast_data)
                    plot_selection(forecast_data, data, target_column)

            elif model_choice == 'Moving Average':
                for target_column in target_columns:
                    forecast_data = moving_average_forecast(data, period, target_column=target_column)
                    st.dataframe(forecast_data)
                    plot_selection(forecast_data, data, target_column)

            elif model_choice == 'Linear Regression':
                for target_column in target_columns:
                    data = data.dropna(subset=[target_column])
                    forecast_data = linear_regression_forecast(data, period, target_column)
                    st.dataframe(forecast_data)
                    plot_selection(forecast_data, data, target_column)

            if forecast_data is not None:
                st.markdown(download_csv(forecast_data), unsafe_allow_html=True)

        else:
            st.warning("Please select valid date and value columns.")
    else:
        st.info("📁 Please upload a CSV file to continue.")


if __name__ == "__main__":
    main()

