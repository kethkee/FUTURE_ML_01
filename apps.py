import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Sales Forecast Dashboard", layout="wide")

st.title("📊 Sales Forecasting Dashboard")
st.markdown("Forecasting monthly sales using Linear Regression (Trend + Seasonality Model)")

# Load Data
df = pd.read_csv("data/Sample - Superstore.csv", encoding='latin-1')
df['Order Date'] = pd.to_datetime(df['Order Date'])

# Aggregate Daily
daily_sales = df.groupby('Order Date')['Sales'].sum().reset_index()

# Monthly Aggregation
daily_sales['YearMonth'] = daily_sales['Order Date'].dt.to_period('M')
monthly_sales = daily_sales.groupby('YearMonth')['Sales'].sum().reset_index()
monthly_sales['YearMonth'] = monthly_sales['YearMonth'].dt.to_timestamp()
monthly_sales = monthly_sales.sort_values('YearMonth').reset_index(drop=True)

# Feature Engineering
monthly_sales['Month'] = monthly_sales['YearMonth'].dt.month
monthly_sales['TimeIndex'] = range(len(monthly_sales))

X = monthly_sales[['TimeIndex', 'Month']]
y = monthly_sales['Sales']

# Train Model
model = LinearRegression()
model.fit(X, y)

# Forecast Future 6 Months
future_months = 6
last_time_index = monthly_sales['TimeIndex'].max()

future_time_index = np.arange(last_time_index + 1, last_time_index + 1 + future_months)
last_month = monthly_sales['Month'].iloc[-1]
future_month_values = [(last_month + i - 1) % 12 + 1 for i in range(1, future_months + 1)]

future_data = pd.DataFrame({
    'TimeIndex': future_time_index,
    'Month': future_month_values
})

future_predictions = model.predict(future_data)

future_dates = pd.date_range(
    start=monthly_sales['YearMonth'].iloc[-1] + pd.DateOffset(months=1),
    periods=future_months,
    freq='MS'
)

# KPI Section
col1, col2, col3 = st.columns(3)

col1.metric("Total Historical Revenue", f"{int(monthly_sales['Sales'].sum()):,}")
col2.metric("Average Monthly Sales", f"{int(monthly_sales['Sales'].mean()):,}")
col3.metric("Next Month Forecast", f"{int(future_predictions[0]):,}")

st.markdown("---")

# Plot
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(monthly_sales['YearMonth'], monthly_sales['Sales'], label="Historical Sales")
ax.plot(future_dates, future_predictions, label="Future Forecast", linestyle='--')
ax.set_title("Sales Forecast")
ax.set_xlabel("Month")
ax.set_ylabel("Sales")
ax.legend()

st.pyplot(fig)

st.markdown("---")

# Future Forecast Table
st.subheader("📅 Future 6-Month Forecast")

forecast_df = pd.DataFrame({
    "Month": future_dates,
    "Predicted Sales": future_predictions.astype(int)
})

st.dataframe(forecast_df)

st.markdown("""
### 📌 Business Insight
- Sales show consistent upward trend from 2014 to 2017.
- Seasonal peaks observed toward year-end.
- Forecast predicts continued growth.
- Useful for inventory planning and staffing decisions.
""")