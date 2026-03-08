import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
df = pd.read_csv("data/Sample - Superstore.csv", encoding='latin-1')
print("\nColumns in dataset:")
print(df.columns)
print(df.head())
print(df.info())
# Convert Order Date to datetime
df['Order Date'] = pd.to_datetime(df['Order Date'])
print("\nData types after conversion:")
print(df.dtypes)
# Aggregate daily sales
daily_sales = df.groupby('Order Date')['Sales'].sum().reset_index()

print(daily_sales.head())



# Create Year-Month column
daily_sales['YearMonth'] = daily_sales['Order Date'].dt.to_period('M')

# Aggregate monthly sales
monthly_sales = daily_sales.groupby('YearMonth')['Sales'].sum().reset_index()

# Convert back to datetime
monthly_sales['YearMonth'] = monthly_sales['YearMonth'].dt.to_timestamp()

print(monthly_sales.head())
print("Number of months:", len(monthly_sales))

# Ensure monthly data is sorted
monthly_sales = monthly_sales.sort_values('YearMonth').reset_index(drop=True)

# Add Year and Month columns
monthly_sales['Year'] = monthly_sales['YearMonth'].dt.year
monthly_sales['Month'] = monthly_sales['YearMonth'].dt.month

# Create numeric time index for regression
monthly_sales['TimeIndex'] = range(len(monthly_sales))

print("\nMonthly Sales with Features:")
print(monthly_sales.head())


X = monthly_sales[['TimeIndex', 'Month']]
y = monthly_sales['Sales']
split_index = int(len(monthly_sales) * 0.8)

X_train = X[:split_index]
X_test = X[split_index:]

y_train = y[:split_index]
y_test = y[split_index:]

print("Training months:", len(X_train))
print("Testing months:", len(X_test))

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("MAE:", mae)
print("MSE:", mse)

print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)


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

print("Future 6 Month Forecast:")
print(future_predictions)

future_dates = pd.date_range(
    start=monthly_sales['YearMonth'].iloc[-1] + pd.DateOffset(months=1),
    periods=future_months,
    freq='MS'
)

plt.figure(figsize=(12,6))

plt.plot(monthly_sales['YearMonth'], monthly_sales['Sales'], label="Historical Sales")
plt.plot(monthly_sales['YearMonth'][split_index:], y_pred, label="Test Predictions")
plt.plot(future_dates, future_predictions, label="Future Forecast", linestyle='--')

plt.title("Sales Forecast Model")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.legend()
plt.show()