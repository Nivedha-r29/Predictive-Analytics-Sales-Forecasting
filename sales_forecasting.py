import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load Dataset
df = pd.read_csv("Sample - Superstore.csv", encoding="latin1")

# Convert Order Date to Date Format
df["Order Date"] = pd.to_datetime(df["Order Date"])

# Monthly Sales Aggregation
monthly_sales = df.groupby(
    pd.Grouper(key="Order Date", freq="M")
)["Sales"].sum().reset_index()

# Create Time Index
monthly_sales["Month_Number"] = range(len(monthly_sales))

# Features and Target
X = monthly_sales[["Month_Number"]]
y = monthly_sales["Sales"]

# Train Linear Regression Model
model = LinearRegression()
model.fit(X, y)

# Historical Predictions
historical_predictions = model.predict(X)

# Model Accuracy
accuracy = r2_score(y, historical_predictions)

print("\nModel Accuracy (R² Score):", round(accuracy, 4))

# Future 12 Months Forecast
future_months = pd.DataFrame({
    "Month_Number": range(
        len(monthly_sales),
        len(monthly_sales) + 12
    )
})

future_sales = model.predict(future_months)

# Future Dates
future_dates = pd.date_range(
    start=monthly_sales["Order Date"].max(),
    periods=13,
    freq="M"
)[1:]

# Forecast DataFrame
forecast_df = pd.DataFrame({
    "Order Date": future_dates,
    "Predicted Sales": future_sales
})

# Export Forecast CSV
forecast_df.to_csv(
    "forecasted_sales.csv",
    index=False
)

print("\nForecast file created successfully!")
print("\nForecast Preview:")
print(forecast_df.head())

# Visualization
plt.figure(figsize=(12, 6))

# Historical Sales
plt.plot(
    monthly_sales["Order Date"],
    monthly_sales["Sales"],
    marker="o",
    label="Historical Sales"
)

# Forecasted Sales
plt.plot(
    forecast_df["Order Date"],
    forecast_df["Predicted Sales"],
    marker="o",
    linestyle="--",
    label="Forecasted Sales"
)

plt.title("Sales Forecasting Using Historical Data")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()