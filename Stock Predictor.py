import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

# Collect stock data
sp500 = yf.Ticker("BTC-USD")
#sp500 = yf.Ticker("QQQ")
#sp500 = sp500.history(period="max")

sp500 = sp500.history(period="max")

# Removes all data before 1990
sp500 = sp500.loc["1990-01-01":].copy()

# Clean data by removing dividends and stock splits columns
del sp500["Dividends"]
del sp500["Stock Splits"]

# Creates column of price of stock next day
sp500["Tomorrow"] = sp500["Close"].shift(-1)

# Creates column to show if price went up or down next day
sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)

# Create model for learning
model = RandomForestClassifier(n_estimators = 100, min_samples_split = 100, random_state = 1)

# Split data for testing
train = sp500.iloc[:-100]
test = sp500.iloc[-100:]

# Define predictors for future stock prices
predictors = ["Close", "Volume", "Open", "High", "Low"]

# Train model to predict target using predictors columns
model.fit(train[predictors], train["Target"])

# Train model
RandomForestClassifier(min_samples_split = 100, random_state = 1)

# Generates predictions from model using test set with predictors
preds = model.predict(test[predictors])

# Turns numpy array into pandas series
preds = pd.Series(preds, index = test.index)

# Combines tests with predictions
combined = pd.concat([test["Target"], preds], axis = 1)

# Conducts testing with model, creates predictions, and returns combined tests and predictions
def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index = test.index, name = "Predictions")
    combined = pd.concat([test["Target"], preds], axis = 1)
    return combined

# At every point in data train model based on past data and test on future data
def backtest(data, model, predictors, start = 2500, step = 250):
    all_predictions = []

    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i + step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
    
    return pd.concat(all_predictions)

# Time intervals to compare current price to price over past time periods
time_intervals = [2,5,60,250,1000]
new_predictors = []

# Calculates rolling averages to compare close prices to past close prices over a past time interval
for interval in time_intervals:
    rolling_averages = sp500.rolling(interval).mean()
    
    # Compares current close to rolling averages of past close
    ratio_column = f"Close_Ratio_{interval}"
    sp500[ratio_column] = sp500["Close"] / rolling_averages["Close"]
    
    # Finds amount of times price went up in past time interval
    trend_column = f"Trend_{interval}"
    sp500[trend_column] = sp500.shift(1).rolling(interval).sum()["Target"]
    
    new_predictors+= [ratio_column, trend_column]

# Removes data rows with not enough past data
sp500 = sp500.dropna()

# Generate predictions
predictions = backtest(sp500, model, new_predictors)

# Print precision of predictions
print(precision_score(predictions["Target"], predictions["Predictions"]))