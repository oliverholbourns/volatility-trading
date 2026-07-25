import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_rolling_volatility(df, window=21):
    returns = df['Close'].pct_change()
    rolling_std = returns.rolling(window=window).std()
    return rolling_std * np.sqrt(252)  # Annualize the volatility

#sigmoid function
def hypothesis(X, theta):
    z = X.dot(theta)
    return 1 / (1 + np.exp(-z))

def cost_function(X, y, theta):
    m = len(y)
    h = hypothesis(X, theta)
    epsilon = 1e-10  # small value to avoid log(0)
    cost = (-1/m) * np.sum(y * np.log(h + epsilon) + (1-y) * np.log(1 - h + epsilon))
    return cost

def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    cost_history = []

    for i in range(iterations):
        h = hypothesis(X, theta)
        gradient = (1/m) * X.transpose().dot(h - y)
        theta -= learning_rate * gradient
        cost = cost_function(X, y, theta)
        cost_history.append(cost)
    
    return theta, cost_history

def label_data(df):
    df['future_return'] = df['Close'].shift(-1) / df['Close'] - 1 #next day return
    df['y'] = (df['future_return'] > 0.015).astype(int)
    return df

def main():
    #download stock data from yahoo finance
    stock = yf.download('AAPL', start='2026-01-01', end='2026-06-30', interval='1d')

    #convert stock to pandas dataframe
    df = pd.DataFrame(stock)
    df.columns = df.columns.droplevel(1) #drop ticker level from multiindex columns

    plt.plot(df['Close'])
    plt.title('AAPL Stock Price')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()

    df['rolling_volatility'] = calculate_rolling_volatility(df)
    df = label_data(df)
    df = df.dropna()

    X = np.column_stack([np.ones(len(df)), df['rolling_volatility'].values])
    y = df['y'].values

    theta = np.zeros(X.shape[1])  # Initialize theta to zeros
    theta, cost_history = gradient_descent(X, y, theta, learning_rate=0.01, iterations=1000)

    annualized_vol = calculate_rolling_volatility(df)

    plt.plot(cost_history)
    plt.title('Cost over iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.show()


if __name__ == "__main__":
    main()