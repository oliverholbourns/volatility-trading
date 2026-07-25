import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_volatility(df):
    returns = df['Close'].pct_change().dropna()
    std_dev = returns.std()
    volatility = std_dev * np.sqrt(252)  # Annualize the volatility
    return volatility

#sigmoid function
def hypothesis(x, fvector):
    z = (fvector.transpose()).dot(x)
    return 1 / (1 + np.exp(-z))

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

    annualized_vol = calculate_volatility(df)
    fvector = [annualized_vol]
    print(annualized_vol)

if __name__ == "__main__":
    main()