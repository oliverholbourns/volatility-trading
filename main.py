import yfinance as yf
import pandas as pd
import numpy as np

stock = yf.download('AAPL', start='2026-01-01', end='2026-06-30', interval='1d')

#convert stock to pandas dataframe
df = pd.DataFrame(stock)
df.columns = df.columns.droplevel(1) #drop ticker level from multiindex columns

def calculate_volatility(df):
    std_dev = df['Close'].std()
    volatility = std_dev * np.sqrt(252)  # Annualize the volatility
    return volatility

annualized_vol = calculate_volatility(df)