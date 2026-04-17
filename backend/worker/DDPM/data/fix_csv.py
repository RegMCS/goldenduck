import yfinance as yf

df = yf.download('AAPL', start='2010-01-01', end='2024-12-31', auto_adjust=True)
df.columns = df.columns.get_level_values(0)
df.index.name = 'Date'
df = df.reset_index()
df.to_csv('worker/DDPM/data/AAPL.csv', index=False)
print(df.columns.tolist())
print(len(df), 'rows')
print(df.head(2))

