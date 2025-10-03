import pandas as pd
import yfinance as yf

# Load Robinhood CSV
df = pd.read_csv("sample_robinhood_data.csv", engine="python", on_bad_lines="skip")

# Keep only Buy/Sell transactions
df = df[df["Trans Code"].isin(["Buy", "Sell"])].copy()

# Clean up Quantity & Price
df["Quantity"] = df["Quantity"].astype(str).str.replace("S", "", regex=False)
df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
df["Price"] = df["Price"].astype(str).str.replace("$", "", regex=False)
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

# Negative quantity for sells
df.loc[df["Trans Code"] == "Sell", "Quantity"] *= -1

# Aggregate into holdings
summary = (
    df.groupby("Instrument")
    .apply(lambda x: pd.Series({
        "Shares": x["Quantity"].sum(),
        "Buy Price": (x["Quantity"] * x["Price"]).sum() / x["Quantity"].sum() if x["Quantity"].sum() > 0 else 0,
        "Date Purchased": x["Activity Date"].min()
    }))
    .reset_index()
)

# Rename Instrument → Ticker first
summary = summary.rename(columns={"Instrument": "Ticker"})

# Add placeholders for dashboard
summary["Current Price"] = None
summary["Sector"] = "Unknown"

# Common ETF tickers
ETF_TICKERS = {
    'SPY', 'QQQ', 'IWM', 'EFA', 'VTI', 'VEA', 'VWO', 'AGG', 'TLT', 'HYG', 'LQD',
    'GLD', 'SLV', 'USO', 'UNG', 'XLF', 'XLK', 'XLE', 'XLI', 'XLV', 'XLY', 'XLP',
    'XLU', 'XLB', 'XLRE', 'XLC', 'SCHD', 'VYM', 'DIV', 'SPHD', 'NOBL', 'DGRO',
    'VIG', 'VOOG', 'VUG', 'MGK', 'SPYG', 'VOOV', 'VTV', 'SPYV', 'VXUS', 'IXUS',
    'ACWI', 'VNQ', 'SCHH', 'IYR', 'BITO', 'ETHE'
}

for i, row in summary.iterrows():
    ticker_symbol = row["Ticker"]
    ticker = yf.Ticker(ticker_symbol)
    
    try:
        # Get latest close price
        summary.at[i, "Current Price"] = ticker.history(period="1d")["Close"].iloc[-1]
        
        # Simple ETF detection
        if ticker_symbol in ETF_TICKERS or ticker.info.get("quoteType") == "ETF":
            summary.at[i, "Sector"] = "ETF"
        else:
            # Regular stock sector
            info = ticker.info
            summary.at[i, "Sector"] = info.get("sector", "Unknown")
            
    except Exception as e:
        print(f"⚠️ Could not fetch data for {ticker_symbol}: {str(e)}")
        # Default classification if we can't fetch data
        if ticker_symbol in ETF_TICKERS:
            summary.at[i, "Sector"] = "ETF"
        else:
            summary.at[i, "Sector"] = "Unknown"
            
# Reorder columns
summary = summary[["Ticker", "Shares", "Buy Price", "Current Price", "Sector", "Date Purchased"]]


# Filter out tickers where shares = 0 (fully sold)
summary = summary[summary["Shares"] > 0]

# Save cleaned file
summary.to_csv("Portfolio.csv", index=False)
print("✅ Portfolio.csv created successfully!")
