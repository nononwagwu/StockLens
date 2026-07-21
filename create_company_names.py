import pandas as pd
import yfinance as yf
import time

# Load 2018 dataset
df = pd.read_csv("data/2018_Financial_Data.csv")

# Get unique tickers from the first column
tickers = (
    df["Unnamed: 0"]
    .dropna()
    .astype(str)
    .unique()
)

company_names = []

print(f"Looking up {len(tickers)} companies...")

for index, ticker in enumerate(tickers, start=1):
    try:
        stock = yf.Ticker(ticker)

        # Try to retrieve the company's name
        info = stock.get_info()

        company_name = (
            info.get("longName")
            or info.get("shortName")
            or ticker
        )

    except Exception:
        # Historical/delisted tickers may not resolve
        company_name = ticker

    company_names.append({
        "Ticker": ticker,
        "Company Name": company_name
    })

    print(
        f"{index}/{len(tickers)}: "
        f"{ticker} -> {company_name}"
    )

    # Small delay to reduce request rate
    time.sleep(0.1)


# Create mapping DataFrame
company_df = pd.DataFrame(company_names)

# Save locally
company_df.to_csv(
    "data/company_names.csv",
    index=False
)

print(
    "\nFinished! Company names saved to "
    "data/company_names.csv"
)