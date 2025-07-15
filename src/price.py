import yfinance as yf


def get_commodity_data(tickers: list, start_date: str, end_date: str) -> dict:
    """
    Download historical price data for a list of commodity tickers.

    Args:
        tickers: List of ticker symbols.
        start_date: Start date in "YYYY-MM-DD" format.
        end_date: End date in "YYYY-MM-DD" format.

    Returns:
        dict: Dictionary with tickers as keys and DataFrames as values.
    """
    data = {}
    for ticker in tickers:
        print(f"Downloading data for {ticker}...")
        df = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False,
            keepna=True,
            multi_level_index=False
            )
        data[ticker] = df
    return data
