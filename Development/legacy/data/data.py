from pandas_datareader import data
import pandas as pd


def get_data(ticker: str, dates: list[str, str]) -> pd.DataFrame:
    """
    The data comes in USD's
    ticker: stock ticker (exampe: apple ticker -> AAPL)
    dates: list of two dates of str type, start date and end_date.
        - from when do you want the data
        - to when you want the data
        - example: ["2012-10-21", "2022-10-20"]
    """

    # API call to get the data as a dataframe
    panel_data = data.DataReader(ticker, "yahoo", dates[0], dates[1])

    # Resample the data to normalize it, the data comes without any data on the (days that the markets are close, weekends)
    panel_data = panel_data.resample("D").max()

    # Fill the data in the closed markets with the data from the previous trading day, because it is their true value
    panel_data = panel_data.fillna(method="ffill")

    print(f"\n{ticker} data acquired")

    return panel_data


# Test the function
# print(get_data(ticker="AAPL", dates=["2012-10-21", "2022-10-20"]))
