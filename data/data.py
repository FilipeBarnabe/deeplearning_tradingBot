from pandas_datareader import data
import pandas as pd


def get_data(ticker: str, dates: list[str, str]) -> pd.DataFrame:
    """
    ticker: stock ticker (exampe: apple ticker -> AAPL)
    dates: list of two dates of str type, start date and end_date.
        - from when do you want the data
        - to when you want the data
        - example: ["2012-10-21", "2022-10-20"]
    """

    panel_data = data.DataReader(ticker, "yahoo", dates[0], dates[1])

    print(f"{ticker} data aquired \n")

    return panel_data


# Test the function
# print(get_data(ticker="AAPL", dates=["2012-10-21", "2022-10-20"]))
