import plotly.express as px
import plotly
import pandas as pd


def stock_line_graph(df: pd.DataFrame, ticker: str = None) -> plotly.graph_objs.Figure:
    """
    - The dataframe needs to:
        - have the datetime on the index
        - have the Close column named as closed
    """

    return px.line(
        x=df.index,
        y=df.Close,
        title=f"{ticker} Stock",
        labels={"x": "Date", "y": f"{ticker} Stock Data in USD"},
    )
