import pandas as pd
from datetime import datetime, timezone
import asyncio
from db import DbProvider

# =========================
# Count Up and Down Candles Function
# =========================
def count_up_down_candles(df):
    """
    Count the number of up and down candles in the DataFrame.
    A candle is considered 'up' if the close is greater than the open.
    A candle is considered 'down' if the close is less than the open.
    """
    # Calculate up and down candles
    df['up'] = (df['close'] > df['open']).astype(int)
    df['down'] = (df['close'] < df['open']).astype(int)

    up_count = df['up'].sum()
    down_count = df['down'].sum()

    return up_count, down_count

# =========================
# Main Function to Count Up and Down Candles in Date Range
# =========================
async def count_candles_in_date_range(start_date, end_date):
    # Convert start and end date to timestamps (in microseconds)
    start_timestamp = int(start_date.timestamp() * 1_000_000)
    end_timestamp = int(end_date.timestamp() * 1_000_000)

    db = DbProvider()
    print("Init Mysql provider")

    # Fetch the data from the database (adjust the query if needed)
    res = await db.select("candles", ["open_time", "open_price", "close_price"], None, 0, "ASC")
    df = pd.DataFrame(res, columns=["open_time", "open", "close"]).reset_index(drop=True)

    # Filter data within the specified date range
    df['open_time'] = pd.to_datetime(df['open_time'], unit='us', utc=True)
    df_filtered = df[(df['open_time'] >= start_date) & (df['open_time'] <= end_date)]

    # Count the up and down candles
    up_count, down_count = count_up_down_candles(df_filtered)

    print(f"Up Candles: {up_count}, Down Candles: {down_count}")

# =========================
# Example Usage (Specify Start and End Dates)
# =========================
if __name__ == "__main__":
    # Specify start and end date (for example)
    start_date = datetime(2024, 1, 1, tzinfo=timezone.utc)  # Start date
    end_date = datetime(2025, 10, 25, tzinfo=timezone.utc)  # End date

    # Run the function to count candles in the given range
    asyncio.run(count_candles_in_date_range(start_date, end_date))
