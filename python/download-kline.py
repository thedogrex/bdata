#!/usr/bin/env python

"""
  script to download klines.
  set the absolute path destination folder for STORE_DIRECTORY, and run

  e.g. STORE_DIRECTORY=/data/ ./download-kline.py

"""
import asyncio
import zipfile
import csv
import io
import sys
from datetime import *
import pandas as pd
import re
from db import DbProvider
from enums import *
from utility import download_file, get_all_symbols, get_parser, get_start_end_date_objects, convert_to_date_object, \
  get_path

db = DbProvider()

async def download_daily_klines(trading_type, symbols, num_symbols, intervals, dates, start_date, end_date, folder):
  current = 0
  date_range = None

  if start_date and end_date:
    date_range = start_date + " " + end_date

  if not start_date:
    start_date = START_DATE
  else:
    start_date = convert_to_date_object(start_date)

  if not end_date:
    end_date = END_DATE
  else:
    end_date = convert_to_date_object(end_date)

  #Get valid intervals for daily
  intervals = list(set(intervals) & set(DAILY_INTERVALS))
  print("Found {} symbols".format(num_symbols))

  for symbol in symbols:
    print("[{}/{}] - start download daily {} klines ".format(current+1, num_symbols, symbol))
    for interval in intervals:
      for date in dates:
        current_date = convert_to_date_object(date)
        if current_date >= start_date and current_date <= end_date:
          path = get_path(trading_type, "klines", "daily", symbol, interval)
          file_name = "{}-{}-{}.zip".format(symbol.upper(), interval, date)
          zip_path = download_file(path, file_name, date_range, folder)

          if zip_path:
            await fill_in_zip_to_db(zip_path)

    current += 1


async def fill_in_zip_to_db(path):
    with zipfile.ZipFile(path, 'r') as z:
        csv_name = find_csv_in_zip(z)
        if csv_name is None:
            print(f"# No file found inside {path}", file=sys.stderr)
            return
        with z.open(csv_name, 'r') as raw:
            # decode as utf-8 (or fallback to latin1)
            try:
                textf = io.TextIOWrapper(raw, encoding='utf-8')
                print(f'process: {textf}')
                await process_file_like(textf)
            except UnicodeDecodeError:
                raw.seek(0)
                textf = io.TextIOWrapper(raw, encoding='latin-1')
                await process_file_like(textf)

def find_csv_in_zip(z: zipfile.ZipFile):
    # prefer .csv files, otherwise first file with lines
    for name in z.namelist():
        if name.lower().endswith('.csv'):
            return name
    # fallback: return first file that looks like text
    for name in z.namelist():
        if not name.endswith('/'):
            return name
    return None

async def process_file_like(f):
    candles = []

    reader = csv.reader(f)
    # skip metadata until we find a row whose first column is a 13-digit timestamp
    start_row = None
    rows = []
    for row in reader:
        # Binance Kline format:
        # [Open time, Open, High, Low, Close, Volume, Close time, ...]
        if not row:
            continue

        if row[0].isdigit():
            open_price = float(row[1])
            close_price = float(row[4])
            direction = "U" if close_price > open_price else "D"
            optime = row[0]

            # convert to nanoseconds
            if(len(optime)==13):
                optime = optime + '000'

            await db.insert_one("candles", fields={"open_time": int(optime),
                                             "open_price" :  open_price,
                                             "close_price" : close_price,
                                             "dir" : direction})

    print(f'candles {candles}')

if __name__ == "__main__":

    folder = '.'
    parser = get_parser('klines')

    symbols = ["BTCUSDT"]
    num_symbols = len(symbols)

    period = convert_to_date_object(datetime.today().strftime('%Y-%m-%d')) - convert_to_date_object(
        PERIOD_START_DATE)
    dates = pd.date_range(end=datetime.today(), periods=period.days + 1).to_pydatetime().tolist()
    dates = [date.strftime("%Y-%m-%d") for date in dates]

    print(dates)

    start_date = '2018-01-01'

    asyncio.run(download_daily_klines('spot', symbols, num_symbols, ["1h"], dates, start_date, None, folder))

