import pandas as pd
import asyncio
from db import DbProvider
import datetime
import sys

db = DbProvider()

# === Constants ===
WIN_COEF = 1.923
BASE_BET = 5 / WIN_COEF

# === Predefined patterns by length ===
PREDEFINED_PATTERNS_BY_LENGTH = {

    4: ["DDDD"],        # 3 patterns for length 4
    5: ["DDDDD"],              # 2 patterns for length 5
    6: ["DDDDDD"],           # 3 patterns for length 6
    7: ["DDDDDDD"],          # 2 patterns for length 7
    8: ["DDDDDDDD"],        # 2 patterns for length 8
    #9: ["DDDDDDDDD"],      # 2 patterns for length 9
    #10: ["DDDDDDDDDD"],   # 2 patterns for length 10
    #11: ["DDDDDDDDDDD"], # 2 patterns for length 11
    #12: ["DDDDDDDDDDDD"]# 2 patterns for length 12
}

"""PREDEFINED_PATTERNS_BY_LENGTH = {
    4: ["UUUU"],        # 3 patterns for length 4
    5: ["UUUUU"],              # 2 patterns for length 5
    6: ["UUUUUU"],           # 3 patterns for length 6
    7: ["UUUUUUU"],          # 2 patterns for length 7
    8: ["UUUUUUUU"],        # 2 patterns for length 8
    9: ["UUUUUUUUU"],      # 2 patterns for length 9
    #10: ["DDDDDDDDDD"],   # 2 patterns for length 10
    #11: ["DDDDDDDDDDD"], # 2 patterns for length 11
    #12: ["DDDDDDDDDDDD"]# 2 patterns for length 12
}"""

# === Utils ===
def to_microseconds_timestamp(date_string: str) -> int:
    dt = datetime.datetime.strptime(date_string, "%Y-%m-%d")
    return int(dt.timestamp() * 1_000_000)

# === Simulation for one pattern length, all months & years ===
async def simulate_length_multi_year(df, years_months, pattern_length):
    log_filename = f"simulation-{pattern_length}.txt"
    log_file = open(log_filename, "w", encoding="utf-8")

    class TeeLogger:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, data):
            for s in self.streams:
                s.write(data)
                s.flush()
        def flush(self):
            for s in self.streams:
                s.flush()

    sys.stdout = TeeLogger(sys.__stdout__, log_file)

    df["open_time"] = pd.to_datetime(df["open_time"], unit="us")

    month_summary = {}

    for year, months in years_months.items():
        for month in months:
            start_dt = datetime.datetime(year, month, 1)
            if month == 12:
                end_dt = datetime.datetime(year, 12, 31, 23, 59, 59)
            else:
                end_dt = datetime.datetime(year, month + 1, 1) - datetime.timedelta(seconds=1)

            month_data = df[(df["open_time"] >= start_dt) & (df["open_time"] <= end_dt)].copy()
            if month_data.empty:
                print(f"No data for {year}-{month:02}\n")
                continue

            print(f"\n=== {year}-{month:02} | Pattern length: {pattern_length} ===")

            patterns_for_length = PREDEFINED_PATTERNS_BY_LENGTH.get(pattern_length, [])
            if not patterns_for_length:
                print(f"No predefined patterns for length {pattern_length}")
                continue

            month_data["direction"] = month_data.apply(lambda x: "U" if x["close"] > x["open"] else "D", axis=1)
            directions = month_data["direction"].tolist()
            dates = month_data["open_time"]

            month_balance = 0.0
            month_max_bet = BASE_BET
            month_min_balance = 0.0

            for pattern in patterns_for_length:
                balance = 0.0
                lose_streak = 0
                wins = 0
                losses = 0
                min_balance = 0.0
                max_balance = 0.0
                max_bet = BASE_BET
                total_bets = 0
                sessions_lost = 0
                current_bet = BASE_BET
                total_losses_amount = 0.0
                daily_stats = {}

                print(f"\n--- Pattern {pattern} ---")

                for i, real in enumerate(directions):
                    current_date = dates.iloc[i]
                    pattern_index = lose_streak % len(pattern)
                    expected = pattern[pattern_index]
                    bet_direction = "D" if expected == "U" else "U"
                    total_bets += 1

                    pattern_display = "".join(
                        [f"[{ch}]" if j == pattern_index else f" {ch} " for j, ch in enumerate(pattern)]
                    )

                    print(
                        f"[{i+1}/{len(directions)}] {current_date.strftime('%Y-%m-%d %H:%M')} "
                        f"Real: {real} | Bet: {bet_direction} | Pattern: {pattern_display} | "
                        f"Bet: {current_bet:.2f} | Balance: {balance:.2f}"
                    )

                    if real == bet_direction:
                        profit = current_bet * (WIN_COEF - 1)
                        balance += profit
                        wins += 1
                        lose_streak = 0
                        total_losses_amount = 0.0
                        current_bet = BASE_BET
                    else:
                        balance -= current_bet
                        losses += 1
                        lose_streak += 1
                        total_losses_amount += current_bet
                        current_bet = (total_losses_amount + BASE_BET) / (WIN_COEF - 1)

                    max_bet = max(max_bet, current_bet)
                    min_balance = min(min_balance, balance)
                    max_balance = max(max_balance, balance)

                    if lose_streak >= len(pattern):
                        sessions_lost += 1
                        lose_streak = 0
                        total_losses_amount = 0.0
                        current_bet = BASE_BET

                month_balance += balance
                month_max_bet = max(month_max_bet, max_bet)
                month_min_balance = min(month_min_balance, min_balance)

                print(f"\nPattern {pattern} summary for {year}-{month:02}:")
                print(f"Bets: {total_bets}, Wins: {wins}, Losses: {losses}")
                print(f"Final balance: {balance:.2f}")
                print(f"Min balance: {min_balance:.2f}, Max balance: {max_balance:.2f}, Max bet: {max_bet:.2f}")
                print(f"Sessions lost: {sessions_lost}")
                print("="*60)

            month_summary[f"{year}-{month:02}"] = {
                "final_balance": month_balance,
                "max_bet": month_max_bet,
                "min_balance": month_min_balance
            }

    # Print monthly summary table
    print("\n=== MONTHLY SUMMARY ===")
    print(f"{'Month':<10}{'Final Balance':>15}{'Max Bet':>12}{'Min Balance':>15}")
    print("-"*55)
    for m, stats in month_summary.items():
        print(f"{m:<10}{stats['final_balance']:>15.2f}{stats['max_bet']:>12.2f}{stats['min_balance']:>15.2f}")
    print("="*55)

    sys.stdout = sys.__stdout__
    log_file.close()
    print(f"✅ Simulation for pattern length {pattern_length} saved to {log_filename}")

# === Data Loader ===
async def load_data():
    res = await db.select("c_15m", ["open_time", "open", "close"], None, 0, "ASC")
    df = pd.DataFrame(res, columns=["open_time", "open", "close"]).reset_index(drop=True)
    return df

# === Main Runner ===
async def run_all_lengths():
    df = await load_data()
    years_months = {
        2022: list(range(1, 13)),  # full year
        2023: list(range(1, 13)),  # full year
        2024: list(range(1, 13)),  # full year
        2025: list(range(1, 12))   # Jan → Nov
    }
    for length in range(4, 9):  # pattern lengths
        await simulate_length_multi_year(df, years_months, length)

if __name__ == "__main__":
    asyncio.run(run_all_lengths())
