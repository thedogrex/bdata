import pandas as pd
import itertools
from collections import Counter
import asyncio
from db import DbProvider
import datetime
import sys
from io import StringIO

db = DbProvider()

# === Constants ===
WIN_COEF = 1.96          # payout multiplier
BASE_BET = 3             # fixed base bet
K_MOST_RARE_PATTERNS = 4 # simulate top-K rarest patterns
MAX_PATTERN_LEN = 4      # max pattern length to search for

# === Utils ===
def to_microseconds_timestamp(date_string: str) -> int:
    dt = datetime.datetime.strptime(date_string, "%Y-%m-%d")
    return int(dt.timestamp() * 1_000_000)

# === Core Simulation with Logging, Daily Stats, and Pattern Tracking ===
async def simulate_betting(df, training_start_date, training_end_date, simulation_start_date, simulation_end_date):

    # --- setup logging to file and console ---
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

    log_file = open("logs.txt", "w", encoding="utf-8")
    sys.stdout = TeeLogger(sys.__stdout__, log_file)

    # === Convert to datetime and slice data ===
    training_start = pd.to_datetime(to_microseconds_timestamp(training_start_date), unit="us")
    training_end = pd.to_datetime(to_microseconds_timestamp(training_end_date), unit="us")
    sim_start = pd.to_datetime(to_microseconds_timestamp(simulation_start_date), unit="us")
    sim_end = pd.to_datetime(to_microseconds_timestamp(simulation_end_date), unit="us")

    df["open_time"] = pd.to_datetime(df["open_time"], unit="us")
    df.set_index("open_time", inplace=True)

    train_data = df[(df.index >= training_start) & (df.index <= training_end)]
    sim_data = df[(df.index >= sim_start) & (df.index <= sim_end)]

    print(f"Loaded {len(sim_data)} simulation candles")

    # === Find rare patterns ===
    rare_patterns_df = await get_rare_patterns(train_data, N=MAX_PATTERN_LEN)
    rare_patterns_df = rare_patterns_df.head(K_MOST_RARE_PATTERNS)

    # === Prepare simulation data ===
    sim_data["direction"] = sim_data.apply(lambda x: "U" if x["close"] > x["open"] else "D", axis=1)
    directions = sim_data["direction"].tolist()
    dates = sim_data.index.date

    # === Per-pattern simulation ===
    pattern_stats = {}

    for _, row in rare_patterns_df.iterrows():
        pattern_tuple = row["pattern"]
        pattern = "".join(["U" if x == 1 else "D" for x in pattern_tuple])

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

        print(f"\n=== Starting Pattern {pattern} ===")
        for i, real in enumerate(directions):
            current_date = dates[i]
            pattern_index = lose_streak % len(pattern)
            expected = pattern[pattern_index]
            bet_direction = "D" if expected == "U" else "U"

            # Visualize pattern progress
            pattern_display = "".join(
                [f"[{ch}]" if j == pattern_index else f" {ch} " for j, ch in enumerate(pattern)]
            )

            total_bets += 1

            # Print last few candles
            print(
                f"[{i+1}/{len(directions)}] {sim_data.index[i].strftime('%Y-%m-%d %H:%M')} "
                f"Real: {real} | Bet: {bet_direction} | Pattern: {pattern_display} | "
                f"Bet: {current_bet:.2f} | Balance: {balance:.2f}"
            )

            # Simulate result
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

            max_bet = max(max_bet, current_bet)
            min_balance = min(min_balance, balance)
            max_balance = max(max_balance, balance)

            # Daily stats
            if current_date not in daily_stats:
                daily_stats[current_date] = {
                    "wins": 0,
                    "losses": 0,
                    "profit": 0.0,
                    "min_balance": balance,
                    "max_balance": balance,
                }

            daily_stats[current_date]["wins"] = wins
            daily_stats[current_date]["losses"] = losses
            daily_stats[current_date]["profit"] = balance
            daily_stats[current_date]["min_balance"] = min(
                daily_stats[current_date]["min_balance"], balance
            )
            daily_stats[current_date]["max_balance"] = max(
                daily_stats[current_date]["max_balance"], balance
            )

            # update current bet if lose
            if real != bet_direction:
                current_bet = (total_losses_amount + BASE_BET) / (WIN_COEF - 1)

            # Session lost (reset to base)
            if lose_streak >= len(pattern):
                sessions_lost += 1
                lose_streak = 0
                total_losses_amount = 0.0
                current_bet = BASE_BET


        # Store pattern results
        pattern_stats[pattern] = {
            "wins": wins,
            "losses": losses,
            "balance": balance,
            "min_balance": min_balance,
            "max_balance": max_balance,
            "max_bet": max_bet,
            "total_bets": total_bets,
            "sessions_lost": sessions_lost,
            "daily_stats": daily_stats,
        }

        # Pattern summary
        print(f"\n=== Pattern {pattern} ===")
        print(f"Bets: {total_bets}, Wins: {wins}, Losses: {losses}")
        print(f"Win rate: {wins / (wins + losses) * 100:.2f}%")
        print(f"Final balance: {balance:.2f}")
        print(f"Min balance: {min_balance:.2f}")
        print(f"Max balance: {max_balance:.2f}")
        print(f"Max bet: {max_bet:.2f}")
        print(f"Sessions lost: {sessions_lost}")
        print("-" * 60)

        # Daily summary
        print("📅 Daily stats:")
        for d, s in daily_stats.items():
            print(
                f"{d}: Profit={s['profit']:.2f}, Wins={s['wins']}, Losses={s['losses']}, "
                f"MinBal={s['min_balance']:.2f}, MaxBal={s['max_balance']:.2f}"
            )
        print("=" * 80)

    # === Global summary ===
    total_wins = sum(p["wins"] for p in pattern_stats.values())
    total_losses = sum(p["losses"] for p in pattern_stats.values())
    total_balance = sum(p["balance"] for p in pattern_stats.values())
    global_min_balance = min(p["min_balance"] for p in pattern_stats.values())
    global_max_bet = max(p["max_bet"] for p in pattern_stats.values())
    total_bets = sum(p["total_bets"] for p in pattern_stats.values())
    global_sessions_lost = sum(p["sessions_lost"] for p in pattern_stats.values())

    print("\n*** FINAL SUMMARY ACROSS ALL PATTERNS ***")
    print(f"Total bets: {total_bets}")
    print(f"Total wins: {total_wins}, Total losses: {total_losses}")
    print(f"Overall win rate: {total_wins / (total_wins + total_losses) * 100:.2f}%")
    print(f"Total balance across bots: {total_balance:.2f}")
    print(f"Lowest balance reached (any bot): {global_min_balance:.2f}")
    print(f"Max bet required (any bot): {global_max_bet:.2f}")
    print(f"Total session losses: {global_sessions_lost}")
    print("=" * 60)

    # restore stdout
    sys.stdout = sys.__stdout__
    log_file.close()
    print("✅ Logs saved to logs.txt")

# === Rare Pattern Finder ===
async def get_rare_patterns(df, N=MAX_PATTERN_LEN):
    df["direction"] = (df["close"] > df["open"]).astype(int)
    all_patterns = list(itertools.product([0, 1], repeat=N))
    observed = [tuple(df["direction"].iloc[i:i+N]) for i in range(len(df) - N + 1)]
    counts = Counter(observed)
    counts.update({p: 0 for p in all_patterns if p not in counts})
    df_counts = pd.DataFrame(counts.items(), columns=["pattern", "count"]).sort_values("count").reset_index(drop=True)
    return df_counts

# === Data Loader ===
async def load_data():
    res = await db.select("c_15m", ["open_time", "open", "close"], None, 0, "ASC")
    df = pd.DataFrame(res, columns=["open_time", "open", "close"]).reset_index(drop=True)
    return df

# === Main Runner ===
async def run_simulation():
    df = await load_data()
    await simulate_betting(
        df,
        training_start_date="2020-01-01",
        training_end_date="2025-01-01",
        simulation_start_date="2025-10-01",
        simulation_end_date="2025-11-01",
    )

if __name__ == "__main__":
    asyncio.run(run_simulation())
