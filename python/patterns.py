from db import DbProvider
import asyncio

db = DbProvider()

async def run():
    res = await db.select("candles", ["open_time", "open_price", "close_price", "dir"],
                          None, 0, "ASC")


    from collections import defaultdict, Counter

    pattern_length = 5
    patterns = []
    pattern_start_times = defaultdict(list)

    # Build patterns of 'dir' values
    for i in range(len(res) - pattern_length + 1):
        seq = ''.join(r[3] for r in res[i:i+pattern_length])
        start_time = res[i][0]
        patterns.append(seq)
        pattern_start_times[seq].append(start_time)

    # Count pattern occurrences
    pattern_counts = Counter(patterns)

    # Sort by frequency (rarest first)
    sorted_patterns = sorted(pattern_counts.items(), key=lambda x: x[1])

    # Print results
    print(f"{'Pattern':<10} {'Count':<5} {'First open_time'}")
    for pattern, count in sorted_patterns:
        first_time = pattern_start_times[pattern][0]
        print(f"{pattern:<10} {count:<5} {first_time}")


    martingale_anti_pattern(res)


from collections import Counter, defaultdict

from collections import Counter

def martingale_anti_pattern(res):
    pattern_length = 5
    total_bets = 24*7*4*12
    base_bet = 40
    multiplier = 2
    max_steps = 5
    START_N = 0
    N_RAREST_PATTERNS = 2

    dirs = [r[3] for r in res]

    START_N = len(res) - total_bets

    # Count all patterns
    pattern_counts = Counter(''.join(dirs[i:i+pattern_length]) for i in range(len(dirs) - pattern_length))
    rarest_patterns = [p for p, _ in pattern_counts.most_common()[:-N_RAREST_PATTERNS-1:-1]]

    print(f"Rarest {len(rarest_patterns)} patterns:", rarest_patterns)

    # Initialize bots
    bots = {}
    for pattern in rarest_patterns:
        bots[pattern] = {
            "pattern": pattern,
            "profit": 0,
            "loss_streak": 0,
            "bet": base_bet,
            "index": 0,  # which character in pattern we're on
            "total_bets": 0,
        }

    for i in range(total_bets):
        if i + START_N + pattern_length >= len(dirs):
            break

        current_window = ''.join(dirs[i+START_N:i+START_N+pattern_length])
        next_dir = dirs[i+START_N+pattern_length]

        print(f'                                               {current_window}')

        for pattern, bot in bots.items():
            # Determine anti-pattern bet: opposite of current index in pattern
            pattern_char = pattern[bot["index"]]
            bet_dir = 'D' if pattern_char == 'U' else 'U'

            #print(f'-------------------------------------------BET {bet_dir}')

            # Check if loss occurs
            # Loss occurs if real next candle matches the pattern's current character
            if next_dir == pattern_char:
                # bot loses
                bot["profit"] -= bot["bet"]
                bot["loss_streak"] += 1


                if bot["loss_streak"] >= max_steps:
                    print(f'[{pattern}] -------------------------------------- STREAK LOSS bet: {bot["bet"]}',
                          flush=True)
                    bot["bet"] = base_bet
                    bot["loss_streak"] = 0
                    bot["index"] = 0
                else:
                    print(f'[{pattern}] loss {bot["bet"]} balance: {bot["profit"]}', flush=True)
                    bot["bet"] *= multiplier
                    bot["index"] = (bot["index"] + 1) % len(pattern)

            else:
                # bot win
                bot["profit"] += bot["bet"]
                bot["loss_streak"] = 0
                bot["index"] = 0
                print(f'[{pattern}] win {bot["bet"]} balance: {bot["profit"]}')
                bot["bet"] = base_bet

            bot["total_bets"] += 1

    # Summary
    total_profit = sum(bot["profit"] for bot in bots.values())
    print("\n=== Simulation Completed ===")
    for pattern, bot in bots.items():
        print(f"[{pattern}] | Profit: {bot['profit']:.2f} | Bets: {bot['total_bets']}")
    print(f"TOTAL PROFIT: {total_profit:.2f}")

    return bots, total_profit
asyncio.run(run())
