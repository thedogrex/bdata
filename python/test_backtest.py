import requests
import json

r = requests.post('http://127.0.0.1:8000/api/backtest', json={
    'strategy': 'xgboost',
    'train_start': '2022-01-01',
    'train_end': '2025-06-30',
    'test_start': '2025-07-01',
    'test_end': '2025-12-31',
    'horizons': [1],
    'table': 'c_5m'
}, timeout=600)

print(f"Status: {r.status_code}")
d = r.json()
if 'error' in d:
    print(f"Error: {d['error']}")
else:
    h = d.get('horizons', {}).get('1', {})
    print(f"Accuracy: {h.get('accuracy_pct')}%")
    print(f"Signals: {h.get('signals')}")
    print(f"Correct: {h.get('correct')}")
    print(f"Skipped: {h.get('skipped')}")
    print(f"Total time: {d.get('total_time_sec')}s")
    print(f"UP accuracy: {h.get('up_accuracy')}%")
    print(f"DOWN accuracy: {h.get('down_accuracy')}%")
    print(f"Max win streak: {h.get('streaks', {}).get('max_win_streak')}")
    print(f"Max lose streak: {h.get('streaks', {}).get('max_lose_streak')}")
    print("\nMonthly:")
    for m in h.get('monthly', []):
        print(f"  {m['month']}: {m['accuracy']}% ({m['correct']}/{m['total']})")
