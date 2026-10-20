"""How much 5-minute history does Yahoo give us, and how fast? (feasibility probe)"""

import time

from app import data

if __name__ == "__main__":
    session = data._fresh_session()
    symbols = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"]
    for period in ("60d", "90d"):
        t = time.time()
        try:
            frames = data._download(symbols, session, period=period, interval="5m")
        except Exception as exc:
            print(period, "FAILED:", str(exc)[:150])
            continue
        for s, f in frames.items():
            days = sorted({d.date() for d in f.index})
            print(f"{period} {s}: {len(f)} bars, {len(days)} sessions, {days[0]} -> {days[-1]}, "
                  f"bars/session median {int(f.groupby(f.index.date).size().median())}")
        print(f"  {time.time() - t:.1f}s")
