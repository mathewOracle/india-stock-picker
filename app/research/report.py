"""Plain-text summaries of research results (CLI output; the web UI renders the same dicts)."""


def fmt_block(b: dict) -> str:
    if not b.get("n"):
        return "n=0"
    pf = f"{b['profit_factor']:.2f}" if b.get("profit_factor") else "-"
    return (f"n={b['n']:<4} win={b['win_rate']:.0%} avg={b['avg_return']:+.2%} med={b['median_return']:+.2%} "
            f"expR={b['expectancy_r']:+.2f} PF={pf} hold={b['avg_bars']:.0f}d | control={b['control_return']:+.2%} "
            f"edge={b['edge']:+.2%} t={b['t_stat']:.2f}")


def print_backtest(s: dict) -> None:
    print(f"\nBACKTEST {s['period']} | {s['evaluations']} evaluations | {s['seconds']}s | "
          f"skipped={s['skipped']} failures={s['failures']}")
    for level, block in s["universe"].items():
        print(f"  {level.upper():6} {fmt_block(block)}")
        if block.get("exits"):
            print(f"         exits: {block['exits']}")
    print(f"  PRE-REGISTERED TEST PASSED: {s['passed']}")
    for half in s["halves"]:
        print(f"  half {half['label']}: {fmt_block(half)}")
    for year in s["by_year"]:
        print(f"  {year['year']}: {fmt_block(year)}")
    stocks = s["stocks"]
    print(f"  stocks passing: {sum(v['passed'] for v in stocks.values())} of {len(stocks)}")
