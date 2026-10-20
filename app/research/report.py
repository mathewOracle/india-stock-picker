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


def print_intraday(result: dict) -> None:
    sessions, first, last = result["sessions"]
    inc = result["increment"]
    print(f"INTRADAY BACKTEST  {sessions} sessions ({first} to {last}) | {result['trades']} trades "
          f"({inc['new_trades']} new) | {result['seconds']}s | failures {len(result['failures'])}")
    print(f"  {'setup':26}{'n':>6}{'days':>5}{'win%':>6}{'avg net':>9}{'avg R':>7}{'t':>6}  status   (pass: t >= {result['bonferroni_t']:.2f})")
    for key, b in sorted(result["setups"].items()):
        if not b["n"]:
            print(f"  {key:26}{0:>6}  never fired with reward/risk >= {result['requirements']['min_rr']}")
            continue
        print(f"  {key:26}{b['n']:>6}{b['days']:>5}{b['win_rate']:>6.0%}{b['avg_return']:>+9.3%}{b['avg_r']:>+7.2f}"
              f"{b['t_stat']:>6.2f}  {b['status']}")
