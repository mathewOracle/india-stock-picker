"""Load FinBERT and measure scoring throughput (also warms the Hugging Face download cache)."""

import time

from app.research import sentiment

if __name__ == "__main__":
    t = time.time()
    model = sentiment.FinBert()
    print(f"loaded {model.name} in {time.time() - t:.0f}s", flush=True)
    titles = ["Reliance profit jumps 18% on retail and Jio strength", "SEBI probes promoter fraud at the firm",
              "Board meeting scheduled on Monday to consider results", "Shares plunge after weak quarterly guidance",
              "Company wins Rs 2,000 crore order from defence ministry"] * 40
    t = time.time()
    out = model.classify(titles)
    rate = len(titles) / (time.time() - t)
    print(f"{len(titles)} titles -> {rate:.0f} titles/sec", flush=True)
    for title, (p, n, u) in list(zip(titles, out))[:5]:
        print(f"  pos={p:.2f} neg={n:.2f} neu={u:.2f} | {title}")
