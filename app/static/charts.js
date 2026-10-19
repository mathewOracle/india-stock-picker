// Candlestick + indicator charts for the stock page (TradingView Lightweight Charts v4).
/* global LightweightCharts */

function renderCharts(d) {
  const { createChart, LineStyle } = LightweightCharts;
  const base = {
    autoSize: true,
    layout: { background: { color: '#ffffff' }, textColor: '#1e293b' },
    grid: { vertLines: { color: '#f1f5f9' }, horzLines: { color: '#f1f5f9' } },
    rightPriceScale: { borderColor: '#cbd5e1' },
    timeScale: { borderColor: '#cbd5e1' },
  };

  const line = (chart, data, color, opts = {}) => {
    const series = chart.addLineSeries({
      color, lineWidth: 1.5, priceLineVisible: false, lastValueVisible: false, ...opts,
    });
    series.setData(data);
    return series;
  };

  // ---- price ----
  const price = createChart(document.getElementById('price-chart'), base);
  const candles = price.addCandlestickSeries({
    upColor: '#15803d', downColor: '#b91c1c', borderVisible: false,
    wickUpColor: '#15803d', wickDownColor: '#b91c1c',
  });
  candles.setData(d.candles);
  line(price, d.bb_upper, '#94a3b8', { lineWidth: 1, lineStyle: LineStyle.Dashed });
  line(price, d.bb_lower, '#94a3b8', { lineWidth: 1, lineStyle: LineStyle.Dashed });
  line(price, d.ema20, '#0284c7', { lineWidth: 1 });
  line(price, d.sma50, '#7c3aed');
  line(price, d.sma200, '#ea580c', { lineWidth: 2 });
  line(price, d.st_line, '#0f766e', { lineWidth: 1, lineStyle: LineStyle.SparseDotted });

  const volume = price.addHistogramSeries({ priceFormat: { type: 'volume' }, priceScaleId: 'vol' });
  price.priceScale('vol').applyOptions({ scaleMargins: { top: 0.82, bottom: 0 } });
  volume.setData(d.volume);

  d.levels.forEach((lv) => candles.createPriceLine({
    price: lv.price, color: '#64748b', lineWidth: 1, lineStyle: LineStyle.Dotted,
    axisLabelVisible: false, title: lv.title,
  }));
  [['entry', '#1d4ed8', 'Entry'], ['stop', '#b91c1c', 'Stop'],
   ['target1', '#15803d', 'T1'], ['target2', '#15803d', 'T2']].forEach(([key, color, title]) =>
    candles.createPriceLine({ price: d.plan[key], color, lineWidth: 2, lineStyle: LineStyle.Solid, title }));

  // ---- RSI ----
  const rsi = createChart(document.getElementById('rsi-chart'), base);
  const rsiLine = line(rsi, d.rsi, '#7c3aed', { lastValueVisible: true });
  [70, 30].forEach((level) => rsiLine.createPriceLine({
    price: level, color: '#94a3b8', lineWidth: 1, lineStyle: LineStyle.Dashed,
    axisLabelVisible: false, title: String(level),
  }));

  // ---- MACD ----
  const macd = createChart(document.getElementById('macd-chart'), base);
  const hist = macd.addHistogramSeries({ priceLineVisible: false, lastValueVisible: false });
  hist.setData(d.macd_hist.map((p) => ({ ...p, color: p.value >= 0 ? '#16a34a99' : '#dc262699' })));
  line(macd, d.macd, '#1d4ed8');
  line(macd, d.macd_signal, '#ea580c');

  // keep all three panes scrolled/zoomed together (guard prevents ping-pong between panes)
  const charts = [price, rsi, macd];
  let syncing = false;
  charts.forEach((source) => source.timeScale().subscribeVisibleLogicalRangeChange((range) => {
    if (!range || syncing) return;
    syncing = true;
    charts.filter((c) => c !== source).forEach((c) => c.timeScale().setVisibleLogicalRange(range));
    syncing = false;
  }));
  price.timeScale().fitContent();
}
