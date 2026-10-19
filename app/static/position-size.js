// Position sizing for intraday setup cards (any element with data-entry / data-stop).

// Pure function (unit-testable with node): shares to buy so a stop-out loses capital x risk%.
function positionSize(capital, riskPct, entry, stop) {
  const perShare = Math.abs(entry - stop);
  const riskQty = perShare > 0 ? Math.floor((capital * riskPct) / 100 / perShare) : 0;
  const affordable = entry > 0 ? Math.floor(capital / entry) : 0;  // no leverage assumed
  const qty = Math.min(riskQty, affordable);
  return { qty, riskQty, capped: riskQty > affordable, rupeesAtRisk: Math.round(qty * perShare) };
}

if (typeof document !== 'undefined') {
  const form = document.getElementById('sizer');
  if (form) {
    const update = () => {
      const capital = Number(form.capital.value) || 0;
      const riskPct = Number(form.risk.value) || 0;
      document.querySelectorAll('[data-entry][data-stop]').forEach((card) => {
        const out = card.querySelector('[data-qty]');
        if (!out) return;
        const size = positionSize(capital, riskPct, Number(card.dataset.entry), Number(card.dataset.stop));
        // say *why* the number is what it is: silent caps look like bugs
        out.textContent = size.capped
          ? `${size.qty} (capital-capped; risk allows ${size.riskQty})`
          : `${size.qty} (risk Rs ${size.rupeesAtRisk.toLocaleString('en-IN')})`;
      });
    };
    form.addEventListener('input', update);
    update();
  }
}

if (typeof module !== 'undefined') module.exports = { positionSize };
