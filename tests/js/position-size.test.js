// Run with: node tests/js/position-size.test.js
const assert = require('node:assert');
const { positionSize } = require('../../app/static/position-size.js');

// risk-limited: 1% of 1,00,000 = 1,000 rupees / 20 per share = 50 shares, affordable at 100
let s = positionSize(100000, 1, 100, 80);
assert.deepStrictEqual([s.qty, s.capped], [50, false]);
assert.strictEqual(positionSize(100000, 2, 100, 80).qty, 100);  // doubling risk doubles qty

// capital-limited (the QA report): a 10,287-rupee stock with a 20-rupee stop on 1 lakh
s = positionSize(100000, 1, 10287, 10267);
assert.deepStrictEqual([s.qty, s.riskQty, s.capped], [9, 50, true]);
assert.strictEqual(positionSize(100000, 2, 10287, 10267).riskQty, 100);  // risk qty still scales

// shorts work too (stop above entry), and a zero-width stop never divides by zero
assert.strictEqual(positionSize(100000, 1, 100, 110).qty, 100);
assert.strictEqual(positionSize(100000, 1, 100, 100).qty, 0);
console.log('position-size: all assertions passed');
