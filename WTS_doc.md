# Wallet Toxicity Scoring Engine (WTS)

---

## 1. What It Does

The Wallet Toxicity Scoring Engine identifies wallets that consistently profit at the expense of Liquidity Providers (LPs) by analyzing on-chain Uniswap V3 swaps and measuring their "markout"—the price change 5 minutes after each trade.

**The Process:**
- Extracts every Swap event from the Uniswap V3 ETH/USDC pool
- For each swap, identifies the **real sender wallet** (not the pool or router)
- Computes whether the trader was "right" or "wrong" about short-term price direction
- Groups trades by wallet and scores each wallet by consistency (high consistency = high skill)
- Ranks wallets from "definitely smart money" (toxicity score 18.77) to "definitely noise" (negative scores)

**Result:** A ranked CSV of 68 wallets, showing which ones systematically extract value from the pool.

---

## 2. Why It Matters

**For Liquidity Providers (Defense):**
- Stop treating all traders equally; identify the 5-10 wallets that drain your pool
- Implement Just-in-Time (JIT) liquidity: remove liquidity seconds before toxic wallets trade, re-add after
- Adjust fees dynamically when toxic wallets are active, capturing part of their profit

**For Traders (Offense):**
- Use the top toxic wallets as a signal generator
- Monitor the mempool for transactions from these wallets
- Front-run or back-run their trades, capturing predictable price moves
- Build systematic copy-trading bots based on their behavior

**For Protocols:**
- Diagnose if a pool is being farmed by a small number of MEV bots (fragile) or distributed fairly (healthy)
- Design fee tiers and incentives based on who actually uses the pool
- Provide LPs with transparency: "Here's who is taking your yield"

---

## 3. How It Works (Core Logic)

**Step 1: Extract Real Traders**
```
For each Uniswap swap:
  - Get on-chain execution price
  - Look up transaction sender (the real wallet, not the router)
  - Record direction (BUY_ETH or SELL_ETH) and size
```

**Step 2: Measure Profitability**
```
For each trade, compute 5-minute markout:
  - Execution price (on Uniswap)
  - Future price (from Binance API, 5 minutes later)
  - If trader bought and price went up → they were right (LP lost)
  - If trader bought and price went down → they were wrong (LP gained)
```

**Step 3: Score Wallets by Consistency**
```
For each wallet, compute:
  - Average markout (profit per trade)
  - Standard deviation (variance/luck)
  - Toxicity = Average / StdDev (skill score)
  
Example:
  Wallet A: $100 profit, $200 variance → Score = 0.50 (lucky)
  Wallet B: $0.70 profit, $0.07 variance → Score = 10.38 (skillful bot)
```

**Result:** Wallet B is ranked higher despite lower absolute profit, because it is predictably right, not randomly lucky.

---

## Key Findings (0.1-Day Sample)

| Metric | Value |
|--------|-------|
| Time Window | 2.4 hours |
| Wallets Identified | 68 |
| Trades Analyzed | 534 |
| Top Wallet Toxicity | 18.77 |
| Bot Detected | `0x4c83...be56c` (12 trades/2.4h, $0.69 per trade, consistent) |
| LP Loss Rate | 100% of top 19 wallets profitable against pool |

---

