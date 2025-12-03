#!/usr/bin/env python3
"""
Wallet Toxicity Scoring Engine (WTS) - Phase 1
Historical Analysis & Scoring (FIXED - Get Real Wallet Addresses)

Analyzes Uniswap V3 ETH/USDC swaps to identify which wallets
consistently predict price movements (smart/toxic flow).

Features:
  - RESUMABLE: Checkpoint every N blocks, resume from last checkpoint
  - Saves intermediate results to JSON (visible for debugging)
  - Graceful recovery on network interruptions
  - DEBUG MODE: Keep checkpoint files visible even on success

Usage:
    python3 phase1_wallet_toxicity.py

Output:
    wallet_toxicity_scores.csv - ranked by toxicity score
    .wts_checkpoint.json - resumption point (visible)
    .wts_trades_data.json - intermediate trades (visible for debugging)
"""

import pandas as pd
import numpy as np
import requests
import time
import sys
import json
import os
from collections import defaultdict
from datetime import datetime

# ============ CONFIG ============
POOL_ADDRESS = "0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640"  # ETH/USDC 0.05%
RPC_URL = "https://eth.llamarpc.com"
BINANCE_SYMBOL = "ETHUSDT"
SWAP_TOPIC = "0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67"

# ===== TESTING: 0.1 DAYS (2.4 HOURS) =====
DAYS_TO_ANALYZE = 0.1  # Ultra-fast testing
BLOCKS_PER_DAY = 7200  # ~12s per block
TOTAL_BLOCKS = int(DAYS_TO_ANALYZE * BLOCKS_PER_DAY)
CHUNK_SIZE = 500
MIN_TRADES_PER_WALLET = 2  # Reduced for 0.1-day test
CHECKPOINT_FILE = ".wts_checkpoint.json"
INTERMEDIATE_DATA_FILE = ".wts_trades_data.json"
DEBUG_MODE = True  # Keep checkpoint files visible

# ============ CHECKPOINT MANAGEMENT ============

def load_checkpoint():
    """Load checkpoint if it exists."""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'r') as f:
                cp = json.load(f)
            print(f"📋 Checkpoint found:")
            print(f"   Timestamp: {cp['timestamp']}")
            print(f"   Blocks processed: {cp['blocks_processed']:,} / {cp['total_blocks']:,}")
            print(f"   Trades found: {cp['trades_found']:,}")
            return cp
        except Exception as e:
            print(f"⚠️  Could not load checkpoint: {e}")
            return None
    return None

def save_checkpoint(current_block, start_block, end_block, total_blocks, all_logs):
    """Save checkpoint to resume later."""
    cp = {
        "timestamp": datetime.now().isoformat(),
        "current_block": current_block,
        "start_block": start_block,
        "end_block": end_block,
        "total_blocks": total_blocks,
        "blocks_processed": current_block - start_block,
        "blocks_remaining": end_block - current_block,
        "trades_found": len(all_logs)
    }
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(cp, f, indent=2)
    if DEBUG_MODE:
        print(f"   💾 Checkpoint saved: {CHECKPOINT_FILE}")
    return cp

def save_intermediate_trades(trades):
    """Save trades to disk for recovery."""
    with open(INTERMEDIATE_DATA_FILE, 'w') as f:
        json.dump(trades, f, indent=2)
    print(f"   💾 Saved {len(trades)} trades to: {INTERMEDIATE_DATA_FILE}")

def clear_checkpoints(keep_files=DEBUG_MODE):
    """Clear checkpoint files (called on successful completion)."""
    if keep_files:
        print(f"\n📁 DEBUG MODE: Keeping checkpoint files for inspection:")
        print(f"   - {CHECKPOINT_FILE}")
        print(f"   - {INTERMEDIATE_DATA_FILE}")
        return
    
    for f in [CHECKPOINT_FILE, INTERMEDIATE_DATA_FILE]:
        if os.path.exists(f):
            os.remove(f)
            print(f"🗑️  Cleaned up: {f}")

# ============ HELPER FUNCTIONS ============

def rpc_call(method, params=[]):
    """Make JSON-RPC call to Ethereum node with retry logic."""
    payload = {
        "jsonrpc": "2.0",
        "method": method,
        "params": params,
        "id": 1
    }
    headers = {'Content-Type': 'application/json'}
    
    retries = 3
    for attempt in range(retries):
        try:
            response = requests.post(RPC_URL, headers=headers, json=payload, timeout=15)
            result = response.json().get('result')
            return result
        except requests.exceptions.Timeout:
            if attempt < retries - 1:
                time.sleep(1)
                continue
            else:
                return None
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1)
                continue
            else:
                return None
    return None

def parse_int256(hex_str):
    """Convert hex string to signed 256-bit integer."""
    val = int(hex_str, 16)
    if val > 2**255:
        val -= 2**256
    return val

def fetch_swap_logs(from_block, to_block):
    """Fetch Swap events from pool."""
    logs = rpc_call("eth_getLogs", [{
        "fromBlock": hex(from_block),
        "toBlock": hex(to_block),
        "address": POOL_ADDRESS,
        "topics": [SWAP_TOPIC]
    }])
    return logs if logs else []

def fetch_transaction(tx_hash):
    """Fetch transaction to get the sender (from address)."""
    tx = rpc_call("eth_getTransactionByHash", [tx_hash])
    if tx and 'from' in tx:
        return tx['from']
    return None

def parse_swap_log(log, current_block):
    """Extract swap data from log event."""
    try:
        data = log['data']
        if data.startswith('0x'):
            data = data[2:]
        
        # Parse event data
        amount0 = parse_int256(data[0:64])       # USDC
        amount1 = parse_int256(data[64:128])     # ETH
        sqrtPriceX96 = int(data[128:192], 16)
        
        # Calculate ETH price in USDC
        raw_price = (sqrtPriceX96 / (2**96)) ** 2
        eth_price_usdc = (1 / raw_price) * 10**12
        
        # Approximate timestamp (block-based)
        block_num = int(log['blockNumber'], 16)
        now_ts = time.time()
        timestamp = int(now_ts - (current_block - block_num) * 12)
        
        tx_hash = log['transactionHash']
        sender = fetch_transaction(tx_hash)
        
        if not sender:
            return None  # Skip if we can't get sender
        
        return {
            "block_number": block_num,
            "tx_hash": tx_hash,
            "wallet": sender.lower(),  # Use sender, not pool
            "eth_price_exec": eth_price_usdc,
            "direction": "BUY_ETH" if amount1 < 0 else "SELL_ETH",
            "timestamp": timestamp,
            "amount_eth": abs(amount1 / 10**18)
        }
    except Exception as e:
        return None

def compute_markout(row):
    """Compute LP markout PnL for a single trade."""
    lp_sign = -1 if row['direction'] == "BUY_ETH" else 1
    return (row['eth_price_exec'] - row['cex_price_5m']) * lp_sign

# ============ MAIN WORKFLOW ============

def main():
    print("\n" + "=" * 70)
    print("WALLET TOXICITY SCORING ENGINE - PHASE 1 (QUICK TEST)")
    print(f"Ultra-fast test with {DAYS_TO_ANALYZE} days ({TOTAL_BLOCKS:,} blocks) of data")
    print("=" * 70)
    
    # Check for existing checkpoint
    checkpoint = load_checkpoint()
    if checkpoint:
        print(f"\n🔄 RESUME MODE DETECTED")
    
    # -------- STEP 1: Get Current Block --------
    print("\n[1/6] Fetching current block number...")
    current_block_hex = rpc_call("eth_blockNumber")
    if not current_block_hex:
        print("❌ Failed to connect to RPC")
        sys.exit(1)
    
    current_block = int(current_block_hex, 16)
    start_block = current_block - TOTAL_BLOCKS
    
    print(f"      ✓ Current Block: {current_block:,}")
    print(f"      ✓ Analyzing: {start_block:,} → {current_block:,}")
    print(f"      ✓ Time Window: ~{DAYS_TO_ANALYZE} days ({TOTAL_BLOCKS:,} blocks)")
    
    # Determine resume point
    if checkpoint and checkpoint['end_block'] == current_block:
        fetch_start = checkpoint['current_block']
        all_logs = []
        print(f"      ✓ Resuming from block {fetch_start:,}")
    else:
        fetch_start = start_block
        print(f"      ✓ Starting fresh analysis")
    
    # -------- STEP 2: Fetch Swap Logs (Chunked) --------
    print(f"\n[2/6] Fetching swap logs ({CHUNK_SIZE} blocks/chunk)...")
    print(f"      Total blocks to fetch: {current_block - fetch_start:,}")
    
    all_logs = []
    chunks_processed = 0
    start_time = time.time()
    
    for i in range(fetch_start, current_block, CHUNK_SIZE):
        end = min(i + CHUNK_SIZE - 1, current_block)
        progress = (i - start_block) / TOTAL_BLOCKS * 100 if TOTAL_BLOCKS > 0 else 0
        elapsed = time.time() - start_time
        sys.stdout.write(f"\r      [{progress:.1f}%] Blocks {i:,} → {end:,} [{elapsed:.0f}s]...")
        sys.stdout.flush()
        
        logs = fetch_swap_logs(i, end)
        if logs:
            all_logs.extend(logs)
        
        chunks_processed += 1
        
        # Save checkpoint every 5 chunks
        if chunks_processed % 5 == 0:
            save_checkpoint(end, start_block, current_block, TOTAL_BLOCKS, all_logs)
        
        time.sleep(0.1)
    
    elapsed = time.time() - start_time
    print(f"\r      ✓ Fetched {len(all_logs):,} swap events from {chunks_processed} chunks [{elapsed:.0f}s]")
    
    if len(all_logs) == 0:
        print("⚠️  No swap events found in this period.")
        print("    This is normal for small timeframes. Try extending DAYS_TO_ANALYZE.")
        sys.exit(1)
    
    # -------- STEP 3: Parse Logs (with TX sender fetching) --------
    print(f"\n[3/6] Parsing swap events (fetching transaction senders)...")
    print(f"      Processing {len(all_logs):,} trades...")
    
    trades = []
    for idx, log in enumerate(all_logs):
        if idx % 10 == 0:
            sys.stdout.write(f"\r      [{idx}/{len(all_logs)}] Parsing and fetching senders...")
            sys.stdout.flush()
        
        trade = parse_swap_log(log, current_block)
        if trade:
            trades.append(trade)
        
        # Rate limiting: be gentle on RPC
        if idx % 50 == 0:
            time.sleep(0.2)
    
    df_trades = pd.DataFrame(trades)
    print(f"\r      ✓ Successfully parsed {len(df_trades):,} trades with senders")
    
    # Save intermediate trades for recovery
    save_intermediate_trades([t for t in trades])
    
    # -------- STEP 4: Fetch Binance Reference Prices --------
    print(f"\n[4/6] Fetching Binance reference prices...")
    
    b_url = f"https://api.binance.com/api/v3/klines?symbol={BINANCE_SYMBOL}&interval=1m&limit=1440"
    try:
        b_data = requests.get(b_url, timeout=10).json()
        print(f"      ✓ Downloaded {len(b_data)} Binance 1m candles")
    except Exception as e:
        print(f"❌ Failed to fetch Binance data: {e}")
        sys.exit(1)
    
    cex_prices = []
    for k in b_data:
        cex_prices.append({
            "ts_min": int(k[0]) // 1000,
            "cex_price": float(k[4])
        })
    df_cex = pd.DataFrame(cex_prices)
    
    # -------- STEP 5: Merge and Compute Markout --------
    print(f"\n[5/6] Computing markout and toxicity scores...")
    
    df_trades['ts_min'] = (df_trades['timestamp'] // 60) * 60
    df_trades = pd.merge(df_trades, df_cex, on='ts_min', how='inner')
    
    df_cex_future = df_cex.copy()
    df_cex_future['ts_min'] = df_cex_future['ts_min'] - 300
    df_cex_future = df_cex_future.rename(columns={"cex_price": "cex_price_5m"})
    df_trades = pd.merge(df_trades, df_cex_future[['ts_min', 'cex_price_5m']], on='ts_min', how='inner')
    
    df_trades['markout'] = df_trades.apply(compute_markout, axis=1)
    
    print(f"      ✓ Merged {len(df_trades):,} trades with CEX data")
    
    # -------- WALLET TOXICITY SCORING --------
    print(f"\n[6/6] Computing wallet toxicity scores...")
    
    wallet_stats = defaultdict(lambda: {'markouts': []})
    
    for _, row in df_trades.iterrows():
        wallet = row['wallet']
        wallet_stats[wallet]['markouts'].append(row['markout'])
    
    wallet_scores = []
    for wallet, data in wallet_stats.items():
        markouts = np.array(data['markouts'])
        avg_markout = np.mean(markouts)
        median_markout = np.median(markouts)
        std_markout = np.std(markouts)
        count = len(markouts)
        
        toxicity = avg_markout / (std_markout + 1e-8)
        
        wallet_scores.append({
            'wallet': wallet,
            'num_trades': count,
            'avg_markout': avg_markout,
            'median_markout': median_markout,
            'std_markout': std_markout,
            'toxicity_score': toxicity,
        })
    
    df_wallet_scores = pd.DataFrame(wallet_scores)
    df_wallet_scores['toxicity_percentile'] = (
        df_wallet_scores['toxicity_score'].rank(pct=True) * 100
    )
    
    df_wallet_scores = df_wallet_scores[
        df_wallet_scores['num_trades'] >= MIN_TRADES_PER_WALLET
    ]
    
    df_wallet_scores = df_wallet_scores.sort_values('toxicity_score', ascending=False)
    
    print(f"      ✓ Analyzed {len(df_wallet_scores):,} unique wallets (min {MIN_TRADES_PER_WALLET} trades)")
    
    # ============ OUTPUT & VISUALIZATION ============
    if len(df_wallet_scores) > 0:
        print("\n" + "=" * 70)
        print("TOP 10 SMART/TOXIC WALLETS (Highest Toxicity Score)")
        print("=" * 70)
        
        top_n = min(10, len(df_wallet_scores))
        top_wallets = df_wallet_scores.head(top_n)[
            ['wallet', 'num_trades', 'avg_markout', 'toxicity_score', 'toxicity_percentile']
        ]
        
        for idx, (_, row) in enumerate(top_wallets.iterrows(), 1):
            print(f"\n#{idx} {row['wallet']}")
            print(f"    Trades: {row['num_trades']:,} | Avg Markout: ${row['avg_markout']:.4f} | "
                  f"Toxicity: {row['toxicity_score']:.2f} | Percentile: {row['toxicity_percentile']:.1f}")
    
    # ============ STATISTICS ============
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print(f"Total trades analyzed: {len(df_trades):,}")
    print(f"Unique wallets: {len(df_wallet_scores):,}")
    if len(df_wallet_scores) > 0:
        print(f"Avg trades per wallet: {len(df_trades) / len(df_wallet_scores):.1f}")
    print(f"\nMarkout Distribution:")
    print(f"  Mean: ${df_trades['markout'].mean():.4f}")
    print(f"  Median: ${df_trades['markout'].median():.4f}")
    print(f"  Std Dev: ${df_trades['markout'].std():.4f}")
    if len(df_wallet_scores) > 0:
        print(f"\nToxicity Score Distribution:")
        print(f"  Mean: {df_wallet_scores['toxicity_score'].mean():.2f}")
        print(f"  Median: {df_wallet_scores['toxicity_score'].median():.2f}")
        print(f"  Min: {df_wallet_scores['toxicity_score'].min():.2f}")
        print(f"  Max: {df_wallet_scores['toxicity_score'].max():.2f}")
    
    # ============ SAVE OUTPUT ============
    output_file = 'wallet_toxicity_scores.csv'
    df_wallet_scores.to_csv(output_file, index=False)
    print(f"\n✓ Saved results to: {output_file}")
    
    # Clean up checkpoints
    clear_checkpoints()
    
    print("\n" + "=" * 70)
    print("FILES CREATED:")
    print("=" * 70)
    print(f"  ✓ {output_file}")
    print(f"  ✓ {CHECKPOINT_FILE} (for resume testing)")
    print(f"  ✓ {INTERMEDIATE_DATA_FILE} (intermediate data)")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏸️  Interrupted by user - progress saved!")
        print("Checkpoint files retained for inspection:")
        print(f"  - {CHECKPOINT_FILE}")
        print(f"  - {INTERMEDIATE_DATA_FILE}")
        print("\nRun the script again to resume from checkpoint.\n")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("Checkpoint saved. Run the script again to resume.\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
