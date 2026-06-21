#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_snapshot_trade_to_lobster.py
====================================
Convert Chinese A-Share Level-2 snapshot + tick-by-tick trade data
into LOBSTER-format CSV pairs for JaxMARL-HFT.

Inputs:
  (1) SSE/SZE Level-2 snapshot CSV  → 10-level orderbook state (~every 3s)
  (2) Transaction CSV               → tick-by-tick trades (逐笔成交)
  (3) (Optional) Index CSV          → index-level context (not used for LOB)

Outputs:
  {Stock}_{Date}_{Start}_{End}_message_10.csv     → LOBSTER message events
  {Stock}_{Date}_{Start}_{End}_orderbook_10.csv   → LOBSTER orderbook snapshots

Data Flow in JaxMARL-HFT:
  1. LoadLOBSTER_resample reads both CSVs
  2. _pre_process_msg_ob: keeps types 1-4, merges same-ts type-4, shifts
  3. _get_inits_day: slices into windows by time
  4. BaseLOBEnv._get_state_from_data: builds initial OB from book_data
  5. step_env: reads messages → scans through jaxob → updates state

LOBSTER Message Format (6 cols):
  time, type, order_id, qty, price, direction

LOBSTER Orderbook Format (20 cols):
  ask_price_1, ask_vol_1, bid_price_1, bid_vol_1, ... (10 levels)

Key Design:
  - Snapshot-anchored: book state resets to real snapshot at each snaptime
  - Trades between snapshots consume liquidity incrementally
  - type-1 events = snapshot boundaries (book marker, not real submission)
  - type-4 events = trade executions
  - Each orderbook row = the true pre-state for the corresponding message
  - message rows and orderbook rows are always 1:1 (required by loader)

Trade-off (vs. full LOBSTER):
  - REAL depth ✓
  - REAL trades ✓
  - No order submissions between snapshots ✗ (depth only decreases via trades)
  - No cancellations ✗
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

# --- Constants ---
A_SHARE_OPEN = 34200.0        # 09:30:00
A_SHARE_LUNCH_START = 41400.0 # 11:30:00
A_SHARE_LUNCH_END = 46800.0   # 13:00:00
A_SHARE_CLOSE = 54000.0       # 15:00:00


# ================================================================
# Time helpers
# ================================================================

def parse_trading_time(t):
    """Parse TradingTime (HHMMSSmmm integer) → seconds since midnight.
    Examples: 92500580 → 09:25:00.580 → 33900.580
    """
    s = str(int(t)).strip().zfill(9)
    h = int(s[0:2])
    m = int(s[2:4])
    sec = int(s[4:6])
    ms = int(s[6:9])
    return float(h * 3600 + m * 60 + sec) + ms / 1000.0


def time_to_sec_ns(t_float):
    sec = int(t_float)
    ns = int(round((t_float - sec) * 1_000_000_000))
    return sec, ns


def is_trading_session(ts):
    if A_SHARE_OPEN <= ts < A_SHARE_LUNCH_START:
        return True
    if A_SHARE_LUNCH_END <= ts <= A_SHARE_CLOSE:
        return True
    return False


# ================================================================
# Data loading
# ================================================================

def load_snapshots(snapshot_path):
    """Load SSE/SZE Level-2 snapshot CSV + parse TradingTime + filter."""
    dtypes = {}
    for i in range(10):
        for col in [f'AskPrice{i}', f'AskVolume{i}', f'BidPrice{i}', f'BidVolume{i}']:
            dtypes[col] = int

    df = pd.read_csv(snapshot_path, dtype=str)
    # Filter out rows where 'Price'-type columns contain the header string
    price_cols = [c for c in df.columns if 'Price' in c or 'Volume' in c]
    for c in price_cols:
        df = df[df[c].str.match(r'^-?\d+$', na=False)].copy()
    for c in price_cols + ['TradingTime']:
        df[c] = df[c].astype(int)
    df['time_sec'] = df['TradingTime'].apply(parse_trading_time)
    df = df[df['time_sec'].apply(is_trading_session)].copy()
    df = df.sort_values('time_sec').reset_index(drop=True)
    print(f"  Snapshots: {len(df)} rows")
    return df


def load_trades(trade_path):
    """Load Transaction CSV + parse + filter.

    TradeBSFlag convention (SSE Level-2):
      1 = buyer-initiated (aggressive buy, eats ask)
      2 = seller-initiated (aggressive sell, eats bid)
      3 = unknown / call auction cross trade (no clear aggressor)

    Only BSFlag 1 and 2 have a known aggressor side, so only these
    can be converted to LOBSTER type-4 execution events.
    BSFlag=3 trades are skipped (would be LOBSTER type-6 cross trades).
    """
    df = pd.read_csv(trade_path, dtype={
        'Price': str, 'Volume': str, 'TradeBSFlag': str, 'TradeType': str,
    })
    # Filter out rows where column names appear as data (duplicate headers)
    df = df[df['Price'].str.match(r'^-?\d+$')].copy()
    for col in ['Price', 'Volume', 'TradeBSFlag', 'TradeType']:
        df[col] = df[col].astype(int)
    df['time_sec'] = df['TradingTime'].apply(parse_trading_time)
    df = df[(df['time_sec'].apply(is_trading_session))
            & (df['Price'] > 0) & (df['Volume'] > 0)].copy()
    df = df.sort_values('time_sec').reset_index(drop=True)

    bs_dist = df['TradeBSFlag'].value_counts().sort_index()
    print(f"  Trades total: {len(df)} rows")
    print(f"  TradeBSFlag distribution: {dict(bs_dist)}")
    print(f"  → Keep BSFlag=1+2: {bs_dist.get(1, 0) + bs_dist.get(2, 0)} "
          f"(aggressive buys/sells)")
    print(f"  → Skip BSFlag=3+  : {sum(bs_dist.get(k, 0) for k in bs_dist.index if k not in (1, 2))} "
          f"(cross/auction trades)")
    return df


# ================================================================
# Book helpers
# ================================================================

def snapshot_to_asks_bids(snap_row):
    """Convert a snapshot row to (asks: {price: vol}, bids: {price: vol}) dicts."""
    asks, bids = {}, {}
    for level in range(10):
        ap = int(snap_row.get(f'AskPrice{level}', 0) or 0)
        av = int(snap_row.get(f'AskVolume{level}', 0) or 0)
        bp = int(snap_row.get(f'BidPrice{level}', 0) or 0)
        bv = int(snap_row.get(f'BidVolume{level}', 0) or 0)
        if ap > 0 and av > 0:
            asks[ap] = av
        if bp > 0 and bv > 0:
            bids[bp] = bv
    return asks, bids


def asks_bids_to_20col(asks, bids):
    """Convert asks/bids dicts → 20-col LOBSTER orderbook row.

    Format: [ask_p1, ask_v1, bid_p1, bid_v1, ..., ask_p10, ask_v10, bid_p10, bid_v10]
    """
    ask_prices = sorted(asks.keys())
    bid_prices = sorted(bids.keys(), reverse=True)
    out = []
    for level in range(10):
        ap = ask_prices[level] if level < len(ask_prices) else 0
        av = asks.get(ap, 0) if level < len(ask_prices) else 0
        bp = bid_prices[level] if level < len(bid_prices) else 0
        bv = bids.get(bp, 0) if level < len(bid_prices) else 0
        out.extend([int(ap), int(av), int(bp), int(bv)])
    return out


def apply_trade(asks, bids, trade_price, trade_qty, bsflag):
    """Incrementally apply one trade to the book. Returns unfilled volume.

    BSFlag=1 (buy aggressive): consume from ask side
    BSFlag=2 (sell aggressive): consume from bid side
    """
    remaining = trade_qty
    if bsflag == 1:
        for ap in sorted(asks.keys()):
            if remaining <= 0:
                break
            avail = asks.get(ap, 0)
            if avail > 0:
                take = min(avail, remaining)
                remaining -= take
                new_vol = avail - take
                if new_vol <= 0:
                    del asks[ap]
                else:
                    asks[ap] = new_vol
    elif bsflag == 2:
        for bp in sorted(bids.keys(), reverse=True):
            if remaining <= 0:
                break
            avail = bids.get(bp, 0)
            if avail > 0:
                take = min(avail, remaining)
                remaining -= take
                new_vol = avail - take
                if new_vol <= 0:
                    del bids[bp]
                else:
                    bids[bp] = new_vol
    return remaining


# ================================================================
# Main conversion
# ================================================================

def convert(snapshot_csv, trade_csv, output_dir, stock, date_str,
            day_start=A_SHARE_OPEN, day_end=A_SHARE_CLOSE):
    """Convert snapshot + trade CSVs → LOBSTER message + orderbook pair."""
    print("=" * 60)
    print("convert_snapshot_trade_to_lobster.py")
    print(f"  Stock: {stock}, Date: {date_str}")
    print("=" * 60)

    snapshots = load_snapshots(snapshot_csv)
    trades = load_trades(trade_csv)
    if len(snapshots) == 0:
        print("ERROR: No valid snapshots in trading hours.")
        sys.exit(1)

    # Validate snapshot time range covers trading hours
    snap_start = snapshots['time_sec'].min()
    snap_end = snapshots['time_sec'].max()
    print(f"  Snapshot range: {snap_start:.2f}s - {snap_end:.2f}s")
    print(f"  Trade range:    {trades['time_sec'].min():.2f}s - {trades['time_sec'].max():.2f}s")

    # ================================================================
    # Build event sequence
    # ================================================================
    messages = []       # [time_s, time_ns, type, dir, qty, price, oid, tid]
    orderbook_rows = [] # 20-col LOBSTER rows

    order_id = 1_000_000
    trader_id = -100

    trade_times = trades['time_sec'].values
    trade_idx = 0
    total_trades_processed = 0

    for snap_idx in range(len(snapshots)):
        snap = snapshots.iloc[snap_idx]
        snap_ts = snap['time_sec']
        next_snap_ts = (snapshots.iloc[snap_idx + 1]['time_sec']
                        if snap_idx + 1 < len(snapshots) else float('inf'))

        # --- Snapshot marker ---
        sec, ns = time_to_sec_ns(snap_ts)
        asks, bids = snapshot_to_asks_bids(snap)

        messages.append([sec, ns, 1, 0, 0, 0, order_id, trader_id])
        order_id += 1
        orderbook_rows.append(asks_bids_to_20col(asks, bids))

        # --- Trades in this interval ---
        while trade_idx < len(trades):
            trade_ts = float(trade_times[trade_idx])
            if trade_ts >= next_snap_ts:
                break
            if trade_ts < snap_ts:
                # Before first snapshot: use first snapshot's book
                trade_idx += 1
                continue

            row = trades.iloc[trade_idx]
            tp = int(row['Price'])
            tq = int(row['Volume'])
            bf = int(row['TradeBSFlag'])

            if bf == 1:
                lobster_dir = -1
            elif bf == 2:
                lobster_dir = 1
            else:
                trade_idx += 1
                continue

            t_sec, t_ns = time_to_sec_ns(trade_ts)
            messages.append([t_sec, t_ns, 4, lobster_dir, tq, tp, 0, trader_id])
            orderbook_rows.append(asks_bids_to_20col(asks, bids))

            apply_trade(asks, bids, tp, tq, bf)
            trade_idx += 1
            total_trades_processed += 1

    skipped = len(trades) - total_trades_processed
    print(f"  Processed: {total_trades_processed} trades (BSFlag=1 or 2)")
    print(f"  Skipped:   {skipped} trades (BSFlag=3+ auction/cross)")

    # ================================================================
    # Validate row counts
    # ================================================================
    n_msg, n_ob = len(messages), len(orderbook_rows)
    print(f"\n  Events: {n_msg} messages + {n_ob} orderbook rows = 1:1 match")

    if n_msg != n_ob:
        print(f"  ERROR: Mismatch! Pad to match...")
        while len(messages) > len(orderbook_rows):
            orderbook_rows.append(orderbook_rows[-1] if orderbook_rows else [0] * 20)
        while len(orderbook_rows) > len(messages):
            last = messages[-1] if messages else [0, 0, 0, 0, 0, 0, 0, 0]
            messages.append([last[0], last[1], 0, 0, 0, 0, 0, 0])
        n_msg, n_ob = len(messages), len(orderbook_rows)

    # ================================================================
    # Write output
    # ================================================================
    out_dir = os.path.join(output_dir, "rawLOBSTER", stock, date_str)
    os.makedirs(out_dir, exist_ok=True)

    # Message CSV (6 cols)
    msg_out = [
        [f"{m[0] + m[1] / 1_000_000_000.0:.9f}", m[2], m[6], m[4], m[5], m[3]]
        for m in messages
    ]
    msg_file = os.path.join(
        out_dir,
        f"{stock}_{date_str}_{int(day_start)}00000_{int(day_end)}00000_message_10.csv"
    )
    pd.DataFrame(msg_out).to_csv(msg_file, index=False, header=False)
    print(f"\n  Wrote: {msg_file}  ({len(msg_out)} rows)")

    ob_file = os.path.join(
        out_dir,
        f"{stock}_{date_str}_{int(day_start)}00000_{int(day_end)}00000_orderbook_10.csv"
    )
    pd.DataFrame(orderbook_rows).to_csv(ob_file, index=False, header=False)
    print(f"  Wrote: {ob_file}  ({len(orderbook_rows)} rows)")

    # Summary
    type_counts = {}
    for m in messages:
        type_counts[m[2]] = type_counts.get(m[2], 0) + 1
    print(f"\n  Event type distribution: {dict(sorted(type_counts.items()))}")
    print(f"  Orderbook row 1 (first 10 vals): {orderbook_rows[0][:10]}")
    print(f"  Orderbook row -1 (first 10 vals): {orderbook_rows[-1][:10]}")
    print("  Done.")

    return msg_file, ob_file


# ================================================================
# CLI
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Convert Chinese A-Share snapshot + trade data to LOBSTER format"
    )
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--trade", required=True)
    parser.add_argument("--output",
                        default=r"C:\Users\吴关洲\.openclaw\workspace\harness\projects\RL-HFT\papers\JaxMARL-HFT-code\data")
    parser.add_argument("--stock", default="600036")
    parser.add_argument("--date", default="20240124")
    parser.add_argument("--day-start", type=float, default=A_SHARE_OPEN)
    parser.add_argument("--day-end", type=float, default=A_SHARE_CLOSE)
    args = parser.parse_args()
    convert(args.snapshot, args.trade, args.output, args.stock, args.date,
            args.day_start, args.day_end)


if __name__ == "__main__":
    main()
