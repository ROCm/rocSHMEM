#!/usr/bin/env python3
"""Parse official DeepEP test_low_latency.py benchmark output.

Extracts D+C avg/min/max, dispatch avg, combine avg from rank 0 output.
Compares strict vs relaxed ordering results.

Usage:
    python3 parse_official_ll_results.py <strict_dir> <relaxed_dir>
"""
import os
import re
import sys
import math
from collections import defaultdict


def parse_log(filepath):
    """Parse a single log file. Returns dict with timing results from rank 0."""
    result = {}
    with open(filepath) as f:
        for line in f:
            # D+C line: [rank 0] Dispatch + combine bandwidth: 83.63 GB/s, avg_t=137.55 us, min_t=127.42 us, max_t=148.27 us
            m = re.search(
                r'\[rank 0\] Dispatch \+ combine bandwidth: ([\d.]+) GB/s, '
                r'avg_t=([\d.]+) us, min_t=([\d.]+) us, max_t=([\d.]+) us',
                line)
            if m:
                result['dc_bw_gbs'] = float(m.group(1))
                result['dc_avg_us'] = float(m.group(2))
                result['dc_min_us'] = float(m.group(3))
                result['dc_max_us'] = float(m.group(4))

            # Dispatch/Combine line (non-hook):
            # [rank 0] Dispatch bandwidth: 72.37 GB/s, avg_t=76.56 us | Combine bandwidth: 95.22 GB/s, avg_t=62.44 us
            m = re.search(
                r'\[rank 0\] Dispatch bandwidth: ([\d.]+) GB/s, avg_t=([\d.]+) us \| '
                r'Combine bandwidth: ([\d.]+) GB/s, avg_t=([\d.]+) us',
                line)
            if m:
                result['disp_bw_gbs'] = float(m.group(1))
                result['disp_avg_us'] = float(m.group(2))
                result['comb_bw_gbs'] = float(m.group(3))
                result['comb_avg_us'] = float(m.group(4))

            # Hook-based dispatch/combine (send/recv split):
            # [rank 0] Dispatch send/recv time: 68.30 + 8.26 us | Combine send/recv time: 55.30 + 7.14 us
            m = re.search(
                r'\[rank 0\] Dispatch send/recv time: ([\d.]+) \+ ([\d.]+) us \| '
                r'Combine send/recv time: ([\d.]+) \+ ([\d.]+) us',
                line)
            if m:
                result['disp_send_us'] = float(m.group(1))
                result['disp_recv_us'] = float(m.group(2))
                result['comb_send_us'] = float(m.group(3))
                result['comb_recv_us'] = float(m.group(4))

    return result


def extract_token_count(filename):
    """Extract token count from filename like run1_tok128.log"""
    m = re.search(r'tok(\d+)', filename)
    return int(m.group(1)) if m else None


def extract_run_num(filename):
    """Extract run number from filename like run1_tok128.log"""
    m = re.search(r'run(\d+)', filename)
    return int(m.group(1)) if m else None


def load_results(result_dir):
    """Load all results from a directory. Returns {token_count: [results_per_run]}"""
    results = defaultdict(list)
    if not os.path.isdir(result_dir):
        print(f"ERROR: Directory not found: {result_dir}")
        sys.exit(1)

    for f in sorted(os.listdir(result_dir)):
        if not f.endswith('.log'):
            continue
        tc = extract_token_count(f)
        if tc is None:
            continue
        r = parse_log(os.path.join(result_dir, f))
        if r:
            results[tc].append(r)
    return results


def avg(values):
    return sum(values) / len(values) if values else 0


def geo_mean(values):
    if not values or any(v <= 0 for v in values):
        return 0
    return math.exp(sum(math.log(v) for v in values) / len(values))


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <strict_dir> <relaxed_dir>")
        sys.exit(1)

    strict_dir = sys.argv[1]
    relaxed_dir = sys.argv[2]

    strict = load_results(strict_dir)
    relaxed = load_results(relaxed_dir)

    all_tokens = sorted(set(strict.keys()) | set(relaxed.keys()))

    if not all_tokens:
        print("No results found!")
        sys.exit(1)

    print("\n" + "=" * 120)
    print("OFFICIAL DeepEP LL Benchmark: Strict vs Relaxed Ordering")
    print(f"Strict dir:  {strict_dir}")
    print(f"Relaxed dir: {relaxed_dir}")
    print("=" * 120)

    # ---- D+C Table ----
    print(f"\n{'='*100}")
    print("Dispatch + Combine (D+C) Latency")
    print(f"{'='*100}")
    print(f"{'Tokens':>8} | {'Strict Avg(us)':>14} {'Strict Min(us)':>14} | "
          f"{'Relaxed Avg(us)':>15} {'Relaxed Min(us)':>15} | "
          f"{'Avg Speedup':>12} {'Min Speedup':>12}")
    print("-" * 100)

    avg_speedups = []
    min_speedups = []

    for tc in all_tokens:
        s_runs = strict.get(tc, [])
        r_runs = relaxed.get(tc, [])

        if not s_runs or not r_runs:
            print(f"{tc:>8} | {'N/A':>14} {'N/A':>14} | {'N/A':>15} {'N/A':>15} | {'N/A':>12} {'N/A':>12}")
            continue

        s_avgs = [r['dc_avg_us'] for r in s_runs if 'dc_avg_us' in r]
        s_mins = [r['dc_min_us'] for r in s_runs if 'dc_min_us' in r]
        r_avgs = [r['dc_avg_us'] for r in r_runs if 'dc_avg_us' in r]
        r_mins = [r['dc_min_us'] for r in r_runs if 'dc_min_us' in r]

        if not s_avgs or not r_avgs:
            continue

        s_avg_val = avg(s_avgs)
        s_min_val = min(s_mins) if s_mins else 0
        r_avg_val = avg(r_avgs)
        r_min_val = min(r_mins) if r_mins else 0

        avg_spd = s_avg_val / r_avg_val if r_avg_val > 0 else 0
        min_spd = s_min_val / r_min_val if r_min_val > 0 else 0

        avg_speedups.append(avg_spd)
        min_speedups.append(min_spd)

        print(f"{tc:>8} | {s_avg_val:>14.2f} {s_min_val:>14.2f} | "
              f"{r_avg_val:>15.2f} {r_min_val:>15.2f} | "
              f"{avg_spd:>11.3f}x {min_spd:>11.3f}x")

    if avg_speedups:
        print("-" * 100)
        print(f"{'GeoMean':>8} | {'':>14} {'':>14} | {'':>15} {'':>15} | "
              f"{geo_mean(avg_speedups):>11.3f}x {geo_mean(min_speedups):>11.3f}x")

    # ---- Dispatch-only Table ----
    print(f"\n{'='*80}")
    print("Dispatch-Only Latency (Kineto)")
    print(f"{'='*80}")
    print(f"{'Tokens':>8} | {'Strict(us)':>12} | {'Relaxed(us)':>12} | {'Speedup':>10}")
    print("-" * 80)

    disp_speedups = []
    for tc in all_tokens:
        s_runs = strict.get(tc, [])
        r_runs = relaxed.get(tc, [])
        s_vals = [r['disp_avg_us'] for r in s_runs if 'disp_avg_us' in r]
        r_vals = [r['disp_avg_us'] for r in r_runs if 'disp_avg_us' in r]
        if not s_vals or not r_vals:
            continue
        s_v = avg(s_vals)
        r_v = avg(r_vals)
        spd = s_v / r_v if r_v > 0 else 0
        disp_speedups.append(spd)
        print(f"{tc:>8} | {s_v:>12.2f} | {r_v:>12.2f} | {spd:>9.3f}x")

    if disp_speedups:
        print("-" * 80)
        print(f"{'GeoMean':>8} | {'':>12} | {'':>12} | {geo_mean(disp_speedups):>9.3f}x")

    # ---- Combine-only Table ----
    print(f"\n{'='*80}")
    print("Combine-Only Latency (Kineto)")
    print(f"{'='*80}")
    print(f"{'Tokens':>8} | {'Strict(us)':>12} | {'Relaxed(us)':>12} | {'Speedup':>10}")
    print("-" * 80)

    comb_speedups = []
    for tc in all_tokens:
        s_runs = strict.get(tc, [])
        r_runs = relaxed.get(tc, [])
        s_vals = [r['comb_avg_us'] for r in s_runs if 'comb_avg_us' in r]
        r_vals = [r['comb_avg_us'] for r in r_runs if 'comb_avg_us' in r]
        if not s_vals or not r_vals:
            continue
        s_v = avg(s_vals)
        r_v = avg(r_vals)
        spd = s_v / r_v if r_v > 0 else 0
        comb_speedups.append(spd)
        print(f"{tc:>8} | {s_v:>12.2f} | {r_v:>12.2f} | {spd:>9.3f}x")

    if comb_speedups:
        print("-" * 80)
        print(f"{'GeoMean':>8} | {'':>12} | {'':>12} | {geo_mean(comb_speedups):>9.3f}x")

    # ---- Hook mode send/recv split ----
    print(f"\n{'='*100}")
    print("Hook Mode: Dispatch Send/Recv Split")
    print(f"{'='*100}")
    print(f"{'Tokens':>8} | {'Strict Send':>12} {'Strict Recv':>12} | {'Relaxed Send':>13} {'Relaxed Recv':>13}")
    print("-" * 100)

    for tc in all_tokens:
        s_runs = strict.get(tc, [])
        r_runs = relaxed.get(tc, [])
        s_send = [r['disp_send_us'] for r in s_runs if 'disp_send_us' in r]
        s_recv = [r['disp_recv_us'] for r in s_runs if 'disp_recv_us' in r]
        r_send = [r['disp_send_us'] for r in r_runs if 'disp_send_us' in r]
        r_recv = [r['disp_recv_us'] for r in r_runs if 'disp_recv_us' in r]
        if not s_send or not r_send:
            continue
        print(f"{tc:>8} | {avg(s_send):>12.2f} {avg(s_recv):>12.2f} | "
              f"{avg(r_send):>13.2f} {avg(r_recv):>13.2f}")

    print(f"\n{'='*120}")
    print("Done.")


if __name__ == '__main__':
    main()
