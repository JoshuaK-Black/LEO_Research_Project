#!/usr/bin/env python3
import os
import re
import math
import json
import argparse
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use('Agg')  # non-interactive backend
    import matplotlib.pyplot as plt
    try:
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    except Exception:
        inset_axes = None
except Exception as e:
    matplotlib = None
    plt = None
    inset_axes = None


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def parse_metrics_from_text(path):
    # Returns (metrics: dict, flags: dict)
    m = {}
    flags = {}
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = [ln.strip() for ln in f]
    except Exception:
        return m, flags

    # FLAGS block
    idx = 0
    if lines and lines[0].startswith('# FLAGS'):
        idx = 1
        while idx < len(lines) and lines[idx]:
            kv = lines[idx]
            if '=' in kv:
                k, v = kv.split('=', 1)
                flags[k.strip()] = v.strip()
            idx += 1

    # DATA METRICS block parsing
    for ln in lines:
        if ln.startswith('TxPkts:'):
            # Example: TxPkts: 1320000  RxPkts: 1303434  Lost: 16566  PDR: 0.98745  AvgDelay(s): 0.0725840  AvgThroughput(Mbps/flow): 0.90218
            # Use regex for robustness
            tx = re.search(r'TxPkts:\s*(\d+)', ln)
            rx = re.search(r'RxPkts:\s*(\d+)', ln)
            lost = re.search(r'Lost:\s*(\d+)', ln)
            pdr = re.search(r'PDR:\s*([0-9.]+)', ln)
            avgd = re.search(r'AvgDelay\(s\):\s*([0-9.]+)', ln)
            thr = re.search(r'AvgThroughput\(Mbps/flow\):\s*([0-9.]+)', ln)
            if tx: m['TxPkts'] = int(tx.group(1))
            if rx: m['RxPkts'] = int(rx.group(1))
            if lost: m['Lost'] = int(lost.group(1))
            if pdr: m['PDR'] = float(pdr.group(1))
            if avgd: m['AvgDelay_s'] = float(avgd.group(1))
            if thr: m['AvgThroughput_Mbps_per_flow'] = float(thr.group(1))
        if ln.startswith('CtrlBytesTx:') or 'CtrlAvgRate(Mbps):' in ln:
            cbytes = re.search(r'CtrlBytesTx:\s*(\d+)', ln)
            crate = re.search(r'CtrlAvgRate\(Mbps\):\s*([0-9.]+)', ln)
            if cbytes: m['CtrlBytesTx'] = int(cbytes.group(1))
            if crate: m['CtrlAvgRate_Mbps'] = float(crate.group(1))
        if ln.startswith('[RL] overrides_applied='):
            val = ln.split('=', 1)[-1].strip()
            if val.isdigit():
                m['RL_overrides_applied'] = int(val)
        if ln.startswith('[RL] pkts_forwarded_via_RL='):
            val = ln.split('=', 1)[-1].strip()
            if val.isdigit():
                m['RL_pkts_forwarded'] = int(val)

    return m, flags


def mean_std(values):
    if not values:
        return None, None
    n = len(values)
    mu = sum(values) / n
    if n == 1:
        return mu, 0.0
    var = sum((x - mu) ** 2 for x in values) / (n - 1)
    return mu, math.sqrt(var)


def aggregate_policy(policy_name, sweep_dir, out_dir, pattern, accept=lambda meta: True):
    # pattern function: returns (group_key tuple, meta dict) from filename
    ensure_dir(out_dir)

    groups = defaultdict(list)  # key -> list of (metrics, meta)

    for fn in os.listdir(sweep_dir):
        path = os.path.join(sweep_dir, fn)
        if not os.path.isfile(path):
            continue
        key_meta = pattern(fn)
        if not key_meta:
            continue
        key, meta = key_meta
        if not accept(meta):
            continue
        # prefer .txt summaries; skip .log when both exist
        if fn.endswith('.log'):
            txt = fn[:-4] + '.txt'
            if os.path.exists(os.path.join(sweep_dir, txt)):
                continue
        metrics, flags = parse_metrics_from_text(path)
        if not metrics:
            continue
        meta = {**meta, **flags}
        groups[key].append((metrics, meta, path))

    # Aggregate per key
    rows = []
    for key, items in sorted(groups.items(), key=lambda kv: kv[0]):
        # collect metric arrays
        acc = defaultdict(list)
        sample_meta = {}
        for metrics, meta, _ in items:
            sample_meta = meta
            for k, v in metrics.items():
                acc[k].append(v)
        row = {'group': key, 'runs': len(items)}
        # include identifiers like pps and variant if present
        if 'pps' in sample_meta:
            row['pps'] = int(sample_meta['pps'])
        if 'variant' in sample_meta:
            row['variant'] = sample_meta['variant']
        for met, vals in acc.items():
            mu, sd = mean_std(vals)
            if mu is None:
                continue
            row[f'{met}_mean'] = mu
            row[f'{met}_std'] = sd
        rows.append(row)

    # Write CSV
    if rows:
        # determine columns
        cols = ['group']
        if any('pps' in r for r in rows):
            cols.append('pps')
        if any('variant' in r for r in rows):
            cols.append('variant')
        cols.append('runs')
        metric_keys = sorted({k.rsplit('_', 1)[0] for r in rows for k in r.keys() if k.endswith('_mean')})
        for base in metric_keys:
            cols.append(f'{base}_mean')
            cols.append(f'{base}_std')

        out_csv = os.path.join(out_dir, f'{policy_name.lower()}_summary_by_pps.csv')
        with open(out_csv, 'w', encoding='utf-8') as f:
            f.write(','.join(cols) + '\n')
            for r in rows:
                f.write(','.join(str(r.get(c, '')) for c in cols) + '\n')

    return rows


def plot_policy(policy_name, rows, out_dir, title_note="", zoom_inset=False, pdr_limits=None):
    if not matplotlib or not rows:
        return []
    # If variant dimension exists (RL), plot per-variant series; otherwise single
    has_variant = any('variant' in r for r in rows)
    # metrics to plot
    # Allow overriding PDR y-axis limits
    pdr_ylim = pdr_limits if pdr_limits else (0.6, 1.02)
    plots = [
        ('PDR_mean', 'Packet Delivery Ratio (PDR)', pdr_ylim, 'PDR'),
        ('AvgDelay_s_mean', 'Average Delay (s)', None, 'Average Delay'),
        ('AvgThroughput_Mbps_per_flow_mean', 'Average Throughput (Mbps/flow)', None, 'Average Throughput'),
    ]

    generated = []
    for metric_key, ylab, ylim, short in plots:
        plt.figure(figsize=(7, 4.2))
        if has_variant:
            variants = sorted({r.get('variant', '') for r in rows})
            for v in variants:
                sub = [r for r in rows if r.get('variant', '') == v]
                sub = sorted(sub, key=lambda r: r.get('pps', 0))
                x = [r.get('pps', None) for r in sub]
                y = [r.get(metric_key, None) for r in sub]
                yerr = [r.get(metric_key.replace('_mean', '_std'), 0.0) for r in sub]
                if any(val is not None for val in y):
                    # Friendly legend for RL variant 'a'
                    if policy_name == 'RL' and v == 'a':
                        label = 'RL agent'
                    else:
                        label = f'variant {v}' if v else policy_name
                    plt.errorbar(x, y, yerr=yerr, marker='o', capsize=3, label=label)
        else:
            sub = sorted(rows, key=lambda r: r.get('pps', 0))
            x = [r.get('pps', None) for r in sub]
            y = [r.get(metric_key, None) for r in sub]
            yerr = [r.get(metric_key.replace('_mean', '_std'), 0.0) for r in sub]
            if any(val is not None for val in y):
                plt.errorbar(x, y, yerr=yerr, marker='o', capsize=3, label=policy_name)
        plt.xlabel('Packets per second (PPS)')
        plt.ylabel(ylab)
        # Clean, simple title (no em-dash)
        if title_note:
            title = f"{policy_name} ({title_note}) {short} vs PPS"
        else:
            title = f"{policy_name} {short} vs PPS"
        plt.title(title)
        if ylim:
            plt.ylim(*ylim)
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Optional zoomed inset to highlight tiny whiskers
        if zoom_inset and inset_axes is not None:
            try:
                ax = plt.gca()
                iax = inset_axes(ax, width="40%", height="40%", loc='lower right', borderpad=1)
                # Re-plot series into inset with tight y-limits
                series = []
                if has_variant:
                    for v in variants:
                        sub2 = [r for r in rows if r.get('variant', '') == v]
                        sub2 = sorted(sub2, key=lambda r: r.get('pps', 0))
                        x2 = [r.get('pps', None) for r in sub2]
                        y2 = [r.get(metric_key, None) for r in sub2]
                        e2 = [r.get(metric_key.replace('_mean', '_std'), 0.0) for r in sub2]
                        iax.errorbar(x2, y2, yerr=e2, marker='o', capsize=2)
                        for yi, ei in zip(y2, e2):
                            if yi is not None:
                                series.append((yi, ei))
                else:
                    x2, y2, e2 = x, y, yerr
                    iax.errorbar(x2, y2, yerr=e2, marker='o', capsize=2)
                    for yi, ei in zip(y2, e2):
                        if yi is not None:
                            series.append((yi, ei))
                if series:
                    ymin = min(max(0.0, y - e) for y, e in series)
                    ymax = max(y + e for y, e in series)
                    if ymin == ymax:
                        pad = 0.001 if ymax == 0 else 0.005 * abs(ymax)
                        ymin, ymax = ymin - pad, ymax + pad
                    iax.set_ylim(ymin, ymax)
                iax.set_xticks([])
                iax.set_yticks([])
                iax.grid(True, alpha=0.3)
                iax.set_title('zoom', fontsize=9)
            except Exception:
                pass
        out_png = os.path.join(out_dir, f'{policy_name.lower()}_{metric_key.replace("_mean","")}.png')
        plt.tight_layout()
        plt.savefig(out_png, dpi=140)
        plt.close()
        generated.append(out_png)
    return generated


def plot_comparative(policies_rows, out_dir, title_note="", pdr_limits=None):
    if not matplotlib:
        return []
    ensure_dir(out_dir)
    # metrics to compare
    pdr_ylim = pdr_limits if pdr_limits else (0.6, 1.02)
    comps = [
        ('PDR_mean', 'Packet Delivery Ratio (PDR)', pdr_ylim, 'PDR'),
        ('AvgDelay_s_mean', 'Average Delay (s)', None, 'Average Delay'),
        ('AvgThroughput_Mbps_per_flow_mean', 'Average Throughput (Mbps/flow)', None, 'Average Throughput'),
    ]
    generated = []
    # Determine common PPS set to align series
    for metric_key, ylab, ylim, short in comps:
        plt.figure(figsize=(7, 4.2))
        for policy_name, rows in policies_rows.items():
            # if RL has per-variant rows, average across variants per PPS
            # build map pps -> list of metric values
            per_pps = defaultdict(list)
            for r in rows:
                pps = r.get('pps', None)
                val = r.get(metric_key, None)
                if pps is not None and val is not None:
                    per_pps[pps].append(val)
            xs = sorted(per_pps.keys())
            ys = [sum(per_pps[p])/len(per_pps[p]) if per_pps[p] else None for p in xs]
            if xs and any(v is not None for v in ys):
                plt.plot(xs, ys, marker='o', label=policy_name)
        plt.xlabel('Packets per second (PPS)')
        plt.ylabel(ylab)
        if title_note:
            title = f"RL vs OSPF vs QR ({title_note}) {short} vs PPS"
        else:
            title = f"RL vs OSPF vs QR {short} vs PPS"
        plt.title(title)
        if ylim:
            plt.ylim(*ylim)
        plt.grid(True, alpha=0.3)
        plt.legend()
        out_png = os.path.join(out_dir, f'comparative_{metric_key.replace("_mean","")}.png')
        plt.tight_layout()
        plt.savefig(out_png, dpi=140)
        plt.close()
        generated.append(out_png)
    return generated


def main():
    ap = argparse.ArgumentParser(description='Aggregate ns-3 sweeps and plot averages')
    ap.add_argument('--round-min', type=int, default=1, help='minimum round to include (default: 1)')
    ap.add_argument('--round-max', type=int, default=5, help='maximum round to include (default: 5)')
    ap.add_argument('--rl-variant', type=str, default='a', choices=['a','both'], help='RL variant to include (default: a)')
    ap.add_argument('--suffix', type=str, default='', help='optional suffix for output dirs (e.g., _r1to3)')
    ap.add_argument('--title-note', type=str, default='', help='optional note to append in plot titles (default: none)')
    ap.add_argument('--pdr-ymin', type=float, default=None, help='override PDR y-axis minimum (e.g., 0.7)')
    ap.add_argument('--pdr-ymax', type=float, default=None, help='override PDR y-axis maximum (default: 1.02)')
    ap.add_argument('--zoom-inset', action='store_true', help='add inset zoom box to per-policy plots')
    args = ap.parse_args()
    # Directories
    rl_dir = os.path.join(ROOT, 'results', 'RL', 'RL_Dup_Sweep_5rounds')
    ospf_dir = os.path.join(ROOT, 'results', 'OSPF', 'Big_Sweep')
    qr_dir = os.path.join(ROOT, 'results', 'QR', 'Big_Sweep')

    suffix = args.suffix
    rl_out = os.path.join(ROOT, 'results', 'RL', 'average' + suffix)
    ospf_out = os.path.join(ROOT, 'results', 'OSPF', 'average' + suffix)
    qr_out = os.path.join(ROOT, 'results', 'QR', 'average' + suffix)
    ensure_dir(rl_out)
    ensure_dir(ospf_out)
    ensure_dir(qr_out)

    # Filename parsers
    def rl_pattern(fn):
        # env_pps65_r3_a.txt or .log
        m = re.match(r'env_pps(?P<pps>\d+)_r(?P<r>\d+)_(?P<variant>[ab])\.(?:txt|log)$', fn)
        if not m:
            return None
        pps = int(m.group('pps'))
        rnd = int(m.group('r'))
        var = m.group('variant')
        return (('RL', pps, var), {'pps': pps, 'round': rnd, 'variant': var})

    def ospf_pattern(fn):
        # ospf_pps55_r1.txt
        m = re.match(r'ospf_pps(?P<pps>\d+)_r(?P<r>\d+)\.(?:txt|log)$', fn)
        if not m:
            return None
        pps = int(m.group('pps'))
        rnd = int(m.group('r'))
        return (('OSPF', pps), {'pps': pps, 'round': rnd})

    def qr_pattern(fn):
        # qr_pps55_r1.txt
        m = re.match(r'qr_pps(?P<pps>\d+)_r(?P<r>\d+)\.(?:txt|log)$', fn)
        if not m:
            return None
        pps = int(m.group('pps'))
        rnd = int(m.group('r'))
        return (('QR', pps), {'pps': pps, 'round': rnd})

    def rl_accept(meta):
        if not (args.round_min <= meta.get('round', 0) <= args.round_max):
            return False
        if args.rl_variant == 'a' and meta.get('variant') != 'a':
            return False
        return True

    def by_round(meta):
        return args.round_min <= meta.get('round', 0) <= args.round_max

    rl_rows = aggregate_policy('RL', rl_dir, rl_out, rl_pattern, rl_accept)
    ospf_rows = aggregate_policy('OSPF', ospf_dir, ospf_out, ospf_pattern, by_round)
    qr_rows = aggregate_policy('QR', qr_dir, qr_out, qr_pattern, by_round)

    # Plot per-policy
    title_note = args.title_note.strip() if args.title_note else ""
    # Optional PDR y-limits
    pdr_limits = None
    if args.pdr_ymin is not None or args.pdr_ymax is not None:
        ymin = args.pdr_ymin if args.pdr_ymin is not None else 0.6
        ymax = args.pdr_ymax if args.pdr_ymax is not None else 1.02
        pdr_limits = (ymin, ymax)
    # Zoom inset toggle
    zoom_flag = bool(args.zoom_inset)
    rl_plots = plot_policy('RL', rl_rows, rl_out, title_note, zoom_inset=zoom_flag, pdr_limits=pdr_limits)
    ospf_plots = plot_policy('OSPF', ospf_rows, ospf_out, title_note, zoom_inset=zoom_flag, pdr_limits=pdr_limits)
    qr_plots = plot_policy('QR', qr_rows, qr_out, title_note, zoom_inset=zoom_flag, pdr_limits=pdr_limits)

    # Comparative plots
    final_out = os.path.join(ROOT, 'results', 'Final_Results' + suffix)
    ensure_dir(final_out)
    comp_plots = plot_comparative({'RL': rl_rows, 'OSPF': ospf_rows, 'QR': qr_rows}, final_out, title_note, pdr_limits=pdr_limits)

    # Build combined comparison CSV (per-PPS, average RL across variants)
    def build_map(rows, key):
        m = defaultdict(list)
        for r in rows:
            pps = r.get('pps', None)
            val = r.get(key, None)
            if pps is not None and val is not None:
                m[pps].append(val)
        # average across variants if needed
        return {pps: (sum(vals)/len(vals) if vals else None) for pps, vals in m.items()}

    pdr_rl = build_map(rl_rows, 'PDR_mean')
    pdr_ospf = build_map(ospf_rows, 'PDR_mean')
    pdr_qr = build_map(qr_rows, 'PDR_mean')
    dly_rl = build_map(rl_rows, 'AvgDelay_s_mean')
    dly_ospf = build_map(ospf_rows, 'AvgDelay_s_mean')
    dly_qr = build_map(qr_rows, 'AvgDelay_s_mean')
    thr_rl = build_map(rl_rows, 'AvgThroughput_Mbps_per_flow_mean')
    thr_ospf = build_map(ospf_rows, 'AvgThroughput_Mbps_per_flow_mean')
    thr_qr = build_map(qr_rows, 'AvgThroughput_Mbps_per_flow_mean')

    pps_all = sorted(set(pdr_rl) | set(pdr_ospf) | set(pdr_qr))
    comp_csv = os.path.join(final_out, 'comparison_combined.csv')
    with open(comp_csv, 'w') as f:
        f.write('pps,PDR_RL,PDR_OSPF,PDR_QR,Delay_RL,Delay_OSPF,Delay_QR,Thr_RL,Thr_OSPF,Thr_QR\n')
        for pps in pps_all:
            f.write(','.join(str(x) for x in [
                pps,
                pdr_rl.get(pps, ''), pdr_ospf.get(pps, ''), pdr_qr.get(pps, ''),
                dly_rl.get(pps, ''), dly_ospf.get(pps, ''), dly_qr.get(pps, ''),
                thr_rl.get(pps, ''), thr_ospf.get(pps, ''), thr_qr.get(pps, ''),
            ]) + '\n')

    # Write manifest json with file paths
    manifest = {
        'rl_csv': os.path.join(rl_out, 'rl_summary_by_pps.csv'),
        'ospf_csv': os.path.join(ospf_out, 'ospf_summary_by_pps.csv'),
        'qr_csv': os.path.join(qr_out, 'qr_summary_by_pps.csv'),
        'rl_plots': rl_plots,
        'ospf_plots': ospf_plots,
        'qr_plots': qr_plots,
        'comparative_plots': comp_plots,
        'combined_csv': comp_csv,
    }
    with open(os.path.join(final_out, 'average_manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)

    print('[DONE] Aggregation complete.')
    print('RL CSV:', manifest['rl_csv'])
    print('OSPF CSV:', manifest['ospf_csv'])
    print('QR CSV:', manifest['qr_csv'])
    print('Comparative plots:', *manifest['comparative_plots'], sep='\n  - ')


if __name__ == '__main__':
    main()
